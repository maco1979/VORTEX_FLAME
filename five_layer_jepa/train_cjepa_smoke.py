#!/usr/bin/env python3
"""Short smoke-run trainer for CJEPAManager using synthetic data.

Usage: python five_layer_jepa/train_cjepa_smoke.py --steps 20 --batch 2

This script performs a short, safe training run on synthetic data to
verify the C-JEPA training loop and exercise GPU usage. It saves per-layer
state_dicts to `five_layer_jepa/outputs/`.
"""

import argparse
import os
import torch
from torch.optim import Adam

from five_layer_jepa.causal_jepa import CJEPAManager


def make_synthetic_for_layer(layer, batch_size, device, n_input=8):
    ctx = layer.context_encoder
    # feature_proj[0] is Linear(input_dim, feature_proj_dim)
    try:
        input_dim = ctx.feature_proj[0].in_features
    except Exception:
        # fallback: try to infer from a sample if structure differs
        input_dim = getattr(layer, "input_dim", 512)

    history_len = getattr(layer, "history_len", 6)
    future_len = getattr(layer, "future_len", 4)

    features_history = torch.randn(batch_size, history_len, n_input, input_dim, device=device)
    features_future = torch.randn(batch_size, future_len, n_input, input_dim, device=device)
    auxiliaries = None
    return features_history, features_future, auxiliaries


def collect_trainable_params(manager):
    params = []
    for layer in manager.layers.values():
        params += [p for p in layer.parameters() if p.requires_grad]
    return params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", default="five_layer_jepa/outputs")
    args = parser.parse_args()

    device = torch.device(args.device)
    mgr = CJEPAManager()
    mgr.set_training_phase(2)

    params = collect_trainable_params(mgr)
    if len(params) == 0:
        raise RuntimeError("No trainable parameters found in CJEPAManager layers")

    optimizer = Adam(params, lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)

    mgr_device_layers = {m: layer.to(device) for m, layer in mgr.layers.items()}

    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        total_loss = None
        losses = {}

        for modality, layer in mgr_device_layers.items():
            features_history, features_future, auxiliaries = make_synthetic_for_layer(
                layer, args.batch, device
            )

            # compute target slots without gradient
            with torch.no_grad():
                target_slots_history, _ = layer.target_encoder(features_history)
                target_slots_future, _ = layer.target_encoder(features_future)

            # forward pass (compute gradients)
            context_slots_history, _ = layer.context_encoder(features_history)
            masked_slots, slot_mask, masked_indices = layer.masker(
                context_slots_history
            )

            recovered_slots, predicted_future_slots = layer.predictor(
                masked_slots, slot_mask=slot_mask, auxiliaries=auxiliaries
            )

            recovery_loss_dict = layer.loss_fn(
                recovered_slots.reshape(-1, layer.slot_dim),
                target_slots_history.reshape(-1, layer.slot_dim),
                slot_mask=slot_mask.reshape(-1) if slot_mask is not None else None,
            )

            T_pred = predicted_future_slots.shape[1]
            T_target = target_slots_future.shape[1]
            T_match = min(T_pred, T_target)
            forward_loss_dict = layer.loss_fn(
                predicted_future_slots[:, :T_match].reshape(-1, layer.slot_dim),
                target_slots_future[:, :T_match].reshape(-1, layer.slot_dim),
            )

            layer_loss = recovery_loss_dict["total"] + 0.5 * forward_loss_dict["total"]

            if total_loss is None:
                total_loss = layer_loss
            else:
                total_loss = total_loss + layer_loss

            losses[modality.value] = {
                "layer_loss": layer_loss.item(),
                "recovery": recovery_loss_dict["total"].item(),
                "forward": forward_loss_dict["total"].item(),
            }

        total_loss.backward()
        optimizer.step()

        # update EMA target encoders
        for layer in mgr.layers.values():
            layer.update_ema()

        print(
            f"step {step}/{args.steps} total_loss {total_loss.item():.6f} "
            + " ".join([f"{k}:{v['layer_loss']:.4f}" for k, v in losses.items()])
        )

    # save per-layer checkpoints
    for modality, layer in mgr.layers.items():
        torch.save(layer.state_dict(), os.path.join(args.save_dir, f"{modality.value}_cjepa.pth"))

    print("Finished smoke-run training. Models saved to:", args.save_dir)


if __name__ == "__main__":
    main()
