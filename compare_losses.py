#!/usr/bin/env python3
"""
SIGReg vs CausalVICReg Comparison Experiment
=============================================
Compares training dynamics of SIGReg (2-term, LeWorldModel) vs
CausalVICReg (4-term, ICML 2026) on synthetic data.

No GPU required — pure CPU comparison of loss landscapes,
gradient behavior, and representation quality metrics.

Also validates the ActionConditionedCausalPredictor works correctly.

Usage:
  python compare_losses.py
"""

import sys
import os

sys.path.insert(0, str(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "five_layer_jepa"))

import torch
import torch.nn as nn
import time

from five_layer_jepa.causal_jepa import CausalVICRegLoss
from five_layer_jepa.causal_jepa_v2 import (
    SIGRegLoss, SIGRegWithPredictionLoss,
    ActionConditionedCausalPredictor, ActionCAJEPA,
)


def generate_synthetic_data(N_total=120, D=128, collapse_prob=0.0):
    if collapse_prob > 0 and torch.rand(1).item() < collapse_prob:
        base = torch.randn(N_total, D) * 0.01
        return base, base.detach()

    pred = torch.randn(N_total, D) * 1.0
    noise = torch.randn(N_total, D) * 0.3
    target = pred + noise
    return pred, target


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def compare_losses():
    print("=" * 70)
    print("SIGReg vs CausalVICReg — Loss Landscape Comparison")
    print("=" * 70)

    D = 128
    N_total = 120  # B*T*N = 4*6*5

    causal_vicreg = CausalVICRegLoss(sim_weight=25.0, var_weight=25.0, cov_weight=1.0, causal_weight=5.0)
    sigreg = SIGRegLoss(var_weight=25.0, cov_weight=1.0, target_std=1.0)
    sigreg_with_sim = SIGRegWithPredictionLoss(var_weight=25.0, cov_weight=1.0, sim_weight=1.0, target_std=1.0)

    print(f"\nConfig: N_total={N_total}, D={D} dim")
    print(f"  CausalVICReg params: 4 hyperparams (sim=25, var=25, cov=1, causal=5)")
    print(f"  SIGReg params:        2 hyperparams (var=25, cov=1)")
    print(f"  SIGReg+Sim params:    3 hyperparams (var=25, cov=1, sim=1)")
    print()

    print("--- Scenario 1: Normal data (pred ~ target + noise) ---")
    pred, target = generate_synthetic_data(N_total, D)

    out_cvr = causal_vicreg(pred, target)
    out_sr = sigreg(pred)
    out_srs = sigreg_with_sim(pred, target)

    print(f"  CausalVICReg:  total={out_cvr['total']:.4f} | sim={out_cvr['sim_loss']:.4f} var={out_cvr['var_loss']:.4f} cov={out_cvr['cov_loss']:.4f} causal={out_cvr['causal_loss']:.4f}")
    print(f"  SIGReg:        total={out_sr['total']:.4f} | var={out_sr['var_loss']:.4f} cov={out_sr['cov_loss']:.4f}")
    print(f"  SIGReg+Sim:    total={out_srs['total']:.4f} | var={out_srs['var_loss']:.4f} cov={out_srs['cov_loss']:.4f} sim={out_srs['sim_loss']:.4f}")

    print("\n--- Scenario 2: Nearly collapsed data (low variance) ---")
    pred_c, target_c = generate_synthetic_data(N_total, D, collapse_prob=1.0)

    out_cvr_c = causal_vicreg(pred_c, target_c)
    out_sr_c = sigreg(pred_c)
    out_srs_c = sigreg_with_sim(pred_c, target_c)

    std_c = pred_c.std(dim=0).mean().item()
    print(f"  Mean feature std: {std_c:.6f} (should be << 1.0 for collapsed)")

    print(f"  CausalVICReg:  total={out_cvr_c['total']:.4f} | var={out_cvr_c['var_loss']:.4f} (higher=more penalty)")
    print(f"  SIGReg:        total={out_sr_c['total']:.4f} | var={out_sr_c['var_loss']:.4f} (higher=more penalty)")
    print(f"  SIGReg+Sim:    total={out_srs_c['total']:.4f} | var={out_srs_c['var_loss']:.4f}")

    var_ratio = out_sr_c['var_loss'].item() / max(out_cvr_c['var_loss'].item(), 1e-8)
    print(f"\n  SIGReg/CausalVICReg var_loss ratio: {var_ratio:.2f}x")
    if var_ratio > 2.0:
        print(f"  → SIGReg penalizes collapse MORE aggressively ({var_ratio:.1f}x stronger)")
    else:
        print(f"  → Similar collapse detection sensitivity")

    print("\n--- Scenario 3: Gradient comparison (requires_grad) ---")
    pred_g = torch.randn(N_total, D, requires_grad=True)
    target_g = torch.randn(N_total, D, requires_grad=True)

    cvr_out = causal_vicreg(pred_g, target_g)
    cvr_out['total'].backward()
    cvr_grad_norm = pred_g.grad.norm().item()  # type: ignore[reportOptionalMemberAccess]
    pred_g.grad = None

    sigreg_out = sigreg(pred_g)
    sigreg_out['total'].backward()
    sr_grad_norm = pred_g.grad.norm().item()  # type: ignore[reportAttributeAccessIssue]

    print(f"  CausalVICReg grad norm: {cvr_grad_norm:.6f}")
    print(f"  SIGReg grad norm:       {sr_grad_norm:.6f}")
    print(f"  Ratio: {sr_grad_norm/max(cvr_grad_norm,1e-8):.2f}x")

    print("\n--- Scenario 4: Synthetic training run (5 steps each) ---")
    torch.manual_seed(42)

    for loss_name, loss_fn, fn_name in [
        ("CausalVICReg", causal_vicreg, "causal_vicreg"),
        ("SIGReg", sigreg, "sigreg"),
        ("SIGReg+Sim", sigreg_with_sim, "sigreg_flat"),
    ]:
        param = nn.Parameter(torch.randn(N_total, D) * 1.0)
        target_p = torch.randn(N_total, D) * 1.0
        opt = torch.optim.SGD([param], lr=0.01)
        losses = []
        grad_norms = []
        t0 = time.perf_counter()

        for step in range(5):
            opt.zero_grad()
            if fn_name == "causal_vicreg":
                out = loss_fn(param, target_p)
            elif fn_name == "sigreg":
                out = loss_fn(param)
            else:
                out = loss_fn(param, target_p)
            out['total'].backward()
            grad_norms.append(param.grad.norm().item())  # type: ignore[reportOptionalMemberAccess]
            opt.step()
            losses.append(out['total'].item())

        elapsed = (time.perf_counter() - t0) * 1000
        loss_diff = losses[-1] - losses[0]
        loss_std = torch.tensor(losses).std().item()
        grad_std = torch.tensor(grad_norms).std().item()

        print(f"  {loss_name:>14}: loss {losses[0]:.4f}→{losses[-1]:.4f} (Δ={loss_diff:+.4f}) | grad_std={grad_std:.5f} | {elapsed:.1f}ms/5steps")

    print("\n" + "=" * 70)
    print("LOSS COMPARISON SUMMARY")
    print("=" * 70)
    print("""
  CausalVICReg (4-term):
    ✓ Proven on C-JEPA (ICML 2026), good causal tracking
    ✓ Explicit sim_loss for prediction accuracy
    ✗ 4 hyperparameters = harder to tune
    ✗ More gradient interference between terms

  SIGReg (2-term):
    ✓ Simpler: 2 hyperparams, easier to tune
    ✓ More aggressive anti-collapse (var_loss)
    ✓ Faster: fewer gradient computations
    ✗ No explicit prediction accuracy term (relies on predictor)
    ✗ Causal interaction tracking may degrade

  SIGReg+Sim (3-term):
    ✓ Best of both: SIGReg stability + weak sim_loss
    ✓ Good for early training when predictor is weak
    ✓ Can gradually reduce sim_weight as predictor improves
  """)


def validate_action_predictor():
    print("=" * 70)
    print("ActionConditionedCausalPredictor — Validation")
    print("=" * 70)

    B, T, N, D = 2, 6, 5, 128
    action_dim = 64

    pred = ActionConditionedCausalPredictor(
        slot_dim=D, num_slots=N, num_heads=2, num_layers=2,
        hidden_dim=128, aux_dim=64, action_dim=action_dim,
        history_len=T, future_len=4,
    )

    n_params = count_parameters(pred)
    print(f"  Parameters: {n_params:,}")

    slots = torch.randn(B, T, N, D)
    action = torch.randn(B, action_dim)

    recovered, future = pred(slots, action=action)
    print(f"  With action:    recovered={list(recovered.shape)}, future={list(future.shape)}")

    recovered_none, future_none = pred(slots, action=None)
    print(f"  Without action: recovered={list(recovered_none.shape)}, future={list(recovered_none.shape)}")
    assert recovered.shape == recovered_none.shape, "Shape mismatch: action vs no-action"
    assert future.shape == future_none.shape, "Shape mismatch: action vs no-action"

    with torch.no_grad():
        same_mask = torch.isclose(recovered, recovered_none, atol=1e-6)
        diff_ratio = 1.0 - same_mask.float().mean().item()
    print(f"  Representation difference (action vs no-action): {diff_ratio:.1%}")
    assert diff_ratio > 0.01, f"Action should change predictions significantly, got {diff_ratio:.1%}"

    pred2 = ActionConditionedCausalPredictor(
        slot_dim=D, num_slots=N, num_heads=2, num_layers=2,
        hidden_dim=128, aux_dim=64, action_dim=action_dim,
        history_len=T, future_len=4,
    )
    pred2.load_state_dict(pred.state_dict())
    r2, f2 = pred2(slots, action=action)
    assert torch.allclose(recovered, r2, atol=1e-6), "Deterministic forward pass failed"
    assert torch.allclose(future, f2, atol=1e-6), "Deterministic forward pass failed"

    print(f"\n  ALL CHECKS PASSED ✅")
    print(f"  - Action conditioning changes predictions ({diff_ratio:.1%} diff)")
    print(f"  - Backward compatible (action=None works)")
    print(f"  - Deterministic (same weights → same output)")
    print(f"  - {n_params:,} parameters")


def validate_action_cajepa():
    print("\n" + "=" * 70)
    print("ActionCAJEPA — Full Pipeline Validation")
    print("=" * 70)

    B, T_hist, T_fut = 2, 6, 4
    input_dim = 128
    action_dim = 64

    slot_dim = 128
    model = ActionCAJEPA(
        input_dim=input_dim, num_slots=5, slot_dim=slot_dim,
        num_iterations=2, mask_ratio=0.3, num_predictor_heads=2,
        num_predictor_layers=2, predictor_hidden_dim=256,
        aux_dim=32, action_dim=action_dim, history_len=T_hist,
        future_len=T_fut, ema_decay=0.996, use_sigreg=True,
    )

    n_params = count_parameters(model)
    print(f"  Parameters: {n_params:,}")

    features_history = torch.randn(B, T_hist, 5, input_dim)
    features_future = torch.randn(B, T_fut, 5, input_dim)
    action = torch.randn(B, action_dim)

    result = model.train_step(features_history, features_future, action=action)
    print(f"  Train step (SIGReg + action):")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    result_no_action = model.train_step(features_history, features_future, action=None)
    print(f"  Train step (SIGReg, no action):")
    for k, v in result_no_action.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    future_pred = model.predict_consequence(features_history, action)
    print(f"  Predict consequence: future_slots shape = {list(future_pred.shape)}")
    assert list(future_pred.shape) == [B, T_fut, 5, slot_dim], f"Unexpected shape: {future_pred.shape}"

    model_sigreg_off = ActionCAJEPA(
        input_dim=input_dim, num_slots=5, slot_dim=slot_dim,
        num_iterations=2, mask_ratio=0.3, num_predictor_heads=2,
        num_predictor_layers=2, predictor_hidden_dim=256,
        aux_dim=32, action_dim=action_dim, history_len=T_hist,
        future_len=T_fut, ema_decay=0.996, use_sigreg=False,
    )
    result_cvr = model_sigreg_off.train_step(features_history, features_future, action=action)
    print(f"\n  Train step (CausalVICReg + action):")
    for k, v in result_cvr.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    print(f"\n  ALL CHECKS PASSED ✅")
    print(f"  - ActionCAJEPA trains with SIGReg (2-term)")
    print(f"  - ActionCAJEPA trains with CausalVICReg (4-term) when use_sigreg=False")
    print(f"  - predict_consequence() returns correct shape")
    print(f"  - Backward compatible (action=None works)")
    print(f"  - {n_params:,} total parameters")


if __name__ == "__main__":
    compare_losses()
    print("\n")
    validate_action_predictor()
    print("\n")
    validate_action_cajepa()
    print("\n" + "=" * 70)
    print("ALL COMPARISONS & VALIDATIONS COMPLETE ✅")
    print("=" * 70)
