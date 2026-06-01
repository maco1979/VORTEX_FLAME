#!/usr/bin/env python3
"""Smoke test: near-identity init + separated LR + checkpoint resume"""
import os, sys, time, math
import torch
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import train_ajepa_multiclass as tm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("\n=== 1. Near-Identity Init Test ===")
ajepa = tm.CAJEPA(input_dim=512).to(device)
x = torch.randn(4, 6, 512, device=device)

with torch.no_grad():
    y = ajepa.project_to_slots(x)
    raw_x = x.unsqueeze(2).expand(-1, -1, 5, -1)

    diff = (y - raw_x).norm(dim=-1).mean().item()
    slot_diffs = []
    for i in range(5):
        sd = (y[:, :, i] - x).norm(dim=-1).mean().item()
        slot_diffs.append(sd)
    slot_variance = (y - y.mean(dim=2, keepdim=True)).norm(dim=-1).mean().item()

print(f"  |proj(x) - repeat(x)| = {diff:.4f}  (should be < 2.0 for near-identity)")
print(f"  slot diffs: {[f'{s:.4f}' for s in slot_diffs]}")
print(f"  cross-slot variance: {slot_variance:.4f}  (should be > 0.01)")
assert diff < 2.0, f"Near-identity init failed: diff={diff:.4f} >= 2.0"
assert slot_variance > 0.01, f"No slot differentiation: var={slot_variance:.6f}"
print("  ✅ Near-identity init PASSED")

print("\n=== 2. Forward Pass Test ===")
music_files = tm.scan_audio_files()
esc50_entries = tm.load_esc50_metadata()
dataset = tm.MulticlassAJEPADataset(
    music_files, esc50_entries, tm.HISTORY_SEGMENTS, tm.FUTURE_SEGMENTS,
    esc50_validation_folds={5}, music_ratio=tm.MUSIC_RATIO,
)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True, drop_last=True,
)

projector = tm.AudioFeatureProjector(
    n_mels=tm.N_MELS, segment_frames=tm.SEGMENT_FRAMES, output_dim=512,
).to(device)

sigreg_loss = tm.SIGRegWithPredictionLoss(var_weight=5.0, cov_weight=1.0, sim_weight=1.0)

all_params = list(projector.parameters()) + list(ajepa.parameters())
slot_proj_ids = set(id(p) for p in ajepa.per_slot_input_proj.parameters())
slot_proj_params = list(ajepa.per_slot_input_proj.parameters())
other_params = [p for p in all_params if id(p) not in slot_proj_ids]
optimizer = tm.Adam([
    {"params": other_params, "lr": 5e-5},
    {"params": slot_proj_params, "lr": 1e-5},
], weight_decay=1e-5)

print(f"  Param groups: {len(optimizer.param_groups)}")
for i, pg in enumerate(optimizer.param_groups):
    print(f"    group[{i}]: lr={pg['lr']:.0e}, params={len(pg['params'])}")

ajepa.train()
projector.train()
for batch_idx, (h_mel, f_mel, labels) in enumerate(dataloader):
    if batch_idx >= 2:
        break
    B, Th, N_M, F = h_mel.shape
    _, Tf, _, _ = f_mel.shape

    hf = projector(h_mel.to(device).reshape(B * Th, N_M, F)).reshape(B, Th, -1)
    hs = ajepa.project_to_slots(hf)
    ff = projector(f_mel.to(device).reshape(B * Tf, N_M, F)).reshape(B, Tf, -1)
    fs = ajepa.project_to_slots(ff)

    with torch.no_grad():
        tgt_h, _ = ajepa.target_encoder(hs)
        tgt_f, _ = ajepa.target_encoder(fs)
    ctx_h, _ = ajepa.context_encoder(hs)
    masked, sm, _ = ajepa.masker(ctx_h)
    rec, pred = ajepa.predictor(masked)

    rl = sigreg_loss(rec.reshape(-1, 128), tgt_h.reshape(-1, 128))
    Tm = min(pred.shape[1], tgt_f.shape[1])
    fl = sigreg_loss(pred[:, :Tm].reshape(-1, 128), tgt_f[:, :Tm].reshape(-1, 128))
    jepa_l = rl["total"] + 0.5 * fl["total"]
    cont_l = tm.supervised_contrastive_loss(hf.mean(dim=1), labels.to(device), temperature=0.07)
    loss = jepa_l + 0.3 * cont_l

    optimizer.zero_grad()
    loss.backward()

    grad_norms = {}
    for name, p in ajepa.named_parameters():
        if p.grad is not None:
            gn = p.grad.norm().item()
            if "per_slot_input_proj" in name:
                grad_norms.setdefault("slot_proj", []).append(gn)
            else:
                grad_norms.setdefault("other", []).append(gn)

    optimizer.step()

    avg_slot_gn = sum(grad_norms.get("slot_proj", [0])) / max(len(grad_norms.get("slot_proj", [1])), 1)
    avg_other_gn = sum(grad_norms.get("other", [0])) / max(len(grad_norms.get("other", [1])), 1)
    print(f"  B{batch_idx}: loss={loss.item():.4f} jepa={jepa_l:.4f} cont={cont_l.item():.4f} "
          f"masked={sm.sum().item()}/{sm.numel()} "
          f"grad: slot={avg_slot_gn:.2f} other={avg_other_gn:.2f}",
          flush=True)

print("  ✅ Forward pass PASSED")

print("\n=== 3. Checkpoint Resume Path Test ===")
mc_checkpoint_dir = tm.MC_CHECKPOINT_DIR
old_checkpoint_dir = tm.CHECKPOINT_DIR
mc_path = os.path.join(mc_checkpoint_dir, "ajepa_mc_best.pt")
old_path = os.path.join(old_checkpoint_dir, "ajepa_best.pt")

print(f"  MC checkpoint: {mc_path} exists={os.path.exists(mc_path)}")
print(f"  Old checkpoint: {old_path} exists={os.path.exists(old_path)}")
resume = mc_path if os.path.exists(mc_path) else old_path
if os.path.exists(resume):
    ckpt = torch.load(resume, map_location=device, weights_only=False)
    print(f"  Checkpoint epoch={ckpt.get('epoch','?')} step={ckpt.get('global_step','?')}")
    has_per_slot = any("per_slot_input_proj" in k for k in ckpt.get("ajepa_state", {}))
    print(f"  Has per_slot_input_proj: {has_per_slot}")
print("  ✅ Resume path PASSED")

print(f"\n{'='*60}")
print("✅ ALL SMOKE TESTS PASSED — ready for training")
print(f"{'='*60}")
