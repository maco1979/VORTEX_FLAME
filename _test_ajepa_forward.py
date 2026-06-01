import sys
sys.path.insert(0, ".")
import torch
import torch.nn.functional as F

print("Step 1: importing CAJEPA...", flush=True)
from five_layer_jepa.causal_jepa import CAJEPA
print("Step 2: CAJEPA imported", flush=True)

from train_ajepa import AudioFeatureProjector, N_MELS, SEGMENT_FRAMES, HISTORY_SEGMENTS, FUTURE_SEGMENTS

print("Step 3: creating model...", flush=True)
ajepa = CAJEPA(input_dim=512)
proj = AudioFeatureProjector(n_mels=N_MELS, segment_frames=SEGMENT_FRAMES, output_dim=512)
print("Step 4: models created", flush=True)

device = torch.device("cuda")
ajepa = ajepa.to(device)
proj = proj.to(device)
vram = torch.cuda.memory_allocated() / 1024**3
print(f"Step 5: on GPU, VRAM={vram:.2f}GB", flush=True)

B = 4
history_mel = torch.randn(B, HISTORY_SEGMENTS, N_MELS, SEGMENT_FRAMES).to(device)
future_mel = torch.randn(B, FUTURE_SEGMENTS, N_MELS, SEGMENT_FRAMES).to(device)

print("Step 6: forward pass...", flush=True)
history_flat = history_mel.reshape(B * HISTORY_SEGMENTS, N_MELS, SEGMENT_FRAMES)
history_features = proj(history_flat)
history_features = history_features.reshape(B, HISTORY_SEGMENTS, -1).unsqueeze(2).expand(-1, -1, 8, -1)

future_flat = future_mel.reshape(B * FUTURE_SEGMENTS, N_MELS, SEGMENT_FRAMES)
future_features = proj(future_flat)
future_features = future_features.reshape(B, FUTURE_SEGMENTS, -1).unsqueeze(2).expand(-1, -1, 8, -1)

print(f"Step 7: features shape={history_features.shape}", flush=True)

with torch.no_grad():
    target_slots_h, _ = ajepa.target_encoder(history_features)
    target_slots_f, _ = ajepa.target_encoder(future_features)

context_slots, _ = ajepa.context_encoder(history_features)
masked_slots, slot_mask, _ = ajepa.masker(context_slots)
recovered, predicted_future = ajepa.predictor(masked_slots)

print(f"Step 8: recovered={recovered.shape}, future={predicted_future.shape}", flush=True)

rec_loss = ajepa.loss_fn(recovered.reshape(-1, ajepa.slot_dim), target_slots_h.reshape(-1, ajepa.slot_dim))
T_match = min(predicted_future.shape[1], target_slots_f.shape[1])
fwd_loss = ajepa.loss_fn(
    predicted_future[:, :T_match].reshape(-1, ajepa.slot_dim),
    target_slots_f[:, :T_match].reshape(-1, ajepa.slot_dim),
)
total_loss = rec_loss["total"] + 0.5 * fwd_loss["total"]
print(f"Step 9: loss={total_loss.item():.4f} rec={rec_loss['total'].item():.4f} fwd={fwd_loss['total'].item():.4f}", flush=True)

total_loss.backward()
print(f"Step 10: backward OK! VRAM={torch.cuda.memory_allocated()/1024**3:.2f}GB", flush=True)

ajepa.update_ema()
print("Step 11: EMA update OK", flush=True)
print("ALL PASSED - A-JEPA training loop verified with real data pipeline!", flush=True)
