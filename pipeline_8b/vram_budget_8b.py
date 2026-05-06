#!/usr/bin/env python3
"""
8B VRAM Budget Calculator & Training Config Recommender
RTX 3060 12GB - Strict VRAM Management for 8B Base
"""
import json

TOTAL_VRAM = 12288  # MiB
SAFETY_MARGIN = 1024  # Reserve 1GB for system/driver overhead
USABLE_VRAM = TOTAL_VRAM - SAFETY_MARGIN  # 11264 MiB

MODEL_4BIT = 4800  # MiB - 8.49B params in 4bit
LORA_ADAPTERS = 300  # MiB - r=16 LoRA on language layers
OPTIMIZER_STATES = 800  # MiB - AdamW states for LoRA params
FRAMEWORK_OVERHEAD = 400  # MiB - PyTorch/transformers overhead

FIXED_COST = MODEL_4BIT + LORA_ADAPTERS + OPTIMIZER_STATES + FRAMEWORK_OVERHEAD  # 6300 MiB
REMAINING = USABLE_VRAM - FIXED_COST  # 4964 MiB for activations + KV cache

ACTIVATION_PER_SEQ_PER_SAMPLE = 9.4  # MiB per token per sample (approximate for 34L model)
KV_CACHE_PER_TOKEN = 1.1  # MiB per token (34L, 4bit, hidden=4096)

GRADIENT_CHECKPOINT_SAVINGS = 0.5  # 50% activation savings


def calc_vram(seq_len, batch_size, grad_accum, gradient_checkpointing=True):
    act_per_sample = ACTIVATION_PER_SEQ_PER_SAMPLE * seq_len
    if gradient_checkpointing:
        act_per_sample *= GRADIENT_CHECKPOINT_SAVINGS
    total_act = act_per_sample * batch_size * grad_accum
    kv = KV_CACHE_PER_TOKEN * seq_len * batch_size
    total = FIXED_COST + total_act + kv
    return {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "gradient_checkpointing": gradient_checkpointing,
        "model_4bit": MODEL_4BIT,
        "lora": LORA_ADAPTERS,
        "optimizer": OPTIMIZER_STATES,
        "framework": FRAMEWORK_OVERHEAD,
        "activations": round(total_act),
        "kv_cache": round(kv),
        "total_estimated": round(total),
        "usable_vram": USABLE_VRAM,
        "headroom": round(USABLE_VRAM - total),
        "headroom_pct": round((USABLE_VRAM - total) / USABLE_VRAM * 100, 1),
        "status": "SAFE" if total < USABLE_VRAM * 0.85 else ("TIGHT" if total < USABLE_VRAM else "OOM"),
    }


PRESETS = {
    "safe": {"seq_len": 128, "batch_size": 1, "grad_accum": 4, "gradient_checkpointing": True},
    "moderate": {"seq_len": 256, "batch_size": 1, "grad_accum": 4, "gradient_checkpointing": True},
    "aggressive": {"seq_len": 256, "batch_size": 1, "grad_accum": 8, "gradient_checkpointing": True},
    "max_context": {"seq_len": 512, "batch_size": 1, "grad_accum": 2, "gradient_checkpointing": True},
    "inference_only": {"seq_len": 1024, "batch_size": 1, "grad_accum": 1, "gradient_checkpointing": False},
}


def recommend_for_stage(stage, data_avg_tokens=300):
    needed_seq = min(data_avg_tokens * 2, 1024)
    print(f"\n  Stage: {stage}")
    print(f"  Data avg tokens: ~{data_avg_tokens}")
    print(f"  Needed seq_len: ~{needed_seq}")

    for name, cfg in PRESETS.items():
        if cfg["seq_len"] >= needed_seq:
            result = calc_vram(**cfg)
            if result["status"] in ["SAFE", "TIGHT"]:
                print(f"  Recommended preset: {name}")
                print(f"    seq_len={cfg['seq_len']}, grad_accum={cfg['grad_accum']}")
                print(f"    Estimated VRAM: {result['total_estimated']} MiB ({result['status']})")
                print(f"    Headroom: {result['headroom']} MiB ({result['headroom_pct']}%)")
                return cfg

    print("  WARNING: No safe preset found! Use seq_len=128 minimum.")
    return PRESETS["safe"]


def print_budget_table():
    print("=" * 80)
    print("  8B VRAM Budget Table - RTX 3060 12GB")
    print("=" * 80)
    print(f"  Total VRAM: {TOTAL_VRAM} MiB | Safety margin: {SAFETY_MARGIN} MiB | Usable: {USABLE_VRAM} MiB")
    print(f"  Fixed costs: Model={MODEL_4BIT}, LoRA={LORA_ADAPTERS}, Optim={OPTIMIZER_STATES}, FW={FRAMEWORK_OVERHEAD}")
    print(f"  Remaining for activations+KV: {REMAINING} MiB")
    print("-" * 80)
    print(f"  {'Preset':<15} {'SeqLen':>6} {'GA':>4} {'Act':>8} {'KV':>6} {'Total':>8} {'Head':>8} {'Status':>8}")
    print("-" * 80)
    for name, cfg in PRESETS.items():
        r = calc_vram(**cfg)
        print(f"  {name:<15} {r['seq_len']:>6} {r['grad_accum']:>4} {r['activations']:>8} {r['kv_cache']:>6} {r['total_estimated']:>8} {r['headroom']:>8} {r['status']:>8}")
    print("-" * 80)

    print("\n  Stage-specific recommendations:")
    recommend_for_stage("stage1", data_avg_tokens=200)
    recommend_for_stage("stage2", data_avg_tokens=300)
    recommend_for_stage("stage3_7b_data", data_avg_tokens=400)
    recommend_for_stage("stage3_8b_data", data_avg_tokens=800)
    recommend_for_stage("inference", data_avg_tokens=600)

    print("\n  CRITICAL NOTES:")
    print("  - 8B long-chain data (800+ tokens) REQUIRES seq_len >= 512")
    print("  - But seq_len=512 only fits with grad_accum=2 (TIGHT)")
    print("  - Solution: Truncate 8B data to 256 tokens for training, full length for evaluation")
    print("  - OR: Use gradient accumulation to simulate larger batches within VRAM budget")
    print("  - NEVER use seq_len > 512 on 12GB GPU with 8B model")


if __name__ == "__main__":
    print_budget_table()
