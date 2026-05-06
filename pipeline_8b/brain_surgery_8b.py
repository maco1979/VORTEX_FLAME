#!/usr/bin/env python3
"""
Brain Surgery for Ministral-8B-Reasoning
3 cuts to fit 3060 12GB with LoRA training room:

Cut 1: Remove vision_tower + projector (save 428M params, ~0.2GB 4bit)
Cut 2: Remove top N language layers (save N*218M each)
Cut 3: Slash max_position_embeddings 262144 -> 1024 (save ~36GB KV cache!)

Anatomy:
  Vision tower: 403M (Pixtral 24-layer)
  Projector:    25M
  Language:     8.49B (34 layers x 218M + embed 537M + lm_head 537M)
  Total:        8.92B

KV cache at 262144 ctx = 36.5GB <- THE REAL KILLER
KV cache at 1024 ctx   = 0.14GB <- AFTER SURGERY

Result (cut vision + 6 layers + ctx 1024):
  4bit weights: ~3.6GB + KV 0.14GB + LoRA 0.5GB + overhead ~1GB = ~5.2GB
  3060 12GB: PLENTY OF ROOM!
"""
import os, json, gc, shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

MODEL_DIR = r"D:\models\Ministral-8B-Reasoning"
OUTPUT_BASE = r"D:\models\Ministral-8B-Reasoning-Text"
CUT_LAYERS = 0
NEW_MAX_CTX = 1024

def main():
    cut = CUT_LAYERS
    new_ctx = NEW_MAX_CTX
    remaining_layers = 34 - cut
    output_dir = OUTPUT_BASE + f"-{remaining_layers}L-ctx{new_ctx}"
    print("=" * 60)
    print(f"  Brain Surgery: Ministral-8B -> Text-Only {remaining_layers}L")
    print(f"  Cut 1: vision_tower + projector (428M)")
    print(f"  Cut 2: top {cut} language layers ({cut*218}M)")
    print(f"  Cut 3: context 262144 -> {new_ctx} (KV cache 36.5GB -> 0.14GB)")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    index_path = os.path.join(MODEL_DIR, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index_data = json.load(f)
    weight_map = index_data["weight_map"]

    keep_layers = set(range(remaining_layers))
    remove_layers = set(range(remaining_layers, 34))
    print(f"  Keep layers: 0-{max(keep_layers)} ({len(keep_layers)} layers)")
    if remove_layers:
        print(f"  Remove layers: {min(remove_layers)}-{max(remove_layers)} ({len(remove_layers)} layers)")
    else:
        print(f"  Remove layers: none")

    all_shards = sorted(set(weight_map.values()))
    print(f"  Shards to process: {len(all_shards)}")

    new_weight_map = {}
    shard_tensors = {}

    for shard_name in all_shards:
        shard_path = os.path.join(MODEL_DIR, shard_name)
        print(f"  Processing {shard_name}...", flush=True)

        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                if key.startswith("vision_tower."):
                    continue
                if key.startswith("multi_modal_projector."):
                    continue

                if key.startswith("language_model.model.layers."):
                    parts = key.split("layers.")[1]
                    layer_num = int(parts.split(".")[0])
                    if layer_num in remove_layers:
                        continue

                tensor = f.get_tensor(key)
                new_weight_map[key] = shard_name
                if shard_name not in shard_tensors:
                    shard_tensors[shard_name] = {}
                shard_tensors[shard_name][key] = tensor

        gc.collect()

    kept_params = sum(t.numel() for tensors in shard_tensors.values() for t in tensors.values())
    removed_params = index_data["metadata"]["total_parameters"] - kept_params
    print(f"\n  Original: {index_data['metadata']['total_parameters']/1e9:.2f}B params")
    print(f"  Removed:  {removed_params/1e9:.2f}B params (vision {428/1e3:.1f}B + {cut} layers {cut*218/1e3:.1f}B)")
    print(f"  Kept:     {kept_params/1e9:.2f}B params")
    print(f"  4bit weights: ~{kept_params*0.5/1e9:.1f}GB")
    print(f"  KV cache ({new_ctx} ctx): ~0.14GB")
    print(f"  Total estimate: ~{kept_params*0.5/1e9+0.14+0.5+1:.1f}GB")

    print(f"\n  Writing new shards...", flush=True)
    for shard_name, tensors in shard_tensors.items():
        out_path = os.path.join(output_dir, shard_name)
        save_file(tensors, out_path)
        print(f"    Saved {shard_name}: {len(tensors)} tensors", flush=True)

    new_index = {
        "metadata": {
            "total_parameters": kept_params,
            "total_size": kept_params * 2
        },
        "weight_map": new_weight_map
    }
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2)

    new_config = {
        "architectures": ["Mistral3ForConditionalGeneration"],
        "model_type": "mistral3",
        "dtype": "bfloat16",
        "text_config": {
            "attention_dropout": 0.0,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 14336,
            "max_position_embeddings": new_ctx,
            "model_type": "ministral3",
            "num_attention_heads": 32,
            "num_hidden_layers": remaining_layers,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-05,
            "rope_parameters": {
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "factor": 1.0,
                "llama_4_scaling_beta": 0.1,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "original_max_position_embeddings": new_ctx,
                "rope_theta": 1000000.0,
                "rope_type": "yarn",
                "type": "yarn"
            },
            "sliding_window": None,
            "use_cache": True,
            "vocab_size": 131072,
            "tie_word_embeddings": False
        },
        "transformers_version": "5.0.0.dev0"
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(new_config, f, indent=2)

    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "tekken.json",
        "tokenizer.model", "chat_template.jinja",
        "generation_config.json"
    ]
    for tf in tokenizer_files:
        src = os.path.join(MODEL_DIR, tf)
        dst = os.path.join(output_dir, tf)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"    Copied {tf}")

    print(f"\n{'='*60}")
    print(f"  Surgery complete!")
    print(f"  Output: {output_dir}")
    print(f"  Params: {kept_params/1e9:.2f}B ({remaining_layers} layers)")
    print(f"  Context: {new_ctx} tokens (was 262144)")
    print(f"  4bit VRAM estimate:")
    print(f"    Weights:  ~{kept_params*0.5/1e9:.1f}GB")
    print(f"    KV cache: ~0.14GB")
    print(f"    LoRA:     ~0.5GB")
    print(f"    Overhead: ~1.0GB")
    print(f"    TOTAL:    ~{kept_params*0.5/1e9+0.14+0.5+1:.1f}GB")
    print(f"  3060 12GB: {'FIT!' if kept_params*0.5/1e9+0.14+0.5+1 < 11 else 'TIGHT'}")
    print(f"{'='*60}")

    del shard_tensors
    gc.collect()


if __name__ == "__main__":
    main()
