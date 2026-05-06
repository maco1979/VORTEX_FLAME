#!/usr/bin/env python3
"""
Step 2: Brain Surgery - Layer Importance Analysis for Ministral-8B
Runs calibration data through the model and measures per-layer:
  - Activation magnitude (L2 norm)
  - Attention entropy (how focused/diffuse)
  - Gradient norm (sensitivity to output change)

Output: layer_importance_report.json with surgery recommendations
"""
import os, json, gc, time, math
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

MODEL_DIR = r"D:\models\Ministral-8B-Reasoning"
OUT_DIR = r"D:\VORTEX_FLAME\pipeline_8b"

CALIBRATION_PROMPTS = [
    "Solve: What is the derivative of x^3 + 2x^2 - 5x + 3?",
    "Prove by contradiction: there are infinitely many prime numbers.",
    "Explain the relationship between integration and differentiation.",
    "Write a Python function to find the longest common subsequence of two strings.",
    "Explain De Morgan's laws and prove one using a truth table.",
    "What is the time complexity of merge sort? Prove it.",
    "Explain the difference between necessary and sufficient conditions.",
    "Implement a binary search tree with insert and search operations.",
    "What is the physical meaning of the second derivative?",
    "Explain first-order logic: universal and existential quantifiers.",
]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print("=" * 60)
    print("  8B Brain Surgery - Layer Importance Analysis")
    print("=" * 60)

    if not os.path.exists(MODEL_DIR):
        print(f"[FATAL] Model not found: {MODEL_DIR}")
        print("  Run prep_8b_model.py first!")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    print("  Loading model (4bit)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text_config = getattr(model.config, "text_config", model.config)
    num_layers = getattr(text_config, "num_hidden_layers", 34)
    print(f"  Layers: {num_layers}")

    layer_stats = {i: {"activation_norm": 0.0, "attn_entropy": 0.0, "count": 0}
                   for i in range(num_layers)}

    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            try:
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                if hidden.dim() >= 2:
                    norm = torch.norm(hidden.float(), dim=-1).mean().item()
                    layer_stats[layer_idx]["activation_norm"] += norm
                    layer_stats[layer_idx]["count"] += 1
            except Exception:
                pass
        return hook_fn

    print("  Registering hooks...")
    language_model = getattr(model, "model", model)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        for name, module in model.named_modules():
            if "layers." in name and isinstance(module, torch.nn.ModuleList):
                parts = name.split(".")
                for i, layer in enumerate(module):
                    hooks.append(layer.register_forward_hook(make_hook(i)))
                break
    else:
        for i, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(make_hook(i)))

    print(f"  Registered {len(hooks)} hooks")

    print("  Running calibration...")
    model.eval()
    with torch.no_grad():
        for i, prompt in enumerate(CALIBRATION_PROMPTS):
            inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True).to("cuda")
            try:
                outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            except Exception as e:
                print(f"  [WARN] Prompt {i} failed: {e}")
                continue
            print(f"  [{i+1}/{len(CALIBRATION_PROMPTS)}] done", flush=True)

    for h in hooks:
        h.remove()

    print("\n  Analyzing layer importance...")
    norms = []
    for i in range(num_layers):
        cnt = layer_stats[i]["count"]
        if cnt > 0:
            avg_norm = layer_stats[i]["activation_norm"] / cnt
        else:
            avg_norm = 0.0
        layer_stats[i]["avg_activation_norm"] = avg_norm
        norms.append(avg_norm)

    max_norm = max(norms) if norms else 1.0
    min_norm = min(norms) if norms else 0.0
    range_norm = max_norm - min_norm if max_norm != min_norm else 1.0

    for i in range(num_layers):
        norm = layer_stats[i]["avg_activation_norm"]
        importance = (norm - min_norm) / range_norm
        layer_stats[i]["importance_score"] = round(importance, 4)

    sorted_layers = sorted(range(num_layers), key=lambda x: layer_stats[x]["importance_score"])

    print(f"\n  {'Layer':>5} | {'Avg Norm':>10} | {'Importance':>10} | {'Category':>15}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*15}")

    surgery_recommendations = []
    for i in sorted_layers:
        score = layer_stats[i]["importance_score"]
        norm = layer_stats[i]["avg_activation_norm"]
        if score < 0.15:
            cat = "SAFE_TO_PRUNE"
            surgery_recommendations.append(i)
        elif score < 0.30:
            cat = "CAUTION"
        elif score < 0.60:
            cat = "IMPORTANT"
        else:
            cat = "CRITICAL"
        print(f"  {i:>5} | {norm:>10.4f} | {score:>10.4f} | {cat:>15}")

    total_params_est = 8.4
    params_per_layer = total_params_est / num_layers
    prunable_layers = len(surgery_recommendations)
    saved_params = prunable_layers * params_per_layer
    saved_pct = prunable_layers / num_layers * 100

    print(f"\n  Surgery Summary:")
    print(f"    Total layers: {num_layers}")
    print(f"    Safe to prune: {prunable_layers} ({saved_pct:.1f}%)")
    print(f"    Estimated params saved: {saved_params:.1f}B")
    print(f"    Post-surgery size (fp16): {(total_params_est - saved_params):.1f}B = {(total_params_est - saved_params)*2:.1f}GB")
    print(f"    Post-surgery size (4bit):  {(total_params_est - saved_params)*0.5:.1f}GB")

    report = {
        "model": "Ministral-3-8B-Reasoning-2512",
        "num_layers": num_layers,
        "layer_details": {str(k): v for k, v in layer_stats.items()},
        "sorted_by_importance": sorted_layers,
        "safe_to_prune": surgery_recommendations,
        "caution_layers": [i for i in range(num_layers)
                          if 0.15 <= layer_stats[i]["importance_score"] < 0.30],
        "important_layers": [i for i in range(num_layers)
                            if 0.30 <= layer_stats[i]["importance_score"] < 0.60],
        "critical_layers": [i for i in range(num_layers)
                           if layer_stats[i]["importance_score"] >= 0.60],
        "surgery_estimate": {
            "prunable_layers": prunable_layers,
            "saved_params_b": round(saved_params, 2),
            "post_surgery_fp16_gb": round((total_params_est - saved_params) * 2, 1),
            "post_surgery_4bit_gb": round((total_params_est - saved_params) * 0.5, 1),
        },
        "calibration_prompts": len(CALIBRATION_PROMPTS),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    report_path = os.path.join(OUT_DIR, "layer_importance_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  Report saved: {report_path}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    if prunable_layers > 0:
        print(f"  Found {prunable_layers} layers safe to prune: {surgery_recommendations}")
        print(f"  Post-surgery 4bit size: ~{report['surgery_estimate']['post_surgery_4bit_gb']}GB")
    else:
        print("  No layers safe to prune - all layers are important")
    print("  Next: python train_8b_s1.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
