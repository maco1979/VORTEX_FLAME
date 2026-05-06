#!/usr/bin/env python3
"""
Step 4: Dual-GPU Inference - 1060 as inference card
Load 8B model on GPU1 (1060 6GB) for exam/regression/AB testing
while GPU0 (3060) is busy training
"""
import os, json, gc, time, argparse
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

MODEL_DIR = r"D:\models\Ministral-8B-Reasoning"
LORA_DIR = r"D:\VORTEX_FLAME\soul_lora_v2"
LOG_DIR = r"D:\VORTEX_FLAME\hermes_logs"


def detect_gpus():
    gpu_info = []
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / 1024**3
        gpu_info.append({"id": i, "name": name, "vram_gb": round(mem, 1)})
    return gpu_info


def pick_inference_gpu():
    gpus = detect_gpus()
    print(f"  Available GPUs: {len(gpus)}")
    for g in gpus:
        print(f"    GPU{g['id']}: {g['name']} ({g['vram_gb']}GB)")

    for g in gpus:
        if "1060" in g["name"]:
            print(f"  Selected GPU{g['id']} (1060) for inference")
            return g["id"]
    for g in gpus:
        if g["vram_gb"] <= 6:
            print(f"  Selected GPU{g['id']} (small VRAM) for inference")
            return g["id"]
    print(f"  Using GPU0 for inference (no 1060 found)")
    return 0


def load_model_on_gpu(gpu_id, lora_path=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"  Loading model on GPU{gpu_id} (4bit)...", flush=True)
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

    if lora_path and os.path.exists(lora_path):
        print(f"  Loading LoRA: {lora_path}", flush=True)
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    vram = torch.cuda.memory_reserved() / 1024**3
    print(f"  Model loaded. VRAM: {vram:.1f}GB", flush=True)
    return model, tokenizer


def run_exam(model, tokenizer, questions, keywords, max_new_tokens=400):
    results = []
    passed = 0
    for i, q in enumerate(questions):
        prompt = f"[INST] {q} [/INST] "
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                    temperature=0.7, do_sample=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
        kw_list = keywords[i] if i < len(keywords) else []
        matched = [kw for kw in kw_list if kw.lower() in answer.lower()]
        ok = len(matched) >= 1 or len(answer) > 200
        if ok:
            passed += 1
        results.append({
            "q": q, "a": answer[:300], "matched": matched,
            "pass": ok, "len": len(answer),
        })
        status = "PASS" if ok else "FAIL"
        print(f"  Q{i+1}/{len(questions)} [{status}] {answer[:80]}...", flush=True)
    rate = passed / max(len(questions), 1)
    return rate, results


def main():
    parser = argparse.ArgumentParser(description="8B Dual-GPU Inference")
    parser.add_argument("--soul", type=str, default="cezanne")
    parser.add_argument("--stage", type=str, default="stage1")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--questions", type=str, default=None,
                       help="JSON file with questions")
    args = parser.parse_args()

    print("=" * 60)
    print("  8B Dual-GPU Inference (1060)")
    print(f"  Soul: {args.soul} | Stage: {args.stage}")
    print("=" * 60)

    gpu_id = args.gpu if args.gpu is not None else pick_inference_gpu()

    lora_path = os.path.join(LORA_DIR, args.soul, args.stage, "final")
    if not os.path.exists(lora_path):
        lora_path = None
        print(f"  No LoRA found, using base model only")

    model, tokenizer = load_model_on_gpu(gpu_id, lora_path)

    if args.questions and os.path.exists(args.questions):
        with open(args.questions, "r", encoding="utf-8") as f:
            exam_data = json.load(f)
        questions = exam_data.get("questions", [])
        keywords = exam_data.get("keywords", [])
    else:
        questions = [
            "Prove that the square root of 2 is irrational.",
            "What is the Fundamental Theorem of Calculus?",
            "Explain De Morgan's Laws.",
            "What is the derivative of sin(x)?",
            "Explain linear independence.",
        ]
        keywords = [
            ["irrational", "contradiction", "sqrt"],
            ["fundamental theorem", "integral", "antiderivative"],
            ["De Morgan", "union", "complement"],
            ["cos", "derivative", "chain"],
            ["linear independence", "combination", "vectors"],
        ]

    print(f"\n  Running exam ({len(questions)} questions)...")
    rate, results = run_exam(model, tokenizer, questions, keywords)
    print(f"\n  Exam result: {rate:.0%}")

    log_data = {
        "soul": args.soul, "stage": args.stage,
        "gpu_id": gpu_id, "gpu_name": torch.cuda.get_device_name(gpu_id),
        "exam_rate": rate, "results": results,
        "lora_path": lora_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    log_path = os.path.join(LOG_DIR, args.soul, f"inference_gpu{gpu_id}_{args.stage}.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    print(f"  Log saved: {log_path}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print(f"  Inference complete: {rate:.0%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
