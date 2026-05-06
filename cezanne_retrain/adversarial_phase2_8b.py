#!/usr/bin/env python3
"""Phase 2: 8B answers same questions → generate training data
Loads 7B answers from Phase 1, then 8B answers and generates training data
"""
import os, json, gc, time, sys
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

BASE_8B = "/mnt/d/models/Ministral-8B-Reasoning-Text-34L-ctx1024"
LORA_8B_S3B = "/mnt/d/VORTEX_FLAME/soul_lora_v2/cezanne_8b/stage3b/final"
LOG_DIR = "/mnt/d/VORTEX_FLAME/hermes_logs/cezanne"
DATA_DIR = "/mnt/d/VORTEX_FLAME/soul_training_data/cezanne"

ANSWERS_PATH = os.path.join(LOG_DIR, "adversarial_7b_answers_v2.json")

from transformers import Mistral3ForConditionalGeneration, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def main():
    with open(ANSWERS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)
    print(f"PHASE 2: 8B answers {len(results)} questions", flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)

    model = Mistral3ForConditionalGeneration.from_pretrained(BASE_8B,
        quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(BASE_8B)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(model, LORA_8B_S3B)
    model.eval()
    print(f"8B loaded", flush=True)

    for i, q in enumerate(results):
        prompt = f"[INST] {q['q']} [/INST] "
        inp = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=400, temperature=0.7,
                                 do_sample=True, pad_token_id=tokenizer.eos_token_id)
        ans = tokenizer.decode(out[0], skip_special_tokens=True)
        ans = ans.replace(prompt, "").strip()

        matched = [k for k in q["kw"] if k.lower() in ans.lower()]
        passed = len(matched) >= 2
        q["answer_8b"] = ans
        q["8b_matched"] = matched
        q["8b_pass"] = passed
        status = "PASS" if passed else "FAIL"
        s7 = "PASS" if q["7b_pass"] else "FAIL"
        print(f"  [{i+1}/{len(results)}] [{q['cat']}] 7B={s7} 8B={status}", flush=True)

        if (i + 1) % 10 == 0:
            with open(ANSWERS_PATH, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(ANSWERS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Generate training data
    training_data = []
    stats = {"7b_only_fail": 0, "both_pass": 0, "both_fail": 0, "8b_only_pass": 0}

    for q in results:
        s7 = q["7b_pass"]
        s8 = q["8b_pass"]

        if not s7 and s8:
            stats["8b_only_pass"] += 1
        elif s7 and not s8:
            stats["7b_only_fail"] += 1
        elif s7 and s8:
            stats["both_pass"] += 1
        else:
            stats["both_fail"] += 1

        if not s7 and q.get("answer_8b") and len(q["answer_8b"]) > 30:
            item = {
                "instruction": q["q"],
                "input": "",
                "output": q["answer_8b"],
                "source": f"adversarial_{q['cat']}",
                "soul": "cezanne",
                "_cat": q["cat"],
                "_7b_pass": s7,
                "_8b_pass": s8,
            }
            training_data.append(item)

    cat_counts = {}
    for item in training_data:
        cat = item["_cat"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    out_path = os.path.join(DATA_DIR, "cezanne_adversarial_training.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

    r_path = os.path.join(LOG_DIR, "adversarial_v2_results.json")
    with open(r_path, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "training_items": len(training_data),
                    "by_category": cat_counts, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
                   f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"  ADVERSARIAL SELF-PLAY v2 COMPLETE", flush=True)
    print(f"  Stats: {stats}", flush=True)
    print(f"  Training data: {len(training_data)} items", flush=True)
    print(f"  By category: {cat_counts}", flush=True)
    print(f"  Saved: {out_path}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
