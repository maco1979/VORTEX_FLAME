#!/usr/bin/env python3
"""
Cezanne Stage1 Math Regression Test
Check if Stage2 logic training caused Stage1 math forgetting
"""
import os, json, gc, time
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

BASE_MODEL = r"E:\models\Mistral-7B-Instruct-v0.1-Cezanne"
S1_LORA = r"D:\VORTEX_FLAME\soul_lora_v2\cezanne\stage1\final"
S2_LORA = r"D:\VORTEX_FLAME\soul_lora_v2\cezanne\stage2\final"
LOG_DIR = r"D:\VORTEX_FLAME\hermes_logs\cezanne"

MATH_QUESTIONS = [
    {"q": "Prove that the square root of 2 is irrational. Label each step.", "kw": ["irrational", "contradiction", "sqrt", "2", "assume"]},
    {"q": "What is the Fundamental Theorem of Calculus? Explain both parts.", "kw": ["fundamental theorem", "integral", "antiderivative", "derivative"]},
    {"q": "Solve: find the derivative of f(x) = x^3 + 2x^2 - 5x + 3.", "kw": ["derivative", "3x^2", "4x", "power rule"]},
    {"q": "Prove by mathematical induction: 1+2+...+n = n(n+1)/2.", "kw": ["induction", "base case", "inductive step", "n(n+1)/2"]},
    {"q": "What is the chain rule in calculus? Apply it to differentiate sin(x^2).", "kw": ["chain rule", "cos", "2x", "inner"]},
    {"q": "Explain the concept of eigenvalues and eigenvectors.", "kw": ["eigenvalue", "eigenvector", "matrix", "characteristic"]},
    {"q": "State and prove De Morgan's Laws for sets.", "kw": ["De Morgan", "union", "intersection", "complement"]},
    {"q": "What is the integral of 1/x? Explain why.", "kw": ["ln", "log", "natural", "integral"]},
    {"q": "Explain the difference between a convergent and divergent series.", "kw": ["convergent", "divergent", "series", "limit"]},
    {"q": "What is Bayes' theorem? Derive it from conditional probability.", "kw": ["Bayes", "conditional", "prior", "posterior"]},
    {"q": "Find the determinant of the matrix [[3,1],[2,4]].", "kw": ["determinant", "10", "3*4", "1*2"]},
    {"q": "What is the Taylor series expansion of e^x?", "kw": ["Taylor", "series", "e^x", "factorial", "n!"]},
    {"q": "Explain the concept of linear independence with an example.", "kw": ["linear independence", "combination", "trivial", "vectors"]},
    {"q": "What is the Mean Value Theorem? State it precisely.", "kw": ["mean value", "derivative", "continuous", "differentiable"]},
    {"q": "Solve the system: x + y = 5, 2x - y = 1.", "kw": ["x=2", "y=3", "substitution", "elimination"]},
]


def run_regression(lora_path, label):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"\n  Loading model + {label}...", flush=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if lora_path and os.path.exists(lora_path):
        model = PeftModel.from_pretrained(model, lora_path)
        print(f"  LoRA loaded: {lora_path}", flush=True)

    model.eval()
    results = []
    passed = 0

    for i, item in enumerate(MATH_QUESTIONS):
        q = item["q"]
        kw_list = item["kw"]
        prompt = f"[INST] {q} [/INST] "
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=400, temperature=0.7, do_sample=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
        matched = [kw for kw in kw_list if kw.lower() in answer.lower()]
        ok = len(matched) >= 1 or len(answer) > 200
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        results.append({"q": q, "a": answer[:300], "matched": matched, "pass": ok, "len": len(answer)})
        print(f"  [{label}] Q{i+1}/15 [{status}] {answer[:80]}...", flush=True)

    rate = passed / len(MATH_QUESTIONS)
    print(f"\n  [{label}] Math Regression: {passed}/15 ({rate:.0%})", flush=True)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return rate, results


def main():
    print("=" * 60)
    print("  Cezanne Stage1 Math Regression Test")
    print("  Compare: Stage1-only vs Stage1+Stage2")
    print("=" * 60)

    os.makedirs(LOG_DIR, exist_ok=True)

    print("\n  === Test 1: Stage1 LoRA only ===", flush=True)
    s1_rate, s1_results = run_regression(S1_LORA, "S1-only")

    print("\n  === Test 2: Stage1+Stage2 LoRA ===", flush=True)
    s2_rate, s2_results = run_regression(S2_LORA, "S1+S2")

    forgetting = s1_rate - s2_rate
    print(f"\n{'='*60}")
    print(f"  COMPARISON:")
    print(f"    Stage1 only:  {s1_rate:.0%}")
    print(f"    Stage1+2:     {s2_rate:.0%}")
    print(f"    Forgetting:   {forgetting:+.0%}")
    if forgetting <= 0:
        print(f"    Verdict: NO FORGETTING - Stage2 improved or maintained math")
    elif forgetting <= 0.1:
        print(f"    Verdict: MINIMAL FORGETTING - acceptable, will fix in Stage3")
    else:
        print(f"    Verdict: SIGNIFICANT FORGETTING - need math review in Stage3")
    print(f"{'='*60}")

    log_data = {
        "test": "stage1_math_regression_after_stage2",
        "soul": "cezanne",
        "s1_only_rate": s1_rate,
        "s1_s2_rate": s2_rate,
        "forgetting": forgetting,
        "s1_results": s1_results,
        "s2_results": s2_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    log_path = os.path.join(LOG_DIR, "stage1_regression_after_s2.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    print(f"  Log saved: {log_path}")


if __name__ == "__main__":
    main()
