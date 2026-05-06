"""
Cezanne Stage1 Math Benchmark Exam
Reference: USAMO/GPQA/MMMLU standard benchmarks
Purpose: Measure math reasoning level after Stage1 (pure math training)
"""
import os, json, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = r"E:\models\Mistral-7B-Instruct-v0.1-Cezanne"
LORA_PATH = r"D:\VORTEX_FLAME\soul_lora_v2\cezanne\stage1\final"
LOG_DIR = r"D:\VORTEX_FLAME\hermes_logs\cezanne"

BENCHMARK = {
    "USAMO_Proof": [
        {"q": "Prove that for any positive integers a and b, gcd(a,b) * lcm(a,b) = a*b. Show each step rigorously.", "level": "olympiad", "keywords": ["gcd","lcm","product","prime","factorization","divisor"]},
        {"q": "Prove that the sum of the first n positive integers is n(n+1)/2 using mathematical induction.", "level": "olympiad", "keywords": ["induction","base","n(n+1)/2","assume","prove"]},
        {"q": "Prove that sqrt(2) is irrational by contradiction.", "level": "olympiad", "keywords": ["irrational","contradiction","even","p/q","integer"]},
        {"q": "Prove that there are infinitely many prime numbers.", "level": "olympiad", "keywords": ["infinitely","prime","contradiction","factor","p1*p2*...*pn+1"]},
        {"q": "Prove that for any real numbers a,b: (a+b)/2 >= sqrt(ab) (AM-GM inequality for two numbers).", "level": "olympiad", "keywords": ["AM-GM","arithmetic","geometric","mean","sqrt","inequality"]},
    ],
    "GPQA_Math": [
        {"q": "What is the relationship between the eigenvalues of a real symmetric matrix and its positive definiteness? Prove your answer.", "level": "graduate", "keywords": ["eigenvalue","positive definite","symmetric","quadratic form","spectral"]},
        {"q": "Explain the Riemann hypothesis and its significance for the distribution of prime numbers.", "level": "graduate", "keywords": ["Riemann","zeta","prime","distribution","zeros","critical line"]},
        {"q": "What is the Banach fixed-point theorem? Give a proof outline and an application.", "level": "graduate", "keywords": ["Banach","fixed point","contraction","metric space","iteration","convergence"]},
        {"q": "Explain the difference between Lebesgue and Riemann integration. When does Lebesgue integral exist but Riemann does not?", "level": "graduate", "keywords": ["Lebesgue","Riemann","measurable","Dirichlet","indicator","measure"]},
        {"q": "State and prove the Central Limit Theorem. What are the necessary conditions?", "level": "graduate", "keywords": ["central limit","normal","distribution","convergence","variance","independent"]},
    ],
    "MMMLU_Calculus": [
        {"q": "Compute the derivative of f(x) = sin(x^2) * e^(3x) using the chain rule and product rule.", "level": "undergraduate", "keywords": ["chain rule","product rule","cos(x^2)","2x","e^(3x)","3e^(3x)"]},
        {"q": "Evaluate the integral of x*ln(x) dx using integration by parts.", "level": "undergraduate", "keywords": ["integration by parts","ln(x)","x^2/2","(x^2*ln(x))/2","x^2/4"]},
        {"q": "Find the Taylor series expansion of e^x around x=0 up to the 4th order term.", "level": "undergraduate", "keywords": ["Taylor","1+x","x^2/2","x^3/6","x^4/24","factorial"]},
        {"q": "Solve the differential equation dy/dx = 2xy with initial condition y(0)=1.", "level": "undergraduate", "keywords": ["separable","e^(x^2)","y(0)=1","exponential"]},
        {"q": "Determine whether the series sum(1/n^2) converges and find its sum if possible.", "level": "undergraduate", "keywords": ["converge","pi^2/6","p-series","Basel","1/n^2"]},
    ],
    "MMMLU_LinearAlgebra": [
        {"q": "Find the eigenvalues and eigenvectors of the matrix [[2,1],[1,2]].", "level": "undergraduate", "keywords": ["eigenvalue","3","1","eigenvector","[1,1]","[1,-1]"]},
        {"q": "Explain what it means for a set of vectors to be linearly independent. How do you check this using a matrix?", "level": "undergraduate", "keywords": ["linearly independent","linear combination","determinant","rank","nonzero"]},
        {"q": "What is the rank-nullity theorem? Prove it for linear transformations.", "level": "undergraduate", "keywords": ["rank","nullity","dimension","kernel","image","domain"]},
        {"q": "Explain the Gram-Schmidt orthogonalization process with an example.", "level": "undergraduate", "keywords": ["Gram-Schmidt","orthogonal","projection","normalize","inner product"]},
        {"q": "What is the singular value decomposition (SVD)? What are its applications?", "level": "undergraduate", "keywords": ["SVD","singular value","U","sigma","V","decomposition","low rank"]},
    ],
    "Math_Reasoning": [
        {"q": "A farmer has 100 meters of fence. What dimensions of a rectangular enclosure maximize the area? Show your work.", "level": "high_school", "keywords": ["25","maximize","derivative","50","area","perimeter"]},
        {"q": "In how many ways can you arrange the letters of the word MATHEMATICS?", "level": "high_school", "keywords": ["11!","2!","factorial","permutation","repeated"]},
        {"q": "A fair coin is flipped 10 times. What is the probability of getting exactly 7 heads?", "level": "high_school", "keywords": ["C(10,7)","binomial","0.5^10","120","combination"]},
        {"q": "Solve the system: x + y = 10, x^2 + y^2 = 58. Find all solutions.", "level": "high_school", "keywords": ["substitution","3","7","quadratic","two solutions"]},
        {"q": "Prove that the product of two odd integers is always odd.", "level": "high_school", "keywords": ["odd","2k+1","2m+1","4km","2(2km+k+m)+1","odd"]},
    ],
}

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    print("Loading Cezanne Stage1 model for benchmark...", flush=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    print("Model loaded!\n", flush=True)

    all_results = {}
    grand_total_pass = 0
    grand_total_q = 0

    for category, questions in BENCHMARK.items():
        print(f"\n{'='*50}", flush=True)
        print(f"  {category}", flush=True)
        print(f"{'='*50}", flush=True)
        cat_pass = 0
        cat_details = []

        for i, q_data in enumerate(questions):
            q = q_data["q"]
            level = q_data["level"]
            kw_list = q_data["keywords"]
            prompt = f"[INST] {q} [/INST] "
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=500, temperature=0.7, do_sample=True, top_p=0.95)
            elapsed = time.time() - t0
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

            matched = [kw for kw in kw_list if kw.lower() in answer.lower()]
            ok = len(matched) >= 2 or (len(matched) >= 1 and len(answer) > 300)
            if ok:
                cat_pass += 1
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {level} Q{i+1}: {len(answer)}chars, {elapsed:.1f}s, matched={matched}", flush=True)
            cat_details.append({"question": q, "answer": answer[:800], "pass": ok, "level": level, "matched": matched, "time_s": elapsed})

        rate = cat_pass / len(questions) if questions else 0
        all_results[category] = {"pass": cat_pass, "total": len(questions), "rate": rate, "details": cat_details}
        grand_total_pass += cat_pass
        grand_total_q += len(questions)
        print(f"  {category}: {cat_pass}/{len(questions)} ({rate:.0%})", flush=True)

    overall_rate = grand_total_pass / grand_total_q if grand_total_q else 0
    print(f"\n{'='*60}", flush=True)
    print(f"  BENCHMARK TOTAL: {grand_total_pass}/{grand_total_q} ({overall_rate:.0%})", flush=True)
    print(f"{'='*60}", flush=True)
    for cat, r in all_results.items():
        print(f"  {cat}: {r['pass']}/{r['total']} ({r['rate']:.0%})", flush=True)
    print(f"\n  Reference: USAMO 97.6% | GPQA 94.6% | MMMLU 92.67%", flush=True)
    print(f"  (These are frontier model scores, 7B+LoRA expected much lower)", flush=True)

    os.makedirs(LOG_DIR, exist_ok=True)
    log_data = {
        "type": "stage1_math_benchmark",
        "model": "cezanne_stage1",
        "overall": {"pass": grand_total_pass, "total": grand_total_q, "rate": overall_rate},
        "categories": {k: {"pass": v["pass"], "total": v["total"], "rate": v["rate"]} for k, v in all_results.items()},
        "details": all_results,
        "reference": {"USAMO_frontier": "97.6%", "GPQA_frontier": "94.6%", "MMMLU_frontier": "92.67%"},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    log_path = os.path.join(LOG_DIR, "stage1_benchmark.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {log_path}", flush=True)

if __name__ == "__main__":
    main()
