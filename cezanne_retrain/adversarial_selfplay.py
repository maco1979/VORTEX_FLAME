#!/usr/bin/env python3
"""7B vs 8B Adversarial Self-Play Learning
Phase 1: 8B generates adversarial questions targeting 7B weaknesses
Phase 2: 7B answers all questions
Phase 3: 8B judges 7B answers + provides corrections
Phase 4: Generate training data from failed Q&A pairs
Output: adversarial_training_data.json for 7B retraining
"""
import os, json, gc, time, sys, re
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

BASE_7B = "/mnt/d/models/Mistral-7B-Instruct-v0.1"
BASE_8B = "/mnt/d/models/Ministral-8B-Reasoning-Text-34L-ctx1024"
LORA_7B_S3 = "/mnt/d/VORTEX_FLAME/soul_lora_v2/cezanne/s3_cs_depth_8k/final"
LORA_8B_S3B = "/mnt/d/VORTEX_FLAME/soul_lora_v2/cezanne_8b/stage3b/final"
LOG_DIR = "/mnt/d/VORTEX_FLAME/hermes_logs/cezanne"
DATA_DIR = "/mnt/d/VORTEX_FLAME/soul_training_data/cezanne"

os.makedirs(LOG_DIR, exist_ok=True)

WEAK_CATEGORIES = {
    "Debug": "7B only scores 3/5 on Debug. Generate tricky bug-finding questions: off-by-one, race conditions, pointer loss, modify-during-iterate, resource leaks, silent exceptions, scope issues.",
    "Logic": "7B only scores 3/5 on Logic. Generate questions on: contrapositive proofs, proof by contradiction, formal syllogisms, necessary vs sufficient conditions, logical fallacies.",
    "Systems": "7B scores 4/5 on Systems. Challenge with: CPU cache hierarchy details, TLB mechanics, virtual memory edge cases, concurrency primitives, memory ordering.",
    "Complexity": "7B scores 4/5 on Complexity. Challenge with: amortized analysis proofs, Big-O class comparisons, space-time tradeoffs, recurrence relations.",
    "Algorithm": "7B scores 4/5 on Algorithm. Challenge with: algorithm correctness proofs, edge cases in Dijkstra/BFS, advanced data structure operations.",
}

NUM_QUESTIONS_PER_CAT = 10
TOTAL_QUESTIONS = NUM_QUESTIONS_PER_CAT * len(WEAK_CATEGORIES)

CHALLENGE_PROMPT = """You are an expert CS professor creating challenging exam questions.
Target the following weakness in a 7B language model: {weakness_desc}

Generate {n} difficult questions that would expose this weakness. Each question should:
1. Require deep understanding, not just surface knowledge
2. Have a specific correct answer that can be verified
3. Be progressively harder (easy to expert level)

Output as JSON array with format:
[{{"difficulty": 1-5, "question": "...", "keywords": ["kw1", "kw2", "kw3"]}}]

Questions:"""

JUDGE_PROMPT = """You are an expert CS judge evaluating a student's answer.

Question: {question}
Expected keywords: {keywords}

Student's answer:
{answer}

Evaluate:
1. Is the answer correct? (yes/no/partial)
2. What is missing or wrong?
3. Provide a complete correct answer.

Output JSON only:
{{"verdict": "pass"/"fail"/"partial", "missing": ["what's missing"], "correction": "complete correct answer"}}"""


def load_model(base_path, lora_path, label):
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"  Loading {label}...", flush=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )

    config_path = os.path.join(base_path, "config.json")
    is_mistral3 = False
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        if cfg.get("model_type") == "mistral3" or "Mistral3" in str(cfg.get("architectures", [])):
            is_mistral3 = True

    if is_mistral3:
        from transformers import Mistral3ForConditionalGeneration
        model = Mistral3ForConditionalGeneration.from_pretrained(
            base_path, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16
        )
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            base_path, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16
        )

    tokenizer = AutoTokenizer.from_pretrained(base_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if lora_path and os.path.exists(lora_path):
        model = PeftModel.from_pretrained(model, lora_path)
        print(f"  {label} LoRA loaded", flush=True)
    else:
        print(f"  {label} base only (no LoRA)", flush=True)
    model.eval()
    return model, tokenizer


def generate_8b(model, tokenizer, prompt, max_tokens=800):
    inp = tokenizer(f"[INST] {prompt} [/INST] ", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_tokens, temperature=0.7,
                             do_sample=True, pad_token_id=tokenizer.eos_token_id)
    ans = tokenizer.decode(out[0], skip_special_tokens=True)
    inst_tag = f"[INST] {prompt} [/INST] "
    if ans.startswith(inst_tag):
        ans = ans[len(inst_tag):]
    return ans.strip()


def generate_7b(model, tokenizer, question, max_tokens=400):
    prompt = f"<s>[INST] {question} [/INST] "
    inp = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_tokens, temperature=0.7,
                             do_sample=True, pad_token_id=tokenizer.eos_token_id)
    ans = tokenizer.decode(out[0], skip_special_tokens=True)
    ans = ans.replace(prompt, "").strip()
    return ans


def parse_json_from_text(text):
    try:
        start = text.index("[")
        end = text.rindex("]") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        pass
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        pass
    return None


def phase1_generate_questions():
    print(f"\n{'='*60}", flush=True)
    print(f"  PHASE 1: 8B generates adversarial questions", flush=True)
    print(f"  Categories: {list(WEAK_CATEGORIES.keys())}", flush=True)
    print(f"  Questions per category: {NUM_QUESTIONS_PER_CAT}", flush=True)
    print(f"{'='*60}", flush=True)

    model, tokenizer = load_model(BASE_8B, LORA_8B_S3B, "8B Cezanne S3b")

    all_questions = []
    for cat, desc in WEAK_CATEGORIES.items():
        print(f"\n  Generating {cat} questions...", flush=True)
        prompt = CHALLENGE_PROMPT.format(weakness_desc=desc, n=NUM_QUESTIONS_PER_CAT)
        raw = generate_8b(model, tokenizer, prompt, max_tokens=2000)
        parsed = parse_json_from_text(raw)

        if parsed and isinstance(parsed, list):
            for item in parsed:
                q = {
                    "cat": cat,
                    "difficulty": item.get("difficulty", 3),
                    "question": item.get("question", ""),
                    "keywords": item.get("keywords", []),
                }
                if q["question"]:
                    all_questions.append(q)
                    print(f"    [{cat} d{q['difficulty']}] {q['question'][:60]}...", flush=True)
        else:
            print(f"    [WARN] Failed to parse {cat} questions, using raw text", flush=True)
            lines = raw.split("\n")
            for line in lines:
                line = line.strip()
                if line and len(line) > 20 and not line.startswith("[") and not line.startswith("{"):
                    all_questions.append({
                        "cat": cat, "difficulty": 3,
                        "question": line.lstrip("0123456789.-) "),
                        "keywords": [],
                    })

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(10)

    q_path = os.path.join(LOG_DIR, "adversarial_questions.json")
    with open(q_path, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)
    print(f"\n  Generated {len(all_questions)} questions total", flush=True)
    print(f"  Saved: {q_path}", flush=True)
    return all_questions


def phase2_7b_answers(questions):
    print(f"\n{'='*60}", flush=True)
    print(f"  PHASE 2: 7B answers adversarial questions", flush=True)
    print(f"  Questions: {len(questions)}", flush=True)
    print(f"{'='*60}", flush=True)

    model, tokenizer = load_model(BASE_7B, LORA_7B_S3, "7B Cezanne S3")

    results = []
    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] [{q['cat']} d{q['difficulty']}] Answering...", flush=True)
        answer = generate_7b(model, tokenizer, q["question"])
        q["answer_7b"] = answer

        if q["keywords"]:
            matched = [k for k in q["keywords"] if k.lower() in answer.lower()]
            q["keyword_matched"] = matched
            q["keyword_pass"] = len(matched) >= 2
        else:
            q["keyword_matched"] = []
            q["keyword_pass"] = None

        status = "PASS" if q["keyword_pass"] else ("?" if q["keyword_pass"] is None else "FAIL")
        print(f"    [{status}] matched={q['keyword_matched']}", flush=True)
        results.append(q)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(10)

    a_path = os.path.join(LOG_DIR, "adversarial_7b_answers.json")
    with open(a_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {a_path}", flush=True)
    return results


def phase3_8b_judges(results):
    print(f"\n{'='*60}", flush=True)
    print(f"  PHASE 3: 8B judges 7B answers + provides corrections", flush=True)
    print(f"{'='*60}", flush=True)

    model, tokenizer = load_model(BASE_8B, LORA_8B_S3B, "8B Cezanne S3b")

    for i, q in enumerate(results):
        print(f"  [{i+1}/{len(results)}] [{q['cat']}] Judging...", flush=True)
        prompt = JUDGE_PROMPT.format(
            question=q["question"],
            keywords=", ".join(q.get("keywords", [])),
            answer=q["answer_7b"][:600],
        )
        raw = generate_8b(model, tokenizer, prompt, max_tokens=600)
        parsed = parse_json_from_text(raw)

        if parsed and isinstance(parsed, dict):
            q["verdict"] = parsed.get("verdict", "unknown")
            q["missing"] = parsed.get("missing", [])
            q["correction"] = parsed.get("correction", "")
        else:
            q["verdict"] = "parse_error"
            q["missing"] = []
            q["correction"] = raw[:400]

        print(f"    Verdict: {q['verdict']}", flush=True)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(10)

    j_path = os.path.join(LOG_DIR, "adversarial_judgments.json")
    with open(j_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {j_path}", flush=True)
    return results


def phase4_generate_training_data(results):
    print(f"\n{'='*60}", flush=True)
    print(f"  PHASE 4: Generate training data from failures", flush=True)
    print(f"{'='*60}", flush=True)

    training_data = []
    stats = {"total": len(results), "pass": 0, "fail": 0, "partial": 0}

    for q in results:
        verdict = q.get("verdict", "unknown")
        if verdict == "pass":
            stats["pass"] += 1
            continue

        stats["fail" if verdict == "fail" else "partial"] += 1

        correction = q.get("correction", "")
        if not correction or len(correction) < 20:
            continue

        item = {
            "instruction": q["question"],
            "input": "",
            "output": correction,
            "source": f"adversarial_{q['cat']}",
            "soul": "cezanne",
            "difficulty": q.get("difficulty", 3),
            "verdict": verdict,
            "_cat": q["cat"],
        }
        training_data.append(item)

    cat_counts = {}
    for item in training_data:
        cat = item["_cat"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    print(f"\n  Stats: {stats}", flush=True)
    print(f"  Training items generated: {len(training_data)}", flush=True)
    print(f"  By category: {cat_counts}", flush=True)

    out_path = os.path.join(DATA_DIR, "cezanne_adversarial_training.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {out_path}", flush=True)

    summary = {
        "total_questions": len(results),
        "pass": stats["pass"],
        "fail": stats["fail"],
        "partial": stats["partial"],
        "training_items": len(training_data),
        "by_category": cat_counts,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    s_path = os.path.join(LOG_DIR, "adversarial_summary.json")
    with open(s_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return training_data, summary


def main():
    print(f"\n{'#'*60}", flush=True)
    print(f"  7B vs 8B ADVERSARIAL SELF-PLAY LEARNING", flush=True)
    print(f"  7B: {BASE_7B} + S3 LoRA", flush=True)
    print(f"  8B: {BASE_8B} + S3b LoRA", flush=True)
    print(f"  Target: {TOTAL_QUESTIONS} adversarial questions", flush=True)
    print(f"{'#'*60}", flush=True)

    t0 = time.time()

    questions = phase1_generate_questions()
    print(f"\n  Phase 1 done: {len(questions)} questions generated", flush=True)

    if not questions:
        print("  [ERROR] No questions generated, aborting", flush=True)
        return

    answered = phase2_7b_answers(questions)
    print(f"\n  Phase 2 done: {len(answered)} answers collected", flush=True)

    judged = phase3_8b_judges(answered)
    print(f"\n  Phase 3 done: {len(judged)} judgments made", flush=True)

    training_data, summary = phase4_generate_training_data(judged)

    elapsed = (time.time() - t0) / 60
    print(f"\n{'#'*60}", flush=True)
    print(f"  ADVERSARIAL SELF-PLAY COMPLETE", flush=True)
    print(f"  Time: {elapsed:.1f} minutes", flush=True)
    print(f"  Questions: {summary['total_questions']}", flush=True)
    print(f"  7B Pass: {summary['pass']}", flush=True)
    print(f"  7B Fail: {summary['fail']}", flush=True)
    print(f"  7B Partial: {summary['partial']}", flush=True)
    print(f"  Training data: {summary['training_items']} items", flush=True)
    print(f"  Categories: {summary['by_category']}", flush=True)
    print(f"{'#'*60}", flush=True)


if __name__ == "__main__":
    main()
