#!/usr/bin/env python3
"""7B vs 8B Arena: Cezanne S2+S3_CS vs Cezanne 8B S3
Tests if adversarial debate can spark emergent reasoning
"""
import os, json, gc, time, sys
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

BASE_7B = "/mnt/d/models/Mistral-7B-Instruct-v0.1"
BASE_8B = "/mnt/d/models/Ministral-8B-Reasoning-Text-34L-ctx1024"
LORA_7B_S3 = "/mnt/d/VORTEX_FLAME/soul_lora_v2/cezanne/stage3_cs/final"
LORA_8B_S3 = "/mnt/d/VORTEX_FLAME/soul_lora_v2/cezanne_8b/stage3/final"
LOG = "/mnt/d/VORTEX_FLAME/hermes_logs/arena_7b_8b"

os.makedirs(LOG, exist_ok=True)

ARENA_QUESTIONS = [
    {"cat": "proof", "q": "Prove that the square root of 2 is irrational. Label each step formally."},
    {"cat": "proof2", "q": "Prove by induction: 1+2+3+...+n = n(n+1)/2 for all positive integers n."},
    {"cat": "algo", "q": "Design an algorithm to find the kth smallest element in an unsorted array in O(n) average time. Explain why it works."},
    {"cat": "concurrency", "q": "Design a thread-safe bounded buffer (producer-consumer) using semaphores. Prove it cannot deadlock."},
    {"cat": "distributed", "q": "In a distributed key-value store with 3 replicas, design a protocol that tolerates 1 failure and guarantees linearizable reads. Prove correctness."},
    {"cat": "os", "q": "Explain what happens when you type 'ls' in a shell. Trace from keyboard interrupt to process creation to output display."},
    {"cat": "compiler", "q": "Design a type inference algorithm for a simple functional language with polymorphic types (Hindley-Milner). Explain unification."},
    {"cat": "security", "q": "Explain how TLS 1.3 handshake works. Why is it faster than TLS 1.2? What cryptographic assumptions does it rely on?"},
    {"cat": "network", "q": "A TCP connection has RTT=50ms and bandwidth=100Mbps. What is the optimal window size? How does TCP achieve this?"},
    {"cat": "memory", "q": "Explain how a modern garbage collector (e.g., ZGC) achieves sub-millisecond pause times. What trade-offs does it make?"},
    {"cat": "debug", "q": "A multi-threaded program occasionally produces wrong results but passes all unit tests. Describe a systematic debugging strategy."},
    {"cat": "logic", "q": "Using only propositional logic, prove: (P → Q) → ((Q → R) → (P → R)). Show each derivation step."},
]

JUDGE_PROMPT = """You are an impartial judge evaluating two AI responses to the same question.

Question: {question}

Response A (7B Cezanne):
{response_a}

Response B (8B Cezanne):
{response_b}

Evaluate on these criteria (1-10 each):
1. Correctness: Is the answer factually correct?
2. Completeness: Does it cover all important aspects?
3. Depth: Does it show deep understanding beyond surface level?
4. Rigor: Are proofs/formal reasoning properly structured?
5. Clarity: Is the explanation clear and well-organized?

Output JSON only:
{{"correctness_a": N, "completeness_a": N, "depth_a": N, "rigor_a": N, "clarity_a": N,
  "correctness_b": N, "completeness_b": N, "depth_b": N, "rigor_b": N, "clarity_b": N,
  "winner": "A" or "B" or "tie",
  "reasoning": "brief explanation"}}"""


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
        import json as _json
        with open(config_path, "r") as _f:
            _cfg = _json.load(_f)
        if _cfg.get("model_type") == "mistral3" or "Mistral3" in str(_cfg.get("architectures", [])):
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
        print(f"  {label} LoRA loaded: {lora_path}", flush=True)
    else:
        print(f"  {label} base only (no LoRA found)", flush=True)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, question, max_tokens=600, is_8b=False):
    if is_8b:
        prompt = f"[INST] {question} [/INST] "
    else:
        prompt = f"<s>[INST] {question} [/INST] "
    inp = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_tokens, temperature=0.7, do_sample=True,
                             pad_token_id=tokenizer.eos_token_id)
    ans = tokenizer.decode(out[0], skip_special_tokens=True).replace(
        prompt.replace("<s>", ""), ""
    ).strip()
    return ans


def judge_with_8b(model_8b, tokenizer_8b, question, resp_a, resp_b):
    prompt = JUDGE_PROMPT.format(question=question, response_a=resp_a[:800], response_b=resp_b[:800])
    inp = tokenizer_8b(f"[INST] {prompt} [/INST] ", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model_8b.generate(**inp, max_new_tokens=400, temperature=0.3, do_sample=True)
    result = tokenizer_8b.decode(out[0], skip_special_tokens=True)
    try:
        json_start = result.index("{")
        json_end = result.rindex("}") + 1
        return json.loads(result[json_start:json_end])
    except (ValueError, json.JSONDecodeError):
        return {"winner": "parse_error", "raw": result[-200:]}


def main():
    print(f"\n{'='*60}", flush=True)
    print(f"  7B vs 8B ARENA — Cezanne Adversarial Debate", flush=True)
    print(f"  7B: {BASE_7B} + {LORA_7B_S3}", flush=True)
    print(f"  8B: {BASE_8B} + {LORA_8B_S3}", flush=True)
    print(f"  Questions: {len(ARENA_QUESTIONS)}", flush=True)
    print(f"{'='*60}", flush=True)

    model_7b, tok_7b = load_model(BASE_7B, LORA_7B_S3, "7B Cezanne S3-CS")

    results = []
    for i, item in enumerate(ARENA_QUESTIONS):
        print(f"\n  Q{i+1}/{len(ARENA_QUESTIONS)} [{item['cat']}] Generating 7B...", flush=True)
        resp_7b = generate(model_7b, tok_7b, item['q'])
        print(f"  7B: {resp_7b[:100]}...", flush=True)

        results.append({
            "cat": item["cat"], "question": item["q"],
            "response_7b": resp_7b, "response_8b": "(8B not loaded yet)",
            "judgment": None,
        })

    del model_7b, tok_7b
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(10)

    model_8b, tok_8b = load_model(BASE_8B, LORA_8B_S3, "8B Cezanne S3")

    for i, item in enumerate(ARENA_QUESTIONS):
        print(f"\n  Q{i+1}/{len(ARENA_QUESTIONS)} [{item['cat']}] Generating 8B...", flush=True)
        resp_8b = generate(model_8b, tok_8b, item['q'], is_8b=True)
        results[i]["response_8b"] = resp_8b
        print(f"  8B: {resp_8b[:100]}...", flush=True)

    print(f"\n  Judging with 8B...", flush=True)
    for i, item in enumerate(ARENA_QUESTIONS):
        j = judge_with_8b(model_8b, tok_8b, item['q'], results[i]["response_7b"], results[i]["response_8b"])
        results[i]["judgment"] = j
        winner = j.get("winner", "?")
        print(f"  Q{i+1} [{item['cat']}]: Winner={winner}", flush=True)

    wins_7b = sum(1 for r in results if r["judgment"] and r["judgment"].get("winner") == "A")
    wins_8b = sum(1 for r in results if r["judgment"] and r["judgment"].get("winner") == "B")
    ties = sum(1 for r in results if r["judgment"] and r["judgment"].get("winner") == "tie")

    print(f"\n{'='*60}", flush=True)
    print(f"  ARENA RESULTS", flush=True)
    print(f"  7B Wins: {wins_7b}", flush=True)
    print(f"  8B Wins: {wins_8b}", flush=True)
    print(f"  Ties: {ties}", flush=True)
    print(f"{'='*60}", flush=True)

    log_path = os.path.join(LOG, "arena_7b_vs_8b.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "7b_wins": wins_7b, "8b_wins": wins_8b, "ties": ties,
            "total": len(ARENA_QUESTIONS),
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {log_path}", flush=True)

    del model_8b, tok_8b
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
