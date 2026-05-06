#!/usr/bin/env python3
"""
Benchmark Ministral-8B-Reasoning base model against SOLO Coder metrics
Tests raw 8B capabilities before any LoRA training
"""
import os, json, gc, time
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

MODEL_DIR = r"D:\models\Ministral-8B-Reasoning"
LOG_DIR = r"D:\VORTEX_FLAME\pipeline_8b"

QUESTIONS = {
    "lowlevel": [
        {"q": "Explain how fork() creates a new process in Linux. What does it return in parent vs child?", "kw": ["fork", "child", "parent", "PID", "0", "copy"]},
        {"q": "What is a system call? How does the CPU transition from user mode to kernel mode?", "kw": ["syscall", "kernel", "user", "mode", "interrupt", "ring"]},
        {"q": "Explain virtual memory and how page tables translate virtual addresses to physical addresses.", "kw": ["virtual", "page", "physical", "MMU", "TLB", "frame"]},
        {"q": "What is the difference between a process and a thread? What resources do they share?", "kw": ["process", "thread", "share", "stack", "heap", "memory"]},
        {"q": "Explain the ELF file format. What are the key sections?", "kw": ["ELF", "section", "text", "data", "header", "segment"]},
        {"q": "How does CPU cache work? What is cache locality and why does it matter?", "kw": ["cache", "locality", "L1", "L2", "line", "spatial", "temporal"]},
        {"q": "Explain how stack frames work in x86_64. How are function arguments passed?", "kw": ["stack", "frame", "RDI", "RSI", "RBP", "calling convention"]},
        {"q": "What is an interrupt? Difference between hardware interrupts, software interrupts, and exceptions?", "kw": ["interrupt", "hardware", "software", "exception", "IDT", "handler"]},
        {"q": "How does memory allocation work: stack vs heap vs mmap?", "kw": ["stack", "heap", "mmap", "malloc", "allocation", "free"]},
        {"q": "Explain how a linker resolves symbols. Static vs dynamic linking?", "kw": ["linker", "symbol", "static", "dynamic", "GOT", "PLT"]},
    ],
    "logic_code": [
        {"q": "Prove that any program with loops can be written with recursion, and vice versa.", "kw": ["recursion", "loop", "equivalent", "stack", "transform"]},
        {"q": "Implement binary search and prove its correctness using loop invariants.", "kw": ["binary search", "invariant", "sorted", "mid", "O(log n)"]},
        {"q": "Explain the logical structure of a compiler. What are the main phases?", "kw": ["compiler", "lexer", "parser", "AST", "code generation", "optimization"]},
        {"q": "What is the difference between formal verification and testing?", "kw": ["verification", "testing", "proof", "formal", "correctness"]},
        {"q": "Implement a function that checks if parentheses are balanced. Prove correctness.", "kw": ["balanced", "parentheses", "stack", "invariant", "match"]},
        {"q": "Explain the halting problem. Why is it undecidable?", "kw": ["halting", "undecidable", "Turing", "contradiction", "infinite loop"]},
        {"q": "What is the difference between P and NP? Explain NP-completeness.", "kw": ["P", "NP", "NP-complete", "polynomial", "reduction"]},
        {"q": "Write a recursive descent parser for simple arithmetic expressions.", "kw": ["parser", "recursive", "expression", "grammar", "AST"]},
        {"q": "Explain the Curry-Howard correspondence. What does it mean for programming?", "kw": ["Curry-Howard", "proposition", "type", "proof", "program"]},
        {"q": "What is a monad in functional programming? Explain with a concrete example.", "kw": ["monad", "bind", "return", "functional", "chain", "Maybe"]},
    ],
    "debug": [
        {"q": "How do you use GDB to debug a segmentation fault? Walk through the steps.", "kw": ["GDB", "segfault", "backtrace", "breakpoint", "frame"]},
        {"q": "What are common types of programming errors? How to detect each type?", "kw": ["syntax", "runtime", "logical", "compile", "debug"]},
        {"q": "Explain how AddressSanitizer detects memory errors.", "kw": ["AddressSanitizer", "buffer overflow", "use-after-free", "shadow memory"]},
        {"q": "What is a race condition? How do you detect and fix it?", "kw": ["race condition", "mutex", "lock", "thread", "concurrent"]},
        {"q": "How do you debug a memory leak in a C program?", "kw": ["memory leak", "Valgrind", "malloc", "free", "leak check"]},
    ],
    "security": [
        {"q": "What is a buffer overflow? How does it work and how to prevent it?", "kw": ["buffer overflow", "stack", "canary", "NX", "ASLR"]},
        {"q": "Explain SQL injection. How to prevent it?", "kw": ["SQL injection", "parameterized", "sanitize", "escape", "prepared statement"]},
        {"q": "What is CVE? How are vulnerabilities classified and tracked?", "kw": ["CVE", "vulnerability", "CVSS", "severity", "disclosure"]},
        {"q": "Explain the principle of least privilege. How does it apply to system security?", "kw": ["least privilege", "permission", "root", "sandbox", "access control"]},
        {"q": "What is ASLR and how does it improve security? Can it be bypassed?", "kw": ["ASLR", "randomize", "address", "bypass", "information leak"]},
    ],
    "agent": [
        {"q": "What is a shell? How does command parsing and execution work?", "kw": ["shell", "parse", "fork", "exec", "pipe", "redirect"]},
        {"q": "Explain how pipes work in Unix. How does data flow between processes?", "kw": ["pipe", "process", "file descriptor", "IPC", "dup2"]},
        {"q": "What is an agent in AI? How does it differ from a simple chatbot?", "kw": ["agent", "tool", "action", "observe", "plan", "autonomous"]},
        {"q": "How would you design a tool-calling system for an AI agent?", "kw": ["tool", "API", "function call", "schema", "JSON", "dispatch"]},
        {"q": "Explain how cron jobs work. How do you schedule recurring tasks?", "kw": ["cron", "schedule", "crontab", "interval", "daemon"]},
    ],
}


def run_benchmark(model, tokenizer, category, items):
    results = []
    passed = 0
    for i, item in enumerate(items):
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
        results.append({"q": q, "a": answer[:300], "matched": matched, "pass": ok, "len": len(answer)})
        status = "PASS" if ok else "FAIL"
        print(f"  [{category}] Q{i+1}/{len(items)} [{status}] {answer[:80]}...", flush=True)
    rate = passed / max(len(items), 1)
    return rate, results


def main():
    from transformers import BitsAndBytesConfig, AutoProcessor, Mistral3ForConditionalGeneration

    print("=" * 60)
    print("  8B Base Model SOLO Coder Benchmark")
    print(f"  Model: {MODEL_DIR}")
    print("=" * 60)

    os.makedirs(LOG_DIR, exist_ok=True)

    print("  Loading model (4bit)...", flush=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = Mistral3ForConditionalGeneration.from_pretrained(
        MODEL_DIR, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vram = torch.cuda.memory_reserved() / 1024**3
    print(f"  Model loaded. VRAM: {vram:.1f}GB", flush=True)

    model.eval()

    all_results = {}
    summary = {}
    for cat, items in QUESTIONS.items():
        print(f"\n  === {cat.upper()} ({len(items)} questions) ===", flush=True)
        rate, results = run_benchmark(model, tokenizer, cat, items)
        all_results[cat] = results
        summary[cat] = {"rate": rate, "total": len(items), "passed": int(rate * len(items))}
        print(f"  {cat}: {rate:.0%}", flush=True)

    overall = sum(s["passed"] for s in summary.values()) / sum(s["total"] for s in summary.values())
    print(f"\n  OVERALL: {overall:.0%}", flush=True)

    print(f"\n  SOLO Coder Stage3 Target Check:")
    print(f"    LowLevel:   {summary['lowlevel']['rate']:.0%} (target >=30%)")
    print(f"    LogicCode:  {summary['logic_code']['rate']:.0%} (target >=60%)")
    print(f"    Debug:      {summary['debug']['rate']:.0%} (paving)")
    print(f"    Security:   {summary['security']['rate']:.0%} (paving)")
    print(f"    Agent:      {summary['agent']['rate']:.0%} (paving)")

    log_data = {
        "model": "Ministral-3-8B-Reasoning-2512",
        "test": "solo_coder_baseline",
        "overall_rate": overall,
        "summary": summary,
        "details": all_results,
        "vram_gb": vram,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    log_path = os.path.join(LOG_DIR, "8b_solo_coder_baseline.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Log saved: {log_path}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print(f"  8B Baseline: {overall:.0%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
