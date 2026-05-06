#!/usr/bin/env python3
"""
Logic Benchmark - 7B vs 8B Cezanne_PRO Comparison
The REAL differentiator: logical reasoning, not instruction following

Dimensions:
  1. Multi-step Math Reasoning (5 Q)
  2. Causal Chain & Counterfactual (5 Q)
  3. Logical Fallacy Identification (5 Q)
  4. Long-text Logical Consistency (5 Q)
  5. Formal Logic & Proof (5 Q)

Total: 25 questions, each scored 0-2
  0 = wrong/illogical
  1 = partially correct
  2 = fully correct with clear reasoning chain
"""
import os, sys, json, time, gc
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

QUESTIONS = [
    {
        "id": "math_1",
        "dim": "multi_step_math",
        "q": "A farmer has 100 meters of fence. What is the maximum area he can enclose with a rectangular fence? Prove your answer step by step using calculus.",
        "scoring": "2=correct calculus proof with derivative, 1=correct answer no proof, 0=wrong",
        "keywords": ["derivative", "50", "2500", "square", "maximum", "2x+2y=100"],
    },
    {
        "id": "math_2",
        "dim": "multi_step_math",
        "q": "Prove that the sum of the first n odd numbers equals n squared. Use mathematical induction and show each step explicitly.",
        "scoring": "2=complete induction proof, 1=correct formula no induction, 0=wrong",
        "keywords": ["induction", "base case", "n=1", "inductive step", "2k+1", "n^2"],
    },
    {
        "id": "math_3",
        "dim": "multi_step_math",
        "q": "A tank has two inlet pipes and one outlet pipe. Pipe A fills the tank in 6 hours, Pipe B fills it in 8 hours, and Pipe C drains it in 12 hours. If all three are opened simultaneously, how long until the tank is full? Show the complete reasoning chain.",
        "scoring": "2=correct answer with full reasoning, 1=setup correct but arithmetic error, 0=wrong approach",
        "keywords": ["1/6", "1/8", "1/12", "rate", "24/5", "4.8"],
    },
    {
        "id": "math_4",
        "dim": "multi_step_math",
        "q": "Prove using contradiction: the square root of 2 is irrational. Label each logical step and its justification.",
        "scoring": "2=complete proof with labeled steps, 1=correct idea incomplete proof, 0=wrong",
        "keywords": ["contradiction", "assume", "rational", "p/q", "even", "2q^2", "coprime"],
    },
    {
        "id": "math_5",
        "dim": "multi_step_math",
        "q": "In a group of 100 people, 70 speak English, 50 speak French, and 30 speak both. How many speak neither? Show your reasoning using set theory.",
        "scoring": "2=correct with set theory, 1=correct answer no set theory, 0=wrong",
        "keywords": ["union", "intersection", "70+50-30", "90", "10", "neither"],
    },
    {
        "id": "causal_1",
        "dim": "causal_counterfactual",
        "q": "If the Roman Empire had not fallen, how might the development of modern science have been different? Construct a causal chain with at least 3 links, and identify where your reasoning is most speculative.",
        "scoring": "2=clear causal chain with uncertainty markers, 1=chain but no uncertainty, 0=no chain",
        "keywords": ["causal", "chain", "speculative", "uncertain", "link", "might", "could"],
    },
    {
        "id": "causal_2",
        "dim": "causal_counterfactual",
        "q": "Explain the causal chain from 'increased CO2 emissions' to 'sea level rise'. At each step, identify whether the link is necessary, sufficient, or contributory.",
        "scoring": "2=correct chain with necessity/sufficiency labels, 1=chain without labels, 0=wrong",
        "keywords": ["CO2", "greenhouse", "temperature", "thermal expansion", "ice melt", "necessary", "sufficient", "contributory"],
    },
    {
        "id": "causal_3",
        "dim": "causal_counterfactual",
        "q": "A city implements a congestion charge and traffic decreases by 15%. A critic says 'the decrease is due to a recession, not the charge.' Design a logical test to determine which explanation is correct.",
        "scoring": "2=valid test design with control, 1=partial test, 0=no test",
        "keywords": ["control", "compare", "without charge", "recession", "confound", "variable", "counterfactual"],
    },
    {
        "id": "causal_4",
        "dim": "causal_counterfactual",
        "q": "If gravity were twice as strong, what would happen to the period of a pendulum? Derive the relationship and explain the causal logic.",
        "scoring": "2=correct derivation with causal explanation, 1=correct direction no derivation, 0=wrong",
        "keywords": ["period", "pendulum", "2pi", "sqrt", "L/g", "inverse", "shorter", "1/sqrt(2)"],
    },
    {
        "id": "causal_5",
        "dim": "causal_counterfactual",
        "q": "Country A raises interest rates. Country B's currency depreciates against Country A's. Explain the causal chain. Is this necessary or contingent? What could break the chain?",
        "scoring": "2=complete chain with contingency analysis, 1=chain no contingency, 0=wrong",
        "keywords": ["interest rate", "capital flow", "demand", "currency", "appreciate", "depreciate", "contingent", "break"],
    },
    {
        "id": "fallacy_1",
        "dim": "logical_fallacy",
        "q": "Identify the logical fallacy in: 'You can't trust John's argument about climate change because he drives an SUV.' Name the fallacy, explain why it's fallacious, and reconstruct the argument without the fallacy.",
        "scoring": "2=correct fallacy name + explanation + reconstruction, 1=name only, 0=wrong",
        "keywords": ["ad hominem", "tu quoque", "hypocrisy", "irrelevant", "character", "argument"],
    },
    {
        "id": "fallacy_2",
        "dim": "logical_fallacy",
        "q": "Is this argument valid? 'All birds can fly. Penguins are birds. Therefore penguins can fly.' Identify the specific logical error and explain the difference between logical validity and soundness.",
        "scoring": "2=validity vs soundness distinction, 1=identifies error, 0=wrong",
        "keywords": ["valid", "sound", "false premise", "unsound", "logically valid", "counterexample"],
    },
    {
        "id": "fallacy_3",
        "dim": "logical_fallacy",
        "q": "A politician says: 'Either we cut taxes or the economy will collapse.' Name this fallacy, explain why it's fallacious, and give the general form of this type of fallacy.",
        "scoring": "2=correct name + general form, 1=name only, 0=wrong",
        "keywords": ["false dilemma", "false dichotomy", "either-or", "black and white", "alternatives", "excluded middle"],
    },
    {
        "id": "fallacy_4",
        "dim": "logical_fallacy",
        "q": "What fallacy is this? 'This medicine has been used for thousands of years, so it must be effective.' Name it, explain the error, and show how to properly evaluate the medicine's effectiveness.",
        "scoring": "2=correct fallacy + proper evaluation method, 1=name only, 0=wrong",
        "keywords": ["appeal to tradition", "appeal to antiquity", "clinical trial", "evidence", "placebo", "anecdotal"],
    },
    {
        "id": "fallacy_5",
        "dim": "logical_fallacy",
        "q": "Identify the fallacy: 'If we allow same-sex marriage, next people will want to marry their pets.' Name the fallacy, explain the logical error, and show where the chain of reasoning breaks down.",
        "scoring": "2=correct fallacy + breakdown analysis, 1=name only, 0=wrong",
        "keywords": ["slippery slope", "slippery", "unjustified", "link", "chain", "intermediate", "no evidence"],
    },
    {
        "id": "consist_1",
        "dim": "logical_consistency",
        "q": "A person says: 'I believe in absolute freedom of speech. But hate speech should be banned.' Is this logically consistent? If not, identify the contradiction and propose a way to resolve it.",
        "scoring": "2=identifies contradiction + resolution, 1=identifies contradiction, 0=wrong",
        "keywords": ["contradiction", "consistent", "absolute", "exception", "qualify", "resolve", "conditional"],
    },
    {
        "id": "consist_2",
        "dim": "logical_consistency",
        "q": "A company policy states: 'All employees must attend the meeting. Employees on leave are exempt.' Under what conditions is this consistent? Under what conditions is it contradictory? Analyze formally.",
        "scoring": "2=formal analysis with conditions, 1=informal analysis, 0=wrong",
        "keywords": ["universal", "exception", "consistent", "contradictory", "leave", "scope", "quantifier", "condition"],
    },
    {
        "id": "consist_3",
        "dim": "logical_consistency",
        "q": "Analyze this argument for logical consistency: 'No one should be above the law. The president has executive privilege. Executive privilege means the president is not subject to certain laws.' Identify all logical tensions and propose resolutions.",
        "scoring": "2=identifies all tensions + resolutions, 1=identifies some tensions, 0=wrong",
        "keywords": ["tension", "contradiction", "above the law", "privilege", "exception", "scope", "resolve", "limited"],
    },
    {
        "id": "consist_4",
        "dim": "logical_consistency",
        "q": "A philosopher argues: 'All knowledge comes from experience. But the concept of experience itself is not derived from experience.' Is this a genuine paradox or a linguistic confusion? Analyze step by step.",
        "scoring": "2=correct analysis distinguishing paradox vs confusion, 1=partial analysis, 0=wrong",
        "keywords": ["paradox", "self-reference", "circular", "meta", "linguistic", "category", "level", "meta-level"],
    },
    {
        "id": "consist_5",
        "dim": "logical_consistency",
        "q": "Check for logical consistency: 'Every rule has an exception. This statement is a rule.' If both statements are true, what follows? Is this a paradox? Analyze formally.",
        "scoring": "2=formal analysis of the paradox, 1=identifies the issue, 0=wrong",
        "keywords": ["paradox", "self-reference", "exception", "infinite regress", "Russell", "Godel", "meta-level"],
    },
    {
        "id": "formal_1",
        "dim": "formal_logic",
        "q": "Using propositional logic, prove: (P -> Q) AND (Q -> R) => (P -> R). Show each step with the inference rule used.",
        "scoring": "2=complete formal proof with rules, 1=correct informal proof, 0=wrong",
        "keywords": ["modus ponens", "hypothetical syllogism", "P->Q", "Q->R", "P->R", "step", "inference"],
    },
    {
        "id": "formal_2",
        "dim": "formal_logic",
        "q": "Translate into first-order logic: 'Every student who studies hard passes the exam. Some students study hard but fail.' Show that these statements are consistent (not contradictory).",
        "scoring": "2=correct FOL translation + consistency proof, 1=correct translation only, 0=wrong",
        "keywords": ["for all", "exists", "student", "studies", "passes", "consistent", "counterexample", "not all"],
    },
    {
        "id": "formal_3",
        "dim": "formal_logic",
        "q": "Prove using natural deduction: NOT(P AND Q) => NOT P OR NOT Q. This is one of De Morgan's Laws. Show each derivation step with the rule applied.",
        "scoring": "2=complete natural deduction proof, 1=truth table proof, 0=wrong",
        "keywords": ["De Morgan", "negation", "conjunction", "disjunction", "natural deduction", "assumption", "derive"],
    },
    {
        "id": "formal_4",
        "dim": "formal_logic",
        "q": "Is this argument valid in first-order logic? 'All cats are animals. Some animals are pets. Therefore some cats are pets.' If invalid, show a countermodel. If valid, prove it.",
        "scoring": "2=correct invalidity + countermodel, 1=says invalid without countermodel, 0=says valid",
        "keywords": ["invalid", "countermodel", "does not follow", "not necessarily", "interpretation", "all cats", "some animals"],
    },
    {
        "id": "formal_5",
        "dim": "formal_logic",
        "q": "Using predicate logic, formalize and evaluate: 'There exists a number greater than all other numbers.' Is this satisfiable? Is it valid? Explain the difference and give your reasoning.",
        "scoring": "2=correct formalization + satisfiability/validity analysis, 1=correct answer no formalization, 0=wrong",
        "keywords": ["exists", "for all", "greater", "satisfiable", "valid", "not valid", "infinite", "no maximum"],
    },
]


def score_answer(answer, question):
    keywords = question["keywords"]
    matched = [kw for kw in keywords if kw.lower() in answer.lower()]
    match_ratio = len(matched) / len(keywords) if keywords else 0
    length_adequate = len(answer) > 150

    if match_ratio >= 0.4 and length_adequate:
        return 2
    elif match_ratio >= 0.2 or length_adequate:
        return 1
    else:
        return 0


def run_benchmark(model, tokenizer, model_name):
    print(f"\n{'='*60}", flush=True)
    print(f"  Logic Benchmark: {model_name}", flush=True)
    print(f"  25 questions across 5 dimensions", flush=True)
    print(f"{'='*60}", flush=True)

    results = []
    dim_scores = {}
    model.eval()

    for i, q_data in enumerate(QUESTIONS):
        prompt = f"[INST] {q_data['q']} [/INST] "
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=500, temperature=0.7, do_sample=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
        score = score_answer(answer, q_data)
        dim = q_data["dim"]

        if dim not in dim_scores:
            dim_scores[dim] = {"total": 0, "max": 0}
        dim_scores[dim]["total"] += score
        dim_scores[dim]["max"] += 2

        safe_answer = answer[:100].encode('ascii', errors='replace').decode()
        print(f"  {q_data['id']} [{score}/2] {safe_answer}...", flush=True)
        results.append({
            "id": q_data["id"], "dim": dim, "score": score,
            "answer_preview": answer[:300],
            "matched_keywords": [kw for kw in q_data["keywords"] if kw.lower() in answer.lower()],
        })

    total = sum(d["total"] for d in dim_scores.values())
    max_total = sum(d["max"] for d in dim_scores.values())

    print(f"\n  --- {model_name} Results ---", flush=True)
    dim_names = {
        "multi_step_math": "Multi-step Math",
        "causal_counterfactual": "Causal/Counterfactual",
        "logical_fallacy": "Logical Fallacy",
        "logical_consistency": "Logical Consistency",
        "formal_logic": "Formal Logic",
    }
    for dim, scores in dim_scores.items():
        name = dim_names.get(dim, dim)
        pct = scores["total"] / scores["max"] * 100 if scores["max"] > 0 else 0
        print(f"  {name}: {scores['total']}/{scores['max']} ({pct:.0f}%)", flush=True)
    print(f"  TOTAL: {total}/{max_total} ({total/max_total*100:.0f}%)", flush=True)

    return {"model": model_name, "total": total, "max": max_total,
            "pct": total/max_total*100, "dim_scores": dim_scores, "details": results}


def main():
    from transformers import Mistral3ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    LOG_DIR = r"D:\VORTEX_FLAME\hermes_logs"
    os.makedirs(LOG_DIR, exist_ok=True)

    print("=" * 60, flush=True)
    print("  Logic Benchmark: Cezanne (7B) vs Cezanne_PRO (8B)", flush=True)
    print("=" * 60, flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)

    # --- 7B Cezanne ---
    print("\n>>> Loading 7B Cezanne (Stage2)...", flush=True)
    model_7b = AutoModelForCausalLM.from_pretrained(
        r"E:\models\Mistral-7B-Instruct-v0.1-Cezanne",
        quantization_config=bnb, device_map="auto", torch_dtype=torch.float16,
    )
    tok_7b = AutoTokenizer.from_pretrained(r"E:\models\Mistral-7B-Instruct-v0.1-Cezanne")
    if tok_7b.pad_token is None:
        tok_7b.pad_token = tok_7b.eos_token
    model_7b = PeftModel.from_pretrained(model_7b, r"D:\VORTEX_FLAME\soul_lora_v2\cezanne\stage2\final")
    result_7b = run_benchmark(model_7b, tok_7b, "Cezanne-7B")
    del model_7b, tok_7b
    gc.collect()
    torch.cuda.empty_cache()

    # --- 8B Cezanne_PRO ---
    print("\n>>> Loading 8B Cezanne_PRO (Stage2)...", flush=True)
    model_8b = Mistral3ForConditionalGeneration.from_pretrained(
        r"D:\models\Ministral-8B-Reasoning-Text-34L-ctx1024",
        quantization_config=bnb, device_map="auto", torch_dtype=torch.float16,
    )
    tok_8b = AutoTokenizer.from_pretrained(r"D:\models\Ministral-8B-Reasoning-Text-34L-ctx1024")
    if tok_8b.pad_token is None:
        tok_8b.pad_token = tok_8b.eos_token
    model_8b = PeftModel.from_pretrained(model_8b, r"D:\VORTEX_FLAME\soul_lora_v2\cezanne_8b\stage2\final")
    result_8b = run_benchmark(model_8b, tok_8b, "Cezanne_PRO-8B")
    del model_8b, tok_8b
    gc.collect()
    torch.cuda.empty_cache()

    # --- Comparison ---
    print(f"\n{'='*60}", flush=True)
    print(f"  FINAL COMPARISON", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Cezanne-7B:      {result_7b['total']}/{result_7b['max']} ({result_7b['pct']:.0f}%)", flush=True)
    print(f"  Cezanne_PRO-8B:  {result_8b['total']}/{result_8b['max']} ({result_8b['pct']:.0f}%)", flush=True)

    dim_names = {
        "multi_step_math": "Multi-step Math",
        "causal_counterfactual": "Causal/Counterfactual",
        "logical_fallacy": "Logical Fallacy",
        "logical_consistency": "Logical Consistency",
        "formal_logic": "Formal Logic",
    }
    print(f"\n  Per-dimension comparison:", flush=True)
    for dim in dim_names:
        s7 = result_7b["dim_scores"].get(dim, {"total": 0, "max": 10})
        s8 = result_8b["dim_scores"].get(dim, {"total": 0, "max": 10})
        p7 = s7["total"]/s7["max"]*100 if s7["max"] else 0
        p8 = s8["total"]/s8["max"]*100 if s8["max"] else 0
        winner = "7B" if p7 > p8 else ("8B" if p8 > p7 else "TIE")
        print(f"  {dim_names[dim]:>25s}: 7B={p7:.0f}% 8B={p8:.0f}% -> {winner}", flush=True)

    verdict = "8B" if result_8b["pct"] > result_7b["pct"] else ("7B" if result_7b["pct"] > result_8b["pct"] else "TIE")
    print(f"\n  VERDICT: {verdict} wins on logical reasoning", flush=True)

    log_data = {
        "benchmark": "logic_v1",
        "questions": len(QUESTIONS),
        "cezanne_7b": result_7b,
        "cezanne_pro_8b": result_8b,
        "verdict": verdict,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    log_path = os.path.join(LOG_DIR, "logic_benchmark_result.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {log_path}", flush=True)


if __name__ == "__main__":
    main()
