#!/usr/bin/env python3
"""
8B Pipeline Runner - Execute all steps in sequence
Step 1: prep_8b_model.py      (copy model to D: SSD)
Step 2: analyze_8b_layers.py   (brain surgery analysis)
Step 3: train_8b_s1.py         (stage1 training)
Step 4: infer_8b_1060.py       (1060 inference test)
"""
import os, sys, json, time, subprocess

PIPELINE_DIR = r"D:\VORTEX_FLAME\pipeline_8b"
LOG_DIR = r"D:\VORTEX_FLAME\hermes_logs"

STEPS = [
    {
        "name": "Model Prep",
        "script": "prep_8b_model.py",
        "desc": "Copy 8B model from E: (HDD) to D: (SSD)",
        "check": lambda: os.path.exists(r"D:\models\Ministral-8B-Reasoning\consolidated.safetensors"),
        "skip_if_done": True,
    },
    {
        "name": "Brain Surgery",
        "script": "analyze_8b_layers.py",
        "desc": "Analyze layer importance for pruning",
        "check": lambda: os.path.exists(os.path.join(PIPELINE_DIR, "layer_importance_report.json")),
        "skip_if_done": True,
    },
    {
        "name": "Stage1 Training",
        "script": "train_8b_s1.py",
        "desc": "Pure math foundation training",
        "check": lambda: os.path.exists(os.path.join(LOG_DIR, os.environ.get("SOUL_NAME", "galileo"), "stage1_result.json")),
        "skip_if_done": False,
    },
    {
        "name": "1060 Inference",
        "script": "infer_8b_1060.py",
        "desc": "Test inference on 1060 GPU",
        "check": None,
        "skip_if_done": False,
    },
]


def run_step(step):
    script_path = os.path.join(PIPELINE_DIR, step["script"])
    if not os.path.exists(script_path):
        print(f"  [FATAL] Script not found: {script_path}")
        return False

    if step["skip_if_done"] and step["check"] and step["check"]():
        print(f"  SKIP (already done): {step['name']}")
        return True

    print(f"\n  Running: {step['name']} - {step['desc']}")
    print(f"  Script: {step['script']}")
    print(f"  {'='*50}")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PIPELINE_DIR,
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"  DONE in {elapsed/60:.1f} min")
        return True
    else:
        print(f"  FAILED (exit code: {result.returncode})")
        return False


def main():
    soul_name = sys.argv[1] if len(sys.argv) > 1 else "galileo"
    os.environ["SOUL_NAME"] = soul_name

    print("=" * 60)
    print("  8B Pipeline Runner")
    print(f"  Soul: {soul_name}")
    print(f"  Steps: {len(STEPS)}")
    print("=" * 60)

    results = []
    for i, step in enumerate(STEPS):
        print(f"\n  [{i+1}/{len(STEPS)}] {step['name']}")
        ok = run_step(step)
        results.append({"step": step["name"], "ok": ok})
        if not ok:
            print(f"\n  Pipeline STOPPED at step {i+1}: {step['name']}")
            break

    print("\n" + "=" * 60)
    print("  Pipeline Summary:")
    for r in results:
        status = "PASS" if r["ok"] else "FAIL"
        print(f"    {r['step']}: {status}")
    passed = sum(1 for r in results if r["ok"])
    print(f"  Total: {passed}/{len(results)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
