#!/usr/bin/env python3
"""
CODE-JEPA Data Collection — Parse source code into AST features
================================================================

Extracts structural code features from Python files for CCODEJEPA training.

Features per code segment (7 slots):
  1. control_flow    — if/for/while/try density
  2. data_structures — class/dict/list/tuple density
  3. api_calls       — function call density
  4. error_handling  — try/except/raise density
  5. side_effects    — IO/global/assign density
  6. type_system     — type hints density
  7. concurrency     — async/await/threading density

Each file is split into segments of LINES_PER_SEGMENT lines.
Each segment produces a feature vector of 7 dimensions.

Usage:
  python collect_code_data.py --source D:\VORTEX_FLAME --output D:\VORTEX_FLAME\code_jepa_data
"""

import argparse
import ast
import json
import os
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch

LINES_PER_SEGMENT = 30
N_FEATURES = 7
DATA_DIR = r"D:\VORTEX_FLAME\code_jepa_data"

FEATURE_NAMES = [
    "control_flow", "data_structures", "api_calls",
    "error_handling", "side_effects", "type_system", "concurrency",
]


class ASTFeatureExtractor(ast.NodeVisitor):
    def __init__(self):
        self.counts = Counter()
        self.n_lines = 0

    def visit_If(self, node):
        self.counts["control_flow"] += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.counts["control_flow"] += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.counts["control_flow"] += 1
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.counts["data_structures"] += 1
        self.generic_visit(node)

    def visit_Dict(self, node):
        self.counts["data_structures"] += 1
        self.generic_visit(node)

    def visit_List(self, node):
        self.counts["data_structures"] += 1
        self.generic_visit(node)

    def visit_Tuple(self, node):
        self.counts["data_structures"] += 1
        self.generic_visit(node)

    def visit_Call(self, node):
        self.counts["api_calls"] += 1
        self.generic_visit(node)

    def visit_Try(self, node):
        self.counts["error_handling"] += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        self.counts["error_handling"] += 1
        self.generic_visit(node)

    def visit_Raise(self, node):
        self.counts["error_handling"] += 1
        self.generic_visit(node)

    def visit_Assign(self, node):
        self.counts["side_effects"] += 1
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self.counts["side_effects"] += 1
        self.generic_visit(node)

    def visit_Global(self, node):
        self.counts["side_effects"] += 1
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self.counts["type_system"] += 1
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.returns is not None:
            self.counts["type_system"] += 1
        for arg in node.args.args:
            if arg.annotation is not None:
                self.counts["type_system"] += 1
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.counts["concurrency"] += 1
        if node.returns is not None:
            self.counts["type_system"] += 1
        self.generic_visit(node)

    def visit_Await(self, node):
        self.counts["concurrency"] += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node):
        self.counts["concurrency"] += 1
        self.generic_visit(node)

    def visit_AsyncWith(self, node):
        self.counts["concurrency"] += 1
        self.generic_visit(node)

    def to_feature_vector(self) -> np.ndarray:
        vec = np.zeros(N_FEATURES, dtype=np.float32)
        total = max(sum(self.counts.values()), 1)
        for i, name in enumerate(FEATURE_NAMES):
            vec[i] = self.counts.get(name, 0) / total
        return vec


def extract_features_from_code(source_code: str) -> np.ndarray:
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return np.zeros(N_FEATURES, dtype=np.float32)

    extractor = ASTFeatureExtractor()
    extractor.visit(tree)
    extractor.n_lines = source_code.count("\n") + 1
    return extractor.to_feature_vector()


def process_file(filepath: str) -> list:
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return []

    segments = []
    for i in range(0, len(lines), LINES_PER_SEGMENT):
        chunk = "".join(lines[i:i + LINES_PER_SEGMENT])
        if len(chunk.strip()) < 10:
            continue
        features = extract_features_from_code(chunk)
        segments.append(features)

    return segments


def scan_directory(root_dir: str, extensions=None) -> list:
    if extensions is None:
        extensions = [".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"]

    all_files = []
    for ext in extensions:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            skip_dirs = {"node_modules", ".git", "__pycache__", "venv", ".venv",
                         "dist", "build", ".tox", "egg-info", ".mypy_cache"}
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for fname in filenames:
                if fname.endswith(ext):
                    all_files.append(os.path.join(dirpath, fname))

    return all_files


def collect_code_data(source_dirs: list, output_dir: str, max_files: int = 500):
    os.makedirs(output_dir, exist_ok=True)

    all_files = []
    for d in source_dirs:
        if os.path.isdir(d):
            files = scan_directory(d)
            all_files.extend(files)
            print(f"  Scanned {d}: {len(files)} files", flush=True)

    all_files = all_files[:max_files]
    print(f"\nProcessing {len(all_files)} files...", flush=True)

    manifest = []
    total_segments = 0

    for i, filepath in enumerate(all_files):
        segments = process_file(filepath)
        if len(segments) < 2:
            continue

        segments_arr = np.stack(segments)
        fname = f"code_{i:04d}.pt"
        fpath = os.path.join(output_dir, fname)

        torch.save({
            "features": torch.from_numpy(segments_arr),
            "source": os.path.basename(filepath),
            "n_segments": len(segments),
        }, fpath)

        manifest.append({
            "file": fname,
            "source": os.path.basename(filepath),
            "n_segments": len(segments),
        })
        total_segments += len(segments)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(all_files)}] {total_segments} segments", flush=True)

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_files": len(manifest),
            "total_segments": total_segments,
            "feature_names": FEATURE_NAMES,
            "lines_per_segment": LINES_PER_SEGMENT,
            "sources": [os.path.basename(d) for d in source_dirs],
            "files": manifest,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nDone! {len(manifest)} files, {total_segments} segments saved to {output_dir}")
    return len(manifest), total_segments


def generate_synthetic_code(n_files=20, n_segments=25):
    print(f"  Generating {n_files} synthetic code patterns...", flush=True)
    results = []

    patterns = [
        "api_heavy", "control_heavy", "data_heavy", "async_heavy",
        "typed", "error_heavy", "balanced", "minimal",
        "callback", "oop", "functional", "script",
    ]

    for i in range(n_files):
        pattern = patterns[i % len(patterns)]
        segments = []
        for s in range(n_segments):
            vec = np.zeros(N_FEATURES, dtype=np.float32)
            noise = np.random.uniform(0, 0.1, N_FEATURES).astype(np.float32)

            if pattern == "api_heavy":
                vec[2] = np.random.uniform(0.4, 0.8)
                vec[4] = np.random.uniform(0.1, 0.3)
            elif pattern == "control_heavy":
                vec[0] = np.random.uniform(0.4, 0.7)
                vec[3] = np.random.uniform(0.1, 0.3)
            elif pattern == "data_heavy":
                vec[1] = np.random.uniform(0.4, 0.7)
                vec[5] = np.random.uniform(0.1, 0.3)
            elif pattern == "async_heavy":
                vec[6] = np.random.uniform(0.4, 0.7)
                vec[2] = np.random.uniform(0.1, 0.3)
            elif pattern == "typed":
                vec[5] = np.random.uniform(0.4, 0.7)
                vec[1] = np.random.uniform(0.1, 0.3)
            elif pattern == "error_heavy":
                vec[3] = np.random.uniform(0.4, 0.7)
                vec[0] = np.random.uniform(0.1, 0.3)
            elif pattern == "balanced":
                vec = np.random.dirichlet(np.ones(N_FEATURES)).astype(np.float32)
            elif pattern == "minimal":
                vec[2] = np.random.uniform(0.5, 0.9)
            elif pattern == "callback":
                vec[2] = np.random.uniform(0.3, 0.5)
                vec[0] = np.random.uniform(0.2, 0.4)
            elif pattern == "oop":
                vec[1] = np.random.uniform(0.3, 0.5)
                vec[5] = np.random.uniform(0.2, 0.4)
            elif pattern == "functional":
                vec[2] = np.random.uniform(0.3, 0.5)
                vec[1] = np.random.uniform(0.1, 0.3)
            else:
                vec[2] = np.random.uniform(0.3, 0.6)

            vec = vec + noise
            vec = vec / max(vec.sum(), 1e-8)
            segments.append(vec)

        results.append({
            "features": np.stack(segments),
            "pattern": pattern,
            "n_segments": n_segments,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", nargs="+", default=[r"D:\VORTEX_FLAME"])
    parser.add_argument("--output", default=DATA_DIR)
    parser.add_argument("--max-files", type=int, default=500)
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("CODE-JEPA Data Collection")
    print("=" * 60)

    if not args.synthetic_only:
        n_files, n_segs = collect_code_data(args.source, args.output, args.max_files)
    else:
        n_files, n_segs = 0, 0

    syn_results = generate_synthetic_code(n_files=20, n_segments=25)
    syn_dir = os.path.join(args.output, "synthetic")
    os.makedirs(syn_dir, exist_ok=True)

    syn_manifest = []
    for i, r in enumerate(syn_results):
        fname = f"syn_{r['pattern']}_{i:03d}.pt"
        fpath = os.path.join(syn_dir, fname)
        torch.save({
            "features": torch.from_numpy(r["features"]),
            "source": f"synthetic_{r['pattern']}",
            "n_segments": r["n_segments"],
        }, fpath)
        syn_manifest.append({"file": fname, "pattern": r["pattern"], "n_segments": r["n_segments"]})

    syn_manifest_path = os.path.join(syn_dir, "manifest.json")
    with open(syn_manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_files": len(syn_manifest),
            "total_segments": sum(r["n_segments"] for r in syn_manifest),
            "type": "synthetic",
            "files": syn_manifest,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nSynthetic: {len(syn_manifest)} files added")
    print(f"Total: {n_files + len(syn_manifest)} files, {n_segs + sum(r['n_segments'] for r in syn_results)} segments")


if __name__ == "__main__":
    main()
