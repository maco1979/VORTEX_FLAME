import os
import sys
import ast

sys.path.insert(0, r"D:\VORTEX_FLAME")
from soul_memory import write

PROJECT_ROOT = r"D:\VORTEX_FLAME"

SKIP_DIRS = {
    "node_modules", "__pycache__", ".git", ".venv", "venv",
    "dist", "build", ".next", "pnpm", ".mypy_cache",
    "five_layer_jepa", "llama.cpp", "tools", "design_specs",
}

CORE_FILES = {
    "soul_memory.py", "soul_orchestrator.py", "soul_knowledge_alignment.py",
    "cli_anything.py", "guardian.py", "harness_runtime.py",
    "skill_registry_auto.py", "mcp_server_registry.py",
}

ADAPTER_FILES = {f for f in os.listdir(PROJECT_ROOT) if f.endswith("_adapter.py")}

def extract_code_structure(filepath):
    structure: dict = {"classes": [], "functions": [], "imports": []}
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                structure["classes"].append({
                    "name": node.name,
                    "methods": methods[:10],
                    "line": node.lineno,
                })
            elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                structure["functions"].append({
                    "name": node.name,
                    "line": node.lineno,
                    "args": [a.arg for a in node.args.args if a.arg != "self"],
                })
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    structure["imports"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    structure["imports"].append(node.module)

    except Exception as e:
        structure["parse_error"] = str(e)

    return structure

def main():
    print("=" * 60)
    print("Indexing VORTEX_FLAME source code → cezanne code_memory")
    print("=" * 60)

    py_files = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        rel_root = os.path.relpath(root, PROJECT_ROOT)
        skip = False
        for sd in SKIP_DIRS:
            if sd in rel_root.split(os.sep):
                skip = True
                break
        if skip:
            continue
        for f in files:
            if f.endswith(".py"):
                fp = os.path.join(root, f)
                rel = os.path.relpath(fp, PROJECT_ROOT)
                py_files.append((fp, rel))

    print(f"\n  Found {len(py_files)} Python files")

    indexed = 0
    errors = 0
    for fp, rel in sorted(py_files, key=lambda x: x[1]):
        fname = os.path.basename(fp)
        is_core = fname in CORE_FILES
        is_adapter = fname in ADAPTER_FILES

        try:
            structure = extract_code_structure(fp)

            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                source = f.read()

            content = {
                "topic": f"VORTEX_FLAME: {rel}",
                "source": "project_source",
                "type": "code_memory",
                "file_path": rel,
                "is_core": is_core,
                "is_adapter": is_adapter,
                "classes": [c["name"] for c in structure.get("classes", [])],
                "functions": [f["name"] for f in structure.get("functions", [])],
                "imports": structure.get("imports", [])[:20],
                "class_details": structure.get("classes", [])[:5],
                "function_details": structure.get("functions", [])[:10],
                "source_preview": source[:3000],
            }

            if len(source) > 3000:
                content["source_extended"] = source[3000:8000]

            importance = 0.9 if is_core else (0.8 if is_adapter else 0.6)
            tags = ["source_code", "python"]
            if is_core:
                tags.append("core_module")
            if is_adapter:
                tags.append("adapter")
            tags.append(os.path.splitext(fname)[0])

            write(
                soul="cezanne",
                category="code_memory",
                content=content,
                importance=importance,
                tags=tags,
            )
            indexed += 1

            cls_count = len(structure.get("classes", []))
            fn_count = len(structure.get("functions", []))
            print(f"  ✓ {rel} ({cls_count} classes, {fn_count} functions)")

        except Exception as e:
            errors += 1
            print(f"  ✗ {rel}: {e}")

    print(f"\n  Result: {indexed} indexed, {errors} errors")

if __name__ == "__main__":
    main()
