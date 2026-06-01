import os
import sys
import time
import ast
import json

sys.path.insert(0, r"D:\VORTEX_FLAME")
from soul_memory import write

COMFY_ROOT = r"E:\ComfyUI"
BATCH_LOG = 50

SKIP_DIRS = {
    ".ci", ".github", "__pycache__", ".git", "node_modules",
    "custom_nodes", "web", "models", "input", "output",
}

ARCH_KEYWORDS = {
    "flux": "Flux DiT架构",
    "sd1": "Stable Diffusion 1.x",
    "sdxl": "Stable Diffusion XL",
    "cascade": "Stable Cascade",
    "controlnet": "ControlNet条件控制",
    "lora": "LoRA低秩适配",
    "vae": "VAE变分自编码器",
    "clip": "CLIP视觉-语言编码器",
    "sam": "SAM分割模型",
    "wan": "Wan视频生成",
    "cogvideo": "CogVideo视频生成",
    "cosmos": "Cosmos世界模型",
    "hunyuan": "混元多模态",
    "genmo": "Genmo Mochi视频",
    "ltx": "LTX视频生成",
    "lumina": "Lumina图像生成",
    "kandinsky": "Kandinsky图像生成",
    "aura": "AuraFlow",
    "t5": "T5文本编码器",
    "dit": "DiT扩散Transformer",
    "mmdit": "MMDiT多模态DiT",
    "diffusion": "扩散模型基础",
    "sampling": "采样器/求解器",
    "quantiz": "量化推理",
    "memory_management": "显存管理",
}

def extract_python_structure(filepath):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
        tree = ast.parse(source)
    except Exception:
        return None

    classes = []
    functions = []
    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            classes.append({"name": node.name, "methods": methods[:20], "line": node.lineno})
        elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
            functions.append({"name": node.name, "line": node.lineno})
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)

    return {
        "classes": classes[:30],
        "top_functions": functions[:30],
        "imports": list(set(imports))[:30],
        "loc": source.count("\n") + 1,
    }

def classify_arch_tags(rel_path, structure):
    tags = set()
    text = rel_path.lower()
    if structure:
        for cls in structure.get("classes", []):
            text += " " + cls["name"].lower()
    for kw, label in ARCH_KEYWORDS.items():
        if kw in text:
            tags.add(label)
    if not tags:
        tags.add("ComfyUI核心")
    return list(tags)

def index_comfyui():
    print("=" * 60)
    print("  Indexing ComfyUI source code")
    print("=" * 60)

    py_files = []
    for root, dirs, files in os.walk(COMFY_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))

    print(f"Found {len(py_files)} Python files")

    count = 0
    for filepath in py_files:
        rel = os.path.relpath(filepath, COMFY_ROOT).replace("\\", "/")
        structure = extract_python_structure(filepath)
        if not structure:
            continue

        arch_tags = classify_arch_tags(rel, structure)

        class_summaries = []
        for cls in structure["classes"]:
            methods_str = ", ".join(cls["methods"][:10])
            class_summaries.append(f"{cls['name']} (L{cls['line']}): [{methods_str}]")

        func_names = [f["name"] for f in structure["top_functions"]]

        content = {
            "topic": f"ComfyUI: {rel}",
            "source": "comfyui_source",
            "file_path": rel,
            "loc": structure["loc"],
            "classes": class_summaries[:15],
            "top_functions": func_names[:15],
            "imports": structure["imports"][:15],
            "arch_tags": arch_tags,
        }

        tags = ["comfyui", "diffusion_model", "source_code"] + arch_tags
        write("cezanne", "code_memory", content, importance=0.6, tags=tags)
        count += 1
        if count % BATCH_LOG == 0:
            print(f"  ComfyUI: {count}/{len(py_files)}")

    print(f"  ComfyUI done: {count} files indexed")

if __name__ == "__main__":
    t0 = time.time()
    index_comfyui()
    print(f"Done in {time.time()-t0:.0f}s")
