#!/usr/bin/env python3
"""
VORTEX FLAME 长期记忆系统 v2 — 代码索引 + 训练日志 + 推理注入

架构:
  Model (short ctx 1024) + FAISS (infinite memory)
  用户问题 → FAISS检索 → Top-K相关代码/日志/知识 → 注入prompt → 模型回答

新增功能 (v2):
  1. 代码目录索引器 — 自动扫描项目→按函数/类分块→向量化
  2. 训练日志自动写入 — 每阶段完成自动记录
  3. 推理注入 — build_rich_prompt() 混合注入身份+训练日志+代码上下文
"""
import os, json, time, hashlib, re, glob as glob_mod, platform
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

if platform.system() == "Linux" and os.path.exists("/mnt/d/VORTEX_FLAME"):
    MEMORY_ROOT = "/mnt/d/VORTEX_FLAME/long-memory"
else:
    MEMORY_ROOT = r"D:\VORTEX_FLAME\long-memory"
ENCODER_NAME = "all-MiniLM-L6-v2"

_encoder = None
_index_cache = {}
_mem_cache = {}


def _get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(ENCODER_NAME)
    return _encoder


def _soul_dir(soul="cezanne"):
    d = os.path.join(MEMORY_ROOT, soul)
    os.makedirs(d, exist_ok=True)
    return d


def _index_dir(soul="cezanne"):
    d = os.path.join(_soul_dir(soul), "faiss_index")
    os.makedirs(d, exist_ok=True)
    return d


def _mem_path(soul="cezanne"):
    p = os.path.join(_soul_dir(soul), "mem.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def _load_mem(soul="cezanne", use_cache=True):
    if use_cache and soul in _mem_cache:
        return _mem_cache[soul]
    p = _mem_path(soul)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"entries": []}
    if use_cache:
        _mem_cache[soul] = data
    return data


def _save_mem(data, soul="cezanne"):
    p = _mem_path(soul)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    _mem_cache[soul] = data


def _invalidate_cache(soul="cezanne"):
    _mem_cache.pop(soul, None)
    _index_cache.pop(soul, None)


def write(soul, category, content, metadata=None):
    """写入一条记忆,返回entry_id"""
    data = _load_mem(soul, use_cache=False)
    eid = hashlib.md5(f"{category}:{content}:{time.time()}".encode()).hexdigest()[:12]
    entry = {
        "id": eid,
        "category": category,
        "content": content,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metadata": metadata or {},
    }
    data["entries"].append(entry)
    _save_mem(data, soul)
    _build_index(soul)
    return eid


def search(soul, query, top_k=5, category=None):
    """向量搜索记忆,返回匹配条目"""
    idx_dir = _index_dir(soul)
    index_path = os.path.join(idx_dir, "index.faiss")
    ids_path = os.path.join(idx_dir, "ids.json")

    if not os.path.exists(index_path):
        _build_index(soul)
    if not os.path.exists(index_path):
        return []

    if soul in _index_cache:
        index, ids = _index_cache[soul]
    else:
        index = faiss.read_index(index_path)
        with open(ids_path, "r") as f:
            ids = json.load(f)
        _index_cache[soul] = (index, ids)

    encoder = _get_encoder()
    q_vec = encoder.encode([query], normalize_embeddings=True).astype("float32")
    k = min(top_k, len(ids))
    if k == 0:
        return []

    distances, indices = index.search(q_vec, k)
    data = _load_mem(soul)
    id_map = {e["id"]: e for e in data["entries"]}
    results = []
    for i in range(k):
        idx = indices[0][i]
        if idx < len(ids) and ids[idx] in id_map:
            entry = id_map[ids[idx]]
            if category is None or entry.get("category") == category:
                results.append({**entry, "score": float(distances[0][i])})
    return results


def _build_index(soul):
    data = _load_mem(soul, use_cache=False)
    if not data["entries"]:
        return

    encoder = _get_encoder()
    texts = [e["content"] for e in data["entries"]]
    ids = [e["id"] for e in data["entries"]]

    vectors = encoder.encode(texts, normalize_embeddings=True).astype("float32")
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    idx_dir = _index_dir(soul)
    faiss.write_index(index, os.path.join(idx_dir, "index.faiss"))
    with open(os.path.join(idx_dir, "ids.json"), "w") as f:
        json.dump(ids, f)
    _index_cache[soul] = (index, ids)


# ============================
#  v2 新增功能
# ============================

def _chunk_python_file(filepath):
    """将Python文件按函数/类分块"""
    chunks = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return chunks

    lines = content.split("\n")
    current_func = None
    func_start = 0
    func_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        # 检测函数/类定义
        if re.match(r"^(def |class |async def )", stripped):
            if current_func and func_lines:
                chunk_text = "\n".join(func_lines).strip()
                if len(chunk_text) > 20:
                    chunks.append({
                        "name": current_func,
                        "file": filepath,
                        "start_line": func_start + 1,
                        "end_line": i,
                        "content": chunk_text,
                    })
            current_func = stripped.split("(")[0].replace("def ", "").replace("class ", "").replace("async ", "").strip()
            func_start = i
            func_lines = [line]
        else:
            func_lines.append(line)

    # 最后一个函数
    if current_func and func_lines:
        chunk_text = "\n".join(func_lines).strip()
        if len(chunk_text) > 20:
            chunks.append({
                "name": current_func,
                "file": filepath,
                "start_line": func_start + 1,
                "end_line": len(lines),
                "content": chunk_text,
            })

    return chunks


def _chunk_generic_file(filepath):
    """通用文件按段落分块"""
    chunks = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return chunks

    paragraphs = content.split("\n\n")
    for i, para in enumerate(paragraphs):
        para = para.strip()
        if len(para) > 30:
            chunks.append({
                "name": os.path.basename(filepath),
                "file": filepath,
                "start_line": 0,
                "end_line": 0,
                "content": para[:2000],
            })
    return chunks


def index_directory(directory, soul="cezanne", recursive=True, patterns=None):
    """扫描目录,分块代码文件,向量化存入记忆库"""
    if patterns is None:
        patterns = ["*.py", "*.json", "*.md", "*.txt", "*.js", "*.ts", "*.html", "*.css", "*.yaml", "*.yml"]

    all_chunks = []

    for pattern in patterns:
        search_pattern = os.path.join(directory, "**", pattern) if recursive else os.path.join(directory, pattern)
        for filepath in glob_mod.glob(search_pattern, recursive=recursive):
            # 跳过虚拟环境/缓存/隐藏目录
            if any(skip in filepath for skip in [".venv", "node_modules", "__pycache__", ".git", ".trae", "faiss_index"]):
                continue
            if os.path.getsize(filepath) > 500_000:  # 跳过大于500KB的文件
                continue

            if filepath.endswith(".py"):
                chunks = _chunk_python_file(filepath)
            else:
                chunks = _chunk_generic_file(filepath)
            all_chunks.extend(chunks)

    # 批量写入记忆
    data = _load_mem(soul, use_cache=False)
    added = 0
    for chunk in all_chunks:
        # 用文件路径+函数名+起止行做去重
        content = chunk["content"]
        eid = hashlib.md5(f"{chunk['file']}:{chunk['name']}:{chunk['start_line']}".encode()).hexdigest()[:12]

        # 检查是否已存在
        existing_ids = {e["id"] for e in data["entries"]}
        if eid in existing_ids:
            continue

        entry = {
            "id": eid,
            "category": "code",
            "content": content,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": {
                "file": chunk["file"],
                "name": chunk["name"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
            },
        }
        data["entries"].append(entry)
        added += 1

    _save_mem(data, soul)
    if added > 0:
        _build_index(soul)

    return {"total_chunks": len(all_chunks), "added": added, "skin_pped": len(all_chunks) - added}


def log_training(soul, stage, loss, exam_rate, elapsed_min, peak_vram_gb, samples, config_info=""):
    """训练完成后自动写入记忆库"""
    content = (
        f"Stage={stage} | Loss={loss:.4f} | Exam={exam_rate:.0%} | "
        f"Time={elapsed_min:.0f}min | VRAM={peak_vram_gb:.1f}GB | Samples={samples}"
    )
    if config_info:
        content += f" | {config_info}"

    eid = write(soul, "training_log", content, {
        "stage": stage, "loss": loss, "exam_rate": exam_rate,
        "elapsed_min": elapsed_min, "samples": samples,
    })
    return eid


def log_benchmark(soul, stage, avg_score, pass_rate, details=""):
    """回测完成后自动写入记忆库"""
    content = f"Stage={stage} benchmark | AvgScore={avg_score:.0%} | PassRate={pass_rate:.0%}"
    if details:
        content += f" | {details}"
    eid = write(soul, "benchmark_log", content, {
        "stage": stage, "avg_score": avg_score, "pass_rate": pass_rate,
    })
    return eid


def log_knowledge(soul, category, content, source=""):
    """写入知识点（如李亚普诺夫、GLM-5.1）"""
    metadata = {"source": source} if source else {}
    return write(soul, f"knowledge_{category}", content, metadata)


def build_rich_prompt(soul, user_question, max_chars=3000):
    """
    构建含记忆的丰富prompt:
      1. 召回对话历史 (conversation memory — 最高优先级)
      2. 召回灵魂知识 (identity/skill/training_log)
      3. 召回相关代码 (code chunks)
    """
    conversation_ctx = recall(soul, user_question, top_k=3, max_chars=800)

    identity = recall(soul, f"{soul} identity skills training", top_k=3, max_chars=800)

    code_context = recall(soul, user_question, top_k=5, max_chars=2000)

    parts = []
    if conversation_ctx:
        parts.append(f"[Recent Conversations]\n{conversation_ctx}\n[/Recent Conversations]")
    if identity:
        parts.append(f"[Soul Knowledge]\n{identity}\n[/Soul Knowledge]")
    if code_context:
        parts.append(f"[Relevant Code]\n{code_context}\n[/Relevant Code]")

    context_block = "\n\n".join(parts) + "\n\n" if parts else ""
    return f"{context_block}<s>[INST] {user_question} [/INST]"


def init_default_memory(soul="cezanne"):
    """初始化灵魂默认记忆"""
    defaults = {
        "cezanne": [
            {"category": "identity", "content": "I am Cezanne_PRO, a structured thinking and coding AI soul (8B). I excel at algorithm design, system architecture, and logical reasoning. Training targets GPT-5.5-level code development abilities."},
            {"category": "preference", "content": "User prefers Chinese responses. Code should use functional style with clear type hints. Target: precision over generality."},
            {"category": "skill", "content": "Core skills: Python, algorithm design, system architecture, code review, debugging (off-by-one, race conditions, memory leaks, deadlocks), performance optimization, formal verification (Hoare logic, program correctness)."},
            {"category": "training", "content": "Training pipeline: Stage1=Math(229K) → Stage2=Logic(3.3K) → Stage3a=CodeAlpaca(7.9K) → Stage3b=Deep+AntiForget+Supplements(1.5K). Base: Ministral-8B-Reasoning(brain surgery, vision removed, 1024 ctx). LoRA r=16 alpha=32, 4-stage chain."},
            {"category": "architecture", "content": "Architecture: Short context(1024 tokens) + FAISS external memory. Model handles thinking, FAISS handles long-term storage. Hardware: RTX 3060 12GB. Training: bitsandbytes 4bit + gradient checkpointing + seq_len=128."},
            {"category": "target", "content": "Target: GPT-5.5/GLM-5.1 code development capabilities. Vertical model, not general chatbot. Precision in code generation, debugging, and algorithmic reasoning is the #1 priority."},
        ],
        "einstein": [
            {"category": "identity", "content": "I am Einstein, a quantum physics and innovative thinking AI soul."},
            {"category": "skill", "content": "Core skills: Physics calculations, quantum mechanics, relativity, thought experiments, control theory (Lyapunov stability)."},
        ],
        "global": [
            {"category": "system", "content": "VORTEX FLAME system: 14 souls. Cezanne(7B/8B code+logic) and Einstein(physics) are training. Arena 4-mode: Code Battle, Attack-Defense, Reasoning Duel, Dual Soul vs Single. Modded Transformer blueprint at E:\\VORTEX_FLAME_Blueprint\\魔改Transformer."},
            {"category": "hardware", "content": "Hardware: RTX 3060 12GB(training) + RTX 3060 6GB(idle). 8B model post brain-surgery: vision removed, context cut to 1024, FAISS handles long memory. Training: serial only, 7B and 8B never parallel."},
            {"category": "data", "content": "Training data: Stage1=MathInstruct 229K, Stage2=Logic+Code 3.3K, Stage3a=CodeAlpaca 7.9K(from 20K filtered: -4K short, -8K no_code), Stage3b=AntiForget+Supplements 1.5K(37 targeted). All deduped, aligned 7B=8B."},
        ],
    }

    entries = defaults.get(soul, [])
    data = _load_mem(soul, use_cache=False)
    existing_contents = {e["content"] for e in data["entries"]}

    added = 0
    for entry in entries:
        if entry["content"] not in existing_contents:
            write_no_rebuild(soul, entry["category"], entry["content"])
            added += 1

    if added > 0:
        _build_index(soul)

    return added


def write_no_rebuild(soul, category, content, metadata=None):
    """写入但不重建索引(批量写入用)"""
    data = _load_mem(soul, use_cache=False)
    eid = hashlib.md5(f"{category}:{content}:{time.time()}".encode()).hexdigest()[:12]
    entry = {
        "id": eid, "category": category, "content": content,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "metadata": metadata or {},
    }
    data["entries"].append(entry)
    _save_mem(data, soul)
    return eid


_knowledge_cache = {}


def _knowledge_dir(soul="cezanne"):
    d = os.path.join(_soul_dir(soul), "knowledge")
    os.makedirs(d, exist_ok=True)
    return d


def _knowledge_data_path(soul="cezanne"):
    return os.path.join(_knowledge_dir(soul), "data.jsonl")


def _knowledge_index_dir(soul="cezanne"):
    d = os.path.join(_knowledge_dir(soul), "faiss_index")
    os.makedirs(d, exist_ok=True)
    return d


_knowledge_file_handles = {}


def write_knowledge(soul, category, content, metadata=None, entry_id=None):
    """追加写入知识条目到JSONL文件（不重建索引，批量写入用）"""
    if entry_id:
        eid = entry_id
    else:
        eid = hashlib.md5(f"{category}:{content}:{time.time()}".encode()).hexdigest()[:12]
    entry = {
        "id": eid, "category": category, "content": content,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "metadata": metadata or {},
    }
    data_path = _knowledge_data_path(soul)
    if soul not in _knowledge_file_handles:
        _knowledge_file_handles[soul] = open(data_path, "a", encoding="utf-8")
    _knowledge_file_handles[soul].write(json.dumps(entry, ensure_ascii=False) + "\n")
    _knowledge_file_handles[soul].flush()
    _knowledge_cache.pop(soul, None)
    return eid


def close_knowledge_handles():
    for soul, fh in _knowledge_file_handles.items():
        try:
            fh.close()
        except Exception:
            pass
    _knowledge_file_handles.clear()


def _load_knowledge(soul="cezanne"):
    if soul in _knowledge_cache:
        return _knowledge_cache[soul]
    data_path = _knowledge_data_path(soul)
    entries = []
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    _knowledge_cache[soul] = entries
    return entries


def build_knowledge_index(soul="cezanne", batch_size=10000):
    """构建知识库FAISS索引（支持大规模数据，分批编码）"""
    entries = _load_knowledge(soul)
    if not entries:
        return 0

    encoder = _get_encoder()
    texts = [e["content"] for e in entries]
    ids = [e["id"] for e in entries]
    total = len(texts)

    all_vectors = []
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        vecs = encoder.encode(batch, normalize_embeddings=True, show_progress_bar=False).astype("float32")
        all_vectors.append(vecs)
        if (i + batch_size) % 50000 == 0 or i + batch_size >= total:
            print(f"    [knowledge_index] Encoded {min(i + batch_size, total)}/{total}", flush=True)

    vectors = np.vstack(all_vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    idx_dir = _knowledge_index_dir(soul)
    faiss.write_index(index, os.path.join(idx_dir, "index.faiss"))
    with open(os.path.join(idx_dir, "ids.json"), "w") as f:
        json.dump(ids, f)

    return total


_knowledge_index_cache = {}
_knowledge_ids_cache = {}


def _get_knowledge_index(soul):
    if soul in _knowledge_index_cache:
        return _knowledge_index_cache[soul], _knowledge_ids_cache.get(soul, [])
    idx_dir = _knowledge_index_dir(soul)
    index_path = os.path.join(idx_dir, "index.faiss")
    ids_path = os.path.join(idx_dir, "ids.json")
    if not os.path.exists(index_path):
        return None, []
    index = faiss.read_index(index_path)
    with open(ids_path, "r") as f:
        ids = json.load(f)
    _knowledge_index_cache[soul] = index
    _knowledge_ids_cache[soul] = ids
    return index, ids


def search_knowledge(soul, query, top_k=5, category=None):
    index, ids = _get_knowledge_index(soul)
    if index is None:
        return []

    encoder = _get_encoder()
    q_vec = encoder.encode([query], normalize_embeddings=True).astype("float32")
    k = min(top_k * 3, len(ids))
    if k == 0:
        return []

    distances, indices = index.search(q_vec, k)

    line_numbers = set()
    line_to_score = {}
    for i in range(k):
        idx = int(indices[0][i])
        if 0 <= idx < len(ids):
            line_numbers.add(idx)
            line_to_score[idx] = float(distances[0][i])

    entries_by_line = _load_knowledge_by_line_numbers(soul, line_numbers)
    results = []
    for line_num, score in line_to_score.items():
        if line_num in entries_by_line:
            entry = entries_by_line[line_num]
            if category is None or entry.get("category") == category:
                results.append({**entry, "score": score})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def _load_knowledge_by_ids(soul, target_ids):
    data_path = _knowledge_data_path(soul)
    result = {}
    if not os.path.exists(data_path):
        return result
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    if entry["id"] in target_ids:
                        result[entry["id"]] = entry
                        if len(result) >= len(target_ids):
                            break
                except json.JSONDecodeError:
                    pass
    return result


def _load_knowledge_by_line_numbers(soul, line_numbers):
    data_path = _knowledge_data_path(soul)
    idx_dir = _knowledge_index_dir(soul)
    offsets_path = os.path.join(idx_dir, "offsets.json")
    result = {}
    if not os.path.exists(data_path):
        return result

    offsets = None
    if os.path.exists(offsets_path):
        with open(offsets_path, "r") as f:
            offsets = json.load(f)

    with open(data_path, "r", encoding="utf-8") as f:
        if offsets:
            for ln in line_numbers:
                if 0 <= ln < len(offsets):
                    f.seek(offsets[ln])
                    line = f.readline().strip()
                    if line:
                        try:
                            result[ln] = json.loads(line)
                        except json.JSONDecodeError:
                            pass
        else:
            max_line = max(line_numbers) if line_numbers else 0
            for line_num, line in enumerate(f):
                if line_num in line_numbers:
                    line = line.strip()
                    if line:
                        try:
                            result[line_num] = json.loads(line)
                        except json.JSONDecodeError:
                            pass
                if line_num >= max_line:
                    break
    return result


# ============================
#  对话记忆库 (Conversation Memory)
#  独立于知识库，专门存储用户与灵魂的对话历史
#  优先级: 对话记忆 → 知识库 → 主记忆库
# ============================

_conversation_cache = {}
_conversation_file_handles = {}
_conversation_index_cache = {}
_conversation_ids_cache = {}


def _conversation_dir(soul="cezanne"):
    d = os.path.join(_soul_dir(soul), "conversation")
    os.makedirs(d, exist_ok=True)
    return d


def _conversation_data_path(soul="cezanne"):
    return os.path.join(_conversation_dir(soul), "data.jsonl")


def _conversation_index_dir(soul="cezanne"):
    d = os.path.join(_conversation_dir(soul), "faiss_index")
    os.makedirs(d, exist_ok=True)
    return d


def write_conversation(soul, role, content, session_id=None, metadata=None):
    """追加写入对话记录到JSONL文件

    Args:
        soul: 灵魂名称
        role: "user" 或 "assistant"
        content: 对话内容
        session_id: 会话ID，用于关联同一会话的多轮对话
        metadata: 额外元数据（如topic, intent等）
    """
    eid = hashlib.md5(f"conv:{role}:{content}:{time.time()}".encode()).hexdigest()[:12]
    entry = {
        "id": eid,
        "category": f"conversation_{role}",
        "role": role,
        "content": content,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": session_id or "",
        "metadata": metadata or {},
    }
    data_path = _conversation_data_path(soul)
    if soul not in _conversation_file_handles:
        _conversation_file_handles[soul] = open(data_path, "a", encoding="utf-8")
    _conversation_file_handles[soul].write(json.dumps(entry, ensure_ascii=False) + "\n")
    _conversation_file_handles[soul].flush()
    _conversation_cache.pop(soul, None)
    _conversation_index_cache.pop(soul, None)
    _conversation_ids_cache.pop(soul, None)
    return eid


def close_conversation_handles():
    for soul, fh in _conversation_file_handles.items():
        try:
            fh.close()
        except Exception:
            pass
    _conversation_file_handles.clear()


def _load_conversation(soul="cezanne"):
    if soul in _conversation_cache:
        return _conversation_cache[soul]
    data_path = _conversation_data_path(soul)
    entries = []
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    _conversation_cache[soul] = entries
    return entries


def build_conversation_index(soul="cezanne", batch_size=10000):
    """构建对话记忆库FAISS索引"""
    entries = _load_conversation(soul)
    if not entries:
        return 0

    encoder = _get_encoder()
    texts = [e["content"] for e in entries]
    ids = [e["id"] for e in entries]
    total = len(texts)

    all_vectors = []
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        vecs = encoder.encode(batch, normalize_embeddings=True, show_progress_bar=False).astype("float32")
        all_vectors.append(vecs)

    vectors = np.vstack(all_vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    idx_dir = _conversation_index_dir(soul)
    faiss.write_index(index, os.path.join(idx_dir, "index.faiss"))
    with open(os.path.join(idx_dir, "ids.json"), "w") as f:
        json.dump(ids, f)

    _conversation_index_cache[soul] = index
    _conversation_ids_cache[soul] = ids
    return total


def _get_conversation_index(soul):
    if soul in _conversation_index_cache:
        return _conversation_index_cache[soul], _conversation_ids_cache.get(soul, [])
    idx_dir = _conversation_index_dir(soul)
    index_path = os.path.join(idx_dir, "index.faiss")
    ids_path = os.path.join(idx_dir, "ids.json")
    if not os.path.exists(index_path):
        return None, []
    index = faiss.read_index(index_path)
    with open(ids_path, "r") as f:
        ids = json.load(f)
    _conversation_index_cache[soul] = index
    _conversation_ids_cache[soul] = ids
    return index, ids


def search_conversation(soul, query, top_k=5, role=None):
    """搜索对话记忆库

    Args:
        soul: 灵魂名称
        query: 查询文本
        top_k: 返回条数
        role: 过滤角色 ("user" 或 "assistant" 或 None)
    """
    index, ids = _get_conversation_index(soul)
    if index is None:
        return []

    encoder = _get_encoder()
    q_vec = encoder.encode([query], normalize_embeddings=True).astype("float32")
    k = min(top_k * 3, len(ids))
    if k == 0:
        return []

    distances, indices = index.search(q_vec, k)

    line_numbers = set()
    line_to_score = {}
    for i in range(k):
        idx = int(indices[0][i])
        if 0 <= idx < len(ids):
            line_numbers.add(idx)
            line_to_score[idx] = float(distances[0][i])

    entries_by_line = _load_conversation_by_line_numbers(soul, line_numbers)
    results = []
    for line_num, score in line_to_score.items():
        if line_num in entries_by_line:
            entry = entries_by_line[line_num]
            if role is None or entry.get("role") == role:
                results.append({**entry, "score": score})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def _load_conversation_by_line_numbers(soul, line_numbers):
    data_path = _conversation_data_path(soul)
    result = {}
    if not os.path.exists(data_path):
        return result

    max_line = max(line_numbers) if line_numbers else 0
    with open(data_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if line_num in line_numbers:
                line = line.strip()
                if line:
                    try:
                        result[line_num] = json.loads(line)
                    except json.JSONDecodeError:
                        pass
            if line_num >= max_line:
                break
    return result


def get_recent_conversation(soul, n=10, session_id=None):
    """获取最近的对话记录（按时间倒序）

    Args:
        soul: 灵魂名称
        n: 返回条数
        session_id: 过滤特定会话，None则不过滤
    """
    entries = _load_conversation(soul)
    if session_id:
        entries = [e for e in entries if e.get("session_id") == session_id]
    entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return entries[:n]


def recall(soul, query, top_k=5, max_chars=2000):
    """召回相关记忆,优先级: 对话记忆 → 知识库 → 主记忆库"""
    conversation_results = search_conversation(soul, query, top_k=top_k)
    knowledge_results = search_knowledge(soul, query, top_k=top_k)
    main_results = search(soul, query, top_k=top_k)

    all_results = conversation_results + knowledge_results + main_results
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

    if not all_results:
        return ""
    chunks = []
    total = 0
    for r in all_results:
        chunk = f"[{r['category']}] {r['content']}"
        if total + len(chunk) > max_chars and chunks:
            break
        chunks.append(chunk)
        total += len(chunk) + 1
    return "\n".join(chunks)


def get_stats(soul="cezanne"):
    """获取记忆库统计（含知识库+对话记忆库）"""
    data = _load_mem(soul, use_cache=False)
    cats = {}
    for e in data["entries"]:
        cats[e["category"]] = cats.get(e["category"], 0) + 1

    knowledge_entries = _load_knowledge(soul)
    k_cats = {}
    for e in knowledge_entries:
        k_cats[e.get("category", "?")] = k_cats.get(e.get("category", "?"), 0) + 1

    conversation_entries = _load_conversation(soul)
    c_cats = {}
    c_roles = {}
    for e in conversation_entries:
        c_cats[e.get("category", "?")] = c_cats.get(e.get("category", "?"), 0) + 1
        c_roles[e.get("role", "?")] = c_roles.get(e.get("role", "?"), 0) + 1

    return {
        "total": len(data["entries"]),
        "categories": cats,
        "knowledge_total": len(knowledge_entries),
        "knowledge_categories": k_cats,
        "conversation_total": len(conversation_entries),
        "conversation_categories": c_cats,
        "conversation_roles": c_roles,
    }


def refresh_all():
    """刷新所有灵魂的索引（含知识库+对话记忆库）"""
    for soul in ["cezanne", "einstein", "global"]:
        _invalidate_cache(soul)
        _build_index(soul)
        knowledge_path = _knowledge_data_path(soul)
        if os.path.exists(knowledge_path):
            build_knowledge_index(soul)
        conversation_path = _conversation_data_path(soul)
        if os.path.exists(conversation_path):
            build_conversation_index(soul)


if __name__ == "__main__":
    print("=" * 60)
    print("  VORTEX FLAME 长期记忆系统 v2 初始化")
    print("=" * 60)

    # 1. 初始化默认记忆
    print("\n[1/3] 初始化灵魂默认记忆...")
    for soul in ["cezanne", "einstein", "global"]:
        added = init_default_memory(soul)
        data = _load_mem(soul, use_cache=False)
        print(f"  {soul}: {added} new, {len(data['entries'])} total")

    # 2. 索引项目代码
    print("\n[2/3] 索引项目代码目录...")
    dirs_to_index = [
        (r"D:\VORTEX_FLAME\pipeline_8b", "cezanne"),
        (r"D:\VORTEX_FLAME\long-memory", "global"),
    ]
    for directory, soul in dirs_to_index:
        if os.path.isdir(directory):
            result = index_directory(directory, soul=soul)
            print(f"  {os.path.basename(directory)} -> {soul}: {result['added']} new chunks")

    # 3. 记录现有训练日志
    print("\n[3/3] 记录已有训练成果...")
    training_logs = [
        ("cezanne", "stage1", 0.5658, 1.0, 60, 10.0, 229911),
        ("cezanne", "stage2", 0.5148, 1.0, 145, 11.9, 3338),
        ("cezanne", "stage3a", 0.7793, 1.0, 201, 11.9, 7963),
        ("cezanne", "stage3b", 0.0, 0.0, 0, 0.0, 1456),
    ]
    for args in training_logs:
        soul, stage, loss, exam, elapsed, vram, samples = args
        if loss > 0:
            log_training(soul, stage, loss, exam, elapsed, vram, samples, "LoRA r=16 alpha=32")

    # Benchmark logs
    benchmarks = [
        ("cezanne", "S1", 0.76, 1.0, "Math baseline"),
        ("cezanne", "S2", 0.74, 1.0, "Debug degraded 40->20%"),
        ("cezanne", "S3", 0.82, 1.0, "Debug recovered to 80%, Sort/Derivative fixed"),
    ]
    for soul, stage, score, rate, details in benchmarks:
        log_benchmark(soul, stage, score, rate, details)

    refresh_all()

    # 4. 显示统计
    print("\n" + "=" * 60)
    print("  记忆库统计")
    print("=" * 60)
    for soul in ["cezanne", "einstein", "global"]:
        stats = get_stats(soul)
        print(f"  {soul}: {stats['total']} entries")
        for cat, n in sorted(stats['categories'].items()):
            print(f"    {cat}: {n}")

    print("\n  Done! 记忆系统 v2 就绪")
