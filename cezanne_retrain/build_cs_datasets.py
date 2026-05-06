#!/usr/bin/env python3
"""Cezanne 7B CS-Focused Dataset Builder v2
Optimized: regex-compiled keyword matching + hand-crafted quality core from 8B
"""
import os, sys, json, random, re, time
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

SEED = 3407
random.seed(SEED)

DATA_DIR = r"D:\VORTEX_FLAME\soul_training_data\cezanne"
DATA_8B_DIR = r"D:\VORTEX_FLAME\soul_training_data\cezanne_8b"
CSAPP_DIR = r"D:\VORTEX_FLAME\knowledge_base\cs_knowledge\CMU-CSAPP"
OUT_DIR = r"D:\VORTEX_FLAME\soul_training_data\cezanne"

TARGET = 8000

CS_MATH_PATTERN = re.compile(
    r'graph|tree|vertex|edge|adjacen|bipartite|spanning|'
    r'shortest path|dijkstra|bellman|floyd|kruskal|prim|'
    r'matrix|eigenvalue|eigenvector|determinant|inverse matrix|'
    r'linear transformation|vector space|basis|'
    r'probability|bayes|markov|random|stochastic|expected value|'
    r'variance|distribution|bernoulli|binomial|poisson|'
    r'combinatorics|permutation|combination|pigeonhole|'
    r'modular|modulo|congruence|gcd|lcm|prime|'
    r'rsa|encrypt|cryptography|hash|'
    r'recurrence|recursion|induction|invariant|'
    r'complexity|big-o|big theta|polynomial|'
    r'binary|hexadecimal|octal|bitwise|'
    r'boolean|logic gate|truth table|karnaugh|'
    r'floating point|ieee 754|overflow|underflow|'
    r'sort|search|merge sort|partition|heap|'
    r'dynamic programming|greedy|backtracking|'
    r'finite automata|regular expression|context-free|'
    r'turing|decidab|reducib|'
    r'hamming|parity|checksum|error correcting|'
    r'entropy|information theory|shannon|'
    r'discrete|counting|inclusion-exclusion|'
    r'divide and conquer|master theorem|'
    r'amortized|worst case|average case|'
    r'np-complete|np-hard|reduction|'
    r'convolution|fft|fourier|'
    r'gradient|derivative|optimization|'
    r'logarithm|exponential|'
    r'图论|树|顶点|邻接|二部图|生成树|最短路径|'
    r'矩阵|特征值|行列式|逆矩阵|'
    r'概率|贝叶斯|马尔可夫|随机|期望|'
    r'组合|排列|鸽巢|'
    r'模运算|同余|最大公约数|素数|'
    r'加密|密码学|哈希|'
    r'递推|递归|归纳|不变式|'
    r'复杂度|大O|多项式|'
    r'二进制|十六进制|位运算|'
    r'布尔|逻辑门|真值表|'
    r'浮点数|溢出|'
    r'排序|搜索|归并|堆|'
    r'动态规划|贪心|回溯|'
    r'有限自动机|正则表达式|上下文无关|'
    r'图灵|可判定|'
    r'汉明|校验|纠错码|'
    r'信息熵|香农|'
    r'离散|计数|容斥|'
    r'分治|主定理|'
    r'摊还|最坏情况|平均情况|'
    r'NP完全|NP难|归约|'
    r'卷积|傅里叶|'
    r'梯度|导数|优化|'
    r'对数|指数',
    re.IGNORECASE
)

CS_LOGIC_PATTERN = re.compile(
    r'boolean|logic gate|truth table|karnaugh|de morgan|'
    r'propositional|predicate|quantifier|forall|exists|'
    r'syllogism|modus ponens|modus tollens|contrapositive|'
    r'contradiction|tautology|satisfiability|'
    r'formal verification|model checking|temporal logic|'
    r'hoare|precondition|postcondition|invariant|assertion|'
    r'type system|type checking|type inference|'
    r'proof|induction|contradiction|construction|'
    r'algorithm correctness|termination|partial correctness|'
    r'loop invariant|weakest precondition|'
    r'debug|bug|fix|error|fault|trace|'
    r'assert|contract|specification|'
    r'state machine|transition|accepting state|'
    r'decidab|undecidable|halting|'
    r'reduction|completeness|soundness|'
    r'concurrent|race condition|deadlock|mutex|semaphore|'
    r'synchronization|atomic|critical section|'
    r'memory model|happens-before|sequential consistency|'
    r'security|injection|sanitiz|validation|'
    r'complexity class|P vs NP|'
    r'recursion|base case|structural induction|'
    r'well-founded|well-ordering|'
    r'algorithm|data structure|function|class|method|'
    r'recursive|iterative|sort|search|hash|'
    r'linked list|stack|queue|heap|priority queue|'
    r'binary search|binary tree|bst|avl|red-black|'
    r'dynamic programming|memoiz|tabulation|'
    r'greedy|backtracking|bfs|dfs|topological|dijkstra|'
    r'regex|parser|lexer|compiler|interpreter|'
    r'thread|process|lock|concurrent|'
    r'socket|http|tcp|udp|protocol|'
    r'sql|database|query|index|transaction|'
    r'memory|pointer|allocation|garbage collection|'
    r'cache|buffer|pipeline|optimization|'
    r'test|unit test|integration|mock|'
    r'design pattern|factory|singleton|observer|'
    r'api|rest|endpoint|middleware|'
    r'oop|inheritance|polymorphism|encapsulation|'
    r'complexity|time complexity|space complexity|'
    r'bit manipulation|bitwise|mask|'
    r'布尔|逻辑门|真值表|卡诺图|德摩根|'
    r'命题|谓词|量词|全称|存在|'
    r'三段论|假言推理|逆否命题|'
    r'矛盾|重言式|可满足性|'
    r'形式化验证|模型检测|时序逻辑|'
    r'霍尔逻辑|前置条件|后置条件|不变式|断言|'
    r'类型系统|类型检查|类型推导|'
    r'证明|归纳|反证|构造性|'
    r'算法正确性|终止性|部分正确性|'
    r'循环不变式|最弱前置条件|'
    r'调试|找bug|修复|错误|故障|追踪|'
    r'断言|契约|规约|'
    r'状态机|转移|接受状态|'
    r'可判定|不可判定|停机问题|'
    r'归约|完备性|可靠性|'
    r'并发|竞态条件|死锁|互斥|信号量|'
    r'同步|原子|临界区|'
    r'内存模型|先行发生|顺序一致性|'
    r'安全|注入|校验|验证|'
    r'递归|基础情形|结构归纳',
    re.IGNORECASE
)

CS_DEPTH_PATTERN = re.compile(
    r'virtual memory|page table|tlb|cache|pipeline|'
    r'process|thread|signal|socket|concurrent|'
    r'linker|loader|relocat|executable|'
    r'buffer overflow|stack canary|aslr|nx bit|'
    r'calling convention|stack frame|'
    r'endianness|big-endian|little-endian|'
    r'malloc|free|heap|garbage collection|'
    r'cuda|gpu|kernel|thread|block|grid|'
    r'mapreduce|distributed|consensus|raft|'
    r'elf|section|symbol|relocation|'
    r'signal|sigint|sigsegv|sigkill|'
    r'shell|fork|exec|pipe|dup2|'
    r'ipc|shared memory|message queue|'
    r'select|poll|epoll|kqueue|'
    r'deadlock|mutex|semaphore|condition variable|'
    r'tcp|udp|congestion|slow start|'
    r'汇编|机器级|虚拟内存|页表|缓存|流水线|'
    r'进程|线程|信号|并发|链接|存储器|'
    r'缓冲区溢出|栈|堆|内存分配|'
    r'端序|大端|小端|'
    r'调用约定|栈帧|'
    r'进程间通信|共享内存|消息队列|'
    r'死锁|互斥|信号量|条件变量|'
    r'网络编程|套接字|并发服务器',
    re.IGNORECASE
)

CODE_PATTERN = re.compile(
    r'python|java|c\+\+|javascript|algorithm|data structure|'
    r'function|class|method|recursive|iterative|'
    r'sort|search|hash|tree|graph|linked list|'
    r'stack|queue|heap|priority queue|deque|'
    r'binary search|binary tree|bst|avl|red-black|'
    r'dynamic programming|memoiz|tabulation|'
    r'greedy|backtracking|branch and bound|'
    r'bfs|dfs|topological|dijkstra|a\*|'
    r'regex|parser|lexer|compiler|interpreter|'
    r'thread|process|mutex|lock|concurrent|'
    r'socket|http|tcp|udp|protocol|'
    r'sql|database|query|index|transaction|'
    r'memory|pointer|allocation|garbage collection|'
    r'cache|buffer|pipeline|optimization|'
    r'test|unit test|integration|mock|'
    r'design pattern|factory|singleton|observer|'
    r'api|rest|endpoint|middleware|'
    r'oop|inheritance|polymorphism|encapsulation|'
    r'complexity|time complexity|space complexity|'
    r'bit manipulation|bitwise|mask|'
    r'def |class |import |function |return |for |while |if |else |',
    re.IGNORECASE
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        if "data" in raw:
            return raw["data"]
        if "samples" in raw:
            return raw["samples"]
        return [raw]
    return raw


def score_text(text, pattern):
    matches = pattern.findall(text.lower())
    return len(matches)


def build_s1():
    print("\n" + "=" * 60, flush=True)
    print("  S1: CS-Related Math", flush=True)
    print("=" * 60, flush=True)

    math_file = os.path.join(DATA_DIR, "cezanne_stage1_math_8k_v3.json")
    if not os.path.exists(math_file):
        print(f"  [ERROR] {math_file} not found!", flush=True)
        return []

    all_items = load_json(math_file)
    print(f"  Loaded: {len(all_items)} items", flush=True)

    t0 = time.time()
    scored = []
    for i, item in enumerate(all_items):
        combined = (item.get("instruction", "") + " " + item.get("output", "")).lower()
        s = score_text(combined, CS_MATH_PATTERN)
        if s > 0:
            scored.append((s, i, item))
        if (i + 1) % 50000 == 0:
            print(f"  Processed {i+1}/{len(all_items)} ({time.time()-t0:.1f}s)", flush=True)

    scored.sort(key=lambda x: x[0], reverse=True)
    print(f"  CS-relevant: {len(scored)} items ({time.time()-t0:.1f}s)", flush=True)

    take = min(len(scored), TARGET)
    result = []
    for s, i, item in scored[:take]:
        result.append({
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
            "output": item.get("output", ""),
            "source": item.get("source", "MathInstruct"),
            "soul": "cezanne",
            "_cat": "CS_Math",
            "_score": s,
        })

    print(f"  S1 final: {len(result)} items", flush=True)
    return result


def build_s2():
    print("\n" + "=" * 60, flush=True)
    print("  S2: CS-Related Logic", flush=True)
    print("=" * 60, flush=True)

    all_items = []
    sources = {}

    logic_file = os.path.join(DATA_DIR, "cezanne_stage2_logic_8k_v3.json")
    if os.path.exists(logic_file):
        items = load_json(logic_file)
        sources["logic"] = len(items)
        all_items.extend(items)

    debug_file = os.path.join(DATA_DIR, "cezanne_s2_logic_debug_v2.json")
    if os.path.exists(debug_file):
        items = load_json(debug_file)
        sources["debug"] = len(items)
        for item in items:
            if "text" in item and "instruction" not in item:
                text = item["text"]
                parts = text.split("A:", 1)
                if len(parts) == 2:
                    all_items.append({
                        "instruction": parts[0].replace("Q:", "").strip(),
                        "input": "",
                        "output": parts[1].strip(),
                        "source": "debug_v2",
                        "soul": "cezanne",
                        "_cat": "Debug",
                    })

    code_file = os.path.join(DATA_DIR, "cezanne_stage3a_code_v1.json")
    if os.path.exists(code_file):
        items = load_json(code_file)
        sources["code"] = len(items)
        all_items.extend(items)

    print(f"  Sources: {sources}", flush=True)
    print(f"  Total candidates: {len(all_items)}", flush=True)

    t0 = time.time()
    scored = []
    for i, item in enumerate(all_items):
        combined = (item.get("instruction", "") + " " + item.get("output", "")).lower()
        logic_s = score_text(combined, CS_LOGIC_PATTERN)
        code_s = score_text(combined, CODE_PATTERN)
        total = logic_s + code_s * 0.3
        if total > 0:
            scored.append((total, i, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    print(f"  CS-logic relevant: {len(scored)} ({time.time()-t0:.1f}s)", flush=True)

    take = min(len(scored), TARGET)
    result = []
    for s, i, item in scored[:take]:
        result.append({
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
            "output": item.get("output", ""),
            "source": item.get("source", "unknown"),
            "soul": "cezanne",
            "_cat": "CS_Logic",
            "_score": round(s, 2),
        })

    print(f"  S2 final: {len(result)} items", flush=True)
    return result


def build_s3():
    print("\n" + "=" * 60, flush=True)
    print("  S3: CS Depth", flush=True)
    print("=" * 60, flush=True)

    all_items = []

    cs_v1 = os.path.join(DATA_DIR, "cezanne_stage3_cs_v1.json")
    if os.path.exists(cs_v1):
        items = load_json(cs_v1)
        print(f"  CS v1: {len(items)}", flush=True)
        for item in items:
            item["_cat"] = "CS_Systems"
            item["source"] = item.get("source", "CSAPP_v1")
            item["soul"] = "cezanne"
        all_items.extend(items)

    cs_v2 = os.path.join(DATA_DIR, "cezanne_s3_cs_depth_v2.json")
    if os.path.exists(cs_v2):
        items = load_json(cs_v2)
        print(f"  CS v2: {len(items)}", flush=True)
        for item in items:
            if "text" in item and "instruction" not in item:
                text = item["text"]
                parts = text.split("A:", 1)
                if len(parts) == 2:
                    all_items.append({
                        "instruction": parts[0].replace("Q:", "").strip(),
                        "input": "",
                        "output": parts[1].strip(),
                        "source": "CSAPP_v2",
                        "soul": "cezanne",
                        "_cat": "CS_Systems_Deep",
                    })

    exclusive = os.path.join(DATA_8B_DIR, "cezanne_pro_8b_exclusive_code_deep.json")
    if os.path.exists(exclusive):
        items = load_json(exclusive)
        print(f"  8B Exclusive: {len(items)}", flush=True)
        for item in items:
            item["source"] = item.get("source", "8B_exclusive")
            item["soul"] = "cezanne"
            item["_cat"] = "CS_Algorithm_Proof"
        all_items.extend(items)

    csapp_qa = generate_csapp_qa()
    if csapp_qa:
        all_items.extend(csapp_qa)
        print(f"  CSAPP notes: {len(csapp_qa)}", flush=True)

    theory_count = len(all_items)
    print(f"  CS theory total: {theory_count}", flush=True)

    code_need = TARGET - theory_count
    if code_need > 0:
        fusion = os.path.join(DATA_DIR, "cezanne_stage3_fusion_8k_v7.json")
        code_f = os.path.join(DATA_DIR, "cezanne_stage3a_code_v1.json")
        code_src = fusion if os.path.exists(fusion) else code_f

        if os.path.exists(code_src):
            items = load_json(code_src)
            print(f"  Code source: {len(items)} items", flush=True)
            t0 = time.time()
            scored = []
            for item in items:
                combined = (item.get("instruction", "") + " " + item.get("output", "")).lower()
                cs_s = score_text(combined, CS_DEPTH_PATTERN)
                code_s = score_text(combined, CODE_PATTERN)
                algo_s = score_text(combined, CS_MATH_PATTERN)
                total = cs_s * 3 + algo_s * 2 + code_s
                if total > 2:
                    scored.append((total, item))
            scored.sort(key=lambda x: x[0], reverse=True)
            take = min(len(scored), code_need)
            for s, item in scored[:take]:
                item["_cat"] = "CS_Code"
                item["source"] = item.get("source", "CodeAlpaca")
                item["soul"] = "cezanne"
                all_items.append(item)
            print(f"  Added {take} CS-code items ({time.time()-t0:.1f}s)", flush=True)

    if len(all_items) > TARGET:
        random.shuffle(all_items)
        all_items = all_items[:TARGET]

    result = []
    for item in all_items:
        result.append({
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
            "output": item.get("output", ""),
            "source": item.get("source", "unknown"),
            "soul": "cezanne",
            "_cat": item.get("_cat", "CS"),
        })

    print(f"  S3 final: {len(result)} items", flush=True)
    return result


def generate_csapp_qa():
    all_qa = []

    notes_dir = os.path.join(CSAPP_DIR, "note", "booknote")
    if os.path.exists(notes_dir):
        for md_file in os.listdir(notes_dir):
            if not md_file.endswith(".md"):
                continue
            fpath = os.path.join(notes_dir, md_file)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
            except:
                continue
            sections = re.split(r'\n#{1,3}\s+', content)
            for section in sections:
                if len(section) < 100:
                    continue
                lines = section.strip().split('\n')
                title = lines[0].strip() if lines else ""
                body = '\n'.join(lines[1:]).strip()
                if len(body) < 50:
                    continue
                if CS_DEPTH_PATTERN.search(title + " " + body):
                    topic = title if title else md_file.replace(".md", "")
                    all_qa.append({
                        "instruction": f"Explain the concept of {topic} in computer systems.",
                        "input": "",
                        "output": body[:2000],
                        "source": "CSAPP_Notes",
                        "soul": "cezanne",
                        "_cat": "CS_Systems_CSAPP",
                    })

    qa_dir = os.path.join(CSAPP_DIR, "HIT课程资料", "期末考试QA")
    if os.path.exists(qa_dir):
        for qa_file in os.listdir(qa_dir):
            if not qa_file.endswith(".md"):
                continue
            fpath = os.path.join(qa_dir, qa_file)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
            except:
                continue
            parts = re.split(r'(?:#\s*●?\s*|##\s*\d+\.?\s*)', content)
            for part in parts:
                part = part.strip()
                if len(part) < 30:
                    continue
                lines = part.split('\n')
                question = lines[0].strip()
                answer = '\n'.join(lines[1:]).strip()
                if len(answer) < 20:
                    continue
                if CS_DEPTH_PATTERN.search(question + " " + answer):
                    all_qa.append({
                        "instruction": question,
                        "input": "",
                        "output": answer[:1500],
                        "source": "CSAPP_Exam_QA",
                        "soul": "cezanne",
                        "_cat": "CS_Systems_Exam",
                    })

    print(f"  CSAPP total Q&A: {len(all_qa)}", flush=True)
    return all_qa


def main():
    print("=" * 60, flush=True)
    print("  Cezanne 7B CS-Focused Dataset Builder v2", flush=True)
    print(f"  Target: {TARGET} items per stage", flush=True)
    print("=" * 60, flush=True)

    s1 = build_s1()
    s2 = build_s2()
    s3 = build_s3()

    for name, data in [("s1_cs_math_8k", s1), ("s2_cs_logic_8k", s2), ("s3_cs_depth_8k", s3)]:
        out_path = os.path.join(OUT_DIR, f"cezanne_{name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n  Saved: {out_path} ({len(data)} items)", flush=True)
        cats = Counter(item.get("_cat", "?") for item in data)
        srcs = Counter(item.get("source", "?") for item in data)
        print(f"  Cats: {dict(cats)}", flush=True)
        print(f"  Top sources: {dict(srcs.most_common(5))}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("  Done!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
