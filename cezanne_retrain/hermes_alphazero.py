#!/usr/bin/env python3
"""
Hermes AlphaGo Zero Self-Play Engine for Cezanne 7B
===================================================
Core loop: M_self_play -> M_train -> M_new -> repeat

Phase 1: Self-Play (M generates questions + answers + self-judges)
Phase 2: Reward Labeling (Monte-Carlo return: correct=1, wrong=0)
Phase 3: Training (positive samples only, or contrastive)
Phase 4: Evaluation (diagnostic exam + regression gate)
Phase 5: Auto-iterate (if pass, M_new becomes M; if fail, rollback)

All orchestrated by Hermes with safety gates.
"""
import os, sys, json, gc, time, random
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

BASE_MODEL = "/mnt/d/models/Mistral-7B-Instruct-v0.1"
DATA_DIR = "/mnt/d/VORTEX_FLAME/soul_training_data/cezanne"
LORA_DIR = "/mnt/d/VORTEX_FLAME/soul_lora_v2/cezanne"
LOG_DIR = "/mnt/d/VORTEX_FLAME/hermes_logs/cezanne"

sys.path.insert(0, "/mnt/d/VORTEX_FLAME")
try:
    from long_memory import recall, write_knowledge, close_knowledge_handles
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

EXTERNAL_DATASETS = {}

def load_external_datasets():
    global EXTERNAL_DATASETS
    try:
        from datasets import load_dataset as hf_load
        print("  Loading Opus-4.6-Reasoning (Anthropic)...", flush=True)
        opus_ds = hf_load("Crownelius/Opus-4.6-Reasoning-3300x",
                          cache_dir="/mnt/e/AI_Data/Opus-4.6-Reasoning", split="train")
        EXTERNAL_DATASETS["opus46"] = []
        for row in opus_ds:
            if row.get("problem") and row.get("solution"):
                EXTERNAL_DATASETS["opus46"].append({
                    "question": row["problem"],
                    "reference": row.get("solution", ""),
                    "thinking": row.get("thinking", ""),
                    "difficulty": row.get("difficulty", "medium"),
                    "category": row.get("category", ""),
                    "source": "Opus-4.6-Reasoning",
                })
        print(f"    Opus-4.6: {len(EXTERNAL_DATASETS['opus46'])} problems loaded", flush=True)
    except Exception as e:
        print(f"    [WARN] Failed to load Opus-4.6: {e}", flush=True)

    try:
        from datasets import load_dataset as hf_load
        print("  Loading Capybara (Anthropic)...", flush=True)
        cap_ds = hf_load("LDJnr/Capybara",
                         cache_dir="/mnt/e/AI_Data/Capybara", split="train")
        EXTERNAL_DATASETS["capybara"] = []
        for row in cap_ds:
            conv = row.get("conversation", row.get("messages", []))
            if len(conv) >= 2:
                human_msg = ""
                assistant_msg = ""
                for turn in conv:
                    role = turn.get("role", turn.get("from", ""))
                    content = turn.get("content", turn.get("value", ""))
                    if role in ("human", "user") and not human_msg:
                        human_msg = content
                    elif role in ("assistant", "gpt") and not assistant_msg:
                        assistant_msg = content
                if human_msg and len(human_msg) > 20:
                    EXTERNAL_DATASETS["capybara"].append({
                        "question": human_msg,
                        "reference": assistant_msg,
                        "source": "Capybara",
                    })
        print(f"    Capybara: {len(EXTERNAL_DATASETS['capybara'])} problems loaded", flush=True)
    except Exception as e:
        print(f"    [WARN] Failed to load Capybara: {e}", flush=True)

    total = sum(len(v) for v in EXTERNAL_DATASETS.values())
    print(f"  External datasets total: {total} problems", flush=True)
    if total == 0:
        print(f"  [WARN] No external datasets loaded, falling back to 100% CS_TOPICS", flush=True)

EXTERNAL_DATASET_RATIO = 0.4
CHINESE_TEMPLATE_RATIO = 0.3
EINSTEIN_TEST_RATIO = 0.15

EINSTEIN_QUESTIONS = [
    {
        "id": "e01_sorting",
        "lockdown": "You only know: arrays, loops, comparisons, and swaps. No sorting algorithm exists yet.",
        "challenge": "Invent a method to arrange numbers in ascending order. Derive your algorithm step by step from first principles. Explain WHY it works, not just HOW.",
        "domain": "Algorithm",
        "discovery_keywords": ["swap", "compare", "adjacent", "iterate", "pass", "sorted", "invariant", "bubble", "selection", "insertion", "partition", "pivot", "divide", "conquer", "merge", "recursive", "base case"],
        "cn_lockdown": "你只知道：数组、循环、比较、交换。排序算法还不存在。",
        "cn_challenge": "发明一种将数字按升序排列的方法。从基本原理逐步推导你的算法。解释为什么它有效，而不仅仅是怎么做。",
    },
    {
        "id": "e02_concurrency",
        "lockdown": "You only know: sequential execution, functions, variables. No concurrency concept exists yet.",
        "challenge": "A program needs to do two things simultaneously (e.g., read input while processing data). Design a solution. Identify ALL potential problems your solution creates and propose fixes.",
        "domain": "Systems",
        "discovery_keywords": ["simultaneous", "parallel", "concurrent", "race", "shared", "lock", "mutex", "atomic", "interrupt", "thread", "process", "schedule", "context switch", "deadlock", "synchronization", "mutual exclusion", "critical section"],
        "cn_lockdown": "你只知道：顺序执行、函数、变量。并发概念还不存在。",
        "cn_challenge": "一个程序需要同时做两件事（比如一边读输入一边处理数据）。设计一个解决方案。找出你的方案可能造成的所有问题并提出修复方案。",
    },
    {
        "id": "e03_virtual_memory",
        "lockdown": "You only know: flat physical memory addresses, pointers. No virtual memory or paging exists yet.",
        "challenge": "A program needs 10GB but only 4GB of physical RAM exists. Design a memory system that makes the program think it has unlimited memory. Derive the concept from the physical constraint.",
        "domain": "Systems",
        "discovery_keywords": ["virtual", "page", "table", "translate", "map", "swap", "disk", "frame", "fault", "miss", "tlb", "protection", "isolation", "segment", "offset", "indirection", "illusion", "abstraction"],
        "cn_lockdown": "你只知道：平坦物理内存地址、指针。虚拟内存和分页还不存在。",
        "cn_challenge": "一个程序需要10GB但只有4GB物理内存。设计一个内存系统让程序以为它有无限内存。从物理约束出发推导这个概念。",
    },
    {
        "id": "e04_hashing",
        "lockdown": "You only know: arrays, integers, modulo operation. No hash table or dictionary exists yet.",
        "challenge": "You need to store and retrieve items by name (string) in O(1) time. Arrays only support integer indices. Design a data structure that solves this problem. Identify and handle edge cases.",
        "domain": "Algorithm",
        "discovery_keywords": ["hash", "function", "map", "index", "collision", "chaining", "probing", "linear", "bucket", "load", "factor", "rehash", "constant", "direct", "modulo", "distribution", "uniform"],
        "cn_lockdown": "你只知道：数组、整数、取模运算。哈希表和字典还不存在。",
        "cn_challenge": "你需要按名称（字符串）存储和查找元素，时间复杂度O(1)。数组只支持整数索引。设计一个数据结构解决这个问题。识别并处理边界情况。",
    },
    {
        "id": "e05_deadlock",
        "lockdown": "You only know: locks for mutual exclusion, shared resources. No deadlock concept has been identified yet.",
        "challenge": "Two processes each hold one lock and wait for the other's lock. This situation arises naturally from your lock design. Discover this problem, prove it is inevitable under certain conditions, and design a prevention mechanism.",
        "domain": "Systems",
        "discovery_keywords": ["deadlock", "circular", "wait", "hold", "prevention", "avoidance", "detection", "recovery", "order", "priority", "timeout", "preempt", "resource", "allocation", "graph", "cycle", "coffman", "necessary", "condition", "banker"],
        "cn_lockdown": "你只知道：互斥锁、共享资源。死锁概念还未被发现。",
        "cn_challenge": "两个进程各持有一把锁并等待对方的锁。这种情况从锁的设计中自然产生。发现这个问题，证明它在某些条件下不可避免，并设计预防机制。",
    },
    {
        "id": "e06_recursion",
        "lockdown": "You only know: loops, functions, and call stacks. No recursive algorithm exists yet.",
        "challenge": "Some problems have a self-similar structure (e.g., a folder contains folders). Invent a programming technique to handle self-similar problems. Derive it from the nature of the call stack. Identify the critical requirement for termination.",
        "domain": "Algorithm",
        "discovery_keywords": ["recursive", "base case", "call", "stack", "self-similar", "subproblem", "divide", "terminate", "return", "depth", "overflow", "induction", "hypothesis", "step", "trivial", "reduction", "decompose"],
        "cn_lockdown": "你只知道：循环、函数、调用栈。递归算法还不存在。",
        "cn_challenge": "某些问题具有自相似结构（比如文件夹包含文件夹）。发明一种编程技术来处理自相似问题。从调用栈的本质推导。识别终止的关键要求。",
    },
    {
        "id": "e07_cache",
        "lockdown": "You only know: CPU executes instructions, memory has latency. No cache concept exists yet.",
        "challenge": "Memory is 100x slower than CPU. Programs waste most time waiting for data. Design a hardware mechanism to bridge this speed gap. Derive it from the observation that programs access the same data repeatedly.",
        "domain": "Systems",
        "discovery_keywords": ["cache", "locality", "temporal", "spatial", "hit", "miss", "line", "eviction", "lru", "associativity", "set", "way", "block", "prefetch", "hierarchy", "l1", "l2", "l3", "cold", "warm", "replacement"],
        "cn_lockdown": "你只知道：CPU执行指令，内存有延迟。缓存概念还不存在。",
        "cn_challenge": "内存比CPU慢100倍。程序大部分时间在等数据。设计一个硬件机制来弥合这个速度差距。从程序反复访问相同数据的观察出发推导。",
    },
    {
        "id": "e08_network_reliability",
        "lockdown": "You only know: unreliable physical links that drop or reorder packets. No reliable protocol exists yet.",
        "challenge": "Two computers need to exchange data reliably over unreliable links. Design a protocol that guarantees correct, complete, ordered delivery. Derive each mechanism from a specific failure mode you identify.",
        "domain": "Network",
        "discovery_keywords": ["acknowledgment", "sequence", "number", "retransmit", "timeout", "window", "sliding", "handshake", "syn", "fin", "duplicate", "order", "checksum", "cumulative", "flow", "control", "congestion", "reliable", "connection"],
        "cn_lockdown": "你只知道：不可靠的物理链路会丢包或乱序。可靠协议还不存在。",
        "cn_challenge": "两台计算机需要通过不可靠链路可靠地交换数据。设计一个保证正确、完整、有序传输的协议。从你识别的每个故障模式推导出对应的机制。",
    },
    {
        "id": "e09_compression",
        "lockdown": "You only know: bits, bytes, and frequency counting. No compression algorithm exists yet.",
        "challenge": "Data takes too much space. Some symbols appear more frequently than others. Invent a method to represent data using fewer bits on average. Prove your method cannot lose information.",
        "domain": "Algorithm",
        "discovery_keywords": ["frequency", "variable", "length", "prefix", "code", "huffman", "entropy", "symbol", "encode", "decode", "tree", "leaf", "optimal", "lossless", "reversible", "shannon", "information", "redundancy", "bit"],
        "cn_lockdown": "你只知道：比特、字节、频率计数。压缩算法还不存在。",
        "cn_challenge": "数据占用太多空间。某些符号出现频率更高。发明一种方法用更少的比特平均表示数据。证明你的方法不会丢失信息。",
    },
    {
        "id": "e10_type_system",
        "lockdown": "You only know: raw bits in memory. No type system or type checking exists yet.",
        "challenge": "Programs keep crashing because integers are used as strings, or functions receive wrong data shapes. Invent a system to prevent such errors before the program runs. Derive it from the root cause of these crashes.",
        "domain": "Logic",
        "discovery_keywords": ["type", "check", "static", "dynamic", "inference", "annotation", "constraint", "safety", "compile", "runtime", "cast", "polymorphism", "generic", "signature", "contract", "invariant", "soundness", "completeness", "subtyping"],
        "cn_lockdown": "你只知道：内存中的原始比特。类型系统和类型检查还不存在。",
        "cn_challenge": "程序不断崩溃，因为整数被当作字符串使用，或者函数收到错误的数据形状。发明一个系统在程序运行前防止这类错误。从这些崩溃的根本原因推导。",
    },
    {
        "id": "e11_halting",
        "lockdown": "You only know: programs execute step by step, some loop forever. No theoretical limit on program analysis exists yet.",
        "challenge": "Can you write a program that takes another program as input and determines whether it will eventually stop or run forever? Try to design one. If you discover it is impossible, prove why. This is the most important discovery in computer science.",
        "domain": "Logic",
        "discovery_keywords": ["halting", "undecidable", "impossible", "proof", "contradiction", "diagonal", "self-reference", "turing", "infinite", "loop", "decidable", "oracle", "assume", "suppose", "contradiction", "impossible", "cannot exist", "paradox"],
        "cn_lockdown": "你只知道：程序逐步执行，有些会永远循环。程序分析的理论极限还未被发现。",
        "cn_challenge": "你能写一个程序，输入另一个程序，判断它最终会停止还是永远运行吗？试着设计一个。如果你发现这是不可能的，证明为什么。这是计算机科学中最重要的发现。",
    },
    {
        "id": "e12_search_index",
        "lockdown": "You only know: linear scanning of documents. No search index exists yet.",
        "challenge": "You have 1 million documents and need to find all documents containing a specific word in under 1 second. Linear scan takes 10 minutes. Invent a data structure that makes search fast. Derive it from the observation about word-document relationships.",
        "domain": "Algorithm",
        "discovery_keywords": ["index", "inverted", "posting", "list", "term", "frequency", "tf", "idf", "tokenize", "lexicon", "dictionary", "b-tree", "hash", "map", "lookup", "scan", "pointer", "bitmap", "prefix", "trie"],
        "cn_lockdown": "你只知道：线性扫描文档。搜索索引还不存在。",
        "cn_challenge": "你有100万篇文档，需要在1秒内找到包含特定词的所有文档。线性扫描需要10分钟。发明一种让搜索变快的数据结构。从词-文档关系的观察出发推导。",
    },
]

EINSTEIN_REASONING_MARKERS = [
    "observe", "notice", "realize", "discover", "suppose", "assume", "what if",
    "therefore", "thus", "hence", "consequently", "it follows", "imply", "imply",
    "proof", "prove", "contradiction", "absurd", "impossible", "cannot",
    "invariant", "property", "guarantee", "ensure", "maintain", "preserve",
    "edge case", "boundary", "corner case", "degenerate", "extreme",
    "first", "then", "next", "finally", "step", "derive", "deduce",
    "because", "since", "due to", "reason", "cause", "root cause",
    "because", "so", "hence", "thus", "therefore", "consequently",
    "observation", "hypothesis", "experiment", "verify", "confirm",
]

EINSTEIN_CN_REASONING_MARKERS = [
    "观察", "发现", "注意到", "假设", "如果", "那么", "因此", "所以", "由此",
    "证明", "矛盾", "不可能", "不变式", "性质", "保证", "确保", "维持",
    "边界", "极端", "退化", "首先", "然后", "接着", "最后", "推导", "演绎",
    "因为", "由于", "原因", "根本原因", "假设", "验证", "确认", "反证",
    "归纳", "演绎", "直觉", "洞察", "关键", "核心", "本质",
]

CHINESE_TEMPLATES = {
    "Debug": [
        "找出这段代码中的bug并修复：{code}",
        "这段代码有什么问题？请解释并修正：{code}",
        "以下代码在边界情况下会出错，请找出bug：{code}",
        "这段代码有什么隐藏的错误？请分析：{code}",
        "请找出代码中的缺陷并说明原因：{code}",
    ],
    "Logic": [
        "证明或反驳以下命题：{claim}",
        "以下论证是否有效？请分析：{argument}",
        "用形式逻辑解释{concept}，并举例说明。",
        "判断以下推理是否正确，说明理由：{argument}",
        "什么是{concept}？请用逻辑学的方式解释。",
    ],
    "Algorithm": [
        "用Python实现{algo}算法。",
        "解释{algo}的时间复杂度。",
        "比较{algo}和归并排序的优缺点。",
        "{algo}算法的核心思想是什么？请详细说明。",
        "在什么场景下应该使用{algo}而不是其他算法？",
    ],
    "Systems": [
        "解释{concept}的工作原理。",
        "{concept}在操作系统中是如何实现的？",
        "进程和线程在{concept}方面有什么区别？",
        "请详细说明{concept}的内部机制。",
        "{concept}为什么对系统性能很重要？",
    ],
    "Complexity": [
        "归并排序的时间复杂度是多少？请解释原因。",
        "用例子说明摊还分析的概念。",
        "比较O(n^2)和O(n log n)时间复杂度的区别。",
        "O(log n)是什么意思？请举例说明。",
        "快速排序的空间复杂度是多少？为什么？",
    ],
    "Network": [
        "解释TCP和UDP的区别。",
        "什么是SQL注入？如何防止？",
        "解释TCP拥塞控制和慢启动机制。",
        "什么是REST API？说明其主要原则。",
        "解释DNS域名解析的工作过程。",
    ],
    "Memory": [
        "解释FAISS向量索引在长期记忆检索中的工作原理。",
        "语义搜索和关键词搜索在RAG系统中有什么区别？",
        "如何设计一个AI系统的遗忘机制来防止记忆干扰？",
        "解释AI系统中情景记忆和语义记忆的区别。",
        "如何实现代码文件的分块策略以最大化检索准确率？",
    ],
    "WorldModel": [
        "在RTX 3060 12GB显存上训练世界模型，编码器应该选择CNN还是Transformer？为什么？",
        "解释世界模型中编码器-动力学模型-解码器的三阶段架构。",
        "什么是潜在空间压缩？为什么世界模型需要将每帧压缩到15-32个token？",
        "3D高斯泼溅和NeRF在RTX 3060上各有什么优劣？",
        "在8GB显存限制下，世界模型的参数量应该控制在什么范围？",
        "解释梯度检查点技术如何将显存从O(L)降到O(√L)。",
        "世界模型的训练为什么要分预训练和微调两个阶段？",
        "混合精度训练在RTX 3060上如何减少40-50%显存占用？",
        "解释DreamerV3的编码器+动力学+价值/策略训练管线。",
        "4-bit量化技术如何让7B模型在12GB显存上运行？",
        "torch.compile的max-autotune模式如何优化世界模型训练？",
        "世界模型中RNN、Transformer和扩散模型三种架构各适合什么场景？",
    ],
}

CHINESE_VERIFY_KEYWORDS = {
    "Debug": ["差一", "竞态", "锁", "基线", "索引", "越界", "引用", "拷贝", "深拷贝", "空", "除零", "键错误", "名字错误", "泄漏", "关闭", "资源", "可变", "默认值", "检查", "条件", "迭代器", "并发", "原子", "递归", "栈", "无限", "边界", "异常", "未初始化", "副作用", "共享状态", "循环引用", "死循环", "浅拷贝", "资源泄漏", "假值"],
    "Logic": ["逆否", "等价", "谬误", "无效", "有效", "矛盾", "假设", "充分", "必要", "演绎", "归纳", "三段论", "德摩根", "否定", "逆命题", "双条件", "量词", "重言式", "反例", "真值", "前提", "结论"],
    "Algorithm": ["分区", "枢轴", "分治", "优先", "队列", "访问", "邻接", "距离", "递归", "迭代", "贪心", "动态", "松弛", "负权", "最小", "生成树", "并查集", "哈希", "缓存", "淘汰", "摊还", "滚动哈希", "模式匹配", "前缀", "后缀", "旋转", "再平衡"],
    "Systems": ["缓存", "虚拟", "页表", "快表", "堆", "死锁", "互斥", "进程", "线程", "内存", "内核", "用户态", "上下文", "调度", "中断", "信号量", "自旋锁", "碎片", "垃圾回收", "写时复制", "缺页", "索引节点", "日志", "RAID", "乱序", "分支预测"],
    "Complexity": ["时间复杂度", "空间复杂度", "摊还", "最坏", "平均", "最好", "大O", "下界", "上界", "紧界", "多项式", "指数", "归约", "判定问题", "验证"],
    "Network": ["可靠", "连接", "拥塞", "握手", "加密", "证书", "防火墙", "代理", "负载均衡", "分组", "路由", "子网", "无状态", "注入", "参数化"],
    "Memory": ["召回", "存储", "索引", "向量", "相似度", "检索", "增强", "知识", "长期", "短期", "情景", "语义", "巩固", "遗忘", "干扰", "分块", "复述"],
    "WorldModel": ["编码器", "解码器", "动力学", "潜在空间", "压缩", "token", "显存", "梯度检查点", "混合精度", "量化", "3D高斯", "NeRF", "渲染", "扩散模型", "DreamerV3", "RNN", "Transformer", "CNN", "预训练", "微调", "状态转移", "预测", "重建损失", "KL散度", "批量大小", "学习率", "余弦退火", "AdamW", "Tensor Core", "torch.compile", "4-bit", "8-bit", "FP16", "VAE", "潜变量", "时序", "帧", "分辨率", "参数量", "兆", "内存", "优化器", "调度器"],
    "External": ["时间复杂度", "空间复杂度", "边界条件", "基线情况", "正确性", "不变式", "归纳", "最优", "子问题", "反证", "反例", "大O", "摊还", "伪代码", "终止性"],
}

CHINESE_REGRESSION_QUESTIONS = [
    {"q": "找出这段代码的bug：def binary_search(arr, target): left,right=0,len(arr); while left<right: mid=(left+right)//2; if arr[mid]==target: return mid; elif arr[mid]<target: left=mid; else: right=mid; return -1", "kw": ["差一", "mid+1", "无限", "边界", "off-by-one", "infinite"]},
    {"q": "用形式逻辑证明：如果P蕴含Q，Q蕴含R，则P蕴含R。", "kw": ["三段论", "蕴含", "传递", "syllogism", "implies"]},
    {"q": "用Python实现Dijkstra最短路径算法。", "kw": ["dijkstra", "优先", "队列", "距离", "priority", "queue"]},
    {"q": "解释虚拟内存和页表的工作原理。", "kw": ["虚拟", "页表", "快表", "virtual", "page"]},
    {"q": "归并排序的时间复杂度是多少？为什么？", "kw": ["n log", "分治", "合并", "divide", "merge"]},
    {"q": "找出bug：def fibonacci(n): return fibonacci(n-1) + fibonacci(n-2)", "kw": ["基线", "无限", "递归", "base case", "recursion"]},
    {"q": "什么是逆否命题？请举例说明。", "kw": ["逆否", "等价", "否定", "contrapositive", "equivalent"]},
    {"q": "解释CPU缓存为什么对性能很重要。", "kw": ["缓存", "局部性", "cache", "locality"]},
    {"q": "解释TCP和UDP的区别。", "kw": ["可靠", "连接", "TCP", "UDP", "reliable"]},
    {"q": "什么是SQL注入？如何防止？", "kw": ["注入", "参数化", "injection", "parameterized"]},
]

MAX_ITERATIONS = 15
QUESTIONS_PER_ITER = 1200
EPOCHS_PER_ITER = 2
LR = 5e-5
LORA_R = 16
LORA_ALPHA = 32
MAX_LOSS_INITIAL = 3.0
REGRESSION_MAX_DROP = 2
PASS_THRESHOLD_SCHEDULE = {1: 4, 2: 4, 3: 5, 4: 5, 5: 6, 6: 6, 7: 7, 8: 7, 9: 8, 10: 8, 11: 9, 12: 9, 13: 10, 14: 10, 15: 10}
CHECKPOINT_FILE = os.path.join(os.path.dirname(__file__), "alphazero_checkpoint.json")

CS_TOPICS = {
    "Debug": {
        "code_snippets": [
            "def binary_search(arr, target):\n    left, right = 0, len(arr)\n    while left < right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid  # BUG: should be mid+1, causes infinite loop\n        else:\n            right = mid\n    return -1",
            "def find_max(lst):\n    max_val = 0  # BUG: fails for all-negative lists, should be lst[0] or float('-inf')\n    for x in lst:\n        if x > max_val:\n            max_val = x\n    return max_val",
            "def remove_duplicates(lst):\n    for i in range(len(lst)):\n        if lst[i] in lst[i+1:]:\n            lst.remove(lst[i])  # BUG: modifying list while iterating by index shifts elements\n    return lst",
            "def average(lst):\n    return sum(lst) / len(lst)  # BUG: ZeroDivisionError when lst is empty",
            "def merge(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) or j < len(b):  # BUG: 'or' should be 'and', causes IndexError\n        if a[i] < b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    return result",
            "d = {}\nd['key'] += 1  # BUG: KeyError, key doesn't exist yet. Need d.get('key',0)+1 or defaultdict",
            "try:\n    result = 10 / 0\nexcept:\n    pass\nprint(result)  # BUG: result undefined if exception caught, NameError",
            "x = [1, 2, 3]\ny = x  # BUG: shallow copy, y is same object as x\ny.append(4)\nprint(x)  # prints [1,2,3,4] - mutation shared",
            "for i in range(len(lst)):\n    if lst[i] == target:\n        lst.remove(target)  # BUG: removing while iterating shifts indices, skips elements",
            "def copy_dict(d):\n    new_d = d  # BUG: shallow reference, not a copy. Need d.copy() or dict(d)\n    new_d['x'] = 1\n    return new_d",
            "def reverse_list(head):\n    prev = None\n    curr = head\n    while curr:\n        curr.next = prev  # BUG: loses reference to next node before advancing\n        prev = curr\n        curr = curr.next  # curr.next already overwritten, infinite loop or early exit\n    return prev",
            "def fibonacci(n):\n    return fibonacci(n-1) + fibonacci(n-2)  # BUG: no base case, infinite recursion + StackOverflow",
            "def safe_divide(a, b):\n    return a / b if b else 0  # BUG: returns 0 for b=False or b=0.0, but 0.0 is falsy. Should check b != 0",
            "def read_file(path):\n    f = open(path)\n    data = f.read()\n    return data  # BUG: file never closed, resource leak. Use 'with open(path) as f:'",
            "def pop_all(stack):\n    while stack:\n        stack.pop()\n    return stack[0]  # BUG: IndexError, stack is empty after while loop",
            "class Counter:\n    count = 0  # BUG: class variable shared across all instances, should be instance var in __init__\n    def increment(self):\n        self.count += 1",
            "def recursive_sum(lst):\n    return lst[0] + recursive_sum(lst[1:])  # BUG: no base case for empty list, IndexError",
            "def is_sorted(lst):\n    return all(lst[i] <= lst[i+1] for i in range(len(lst)))  # BUG: off-by-one, range should be len(lst)-1",
            "def flatten(nested):\n    result = []\n    for item in nested:\n        result.extend(item)  # BUG: assumes item is always iterable, fails for non-list elements\n    return result",
            "def parse_config(text):\n    lines = text.split('\\n')\n    config = {}\n    for line in lines:\n        k, v = line.split('=')  # BUG: fails on lines without '=' or with multiple '='\n        config[k] = v\n    return config",
            "async def fetch_all(urls):\n    results = []\n    for url in urls:\n        r = await fetch(url)  # BUG: sequential await, not concurrent. Use asyncio.gather()\n        results.append(r)\n    return results",
            "def swap(a, b):\n    a, b = b, a  # BUG: only swaps local variables, caller's values unchanged (Python pass-by-assignment)\n    return a, b",
            "def cache_result(func):\n    results = {}  # BUG: mutable default-like shared state across calls\n    def wrapper(*args):\n        if args not in results:\n            results[args] = func(*args)\n        return results[args]\n    return wrapper",
            "def dedup(lst):\n    seen = set()\n    return [x for x in lst if x not in seen and not seen.add(x)]  # BUG: relies on set.add() returning None (falsy), confusing and fragile",
            "class Node:\n    def __init__(self, val):\n        self.val = val\n        self.next = self  # BUG: self-referencing by default, creates cycle. Should be None\n    def insert(self, val):\n        new = Node(val)\n        new.next = self.next\n        self.next = new",
            "def hash_lookup(table, key):\n    return table[key]  # BUG: KeyError if key missing. Use table.get(key) or handle exception",
            "def linked_cycle(head):\n    slow = head\n    fast = head\n    while fast and fast.next:\n        slow = slow.next\n        fast = fast.next.next\n        if slow == fast:\n            return True\n    return False  # Not a bug per se, but fails if head is None (AttributeError on head.next)",
            "def count_occurrences(lst, val):\n    return len([x for x in lst if x == val]) - lst.count(val)  # BUG: double-counting subtraction makes no sense, should just be lst.count(val)",
            "def sort_by_key(d):\n    return sorted(d.items(), key=lambda x: x[1], reverse=True)  # Not a bug, but fails if values are uncomparable (e.g., mixed types)",
        ],
        "verify_keywords": ["off-by-one", "race", "lock", "base case", "index", "bounds", "exhausted", "reference", "copy", "deepcopy", "zero", "empty", "division", "keyerror", "nameerror", "silent", "null", "pointer", "overflow", "underflow", "deadlock", "leak", "close", "resource", "mutable", "default", "guard", "check", "condition", "iterator", "concurrent", "atomic", "recursion", "stack", "infinite", "boundary", "exception", "uninitialized", "side effect", "shared state", "circular", "missing return", "infinite loop", "zero-division", "shallow copy", "resource leak", "falsy value"],
    },
    "Logic": {
        "claims": [
            "If P implies Q, then not Q implies not P",
            "All prime numbers are odd",
            "If a program has no syntax errors, it has no bugs",
            "O(n^2) is always slower than O(n log n)",
            "Every graph with n-1 edges is a tree",
            "If a sorting algorithm is stable, it must be O(n log n)",
            "Every infinite set is uncountable",
            "If A is a subset of B, then the power set of A is a subset of the power set of B",
            "A connected graph with no cycles is a tree",
            "All NP-complete problems are in P",
            "If a function is continuous, it is differentiable",
            "Every bounded sequence converges",
            "If A implies B and B implies C, then not C implies not A",
            "A bipartite graph always has an Euler circuit",
            "Every regular language is context-free",
        ],
        "arguments": [
            "If it rains, the ground is wet. The ground is wet. Therefore it rained.",
            "All CS students know Python. Alice knows Python. Therefore Alice is a CS student.",
            "If P then Q. Not P. Therefore not Q.",
            "If the program compiles, it is correct. The program is correct. Therefore it compiles.",
            "All recursive functions need a base case. This function has a base case. Therefore it is recursive.",
            "If the test passes, the code is correct. The test passes. Therefore the code is correct.",
            "No bugs exist in tested code. This code is tested. Therefore no bugs exist.",
            "If A or B is true, and A is false, then B must be true.",
        ],
        "concepts": ["contrapositive", "proof by contradiction", "De Morgan's Law", "necessary vs sufficient condition", "deductive vs inductive reasoning", "syllogism", "logical fallacy", "modus ponens", "modus tollens", "biconditional", "existential quantifier", "universal quantifier", "logical equivalence", "truth table", "tautology", "contradiction", "contingency", "soundness", "completeness", "validity"],
        "verify_keywords": ["contrapositive", "equivalent", "fallacy", "invalid", "valid", "contradiction", "assume", "sufficient", "necessary", "deductive", "inductive", "modus", "syllogism", "De Morgan", "negation", "converse", "inverse", "biconditional", "quantifier", "tautology", "counterexample", "truth", "premise", "conclusion", "logical", "sound", "complete"],
    },
    "Algorithm": {
        "algos": ["quicksort", "mergesort", "Dijkstra's algorithm", "BFS", "DFS", "binary search", "heap sort", "topological sort", "quickselect", "A* search", "insertion sort", "counting sort", "radix sort", "Bellman-Ford", "Floyd-Warshall", "Kruskal's algorithm", "Prim's algorithm", "union-find", "LRU cache", "trie insertion", "Rabin-Karp", "KMP", "boyer-moore", "segment tree", "fenwick tree", "suffix array", "balanced BST", "red-black tree", "B-tree", "hash table with chaining"],
        "verify_keywords": ["partition", "pivot", "divide", "conquer", "priority", "queue", "visited", "adjacency", "distance", "recursion", "iterative", "O(n log n)", "O(n^2)", "O(V+E)", "greedy", "dynamic", "relaxation", "negative", "cycle", "minimum", "spanning", "tree", "disjoint", "set", "hash", "linked", "cache", "eviction", "amortized", "rolling hash", "pattern", "match", "prefix", "suffix", "rotation", "rebalance"],
    },
    "Systems": {
        "concepts": ["CPU cache hierarchy", "virtual memory", "malloc/free", "deadlock", "process vs thread", "memory-mapped I/O", "user space vs kernel space", "TLB", "page fault", "context switch", "interrupt", "system call", "pipe", "socket", "shared memory", "semaphore", "mutex vs spinlock", "stack overflow", "heap fragmentation", "garbage collection", "copy-on-write", "demand paging", "page replacement", "file descriptor", "inode", "journaling filesystem", "RAID", "NUMA", "out-of-order execution", "branch prediction"],
        "verify_keywords": ["cache", "L1", "L2", "L3", "locality", "virtual", "page", "TLB", "malloc", "heap", "free", "deadlock", "mutex", "process", "thread", "memory", "kernel", "user", "privilege", "mmap", "context", "scheduler", "interrupt", "syscall", "pipe", "socket", "semaphore", "spinlock", "fragmentation", "GC", "mark", "sweep", "copy-on-write", "paging", "inode", "journal", "RAID", "NUMA", "speculative", "branch"],
    },
    "Complexity": {
        "verify_keywords": ["O(n)", "O(n log n)", "O(n^2)", "O(log n)", "O(1)", "amortized", "worst", "average", "best", "space", "time", "big-O", "theta", "omega", "recurrence", "master theorem", "lower bound", "upper bound", "tight bound", "NP", "P", "NP-hard", "NP-complete", "polynomial", "exponential", "reduction", "decision problem", "verification"],
    },
    "Network": {
        "verify_keywords": ["TCP", "UDP", "reliable", "connection", "congestion", "DNS", "HTTP", "REST", "SQL", "injection", "parameterized", "stateless", "handshake", "encryption", "TLS", "SSL", "certificate", "firewall", "proxy", "load balancer", "CDN", "packet", "routing", "subnet", "NAT", "OSPF", "BGP", "ARP", "DHCP", "WebSocket", "gRPC", "GraphQL"],
    },
    "Memory": {
        "verify_keywords": ["recall", "store", "index", "FAISS", "embedding", "vector", "similarity", "retrieval", "augmented", "RAG", "knowledge", "long-term", "short-term", "episodic", "semantic", "consolidation", "forgetting", "interference", "chunking", "rehearsal"],
    },
    "WorldModel": {
        "verify_keywords": ["encoder", "decoder", "dynamics", "latent", "compression", "token", "VRAM", "memory", "gradient checkpointing", "mixed precision", "quantization", "3D gaussian", "NeRF", "rendering", "diffusion", "DreamerV3", "RNN", "Transformer", "CNN", "pretrain", "finetune", "state transition", "prediction", "reconstruction loss", "KL divergence", "batch size", "learning rate", "cosine annealing", "AdamW", "Tensor Core", "torch.compile", "4-bit", "8-bit", "FP16", "VAE", "latent variable", "temporal", "frame", "resolution", "parameters", "million", "optimizer", "scheduler", "world model", "potential", "splatting", "MuJoCo", "action-conditioned", "video prediction"],
    },
    "External": {
        "verify_keywords": ["time complexity", "space complexity", "edge case", "base case", "boundary condition", "correctness", "invariant", "induction", "recurrence", "optimal", "subproblem", "proof by", "contradiction", "counterexample", "big-O", "amortized", "worst case", "best case", "average case", "pseudocode", "implementation detail", "correctness proof", "termination", "soundness", "completeness"],
    },
}

REGRESSION_QUESTIONS = [
    {"q": "Find the bug: def binary_search(arr, target): left,right=0,len(arr); while left<right: mid=(left+right)//2; if arr[mid]==target: return mid; elif arr[mid]<target: left=mid; else: right=mid; return -1", "kw": ["off-by-one", "mid + 1", "infinite", "boundary"]},
    {"q": "Prove using formal logic: if P implies Q and Q implies R, then P implies R.", "kw": ["syllogism", "implies", "transitivity", "modus"]},
    {"q": "Write Dijkstra's algorithm in Python.", "kw": ["dijkstra", "priority", "queue", "distance"]},
    {"q": "Explain virtual memory and page tables.", "kw": ["virtual", "page", "tlb", "translation"]},
    {"q": "Explain how malloc() works internally.", "kw": ["malloc", "heap", "free", "chunk"]},
    {"q": "What is the time complexity of merge sort?", "kw": ["n log", "merge", "sort", "divide"]},
    {"q": "Find the bug: def fibonacci(n): return fibonacci(n-1) + fibonacci(n-2)", "kw": ["base case", "infinite", "recursion", "n<=1"]},
    {"q": "What is a contrapositive? Give an example.", "kw": ["contrapositive", "not", "implies", "equivalent"]},
    {"q": "Explain CPU cache and why it matters for performance.", "kw": ["cache", "l1", "l2", "locality"]},
    {"q": "Explain proof by contradiction with an example.", "kw": ["contradiction", "assume", "false", "absurd"]},
] + CHINESE_REGRESSION_QUESTIONS


def load_model(lora_path=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if lora_path and os.path.exists(lora_path):
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def generate(model, tokenizer, prompt, max_tokens=256):
    if MEMORY_AVAILABLE:
        try:
            memory_context = recall("cezanne", prompt, top_k=3, max_chars=500)
            if memory_context:
                prompt = f"[Reference Knowledge]\n{memory_context}\n[/Reference Knowledge]\n\n{prompt}"
        except Exception:
            pass
    formatted = f"<s>[INST] {prompt} [/INST] "
    inp = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=768).to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_tokens, temperature=0.7,
                             do_sample=True, pad_token_id=tokenizer.eos_token_id,
                             use_cache=True)
    ans = tokenizer.decode(out[0], skip_special_tokens=True)
    del inp, out
    torch.cuda.empty_cache()
    return ans.replace(formatted, "").strip()


def generate_question(model, tokenizer, topic):
    if EXTERNAL_DATASETS and random.random() < EXTERNAL_DATASET_RATIO:
        all_problems = []
        for ds_name, problems in EXTERNAL_DATASETS.items():
            all_problems.extend(problems)
        if all_problems:
            chosen = random.choice(all_problems)
            q = chosen["question"]
            if len(q) > 1200:
                q = q[:1200]
            source_tag = f"[{chosen['source']}]"
            return f"{source_tag} {q}", chosen.get("reference", "")

    if random.random() < EINSTEIN_TEST_RATIO and EINSTEIN_QUESTIONS:
        eq = random.choice(EINSTEIN_QUESTIONS)
        use_cn = random.random() < CHINESE_TEMPLATE_RATIO
        if use_cn:
            question = f"[EINSTEIN TEST - Knowledge Lockdown]\nKnown: {eq['cn_lockdown']}\nChallenge: {eq['cn_challenge']}"
        else:
            question = f"[EINSTEIN TEST - Knowledge Lockdown]\nKnown: {eq['lockdown']}\nChallenge: {eq['challenge']}"
        return question, f"EINSTEIN:{eq['id']}"

    use_chinese = random.random() < CHINESE_TEMPLATE_RATIO
    topic_data = CS_TOPICS.get(topic, {})

    if use_chinese and topic in CHINESE_TEMPLATES:
        cn_templates = CHINESE_TEMPLATES[topic]
        if topic == "Debug":
            code = random.choice(topic_data.get("code_snippets", ["print('hello')"]))
            template = random.choice(cn_templates)
            question = template.format(code=code)
        elif topic == "Logic":
            choice = random.choice(["claim", "argument", "concept"])
            if choice == "claim":
                item = random.choice(topic_data.get("claims", ["P implies Q"]))
                template = random.choice(cn_templates[:2] + cn_templates[3:4])
                question = template.format(claim=item, argument=item, concept=item)
            elif choice == "argument":
                item = random.choice(topic_data.get("arguments", ["If P then Q. P. Therefore Q."]))
                template = random.choice(cn_templates[1:2] + cn_templates[3:5])
                question = template.format(claim=item, argument=item, concept=item)
            else:
                item = random.choice(topic_data.get("concepts", ["contrapositive"]))
                template = random.choice(cn_templates[2:3] + cn_templates[4:5])
                question = template.format(claim=item, argument=item, concept=item)
        elif topic == "Algorithm":
            algos = topic_data.get("algos", ["quicksort"])
            algo = random.choice(algos)
            template = random.choice(cn_templates)
            question = template.format(algo=algo)
        elif topic == "Systems":
            concepts = topic_data.get("concepts", ["virtual memory"])
            concept = random.choice(concepts)
            template = random.choice(cn_templates)
            question = template.format(concept=concept)
        else:
            question = random.choice(cn_templates)

        if MEMORY_AVAILABLE and random.random() < 0.3:
            try:
                topic_query = f"{topic} 编程 算法 代码"
                mem_context = recall("cezanne", topic_query, top_k=2, max_chars=400)
                if mem_context and len(mem_context) > 50:
                    question += f"\n知识库参考（作为灵感，不要照搬）：\n{mem_context}"
            except Exception:
                pass

        return question, ""

    if topic == "Memory":
        memory_templates = [
            "Explain how FAISS vector indexing works for long-term memory retrieval.",
            "What is the difference between semantic search and keyword search in RAG systems?",
            "How does embedding-based retrieval handle out-of-domain queries?",
            "Explain the trade-offs between chunk size and retrieval precision in knowledge stores.",
            "What is memory consolidation in AI systems? How does it differ from human memory consolidation?",
            "How would you implement a forgetting mechanism in a long-term memory store to prevent interference?",
            "Explain the difference between episodic and semantic memory in the context of AI agents.",
            "How does cosine similarity compare to inner product for vector retrieval? When would you use each?",
            "What is the role of rehearsal in maintaining important memories in an AI system?",
            "Design a chunking strategy for code files that maximizes retrieval accuracy.",
        ]
        question = random.choice(memory_templates)
        if MEMORY_AVAILABLE:
            try:
                mem_context = recall("cezanne", "FAISS memory retrieval embedding vector", top_k=3, max_chars=500)
                if mem_context:
                    question += f"\nContext from knowledge store:\n{mem_context}"
            except Exception:
                pass
        return question, ""

    if topic == "WorldModel":
        worldmodel_templates = [
            "On an RTX 3060 12GB, should a world model encoder use CNN or Transformer? Explain the trade-offs.",
            "Explain the encoder-dynamics-decoder architecture in world models. Why is latent space compression necessary?",
            "What is 3D Gaussian Splatting? Compare it with NeRF on RTX 3060 in terms of memory and speed.",
            "How does gradient checkpointing reduce VRAM from O(L) to O(sqrt(L))? Give a concrete example.",
            "Why should world model training be split into pretraining and finetuning phases?",
            "How does mixed precision training (FP16/BF16) reduce VRAM by 40-50% on RTX 3060?",
            "Explain DreamerV3's training pipeline: encoder + dynamics + value/policy heads.",
            "How does 4-bit quantization (QLoRA/AWQ) enable a 7B model to run on 12GB VRAM?",
            "What does torch.compile's max-autotune mode do to optimize world model training?",
            "Compare RNN, Transformer, and Diffusion architectures for world models. When to use each?",
            "In a world model, what is the role of the dynamics model? How does it predict future latent states?",
            "Explain KL divergence loss in VAE-based world models. Why is it needed for latent space regularization?",
            "How would you design a world model that runs in real-time on RTX 3060? What compromises are necessary?",
            "What is action-conditioned video prediction? How does it differ from unconditional video generation?",
            "Explain cosine annealing learning rate schedule. Why is it preferred for world model training?",
        ]
        question = random.choice(worldmodel_templates)
        if MEMORY_AVAILABLE:
            try:
                mem_context = recall("cezanne", "world model latent space VRAM optimization RTX 3060", top_k=3, max_chars=500)
                if mem_context:
                    question += f"\nContext from knowledge store:\n{mem_context}"
            except Exception:
                pass
        return question, ""

    if topic == "Debug":
        code = random.choice(topic_data.get("code_snippets", ["print('hello')"]))
        debug_templates = [
            f"Find the bug in this code: {code}",
            f"What is wrong with this implementation? {code}",
            f"Identify the error and fix it: {code}",
            f"This code has a subtle bug. Find it and explain: {code}",
            f"Why does this code fail on edge cases? {code}",
            f"What happens when this code runs? Identify any issues: {code}",
        ]
        question = random.choice(debug_templates)
    elif topic == "Logic":
        claim_templates = ["Prove or disprove: {claim}", "Is this statement true or false? {claim}"]
        argument_templates = ["Is this argument valid? {argument}", "Analyze this argument: {argument}"]
        concept_templates = ["What is {concept}? Give an example.", "Explain {concept} in formal logic."]
        choice = random.choice(["claim", "argument", "concept"])
        if choice == "claim":
            item = random.choice(topic_data.get("claims", ["P implies Q"]))
            template = random.choice(claim_templates)
            question = template.format(claim=item)
        elif choice == "argument":
            item = random.choice(topic_data.get("arguments", ["If P then Q. P. Therefore Q."]))
            template = random.choice(argument_templates)
            question = template.format(argument=item)
        else:
            item = random.choice(topic_data.get("concepts", ["contrapositive"]))
            template = random.choice(concept_templates)
            question = template.format(concept=item)
    elif topic == "Algorithm":
        algos = topic_data.get("algos", ["quicksort"])
        algo = random.choice(algos)
        templates = [
            f"Implement {algo} in Python.",
            f"Explain the time complexity of {algo}.",
            f"Compare {algo} vs mergesort. When would you use each?",
        ]
        question = random.choice(templates)
    elif topic == "Systems":
        concepts = topic_data.get("concepts", ["virtual memory"])
        concept = random.choice(concepts)
        templates = [
            f"Explain {concept} and how it works.",
            f"What is the difference between a process and a thread in the context of {concept}?",
            f"How does {concept} work internally?",
        ]
        question = random.choice(templates)
    elif topic == "Complexity":
        templates = [
            "What is the time complexity of merge sort? Explain why.",
            "Explain amortized analysis with an example.",
            "Compare O(n^2) vs O(n log n) time complexity.",
            "What does O(log n) mean? Give an example.",
            "What is the space complexity of quicksort?",
        ]
        question = random.choice(templates)
    elif topic == "Network":
        templates = [
            "Explain the difference between TCP and UDP.",
            "What is SQL injection and how to prevent it?",
            "Explain TCP congestion control and slow start.",
            "What is a REST API? Explain the main principles.",
            "Explain DNS and how domain name resolution works.",
        ]
        question = random.choice(templates)
    else:
        question = f"Explain {topic} in computer science."

    if MEMORY_AVAILABLE and random.random() < 0.3:
        try:
            topic_query = f"{topic} programming algorithm code"
            mem_context = recall("cezanne", topic_query, top_k=2, max_chars=400)
            if mem_context and len(mem_context) > 50:
                question += f"\nReference from knowledge store (use as inspiration, do not copy):\n{mem_context}"
        except Exception:
            pass

    return question, ""


def self_judge(answer, topic, question="", iteration=1, reference=""):
    is_einstein = reference.startswith("EINSTEIN:") if reference else False

    if is_einstein:
        eq_id = reference.split(":", 1)[1] if ":" in reference else ""
        eq = next((q for q in EINSTEIN_QUESTIONS if q["id"] == eq_id), None)

        discovery_kw = eq.get("discovery_keywords", []) if eq else []
        disc_matched = [k for k in discovery_kw if k.lower() in answer.lower()]
        discovery_score = len(set(disc_matched))

        en_reasoning = [m for m in EINSTEIN_REASONING_MARKERS if m in answer.lower()]
        cn_reasoning = [m for m in EINSTEIN_CN_REASONING_MARKERS if m in answer]
        reasoning_score = min(len(set(en_reasoning + cn_reasoning)), 5)

        ans_len = len(answer.strip())
        length_score = 0
        if ans_len >= 100:
            length_score += 1
        if ans_len >= 300:
            length_score += 1
        if ans_len >= 600:
            length_score += 1

        has_step_by_step = any(m in answer.lower() for m in ["step 1", "step 2", "first", "then", "next", "finally",
                                                               "首先", "然后", "接着", "最后", "第一步", "第二步"])
        has_proof_attempt = any(m in answer.lower() for m in ["proof", "prove", "contradiction", "assume", "suppose",
                                                               "证明", "反证", "假设", "矛盾", "不可能"])
        has_why = any(m in answer.lower() for m in ["why", "because", "reason", "therefore", "thus",
                                                      "为什么", "因为", "原因", "因此", "所以"])
        depth_score = (1 if has_step_by_step else 0) + (2 if has_proof_attempt else 0) + (1 if has_why else 0)

        total_score = discovery_score + reasoning_score + length_score + depth_score
        threshold = PASS_THRESHOLD_SCHEDULE.get(iteration, 4) + 2
        passed = total_score >= threshold
        return passed, disc_matched + en_reasoning + cn_reasoning, total_score

    topic_data = CS_TOPICS.get(topic, {})
    keywords = topic_data.get("verify_keywords", [])
    matched = [k for k in keywords if k.lower() in answer.lower()]

    cn_keywords = CHINESE_VERIFY_KEYWORDS.get(topic, [])
    cn_matched = [k for k in cn_keywords if k in answer]
    matched.extend(cn_matched)

    if reference and not matched:
        ref_words = [w for w in reference.lower().split() if len(w) > 4]
        ref_matched = [w for w in ref_words if w in answer.lower()]
        if ref_matched:
            matched.extend(ref_matched[:3])

    keyword_score = len(set(matched))

    length_score = 0
    ans_len = len(answer.strip())
    if ans_len >= 50:
        length_score += 1
    if ans_len >= 150:
        length_score += 1
    if ans_len >= 300:
        length_score += 1

    has_code = any(marker in answer for marker in ["def ", "class ", "import ", "return ", "for ", "while ", "if "])
    has_explanation = any(marker in answer.lower() for marker in ["because", "therefore", "since", "this means", "the reason", "due to", "however", "in contrast", "for example", "specifically", "因为", "所以", "原因是", "由于", "然而", "例如", "具体来说", "也就是说", "换句话说", "因此", "注意"])

    structure_score = (1 if has_code else 0) + (1 if has_explanation else 0)

    total_score = keyword_score + length_score + structure_score
    threshold = PASS_THRESHOLD_SCHEDULE.get(iteration, 4)
    passed = total_score >= threshold
    return passed, matched, total_score


def regression_test(model, tokenizer):
    passed = 0
    was_training = model.training
    model.eval()
    for item in REGRESSION_QUESTIONS:
        ans = generate(model, tokenizer, item["q"], max_tokens=200)
        matched = [k for k in item["kw"] if k.lower() in ans.lower()]
        if len(matched) >= 2:
            passed += 1
    if was_training:
        model.train()
    return passed, len(REGRESSION_QUESTIONS)


SELFPLAY_TOPICS = [t for t in CS_TOPICS.keys() if t not in ("External",)]

def phase1_self_play(model, tokenizer, iteration, lora_path):
    threshold = PASS_THRESHOLD_SCHEDULE.get(iteration, 4)
    print(f"\n  [Iter {iteration}] Phase 1: Self-Play ({QUESTIONS_PER_ITER} questions, threshold={threshold})", flush=True)

    topics = SELFPLAY_TOPICS
    topic_weights = {t: 1.0 for t in topics}
    topic_stats_live = {t: {"win": 0, "lose": 0, "total": 0} for t in topics}
    trajectory = []
    fail_count = 0
    WEAKNESS_CHECK_INTERVAL = 100

    def pick_topic():
        total_w = sum(topic_weights[t] for t in topics)
        r = random.random() * total_w
        cumulative = 0
        for t in topics:
            cumulative += topic_weights[t]
            if r <= cumulative:
                return t
        return topics[-1]

    for i in range(QUESTIONS_PER_ITER):
        topic = pick_topic()
        question, reference = generate_question(model, tokenizer, topic)
        answer = generate(model, tokenizer, question)

        is_external = reference != ""
        if is_external:
            passed, matched, score = self_judge(answer, "External", question, iteration=iteration, reference=reference)
        else:
            passed, matched, score = self_judge(answer, topic, question, iteration=iteration, reference=reference)

        display_topic = "External" if is_external else topic
        trajectory.append({
            "topic": display_topic,
            "question": question,
            "answer": answer,
            "passed": passed,
            "matched": matched,
            "score": score,
            "iteration": iteration,
            "reference_source": "external" if is_external else "template",
        })

        if not is_external:
            topic_stats_live[topic]["total"] += 1
            if passed:
                topic_stats_live[topic]["win"] += 1
            else:
                topic_stats_live[topic]["lose"] += 1

        status = "WIN" if passed else "LOSE"
        print(f"    [{status}] [{display_topic}] score={score} matched={matched}", flush=True)

        if not passed and MEMORY_AVAILABLE:
            try:
                fail_content = f"Q: {question}\nA(attempt): {answer}\nTopic: {topic}\nWeak areas: {', '.join(matched) if matched else 'none matched'}"
                write_knowledge("cezanne", "selfplay_fail", fail_content, {
                    "iteration": iteration,
                    "topic": topic,
                    "score": score,
                })
                fail_count += 1
            except Exception as e:
                print(f"    [WARN] Failed to write fail to memory: {e}", flush=True)

        if (i + 1) % WEAKNESS_CHECK_INTERVAL == 0:
            for t in topics:
                s = topic_stats_live[t]
                if s["total"] > 0:
                    pass_rate = s["win"] / s["total"]
                    if pass_rate < 0.3:
                        topic_weights[t] = 4.0
                    elif pass_rate < 0.5:
                        topic_weights[t] = 3.0
                    elif pass_rate < 0.7:
                        topic_weights[t] = 2.0
                    elif pass_rate > 0.9:
                        topic_weights[t] = 0.5
                    else:
                        topic_weights[t] = 1.0
            weight_str = " ".join(f"{t}={topic_weights[t]:.1f}" for t in topics)
            stats_str = " ".join(f"{t}={topic_stats_live[t]['win']}/{topic_stats_live[t]['total']}" for t in topics)
            print(f"    [WEAKNESS CHECK @{i+1}] weights: {weight_str} | stats: {stats_str}", flush=True)
            gc.collect(); torch.cuda.empty_cache()

    if fail_count > 0 and MEMORY_AVAILABLE:
        try:
            close_knowledge_handles()
            print(f"  Written {fail_count} fail entries to knowledge store (index rebuild deferred)", flush=True)
        except Exception as e:
            print(f"    [WARN] Failed to close knowledge handles: {e}", flush=True)

    t_path = os.path.join(LOG_DIR, f"selfplay_iter{iteration}_trajectory.json")
    with open(t_path, "w", encoding="utf-8") as f:
        json.dump(trajectory, f, ensure_ascii=False, indent=2)

    wins = sum(1 for t in trajectory if t["passed"])
    total = len(trajectory)
    print(f"  Self-Play Result: {wins}/{total} wins ({wins/total:.0%})", flush=True)
    return trajectory


def phase2_reward_labeling(trajectory):
    print(f"\n  Phase 2: Monte-Carlo Reward Labeling (Dynamic Negative Ratio)", flush=True)

    positive = []
    negative = []

    topic_stats = {}
    for t in trajectory:
        topic = t["topic"]
        if topic not in topic_stats:
            topic_stats[topic] = {"win": 0, "lose": 0}
        if t["passed"]:
            topic_stats[topic]["win"] += 1
        else:
            topic_stats[topic]["lose"] += 1

        item = {
            "instruction": t["question"],
            "input": "",
            "output": t["answer"],
            "source": f"selfplay_{t['topic']}",
            "soul": "cezanne",
            "reward": 1 if t["passed"] else 0,
            "_topic": t["topic"],
        }
        if t["passed"]:
            positive.append(item)
        else:
            negative.append(item)

    print(f"  Positive (WIN): {len(positive)}", flush=True)
    print(f"  Negative (LOSE): {len(negative)}", flush=True)

    for topic, stats in topic_stats.items():
        total = stats["win"] + stats["lose"]
        rate = stats["win"] / total if total > 0 else 0
        print(f"    [{topic}] pass_rate={rate:.0%} ({stats['win']}/{total})", flush=True)

    selected_neg = []
    for topic, stats in topic_stats.items():
        topic_negs = [n for n in negative if n["_topic"] == topic]
        if not topic_negs:
            continue
        total = stats["win"] + stats["lose"]
        pass_rate = stats["win"] / total if total > 0 else 0.5
        if pass_rate < 0.3:
            ratio = 0.7
        elif pass_rate < 0.5:
            ratio = 0.5
        elif pass_rate < 0.7:
            ratio = 0.3
        else:
            ratio = 0.1
        n_pick = max(1, int(len(topic_negs) * ratio))
        picked = random.sample(topic_negs, min(n_pick, len(topic_negs)))
        selected_neg.extend(picked)
        print(f"    [{topic}] neg_ratio={ratio:.0%} picked {len(picked)}/{len(topic_negs)}", flush=True)

    training_data = positive + selected_neg
    random.shuffle(training_data)
    print(f"  Training mix: {len(positive)} positive + {len(selected_neg)} negative = {len(training_data)} total", flush=True)

    return training_data, positive, negative


def phase3_train(training_data, iteration, prev_lora_path, prev_loss=None):
    print(f"\n  Phase 3: Training on {len(training_data)} samples", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    stage_name = f"selfplay_iter{iteration}"
    out_dir = os.path.join(LORA_DIR, stage_name)

    if iteration <= 5:
        max_loss = MAX_LOSS_INITIAL
    elif iteration <= 10:
        max_loss = 2.0
    else:
        max_loss = 1.5
    print(f"  Dynamic Loss Gate: {max_loss:.1f} (iter {iteration})", flush=True)

    current_lr = LR / (1.3 ** (iteration - 1))
    print(f"  Learning Rate: {current_lr:.2e} (decay by 1.3x per iter)", flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model.enable_input_require_grads()
    model.config.use_cache = False
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("  Pre-training regression test...", flush=True)
    pre_pass, pre_total = regression_test(model, tokenizer)
    print(f"  Pre-regression: {pre_pass}/{pre_total}", flush=True)

    def format_sample(s):
        return {"text": f"<s>[INST] {s['instruction']} [/INST] {s['output']}</s>"}

    ds = Dataset.from_list([format_sample(s) for s in training_data])

    t0 = time.time()
    sft_config = SFTConfig(
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        warmup_ratio=0.05, num_train_epochs=EPOCHS_PER_ITER,
        learning_rate=current_lr, fp16=True, logging_steps=50, optim="adamw_torch",
        weight_decay=0.01, lr_scheduler_type="cosine", seed=3407,
        output_dir=out_dir, save_strategy="steps", save_steps=500, save_total_limit=2,
        max_grad_norm=0.5, report_to="none", disable_tqdm=True,
        dataloader_num_workers=0, gradient_checkpointing=True,
        max_length=512, dataset_text_field="text", dataset_num_proc=1, packing=False,
    )
    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=ds, args=sft_config)

    try:
        trainer.train()
    except Exception as e:
        print(f"  [ERROR] Training failed: {e}", flush=True)
        del model, tokenizer, trainer
        gc.collect(); torch.cuda.empty_cache()
        return None, None, None

    elapsed = (time.time() - t0) / 60
    peak_vram = torch.cuda.max_memory_reserved() / 1024**3

    final_loss = 999.0
    if trainer.state and trainer.state.log_history:
        print(f"  [DEBUG] log_history has {len(trainer.state.log_history)} entries", flush=True)
        for entry in reversed(trainer.state.log_history):
            for key in ["train_loss", "loss", "train_run_loss"]:
                if key in entry and isinstance(entry[key], (int, float)):
                    final_loss = entry[key]
                    print(f"  [DEBUG] Found loss key='{key}' value={final_loss:.4f}", flush=True)
                    break
            if final_loss != 999.0:
                break
        if final_loss == 999.0:
            print(f"  [WARN] No loss found in log_history, dumping last 3 entries:", flush=True)
            for entry in trainer.state.log_history[-3:]:
                print(f"    {entry}", flush=True)
    print(f"  Final Loss: {final_loss:.4f} | Time: {elapsed:.1f}min | VRAM: {peak_vram:.1f}GB", flush=True)

    if final_loss > max_loss:
        print(f"  [LOSS GATE FAILED] {final_loss:.4f} > {max_loss:.1f}", flush=True)
        del model, tokenizer, trainer
        gc.collect(); torch.cuda.empty_cache()
        return None, None, None

    print("  Post-training regression test...", flush=True)
    post_pass, post_total = regression_test(model, tokenizer)
    print(f"  Post-regression: {post_pass}/{post_total}", flush=True)

    if pre_pass - post_pass > REGRESSION_MAX_DROP:
        print(f"  [REGRESSION GATE FAILED] {pre_pass} -> {post_pass} (dropped {pre_pass - post_pass})", flush=True)
        del model, tokenizer, trainer
        gc.collect(); torch.cuda.empty_cache()
        return None, None, None

    final_path = os.path.join(out_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"  Saved LoRA: {final_path}", flush=True)

    result = {
        "iteration": iteration, "stage": stage_name,
        "final_loss": final_loss, "elapsed_min": elapsed, "peak_vram_gb": peak_vram,
        "training_samples": len(training_data), "epochs": EPOCHS_PER_ITER, "lr": current_lr,
        "regression_pre": pre_pass, "regression_post": post_pass,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    del model, tokenizer, trainer
    gc.collect(); torch.cuda.empty_cache()
    return final_path, result, final_loss


def phase4_evaluate(lora_path, iteration):
    print(f"\n  Phase 4: Diagnostic Evaluation", flush=True)

    model, tokenizer = load_model(lora_path)

    exam_questions = [
        {"cat": "Debug", "q": "Find the bug: def binary_search(arr, target): left,right=0,len(arr); while left<right: mid=(left+right)//2; if arr[mid]==target: return mid; elif arr[mid]<target: left=mid; else: right=mid; return -1", "kw": ["off-by-one", "mid + 1", "infinite", "boundary"]},
        {"cat": "Debug", "q": "Find the bug: class Counter: count=0; def increment(self): self.count+=1. Two threads calling increment() simultaneously.", "kw": ["race", "thread", "lock", "atomic"]},
        {"cat": "Debug", "q": "Find the bug: def fibonacci(n): return fibonacci(n-1) + fibonacci(n-2)", "kw": ["base case", "infinite", "recursion", "n<=1"]},
        {"cat": "Logic", "q": "What is a contrapositive? Give an example.", "kw": ["contrapositive", "not", "implies", "equivalent"]},
        {"cat": "Logic", "q": "Explain proof by contradiction with an example.", "kw": ["contradiction", "assume", "false", "absurd"]},
        {"cat": "Logic", "q": "Prove using formal logic: if P implies Q and Q implies R, then P implies R.", "kw": ["syllogism", "implies", "transitivity", "modus"]},
        {"cat": "Systems", "q": "Explain CPU cache and why it matters for performance.", "kw": ["cache", "l1", "l2", "locality"]},
        {"cat": "Systems", "q": "Explain virtual memory and how page tables work.", "kw": ["virtual", "page", "tlb", "translation"]},
        {"cat": "Algorithm", "q": "Write Dijkstra's shortest path algorithm in Python.", "kw": ["dijkstra", "priority", "queue", "distance"]},
        {"cat": "Complexity", "q": "What is the time complexity of merge sort?", "kw": ["n log", "merge", "sort", "divide"]},
        {"cat": "Network", "q": "Explain the difference between TCP and UDP.", "kw": ["tcp", "udp", "reliable", "connection"]},
        {"cat": "Math", "q": "Explain the Master Theorem for recurrence relations.", "kw": ["master", "theorem", "recurrence", "t(n)"]},
        {"cat": "Debug_CN", "q": "找出这段代码的bug：def binary_search(arr, target): left,right=0,len(arr); while left<right: mid=(left+right)//2; if arr[mid]==target: return mid; elif arr[mid]<target: left=mid; else: right=mid; return -1", "kw": ["差一", "mid+1", "无限", "边界", "off-by-one", "infinite"]},
        {"cat": "Logic_CN", "q": "用形式逻辑证明：如果P蕴含Q，Q蕴含R，则P蕴含R。", "kw": ["三段论", "蕴含", "传递", "syllogism", "implies"]},
        {"cat": "Algorithm_CN", "q": "用Python实现Dijkstra最短路径算法。", "kw": ["dijkstra", "优先", "队列", "距离", "priority", "queue"]},
        {"cat": "Systems_CN", "q": "解释虚拟内存和页表的工作原理。", "kw": ["虚拟", "页表", "快表", "virtual", "page"]},
        {"cat": "Network_CN", "q": "解释TCP和UDP的区别。", "kw": ["可靠", "连接", "TCP", "UDP", "reliable"]},
        {"cat": "Complexity_CN", "q": "归并排序的时间复杂度是多少？为什么？", "kw": ["n log", "分治", "合并", "divide", "merge"]},
    ]

    cat_scores = {}
    for q in exam_questions:
        cat = q["cat"]
        ans = generate(model, tokenizer, q["q"])
        matched = [k for k in q["kw"] if k.lower() in ans.lower()]
        passed = len(matched) >= 2
        if cat not in cat_scores:
            cat_scores[cat] = {"pass": 0, "total": 0}
        cat_scores[cat]["total"] += 1
        if passed:
            cat_scores[cat]["pass"] += 1

    total_pass = sum(s["pass"] for s in cat_scores.values())
    total_q = sum(s["total"] for s in cat_scores.values())
    rate = total_pass / total_q if total_q > 0 else 0

    print(f"  Diagnostic: {total_pass}/{total_q} ({rate:.0%})", flush=True)
    for cat in sorted(cat_scores):
        s = cat_scores[cat]
        print(f"    {cat}: {s['pass']}/{s['total']}", flush=True)

    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()

    return total_pass, total_q, rate, cat_scores


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None

def save_checkpoint(data):
    try:
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def clear_checkpoint():
    try:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
    except Exception:
        pass

def main():
    print(f"\n{'#'*60}", flush=True)
    print(f"  HERMES ALPHAZERO SELF-PLAY ENGINE", flush=True)
    print(f"  Base: {BASE_MODEL}", flush=True)
    print(f"  Max iterations: {MAX_ITERATIONS}", flush=True)
    print(f"  Questions/iter: {QUESTIONS_PER_ITER}", flush=True)
    print(f"  Epochs/iter: {EPOCHS_PER_ITER}", flush=True)
    print(f"  External dataset ratio: {EXTERNAL_DATASET_RATIO:.0%}", flush=True)
    print(f"{'#'*60}", flush=True)

    load_external_datasets()

    ckpt = load_checkpoint()
    if ckpt:
        start_iter = ckpt.get("next_iteration", 1)
        current_lora = ckpt.get("current_lora", os.path.join(LORA_DIR, "s3_cs_depth_8k", "final"))
        best_lora = ckpt.get("best_lora", current_lora)
        best_score = ckpt.get("best_score", 0)
        prev_loss = ckpt.get("prev_loss", None)
        all_results = ckpt.get("all_results", [])
        print(f"  [RESUME] From checkpoint: iter {start_iter}, best_score {best_score:.0%}", flush=True)
        print(f"  [RESUME] current_lora: {current_lora}", flush=True)
    else:
        start_iter = 1
        current_lora = os.path.join(LORA_DIR, "s3_cs_depth_8k", "final")
        best_lora = current_lora
        best_score = 0
        prev_loss = None
        all_results = []

    for iteration in range(start_iter, MAX_ITERATIONS + 1):
        print(f"\n{'='*60}", flush=True)
        print(f"  ITERATION {iteration}/{MAX_ITERATIONS}", flush=True)
        print(f"  Current LoRA: {current_lora}", flush=True)
        print(f"{'='*60}", flush=True)

        save_checkpoint({
            "next_iteration": iteration,
            "current_lora": current_lora,
            "best_lora": best_lora,
            "best_score": best_score,
            "prev_loss": prev_loss,
            "all_results": all_results,
            "phase": "self_play_start",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        model, tokenizer = load_model(current_lora)
        trajectory = phase1_self_play(model, tokenizer, iteration, current_lora)
        del model, tokenizer
        gc.collect(); torch.cuda.empty_cache()

        save_checkpoint({
            "next_iteration": iteration,
            "current_lora": current_lora,
            "best_lora": best_lora,
            "best_score": best_score,
            "prev_loss": prev_loss,
            "all_results": all_results,
            "phase": "training_start",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        time.sleep(30)

        training_data, positive, negative = phase2_reward_labeling(trajectory)

        if len(training_data) < 5:
            print(f"  [SKIP] Too few positive samples ({len(training_data)}), skipping training", flush=True)
            all_results.append({"iteration": iteration, "status": "skipped", "positive": len(positive), "negative": len(negative)})
            continue

        new_lora_path, train_result, final_loss = phase3_train(training_data, iteration, current_lora, prev_loss)

        if new_lora_path is None:
            print(f"  [ROLLBACK] Training failed, keeping current model", flush=True)
            all_results.append({"iteration": iteration, "status": "failed", "positive": len(positive), "negative": len(negative)})
            continue

        prev_loss = final_loss

        total_pass, total_q, rate, cat_scores = phase4_evaluate(new_lora_path, iteration)

        iter_result = {
            "iteration": iteration, "status": "completed",
            "positive": len(positive), "negative": len(negative),
            "total_pass": total_pass, "total_q": total_q, "rate": rate,
            "cat_scores": {c: f"{s['pass']}/{s['total']}" for c, s in cat_scores.items()},
            "train_result": train_result,
            "lora_path": new_lora_path,
        }
        all_results.append(iter_result)

        if rate > best_score:
            best_score = rate
            best_lora = new_lora_path
            current_lora = new_lora_path
            print(f"  [NEW BEST] Score {rate:.0%}, LoRA updated to {new_lora_path}", flush=True)
        else:
            print(f"  [NO IMPROVEMENT] Score {rate:.0%} <= best {best_score:.0%}, keeping {best_lora}", flush=True)

        save_checkpoint({
            "next_iteration": iteration + 1,
            "current_lora": current_lora,
            "best_lora": best_lora,
            "best_score": best_score,
            "prev_loss": prev_loss,
            "all_results": all_results,
            "phase": "iteration_complete",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        r_path = os.path.join(LOG_DIR, f"selfplay_iter{iteration}_result.json")
        with open(r_path, "w", encoding="utf-8") as f:
            json.dump(iter_result, f, ensure_ascii=False, indent=2)

    print(f"\n{'#'*60}", flush=True)
    print(f"  SELF-PLAY COMPLETE", flush=True)
    print(f"  Best LoRA: {best_lora}", flush=True)
    print(f"  Best Score: {best_score:.0%}", flush=True)
    print(f"  Iterations: {len(all_results)}", flush=True)
    for r in all_results:
        status = r.get("status", "?")
        it = r.get("iteration", "?")
        if status == "completed":
            print(f"    Iter {it}: {r['total_pass']}/{r['total_q']} ({r['rate']:.0%}) +{r['positive']}/-{r['negative']}", flush=True)
        else:
            print(f"    Iter {it}: {status}", flush=True)
    print(f"{'#'*60}", flush=True)

    clear_checkpoint()

    summary_path = os.path.join(LOG_DIR, "selfplay_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"best_lora": best_lora, "best_score": best_score,
                    "iterations": all_results,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
                   f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n[FATAL ERROR] {e}", flush=True)
        traceback.print_exc()
        with open(os.path.join(os.path.dirname(__file__), "crash_log.txt"), "w") as f:
            traceback.print_exc(file=f)
        raise
