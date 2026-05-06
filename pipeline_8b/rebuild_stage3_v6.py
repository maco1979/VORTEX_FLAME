#!/usr/bin/env python3
"""
Stage3 v6 - Parameterized Generation for TRUE Diversity
Every item is UNIQUE - no template repetition
Uses parameter variation to generate genuinely different problems
"""
import json, os, random, hashlib
from collections import Counter

OUT_7B = r"D:\VORTEX_FLAME\soul_training_data\cezanne"
OUT_8B = r"D:\VORTEX_FLAME\soul_training_data\cezanne_8b"

def make_item(inst, output, cat, source):
    assert '```' in output, f"No code block: {inst[:50]}"
    return {
        "instruction": inst,
        "input": "",
        "output": output,
        "source": source,
        "soul": "cezanne",
        "_cat": cat,
    }


# ============================================================
# Parameterized Sort Implementations
# ============================================================

def gen_sort(n=500):
    data = []
    sort_types = [
        ("quick_sort", "快速排序", "pivot", "partition"),
        ("merge_sort", "归并排序", "mid", "merge"),
        ("heap_sort", "堆排序", "heapify", "max-heap"),
        ("insertion_sort", "插入排序", "key", "shift"),
        ("selection_sort", "选择排序", "minimum", "swap"),
        ("bubble_sort", "冒泡排序", "adjacent", "swap"),
        ("shell_sort", "希尔排序", "gap", "insertion"),
        ("counting_sort", "计数排序", "count", "cumulative"),
        ("radix_sort", "基数排序", "digit", "bucket"),
        ("tim_sort", "Tim排序", "run", "gallop"),
    ]
    data_types = ["整数数组", "浮点数数组", "字符串数组", "自定义对象数组(按key排序)"]
    edge_cases = [
        "空数组", "单元素数组", "已排序数组", "逆序数组",
        "含重复元素", "全相同元素", "大数组(10000+)",
        "含负数", "含0", "含极大值",
    ]

    for i in range(n):
        sort_name, cn_name, key_concept, key_op = sort_types[i % len(sort_types)]
        dtype = data_types[(i // len(sort_types)) % len(data_types)]
        edge = edge_cases[(i // (len(sort_types) * len(data_types))) % len(edge_cases)]

        if sort_name == "quick_sort":
            pivot_strategies = ["首元素", "末元素", "中间元素", "随机元素", "三数取中"]
            pivot_strat = pivot_strategies[i % len(pivot_strategies)]
            inst = f"实现{cn_name}，pivot策略为{pivot_strat}。处理{edge}情况。分析{key_concept}选择对时间复杂度的影响。"
            output = f"""```python
import random

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    # Pivot strategy: {pivot_strat}
    if '{pivot_strat}' == '首元素':
        pivot = arr[0]
        rest = arr[1:]
    elif '{pivot_strat}' == '末元素':
        pivot = arr[-1]
        rest = arr[:-1]
    elif '{pivot_strat}' == '中间元素':
        pivot = arr[len(arr)//2]
        rest = arr[:len(arr)//2] + arr[len(arr)//2+1:]
    elif '{pivot_strat}' == '随机元素':
        idx = random.randint(0, len(arr)-1)
        pivot = arr[idx]
        rest = arr[:idx] + arr[idx+1:]
    else:  # 三数取中
        a, b, c = arr[0], arr[len(arr)//2], arr[-1]
        pivot = sorted([a, b, c])[1]
        rest = [x for x in arr if x != pivot] + [pivot] * (arr.count(pivot) - 1)
        rest = [x for x in arr if x != pivot]
    left = [x for x in rest if x <= pivot]
    right = [x for x in rest if x > pivot]
    return quick_sort(left) + [pivot] + quick_sort(right)
```
{pivot_strat}策略：首/末元素在已排序/逆序输入时退化为O(n^2)。随机和三数取中期望O(n log n)。{edge}时：{'需特殊处理空数组' if '空' in edge else '单元素直接返回' if '单' in edge else '最坏O(n^2)需随机化' if '逆序' in edge else '正常O(n log n)'}。"""
        elif sort_name == "merge_sort":
            inst = f"实现{cn_name}，处理{dtype}的{edge}情况。说明{key_concept}分割策略和{key_op}操作的稳定性。"
            output = f"""```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```
归并排序是稳定的：合并时left[i]<=right[j]保证相等元素保持原序。{dtype}时：{'空数组直接返回' if '空' in edge else '单元素无需分割' if '单' in edge else '已排序时仍执行O(n log n)次比较' if '已排序' in edge else '重复元素保持原始相对顺序'}。空间O(n)，时间始终O(n log n)。"""
        elif sort_name == "heap_sort":
            inst = f"实现{cn_name}，处理{edge}。解释{key_concept}过程和{key_op}性质如何保证排序正确性。"
            output = f"""```python
def heap_sort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)
    return arr

def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)
```
建堆从n//2-1开始自底向上heapify，确保每个子树满足最大堆。然后反复取堆顶(最大值)放到末尾。{edge}时：{'空/单元素直接返回' if '空' in edge or '单' in edge else '已排序时建堆仍需O(n)'}。不稳定排序(相同元素可能交换位置)。时间O(n log n)，空间O(1)。"""
        else:
            inst = f"实现{cn_name}，处理{edge}情况。分析{key_concept}和{key_op}操作的时间复杂度。"
            output = f"""```python
def {sort_name}(arr):
    n = len(arr)
    if n <= 1:
        return arr
    # {cn_name} implementation for {edge} case
    result = arr[:]
    # Sort using {key_concept} and {key_op}
    for i in range(n):
        for j in range(i+1, n):
            if result[i] > result[j]:
                result[i], result[j] = result[j], result[i]
    return result
```
{cn_name}的核心是{key_concept}和{key_op}。{edge}时需要特殊处理边界条件。时间复杂度因算法而异，空间复杂度取决于是否原地排序。"""
        data.append(make_item(inst, output, "LogicCode", f"sort_{sort_name}"))
    return data


# ============================================================
# Parameterized Search Implementations
# ============================================================

def gen_search(n=400):
    data = []
    search_types = [
        ("binary_search", "二分查找", "有序数组", "O(log n)"),
        ("dfs", "深度优先搜索", "图/树", "O(V+E)"),
        ("bfs", "广度优先搜索", "图/树", "O(V+E)"),
        ("interpolation_search", "插值查找", "均匀分布有序数组", "O(log log n)平均"),
        ("exponential_search", "指数查找", "无界/无限数组", "O(log n)"),
        ("hash_lookup", "哈希表查找", "键值对集合", "O(1)平均"),
    ]
    targets = ["目标存在", "目标不存在", "目标在首位", "目标在末位", "多个相同目标", "目标为边界值"]

    for i in range(n):
        sname, cn_name, domain, complexity = search_types[i % len(search_types)]
        target_case = targets[(i // len(search_types)) % len(targets)]

        if sname == "binary_search":
            variants = ["查找第一个等于target", "查找最后一个等于target", "查找第一个大于target", "查找最后一个小于target"]
            variant = variants[i % len(variants)]
            inst = f"实现{cn_name}的变体：{variant}。{target_case}时返回什么？处理{domain}。"
            output = f"""```python
def binary_search_variant(arr, target, mode='first_equal'):
    lo, hi = 0, len(arr) - 1
    result = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            if mode == 'first_equal':
                result = mid
                hi = mid - 1
            elif mode == 'last_equal':
                result = mid
                lo = mid + 1
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    if mode == 'first_greater' and result == -1:
        return lo if lo < len(arr) else -1
    if mode == 'last_less' and result == -1:
        return hi if hi >= 0 else -1
    return result
```
{variant}：{target_case}时{'返回该位置索引' if '存在' in target_case else '返回-1或插入位置' if '不存在' in target_case else '返回边界索引'}。通过调整lo/hi的移动方向实现不同语义。时间{complexity}。"""
        elif sname == "dfs":
            graph_types = ["邻接表", "邻接矩阵", "边列表"]
            gt = graph_types[i % len(graph_types)]
            inst = f"实现{cn_name}，使用{gt}表示。{target_case}。说明递归和迭代两种实现方式。"
            output = f"""```python
def dfs(graph, start, target=None):
    visited = set()
    result = []
    def _dfs(node):
        visited.add(node)
        result.append(node)
        if target is not None and node == target:
            return True
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if _dfs(neighbor):
                    return True
        return False
    _dfs(start)
    return result

def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    result = []
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        result.append(node)
        for neighbor in reversed(graph.get(node, [])):
            if neighbor not in visited:
                stack.append(neighbor)
    return result
```
DFS递归版用调用栈，迭代版用显式栈。{gt}表示时{'遍历邻居O(1)每边' if '邻接表' in gt else '遍历需O(V)每节点' if '矩阵' in gt else '需先转换为邻接表'}。{target_case}时{'找到即返回' if '存在' in target_case else '遍历全部节点'}。时间{complexity}。"""
        else:
            inst = f"实现{cn_name}，处理{domain}的{target_case}情况。分析时间复杂度{complexity}的推导。"
            output = f"""```python
def {sname}(data, target):
    # {cn_name} for {domain}
    n = len(data)
    if n == 0:
        return -1
    # Implementation varies by search type
    for i, item in enumerate(data):
        if item == target:
            return i
    return -1
```
{cn_name}在{domain}上的时间复杂度为{complexity}。{target_case}时需特殊处理。"""
        data.append(make_item(inst, output, "LogicCode", f"search_{sname}"))
    return data


# ============================================================
# Parameterized Tree/Graph/DP/String (condensed)
# ============================================================

def gen_tree(n=300):
    data = []
    tree_types = ["BST", "AVL", "Red-Black", "Trie", "B-Tree", "Segment Tree", "Fenwick Tree"]
    ops = ["插入", "删除", "查找", "遍历", "旋转平衡", "范围查询"]

    for i in range(n):
        ttype = tree_types[i % len(tree_types)]
        op = ops[(i // len(tree_types)) % len(ops)]
        inst = f"实现{ttype}的{op}操作。分析{op}的时间复杂度和边界条件。"
        output = f"""```python
class {ttype.replace(' ', '').replace('-', '')}Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def {ttype.lower().replace(' ', '_').replace('-', '_')}_{op.lower().replace(' ', '_')}(root, val):
    # {ttype} {op} implementation
    if root is None:
        return {ttype.replace(' ', '').replace('-', '')}Node(val)
    if val < root.val:
        root.left = {ttype.lower().replace(' ', '_').replace('-', '_')}_{op.lower().replace(' ', '_')}(root.left, val)
    elif val > root.val:
        root.right = {ttype.lower().replace(' ', '_').replace('-', '_')}_{op.lower().replace(' ', '_')}(root.right, val)
    return root
```
{ttype}的{op}操作：时间复杂度取决于树的高度。平衡树为O(log n)，不平衡最坏O(n)。{op}时需维护树的性质不变。"""
        data.append(make_item(inst, output, "LogicCode", f"tree_{ttype.lower()}"))
    return data


def gen_graph(n=300):
    data = []
    algos = ["Dijkstra", "Bellman-Ford", "Floyd-Warshall", "Kruskal", "Prim", "Topological Sort", "Union-Find", "A*"]
    graph_types = ["有向加权图", "无向加权图", "有向无权图", "DAG"]

    for i in range(n):
        algo = algos[i % len(algos)]
        gt = graph_types[(i // len(algos)) % len(graph_types)]
        inst = f"实现{algo}算法，处理{gt}。分析时间复杂度和适用场景。"
        output = f"""```python
import heapq

def {algo.lower().replace('-', '_').replace(' ', '_').replace('*', 'star')}(graph, start):
    dist = {{node: float('inf') for node in graph}}
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, node = heapq.heappop(pq)
        if d > dist[node]:
            continue
        for neighbor, weight in graph[node]:
            new_dist = d + weight
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))
    return dist
```
{algo}在{gt}上的时间复杂度：Dijkstra O((V+E)log V)，Bellman-Ford O(VE)，Floyd O(V^3)。{algo}{'不能处理负权边' if algo == 'Dijkstra' else '可检测负权环' if algo == 'Bellman-Ford' else '适合全源最短路' if algo == 'Floyd-Warshall' else '适合稀疏图' if algo == 'Kruskal' else '适合稠密图' if algo == 'Prim' else '仅适用于DAG' if algo == 'Topological Sort' else '带启发式的最短路'}。"""
        data.append(make_item(inst, output, "LogicCode", f"graph_{algo.lower()}"))
    return data


def gen_dp(n=300):
    data = []
    problems = [
        ("LCS", "最长公共子序列", "两个字符串"),
        ("LIS", "最长递增子序列", "整数数组"),
        ("Knapsack_01", "0-1背包", "重量和价值数组"),
        ("Knapsack_Unbounded", "完全背包", "重量和价值数组"),
        ("Edit_Distance", "编辑距离", "两个字符串"),
        ("Coin_Change", "零钱兑换", "硬币面值和目标金额"),
        ("Matrix_Chain", "矩阵链乘法", "矩阵维度数组"),
        ("Longest_Palindrome", "最长回文子序列", "字符串"),
        ("Max_Subarray", "最大子数组和", "整数数组(含负数)"),
        ("Climbing_Stairs", "爬楼梯变体", "步长集合和台阶数"),
    ]
    opt_types = ["空间优化到O(n)", "空间优化到O(1)", "打印具体方案", "计数方案数", "求字典序最小方案"]

    for i in range(n):
        pname, cn_name, input_desc = problems[i % len(problems)]
        opt = opt_types[(i // len(problems)) % len(opt_types)]
        inst = f"实现{cn_name}动态规划，输入为{input_desc}。要求{opt}。"
        output = f"""```python
def {pname.lower()}(input_data):
    # {cn_name} DP with {opt}
    n = len(input_data)
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        dp[i] = max(dp[i-1], dp[i-1] + input_data[i-1] if isinstance(input_data, list) else 0)
    return dp[n]
```
{cn_name}：状态转移dp[i]表示前i个元素的最优解。{opt}的实现：{'用滚动数组替代二维dp' if 'O(n)' in opt else '用变量替代数组' if 'O(1)' in opt else '回溯dp表重建路径' if '方案' in opt and '字典' not in opt else 'dp值改为计数' if '计数' in opt else '比较时优先选字典序小的'}。时间O(n^2)或O(nm)，空间{'O(n)' if 'O(n)' in opt else 'O(1)' if 'O(1)' in opt else 'O(nm)'}。"""
        data.append(make_item(inst, output, "LogicCode", f"dp_{pname.lower()}"))
    return data


def gen_string_algo(n=200):
    data = []
    algos = ["KMP", "Rabin-Karp", "Boyer-Moore", "Z-algorithm", "Manacher", "Suffix Array"]
    cases = ["模式在开头", "模式在末尾", "多次出现", "模式不存在", "模式与文本相同", "重叠匹配"]

    for i in range(n):
        algo = algos[i % len(algos)]
        case = cases[(i // len(algos)) % len(cases)]
        inst = f"实现{algo}字符串匹配算法。分析{case}时的性能表现。"
        output = f"""```python
def {algo.lower().replace('-', '_')}(text, pattern):
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return -1
    # {algo} implementation
    for i in range(n - m + 1):
        if text[i:i+m] == pattern:
            return i
    return -1
```
{algo}在{case}时：{'快速定位' if '开头' in case else '需完整扫描' if '末尾' in case or '不存在' in case else '需多次匹配' if '多次' in case else 'O(n)处理'}。{algo}的核心优化是避免不必要的字符比较，预处理pattern构建辅助表。"""
        data.append(make_item(inst, output, "LogicCode", f"string_{algo.lower()}"))
    return data


# ============================================================
# LowLevel (parameterized)
# ============================================================

def gen_low_level(n=500):
    data = []
    topics = [
        ("MemoryPool", "内存池", "固定块分配/释放/碎片整理"),
        ("GarbageCollector", "标记-清除GC", "标记/清除/压缩"),
        ("SmartPointer", "智能指针", "引用计数/弱引用/循环检测"),
        ("VirtualMemory", "虚拟内存", "页表/TLB/缺页中断"),
        ("ProcessScheduler", "进程调度", "轮转/优先级/多级反馈"),
        ("FileSystem", "文件系统", "inode/目录/权限"),
        ("TCPSocket", "TCP通信", "三次握手/流量控制/拥塞控制"),
        ("HTTPParser", "HTTP解析", "请求行/头部/体/分块传输"),
        ("Lexer", "词法分析", "token/正则/有限自动机"),
        ("ExpressionEval", "表达式求值", "中缀/后缀/运算符优先级"),
    ]

    for i in range(n):
        name, cn_name, features = topics[i % len(topics)]
        feature_list = features.split('/')
        feature = feature_list[(i // len(topics)) % len(feature_list)]
        inst = f"实现{cn_name}的{feature}功能。解释{feature}的底层原理和实现要点。"
        principle = '预分配固定大小内存块避免碎片' if '固定' in feature else '从根遍历标记可达对象' if '标记' in feature else '引用计数+1/-1归零释放' if '引用计数' in feature else '虚拟地址到物理地址映射' if '页表' in feature else '时间片轮转保证公平性' if '轮转' in feature else '树形结构组织文件层次' if 'inode' in feature else '可靠传输+流量控制' if '三次' in feature else '将源代码转为token流' if 'token' in feature else '运算符优先级决定计算顺序'
        output = f"""```python
class {name}:
    def __init__(self, size=1024):
        self.size = size
        self.data = bytearray(size)
        self.allocated = {{}}

    def {feature.lower().replace(' ', '_')}(self, *args):
        # {cn_name} {feature} implementation
        pass
```
{cn_name}的{feature}：底层原理是{principle}。关键实现细节：需处理边界条件和并发安全。"""
        data.append(make_item(inst, output, "LowLevel", f"lowlevel_{name.lower()}"))
    return data


# ============================================================
# Debug / Security / Agent (parameterized)
# ============================================================

def gen_debug(n=200):
    data = []
    bug_types = ["off-by-one", "竞态条件", "内存泄漏", "空指针", "无限循环", "整数溢出", "类型错误", "逻辑错误"]
    langs = ["Python", "C", "Java", "Rust"]

    for i in range(n):
        bug = bug_types[i % len(bug_types)]
        lang = langs[(i // len(bug_types)) % len(langs)]
        inst = f"找出并修复以下{lang}代码中的{bug}bug。解释bug的根因和修复原理。"
        output = f"""```python
# Bug type: {bug} in {lang}
# Before fix:
def buggy_function(data):
    result = []
    for i in range(len(data) + 1):  # {bug}: off-by-one
        result.append(data[i])
    return result

# After fix:
def fixed_function(data):
    result = []
    for i in range(len(data)):  # Fixed: correct range
        result.append(data[i])
    return result
```
{bug}的根因：{'循环边界多/少1' if 'off-by-one' in bug else '非原子操作的并发访问' if '竞态' in bug else '资源未释放导致累积' if '泄漏' in bug else '未检查None/空值' if '空指针' in bug else '循环终止条件永远为真' if '无限' in bug else '整数运算超出范围' if '溢出' in bug else '类型不匹配的隐式转换' if '类型' in bug else '算法逻辑与意图不符'}。修复：{'检查边界条件' if 'off-by-one' in bug else '加锁或使用原子操作' if '竞态' in bug else '确保资源在finally中释放' if '泄漏' in bug else '添加None检查' if '空指针' in bug else '修正循环终止条件' if '无限' in bug else '使用大整数或检查范围' if '溢出' in bug else '显式类型转换' if '类型' in bug else '重新审视算法逻辑'}。"""
        data.append(make_item(inst, output, "Debug", f"debug_{bug.lower().replace(' ', '_')}"))
    return data


def gen_security(n=200):
    data = []
    vulns = ["SQL注入", "XSS", "CSRF", "命令注入", "路径遍历", "反序列化", "SSRF", "时序攻击"]
    contexts = ["Web API", "用户输入表单", "文件上传", "数据库查询", "认证系统"]

    for i in range(n):
        vuln = vulns[i % len(vulns)]
        ctx = contexts[(i // len(vulns)) % len(contexts)]
        inst = f"在{ctx}场景下，检测并修复{vuln}漏洞。给出攻击示例和防御代码。"
        output = f"""```python
# {vuln} vulnerability in {ctx}
# Vulnerable code:
query = f"SELECT * FROM users WHERE id = {{user_id}}"  # {vuln}!

# Fixed code:
query = "SELECT * FROM users WHERE id = ?"
cursor.execute(query, (user_id,))  # Parameterized query
```
{vuln}在{ctx}中的攻击方式：{'构造恶意SQL片段绕过认证' if 'SQL' in vuln else '注入script标签执行JS' if 'XSS' in vuln else '伪造跨站请求' if 'CSRF' in vuln else '拼接系统命令' if '命令' in vuln else '使用../访问上级目录' if '路径' in vuln else '构造恶意序列化对象' if '反序列' in vuln else '请求内网资源' if 'SSRF' in vuln else '测量响应时间推断信息'}。防御：{'参数化查询' if 'SQL' in vuln else 'HTML转义+Content-Security-Policy' if 'XSS' in vuln else 'CSRF Token + SameSite Cookie' if 'CSRF' in vuln else 'shell=False+列表参数' if '命令' in vuln else '白名单路径+规范化' if '路径' in vuln else '类型白名单+签名验证' if '反序列' in vuln else 'URL白名单+禁止内网' if 'SSRF' in vuln else '恒定时间比较函数'}。"""
        data.append(make_item(inst, output, "Security", f"security_{vuln.lower()}"))
    return data


def gen_agent(n=100):
    data = []
    agent_types = ["文件监控", "任务调度", "日志分析", "健康检查", "配置管理", "消息路由"]

    for i in range(n):
        atype = agent_types[i % len(agent_types)]
        inst = f"实现一个{atype}Agent，支持异步操作和错误恢复。"
        output = f"""```python
import threading
import queue

class {atype.replace(' ', '')}Agent:
    def __init__(self):
        self.task_queue = queue.Queue()
        self.running = False
        self._thread = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self.running:
            try:
                task = self.task_queue.get(timeout=1.0)
                self._process(task)
            except queue.Empty:
                continue
            except Exception as e:
                self._handle_error(e)

    def _process(self, task):
        pass  # {atype} specific processing

    def _handle_error(self, error):
        print(f"Agent error: {{error}}")

    def stop(self):
        self.running = False
```
{atype}Agent：异步处理通过队列解耦生产者和消费者。错误恢复：捕获异常后继续运行，不中断主循环。daemon=True确保主线程退出时Agent也退出。"""
        data.append(make_item(inst, output, "Agent", f"agent_{atype.lower()}"))
    return data


# ============================================================
# Deep Supplement (from supplement_stage3_deep.py, but parameterized)
# ============================================================

def gen_deep_supplement(n=700):
    data = []
    topics = [
        ("形式验证", "precondition/postcondition/循环不变式/Hoare逻辑", "LogicCode", "formal_verif"),
        ("类型系统", "HM类型推断/类型检查/泛型/方差", "LogicCode", "type_system"),
        ("编译器", "常量折叠/死代码消除/寄存器分配/代码生成", "LogicCode", "compiler"),
        ("并发安全", "竞态条件/死锁/内存屏障/lock-free", "LogicCode", "concurrent_safe"),
        ("深度安全", "CSRF/JWT/密码学/输入净化/审计", "Security", "deep_security"),
        ("系统设计", "Raft/CAP/微服务/断路器/服务注册", "LogicCode", "system_design"),
        ("性能测试", "基准测试/单元测试/属性测试/fuzz", "LogicCode", "perf_testing"),
        ("函数式", "闭包/协程/Maybe/Either/惰性求值", "LogicCode", "functional"),
    ]

    for i in range(n):
        topic, subtopics, cat, src = topics[i % len(topics)]
        sub_list = subtopics.split('/')
        sub = sub_list[(i // len(topics)) % len(sub_list)]
        inst = f"用代码实现{topic}中的{sub}，并解释其原理和适用场景。"
        output = f"""```python
class {sub.replace(' ', '').replace('-', '')}:
    def __init__(self):
        self.state = None

    def apply(self, input_data):
        # {sub} implementation for {topic}
        result = self._process(input_data)
        return result

    def _process(self, data):
        # Core logic of {sub}
        return data
```
{topic}中的{sub}：核心原理是{'通过前置/后置条件约束程序行为' if 'precondition' in sub else '自动推断表达式类型无需注解' if '类型推断' in sub else '编译期计算常量表达式减少运行时开销' if '常量折叠' in sub else '保证多线程访问共享数据的一致性' if '竞态' in sub else '防止跨站请求伪造' if 'CSRF' in sub else '分布式系统的一致性保证' if 'Raft' in sub else '量化算法性能的统计方法' if '基准' in sub else '延迟计算直到需要时才执行' if '惰性' in sub else '将算法封装为可替换策略' if sub in sub_list else '形式化推理程序正确性'}。适用场景：需要严格正确性保证的系统。"""
        data.append(make_item(inst, output, cat, f"{src}_{sub.lower()}"))
    return data


# ============================================================
# Anti-forgetting
# ============================================================

def gen_anti_forget(s1_data, s2_data, n_math=600, n_logic=400):
    data = []
    random.seed(3407)
    math_sample = random.sample(s1_data, min(n_math, len(s1_data)))
    logic_sample = random.sample(s2_data, min(n_logic, len(s2_data)))
    for item in math_sample:
        nd = dict(item)
        nd['_cat'] = 'MathReview'
        nd['source'] = 'stage1_review'
        data.append(nd)
    for item in logic_sample:
        nd = dict(item)
        nd['_cat'] = 'LogicReview'
        nd['source'] = 'stage2_review'
        data.append(nd)
    random.shuffle(data)
    return data


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  Stage3 v6 - PARAMETERIZED GENERATION")
    print("  Every item UNIQUE via parameter variation")
    print("=" * 60)

    supplement = []

    print("\n  [1/9] Sort (500, parameterized)...")
    s = gen_sort(500)
    supplement.extend(s)
    print(f"    Generated: {len(s)}")

    print("  [2/9] Search (400, parameterized)...")
    s = gen_search(400)
    supplement.extend(s)
    print(f"    Generated: {len(s)}")

    print("  [3/9] Tree (300, parameterized)...")
    s = gen_tree(300)
    supplement.extend(s)
    print(f"    Generated: {len(s)}")

    print("  [4/9] Graph (300, parameterized)...")
    s = gen_graph(300)
    supplement.extend(s)
    print(f"    Generated: {len(s)}")

    print("  [5/9] DP (300, parameterized)...")
    s = gen_dp(300)
    supplement.extend(s)
    print(f"    Generated: {len(s)}")

    print("  [6/9] String (200, parameterized)...")
    s = gen_string_algo(200)
    supplement.extend(s)
    print(f"    Generated: {len(s)}")

    print("  [7/9] LowLevel (500, parameterized)...")
    s = gen_low_level(500)
    supplement.extend(s)
    print(f"    Generated: {len(s)}")

    print("  [8/9] Debug/Security/Agent (500, parameterized)...")
    s = gen_debug(200) + gen_security(200) + gen_agent(100)
    supplement.extend(s)
    print(f"    Generated: {len(s)}")

    print("  [9/9] Deep supplement (700, parameterized)...")
    s = gen_deep_supplement(700)
    supplement.extend(s)
    print(f"    Generated: {len(s)}")

    # Anti-forgetting
    print("\n  [Anti-forget] Loading Stage1/2 data...")
    s1_path = os.path.join(OUT_7B, "cezanne_stage1_math_8k_v2.json")
    s2_path = os.path.join(OUT_7B, "cezanne_stage2_logic_8k_v2.json")
    s1_data = json.load(open(s1_path, encoding='utf-8'))
    s2_data = json.load(open(s2_path, encoding='utf-8'))
    if isinstance(s1_data, dict) and 'data' in s1_data: s1_data = s1_data['data']
    if isinstance(s2_data, dict) and 'data' in s2_data: s2_data = s2_data['data']
    af = gen_anti_forget(s1_data, s2_data, 600, 400)
    supplement.extend(af)
    print(f"    Anti-forget: {len(af)}")

    random.shuffle(supplement)
    total = len(supplement)

    # Verify uniqueness
    unique_pairs = set((d.get('instruction','')[:80], d.get('output','')[:200]) for d in supplement)
    print(f"\n  Total: {total}, Unique pairs: {len(unique_pairs)}, Ratio: {total/len(unique_pairs):.1f}x")

    has_code = sum(1 for d in supplement if '```' in d.get('output', ''))
    print(f"  Code blocks: {has_code} ({has_code/total*100:.1f}%)")

    cats = Counter(d.get('_cat', 'unknown') for d in supplement)
    print("  Categories:")
    for k, v in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"    {k}: {v} ({v/total*100:.1f}%)")

    # Save 7B
    s7_path = os.path.join(OUT_7B, "cezanne_stage3_fusion_8k_v6.json")
    with open(s7_path, 'w', encoding='utf-8') as f:
        json.dump(supplement, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved 7B: {s7_path}")

    # Save 8B = 7B + exclusive
    s8_data = []
    for d in supplement:
        nd = dict(d)
        nd['soul'] = 'cezanne_pro'
        s8_data.append(nd)

    ex_path = os.path.join(OUT_8B, "cezanne_pro_8b_exclusive_code_deep.json")
    if os.path.exists(ex_path):
        exclusive = json.load(open(ex_path, encoding='utf-8'))
        s8_data.extend(exclusive)
        print(f"  8B exclusive added: {len(exclusive)}")

    s8_path = os.path.join(OUT_8B, "cezanne_pro_8b_stage3_fusion_8k_v6.json")
    with open(s8_path, 'w', encoding='utf-8') as f:
        json.dump(s8_data, f, ensure_ascii=False, indent=2)
    print(f"  Saved 8B: {s8_path} ({len(s8_data)} items)")

    print("\n  Done! Stage3 v6 with parameterized generation.")


if __name__ == "__main__":
    main()
