"""
Cezanne 5阶段数据准备
Stage1: 纯数学 (8000条)
Stage2: 纯逻辑学 (8000条)
Stage3: 数学+逻辑+编程基础融合 (8000条)
Stage4: 行业落地-代码工程/嵌入式/IoT (4000条)
Stage5: 工具链+防遗忘 (自动生成)
"""
import json, os, gc, random

DATA_DIR = r"D:\VORTEX_FLAME\soul_training_data\cezanne"
V3_PATH = os.path.join(DATA_DIR, "cezanne_55gb_v3.json")

MATH_KW = ['数学','方程','微积分','概率','统计','矩阵','线性代数','集合','证明','定理',
           '积分','微分','导数','极限','数列','级数','向量','行列式','特征值','正交',
           '函数','多项式','三角','几何','拓扑','群论','环论','域论','数论','组合数学',
           '离散数学','图论','欧拉','拉格朗日','泰勒','傅里叶','拉普拉斯','黎曼']
LOGIC_KW = ['逻辑','推理','证明','真值','命题','蕴含','矛盾','归纳','演绎','布尔',
            '集合运算','逻辑门','与非','或非','充分必要','充要','反证','逆否',
            '一阶逻辑','谓词','量词','存在','任意','有效推理','形式化','符号逻辑']
CODE_KW = ['代码','编程','python','java','程序','函数','算法','数据结构','递归',
           '面向对象','编译','变量','循环','条件','排序','搜索','二叉树','链表',
           '哈希','栈','队列','图','动态规划','贪心','分治','复杂度','指针',
           '数据库','SQL','API','框架','调试','测试','部署','架构','微服务',
           '嵌入式','IoT','传感器','单片机','固件','驱动','RTOS']
INDUSTRY_KW = ['嵌入式','IoT','物联网','传感器','单片机','固件','驱动','RTOS',
               '智能硬件','边缘计算','工业控制','PLC','SCADA','通信协议',
               'MQTT','CoAP','蓝牙','WiFi','ZigBee','LoRa','5G',
               'Cursor','IDE','DevOps','CI/CD','Docker','Kubernetes']

def classify(item):
    inst = item.get('instruction','') + ' ' + item.get('output','')
    has_math = any(kw in inst for kw in MATH_KW)
    has_logic = any(kw in inst for kw in LOGIC_KW)
    has_code = any(kw in inst for kw in CODE_KW)
    has_industry = any(kw in inst for kw in INDUSTRY_KW)
    if has_industry:
        return 'industry'
    if has_logic:
        return 'logic'
    if has_math:
        return 'math'
    if has_code:
        return 'code'
    return 'other'

print("Loading all Cezanne data...", flush=True)
all_items = []

for fname in ["cezanne_4k.json", "cezanne_desktop_supplement.json", "cezanne_hq_10k.json"]:
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        continue
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and len(item.get("output", "")) >= 50:
                all_items.append(item)
        print(f"  {fname}: loaded", flush=True)

print(f"  Loading v3 (473MB)...", flush=True)
with open(V3_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
if isinstance(data, dict) and "data" in data:
    data = data["data"]
if isinstance(data, list):
    for item in data:
        if isinstance(item, dict) and len(item.get("output", "")) >= 50:
            all_items.append(item)
del data
gc.collect()
print(f"  Total raw items: {len(all_items)}", flush=True)

print("Classifying...", flush=True)
buckets = {'math': [], 'logic': [], 'code': [], 'industry': [], 'other': []}
seen = set()
for item in all_items:
    key = item.get('instruction', '')[:100]
    if key in seen:
        continue
    seen.add(key)
    cat = classify(item)
    buckets[cat].append(item)

for cat, items in buckets.items():
    print(f"  {cat}: {len(items)} unique items", flush=True)

random.seed(3407)

s1 = buckets['math'][:8000]
random.shuffle(s1)
s1 = s1[:8000]
print(f"\nStage1 (Math): {len(s1)} items", flush=True)

s2 = buckets['logic'][:8000]
if len(s2) < 8000:
    s2.extend(random.sample(buckets['code'], min(8000 - len(s2), len(buckets['code']))))
random.shuffle(s2)
s2 = s2[:8000]
print(f"Stage2 (Logic): {len(s2)} items", flush=True)

s3_math = random.sample(buckets['math'], min(2500, len(buckets['math'])))
s3_logic = random.sample(buckets['logic'], min(2500, len(buckets['logic'])))
s3_code = random.sample(buckets['code'], min(3000, len(buckets['code'])))
s3 = s3_math + s3_logic + s3_code
random.shuffle(s3)
s3 = s3[:8000]
print(f"Stage3 (Math+Logic+Code): {len(s3)} items", flush=True)

s4 = buckets['industry'][:4000]
if len(s4) < 4000:
    s4.extend(random.sample(buckets['code'], min(4000 - len(s4), len(buckets['code']))))
random.shuffle(s4)
s4 = s4[:4000]
print(f"Stage4 (Industry): {len(s4)} items", flush=True)

for stage_name, stage_data in [("stage1_math_8k", s1), ("stage2_logic_8k", s2),
                                ("stage3_fusion_8k", s3), ("stage4_industry_4k", s4)]:
    path = os.path.join(DATA_DIR, f"cezanne_{stage_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stage_data, f, ensure_ascii=False, indent=2)
    sz = os.path.getsize(path) / 1024 / 1024
    print(f"  Saved {stage_name}: {len(stage_data)} items, {sz:.1f}MB", flush=True)

print("\nDone! All 4 stages ready.")
