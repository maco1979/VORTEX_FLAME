import json
import os
import sys
import copy
from datetime import datetime, timedelta
from pathlib import Path

REGISTRY_PATH = Path(__file__).parent / "industry_knowledge_graph" / "tool_template_registry.json"
TRAINING_DATA_DIR = Path(__file__).parent / "soul_training_data"
REGISTRY = None


def load_registry():
    global REGISTRY
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        REGISTRY = json.load(f)
    return REGISTRY


def save_registry(reg=None):
    data = reg or REGISTRY
    data["meta"]["last_updated"] = datetime.now().strftime("%Y-%m-%d")
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_all_tools(reg=None):
    data = reg or REGISTRY
    tools = []
    for soul_id, soul_data in data["souls"].items():
        for tool_id, tool_data in soul_data["tools"].items():
            entry = copy.deepcopy(tool_data)
            entry["soul_id"] = soul_id
            entry["soul_name"] = soul_data["name"]
            entry["tool_id"] = tool_id
            tools.append(entry)
    return tools


def get_tools_by_soul(soul_id, reg=None):
    data = reg or REGISTRY
    soul = data["souls"].get(soul_id)
    if not soul:
        return []
    tools = []
    for tool_id, tool_data in soul["tools"].items():
        entry = copy.deepcopy(tool_data)
        entry["tool_id"] = tool_id
        tools.append(entry)
    return tools


def get_top_templates(soul_id, top_n=10, reg=None):
    data = reg or REGISTRY
    soul = data["souls"].get(soul_id)
    if not soul:
        return []
    all_templates = []
    for tool_id, tool_data in soul["tools"].items():
        for tpl in tool_data.get("templates", []):
            entry = copy.deepcopy(tpl)
            entry["tool_id"] = tool_id
            entry["tool_name"] = tool_data["name"]
            entry["tool_name_cn"] = tool_data["name_cn"]
            entry["category"] = tool_data["category"]
            entry["difficulty"] = tpl.get("difficulty", "basic")
            all_templates.append(entry)
    all_templates.sort(key=lambda x: x["popularity"], reverse=True)
    return all_templates[:top_n]


def check_sunset_candidates(reg=None):
    data = reg or REGISTRY
    threshold = data["meta"]["sunset_threshold"]
    candidates = []
    for soul_id, soul_data in data["souls"].items():
        for tool_id, tool_data in soul_data["tools"].items():
            flags = []
            if tool_data["popularity"] < threshold["popularity_below"]:
                flags.append(f"popularity={tool_data['popularity']}<{threshold['popularity_below']}")
            last = datetime.strptime(tool_data["last_updated"], "%Y-%m-%d")
            months_old = (datetime.now() - last).days / 30
            if months_old > threshold["no_update_months"]:
                flags.append(f"no_update={months_old:.0f}months>{threshold['no_update_months']}months")
            if flags:
                candidates.append({
                    "soul_id": soul_id,
                    "tool_id": tool_id,
                    "tool_name": tool_data["name"],
                    "flags": flags,
                    "popularity": tool_data["popularity"],
                    "status": tool_data["status"],
                    "last_updated": tool_data["last_updated"]
                })
    return candidates


def update_popularity(tool_id, soul_id, new_popularity, reg=None):
    data = reg or REGISTRY
    soul = data["souls"].get(soul_id)
    if not soul or tool_id not in soul["tools"]:
        return False
    soul["tools"][tool_id]["popularity"] = new_popularity
    soul["tools"][tool_id]["last_updated"] = datetime.now().strftime("%Y-%m-%d")
    if new_popularity < 20:
        soul["tools"][tool_id]["status"] = "sunset_review"
    return True


def add_tool(soul_id, tool_id, tool_data, reg=None):
    data = reg or REGISTRY
    soul = data["souls"].get(soul_id)
    if not soul:
        return False
    if tool_id in soul["tools"]:
        return False
    tool_data["last_updated"] = datetime.now().strftime("%Y-%m-%d")
    tool_data["status"] = "active"
    soul["tools"][tool_id] = tool_data
    return True


def remove_tool(soul_id, tool_id, reg=None):
    data = reg or REGISTRY
    soul = data["souls"].get(soul_id)
    if not soul or tool_id not in soul["tools"]:
        return False
    del soul["tools"][tool_id]
    return True


def generate_training_samples(soul_id, samples_per_template=3, reg=None):
    data = reg or REGISTRY
    soul = data["souls"].get(soul_id)
    if not soul:
        return []

    samples = []
    for tool_id, tool_data in soul["tools"].items():
        if tool_data["status"] != "active":
            continue
        for tpl in tool_data.get("templates", []):
            for i in range(samples_per_template):
                difficulty_map = {"basic": "基础", "intermediate": "进阶", "advanced": "高级"}
                diff_cn = difficulty_map.get(tpl.get("difficulty", "basic"), "基础")

                instructions = [
                    f"请用{tool_data['name_cn']}完成{tpl['name']}，{tpl['desc']}",
                    f"如何在{tool_data['name']}中实现{tpl['name']}？请给出{diff_cn}步骤",
                    f"我需要用{tool_data['name']}做{tpl['name']}，请指导操作流程和注意事项"
                ]

                output_parts = [
                    f"## {tool_data['name_cn']} - {tpl['name']}\n",
                    f"**工具**: {tool_data['name']} ({tool_data['category']})",
                    f"**难度**: {diff_cn}",
                    f"**用途**: {tpl['desc']}\n",
                    f"### 操作步骤",
                ]

                if diff_cn == "基础":
                    output_parts.extend([
                        f"1. 打开{tool_data['name']}，新建项目",
                        f"2. 选择{tpl['name']}模板或向导",
                        f"3. 配置基本参数（根据具体需求调整）",
                        f"4. 运行计算/生成/分析",
                        f"5. 查看结果并导出报告",
                    ])
                elif diff_cn == "进阶":
                    output_parts.extend([
                        f"1. 前置准备：确认输入数据格式和质量",
                        f"2. 在{tool_data['name']}中创建{tpl['name']}项目",
                        f"3. 高级参数配置：根据物理/业务模型调整关键参数",
                        f"4. 网格/采样/精度设置：平衡计算精度与效率",
                        f"5. 运行并监控收敛/进度",
                        f"6. 后处理：结果验证与敏感性分析",
                        f"7. 导出并生成技术报告",
                    ])
                else:
                    output_parts.extend([
                        f"1. 需求分析：明确{tpl['name']}的技术目标和约束条件",
                        f"2. 方案设计：选择合适的算法/方法论",
                        f"3. 在{tool_data['name']}中搭建高级工作流",
                        f"4. 参数调优：基于经验法则和试算优化关键参数",
                        f"5. 多方案对比：至少2种方案交叉验证",
                        f"6. 结果审查：物理/业务合理性检查",
                        f"7. 文档化：记录假设、参数、结论和局限性",
                    ])

                output_parts.append(f"\n**平台**: {', '.join(tool_data.get('platform', []))}")
                if tool_data.get("license"):
                    license_map = {"commercial": "商业授权", "open_source": "开源免费", "freemium": "免费增值", "subscription": "订阅制", "free_research": "学术免费", "open_weight": "开放权重"}
                    output_parts.append(f"**授权**: {license_map.get(tool_data['license'], tool_data['license'])}")

                sample = {
                    "instruction": instructions[i % len(instructions)],
                    "output": "\n".join(output_parts),
                    "source": f"tool_{tool_id}_{tpl['id']}",
                    "tool": tool_data["name"],
                    "template": tpl["name"],
                    "difficulty": tpl.get("difficulty", "basic"),
                    "popularity": tpl["popularity"]
                }
                samples.append(sample)

    samples.sort(key=lambda x: x["popularity"], reverse=True)
    return samples


def generate_soul_training_data(soul_id, max_samples=2000, samples_per_template=3, reg=None):
    samples = generate_training_samples(soul_id, samples_per_template, reg)

    existing_path = TRAINING_DATA_DIR / soul_id / f"{soul_id}_4k.json"
    existing = []
    if existing_path.exists():
        with open(existing_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    seen_instructions = set(s.get("instruction", "")[:100] for s in existing)
    new_unique = []
    for s in samples:
        key = s["instruction"][:100]
        if key not in seen_instructions:
            seen_instructions.add(key)
            new_unique.append(s)

    combined = existing + new_unique
    if len(combined) > max_samples:
        tool_samples = [s for s in combined if s.get("source", "").startswith("tool_")]
        other_samples = [s for s in combined if not s.get("source", "").startswith("tool_")]
        tool_budget = int(max_samples * 0.3)
        other_budget = max_samples - tool_budget
        combined = tool_samples[:tool_budget] + other_samples[:other_budget]

    out_dir = TRAINING_DATA_DIR / soul_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{soul_id}_tool_enhanced.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"[{soul_id}] 工具模板训练数据生成完成:")
    print(f"  原有样本: {len(existing)}")
    print(f"  新增工具样本: {len(new_unique)}")
    print(f"  合并后总数: {len(combined)}")
    print(f"  输出: {out_path}")
    return combined


def quarterly_review(reg=None):
    data = reg or REGISTRY
    print("=" * 60)
    print("  工具模板季度审查报告")
    print(f"  审查日期: {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 60)

    sunset_candidates = check_sunset_candidates(data)
    print(f"\n[!] 淘汰候选 ({len(sunset_candidates)} 个):")
    for c in sunset_candidates:
        print(f"  [{c['soul_id']}] {c['tool_name']} (pop={c['popularity']}, flags={c['flags']})")

    all_tools = get_all_tools(data)
    tier_s = [t for t in all_tools if t["tier"] == "S"]
    tier_a = [t for t in all_tools if t["tier"] == "A"]
    tier_b = [t for t in all_tools if t["tier"] == "B"]

    print(f"\n[Tier] 工具分布:")
    print(f"  S级(核心): {len(tier_s)} 个")
    print(f"  A级(重要): {len(tier_a)} 个")
    print(f"  B级(补充): {len(tier_b)} 个")
    print(f"  总计: {len(all_tools)} 个工具, {sum(len(t.get('templates',[])) for t in all_tools)} 个模板")

    print(f"\n[HOT] 各灵魂热门模板 TOP3:")
    for soul_id, soul_data in data["souls"].items():
        top3 = get_top_templates(soul_id, 3, data)
        print(f"  [{soul_data['name']}]")
        for t in top3:
            print(f"    {t['tool_name']}->{t['name']} (pop={t['popularity']})")

    return sunset_candidates


def generate_all_souls_training_data(reg=None):
    data = reg or REGISTRY
    for soul_id in data["souls"]:
        generate_soul_training_data(soul_id, reg=data)
    print("\n✅ 全部14灵魂工具训练数据生成完成")


if __name__ == "__main__":
    load_registry()

    if len(sys.argv) < 2:
        print("用法: python tool_lifecycle.py <command> [args]")
        print("命令:")
        print("  review                    - 季度审查报告")
        print("  sunset                    - 查看淘汰候选")
        print("  top <soul_id> [n]         - 查看灵魂热门模板")
        print("  train <soul_id> [n]       - 生成灵魂工具训练数据")
        print("  train_all                 - 生成全部灵魂工具训练数据")
        print("  add <soul_id> <tool_id>   - 添加新工具(交互式)")
        print("  update <soul_id> <tool_id> <popularity> - 更新热度")
        print("  remove <soul_id> <tool_id> - 移除工具")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "review":
        quarterly_review()
    elif cmd == "sunset":
        candidates = check_sunset_candidates()
        if not candidates:
            print("✅ 没有淘汰候选")
        for c in candidates:
            print(f"⚠️ [{c['soul_id']}] {c['tool_name']} pop={c['popularity']} flags={c['flags']}")
    elif cmd == "top":
        soul = sys.argv[2] if len(sys.argv) > 2 else "einstein"
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        templates = get_top_templates(soul, n)
        for t in templates:
            print(f"  {t['tool_name']} → {t['name']} (pop={t['popularity']}, diff={t['difficulty']})")
    elif cmd == "train":
        soul = sys.argv[2] if len(sys.argv) > 2 else "einstein"
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        generate_soul_training_data(soul, samples_per_template=n)
    elif cmd == "train_all":
        generate_all_souls_training_data()
    elif cmd == "update":
        soul = sys.argv[2]
        tool = sys.argv[3]
        pop = int(sys.argv[4])
        if update_popularity(tool, soul, pop):
            save_registry()
            print(f"✅ {soul}/{tool} popularity → {pop}")
        else:
            print(f"❌ 未找到 {soul}/{tool}")
    elif cmd == "remove":
        soul = sys.argv[2]
        tool = sys.argv[3]
        if remove_tool(soul, tool):
            save_registry()
            print(f"✅ 已移除 {soul}/{tool}")
        else:
            print(f"❌ 未找到 {soul}/{tool}")
    else:
        print(f"未知命令: {cmd}")
