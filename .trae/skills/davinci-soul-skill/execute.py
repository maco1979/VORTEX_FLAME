#!/usr/bin/env python3
import json
import sys
sys.path.insert(0, 'd:/贾维斯')

from soul_knowledge_base import KnowledgeBase

def get_davinci_status():
    kb = KnowledgeBase()
    knowledge = kb.get_knowledge_for_soul('davinci-soul', count=5)
    return {
        "soul_name": "davinci-soul",
        "type": "cross",
        "domain": "跨界创新/达芬奇风格",
        "knowledge": knowledge,
        "abilities": [
            "工程架构设计",
            "人体解剖素描",
            "机械原理应用",
            "技术创新思维"
        ],
        "triggered_skills": ["davinci-cross-soul", "multi-soul-collaboration"],
        "creativity_weight": 1.5,
        "inspiration_range": [0.04, 0.11]
    }

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "status"

    if action == "status":
        result = get_davinci_status()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"error": f"未知操作: {action}"}, ensure_ascii=False))