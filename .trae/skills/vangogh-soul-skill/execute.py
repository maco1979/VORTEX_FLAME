#!/usr/bin/env python3
import json
import sys
sys.path.insert(0, 'd:/贾维斯')

from soul_knowledge_base import KnowledgeBase

def get_vangogh_status():
    kb = KnowledgeBase()
    knowledge = kb.get_knowledge_for_soul('vangogh-soul', count=5)
    return {
        "soul_name": "vangogh-soul",
        "type": "art",
        "domain": "情绪美学/视觉艺术",
        "knowledge": knowledge,
        "abilities": [
            "情绪驱动创作",
            "色彩碰撞与表达",
            "后印象派风格生成",
            "星空/向日葵/鸢尾花意象"
        ],
        "triggered_skills": ["museum-art-integration", "van-gogh-fkj-video-workflow"],
        "creativity_weight": 1.0,
        "inspiration_range": [0.03, 0.10]
    }

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "status"

    if action == "status":
        result = get_vangogh_status()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"error": f"未知操作: {action}"}, ensure_ascii=False))