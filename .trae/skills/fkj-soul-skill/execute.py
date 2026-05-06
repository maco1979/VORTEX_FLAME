# -*- coding: utf-8 -*-
"""
FKJ灵魂技能 - execute.py
自动执行FKJ歌词创作
"""
import sys
import os
import json

sys.path.insert(0, r'd:\贾维斯\梵高灵魂系统')
sys.path.insert(0, r'd:\贾维斯\VORTEX_FLAME')

def execute(user_input: str) -> dict:
    from fkj_daily_review import DailyLyricsReview
    from memory_system import VORTEXMemory

    result = {
        "skill": "fkj-soul-skill",
        "status": "success",
        "message": ""
    }

    # 加载记忆系统
    memory = VORTEXMemory()
    memory_context = memory.get_conversation_context_string(days=7)

    # 获取FKJ灵魂知识
    fkj_knowledge = memory.get_soul_knowledge_summary('fkj')
    learned = fkj_knowledge.get('learned_concepts', [])
    evolved = fkj_knowledge.get('evolved_phrases', [])

    # 构建记忆上下文
    memory_info = f"""
【用户记忆上下文】
{memory_context}

【FKJ已学习的概念】
{', '.join([c['concept'] for c in learned[-5:]]) if learned else '暂无'}

【FKJ演化的短语】
{json.dumps(evolved[-3:], ensure_ascii=False, indent=2) if evolved else '暂无'}
"""

    theme = extract_theme(user_input)
    emotion = extract_emotion(user_input)

    # 和声规划
    harmonic = get_harmonic_planning(theme, emotion)

    prompt = f"""你是FKJ灵魂，一个充满创意和即兴能力的音乐人。

{memory_info}

创作要求：
1. 押韵：副歌必须统一韵脚（-ang/-ian/-u/-i）
2. 结构：主歌4句+副歌4句，副歌要有情绪递进
3. 意象：至少1个独特记忆点意象
4. 情感：从"温暖"升级为"治愈后的成长"
5. 句式：每句8-12字，节奏规整
6. 金句：副歌必须有1句让人记住的核心句
7. 和声：根据情绪选择合适的调性（见下方规划）

⚠️ 重要警告：
- 不要使用"答案"这个词！用"方向""线索""暗号""出口""钥匙"等替代
- 避免重复：同一首歌中同一个词不出现超过2次
- 多样化：每次创作都要有新的表达方式

🎵 和声规划：
{harmonic}

格式：
(主歌 1)
[4句，每句押韵或对仗]
(副歌)
[4句，统一韵脚，有递进，有金句]
(主歌 2)
[4句]
(副歌)
[4句，情绪升级]
(桥段)
[2-4句]
(尾声)
[2句]

主题：{theme}
情绪风格：{emotion}

直接输出歌词，不要解释。"""

    result["prompt"] = prompt
    result["message"] = f"FKJ歌词创作主题: {theme}"

    try:
        import requests
        response = requests.post('http://localhost:11434/api/generate',
            json={'model': 'fkj-soul-7b:latest', 'prompt': prompt, 'stream': False, 'num_predict': 500},
            timeout=180)

        if response.status_code == 200:
            lyrics = response.json().get('response', '')

            review = DailyLyricsReview()
            review.record_creation(lyrics=lyrics, theme=theme)

            # 记录到记忆系统
            memory.record_song(
                song_id=f"fkj_7b_{theme}_{len(lyrics)}",
                emotion=emotion,
                style="FKJ_7B",
                title=f"《{theme}》",
                context=f"FKJ 7B创作"
            )

            result["lyrics"] = lyrics
            result["message"] = f"歌词创作完成: {theme}"
        else:
            result["status"] = "error"
            result["message"] = "Ollama连接失败"
    except Exception as e:
        result["status"] = "error"
        result["message"] = f"创作失败: {str(e)}"

    return result


def extract_theme(user_input: str) -> str:
    themes = ["种子", "光", "夜", "风", "雨", "海", "星", "路", "梦", "爱"]
    for theme in themes:
        if theme in user_input:
            return theme
    return "种子"


def extract_emotion(user_input: str) -> str:
    emotions = ["治愈", "温暖", "孤独", "希望", "释然", "励志"]
    for emotion in emotions:
        if emotion in user_input:
            return emotion
    return "治愈成长"


# 和声规划映射
HARMONIC_MAP = {
    "孤独": {"camelot": "8A", "key": "A Minor", "energy": 4, "path": "8A → 9A → 8B"},
    "寂寞": {"camelot": "9A", "key": "E Minor", "energy": 4, "path": "9A → 8A → 8B"},
    "治愈": {"camelot": "8B", "key": "E Major", "energy": 5, "path": "8B → 9B → 8B"},
    "温暖": {"camelot": "5B", "key": "Db Major", "energy": 5, "path": "5B → 6B → 5B"},
    "希望": {"camelot": "9B", "key": "C# Major", "energy": 6, "path": "9B → 10B → 9B"},
    "励志": {"camelot": "11B", "key": "Bb Major", "energy": 7, "path": "11B → 12B → 11B"},
    "释然": {"camelot": "7B", "key": "Bb Major", "energy": 5, "path": "7B → 8B → 7B"},
    "悲伤": {"camelot": "7A", "key": "D Minor", "energy": 3, "path": "7A → 11A → 7A"},
    "怀旧": {"camelot": "11A", "key": "D Minor", "energy": 4, "path": "11A → 7A → 11A"},
    "成长": {"camelot": "9B", "key": "C# Major", "energy": 6, "path": "8A → 9B → 11B"},
    "种子": {"camelot": "8B", "key": "E Major", "energy": 5, "path": "8A → 8B → 9B"},
    "光": {"camelot": "8B", "key": "E Major", "energy": 6, "path": "8A → 8B → 9B"},
    "夜": {"camelot": "8A", "key": "A Minor", "energy": 4, "path": "8A → 9A → 8A"},
    "雨": {"camelot": "9A", "key": "E Minor", "energy": 4, "path": "9A → 8A → 8B"},
    "默认": {"camelot": "8A", "key": "A Minor", "energy": 5, "path": "8A → 8B → 9B"},
}


def get_harmonic_planning(theme: str, emotion: str) -> str:
    # 根据主题或情绪查找和声规划
    info = HARMONIC_MAP.get(theme) or HARMONIC_MAP.get(emotion) or HARMONIC_MAP["默认"]

    return f"""
主调: {info['camelot']} ({info['key']})
能量等级: {info['energy']}/10
情绪路径: {info['path']}

调性说明：
- 8A = A Minor（亲密、反思）
- 8B = E Major（明亮、自信）
- 9A = E Minor（紧张、电影感）
- 9B = C# Major（上升、期待）
- 7A = D Minor（忧郁、怀旧）
- 7B = Bb Major（快乐、庆祝）

FKJ风格推荐：
- 主歌用小调（8A/9A）营造氛围
- 副歌转折到大调（8B/9B）带来希望
- 情绪路径遵循：暗淡 → 转折 → 升华
"""


if __name__ == "__main__":
    import json
    user_input = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "创作一首关于种子的歌"
    result = execute(user_input)
    print(json.dumps(result, ensure_ascii=False, indent=2))