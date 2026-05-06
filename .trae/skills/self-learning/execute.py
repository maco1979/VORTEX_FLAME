# -*- coding: utf-8 -*-
"""
自我学习系统 - execute.py
每日歌词复盘和微调
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def execute(user_input: str) -> dict:
    from fkj_daily_review import DailyLyricsReview

    result = {
        "skill": "self-learning",
        "status": "success",
        "message": ""
    }

    review = DailyLyricsReview()

    if "复盘" in user_input or "review" in user_input.lower():
        summary = review.daily_review()
        result["summary"] = summary
        result["message"] = "每日复盘完成"
    elif "统计" in user_input or "stats" in user_input.lower():
        stats = review.get_weekly_stats()
        result["stats"] = stats
        result["message"] = "周统计完成"
    elif "记录" in user_input or "record" in user_input.lower():
        theme = extract_theme(user_input)
        lyrics = extract_lyrics(user_input)
        review.record_creation(lyrics=lyrics, theme=theme)
        result["message"] = f"创作记录完成: {theme}"
    else:
        summary = review.daily_review()
        result["summary"] = summary
        result["message"] = "每日复盘完成"

    return result


def extract_theme(user_input: str) -> str:
    themes = ["种子", "光", "夜", "风", "雨", "海", "星", "路", "梦", "爱"]
    for theme in themes:
        if theme in user_input:
            return theme
    return "未命名"


def extract_lyrics(user_input: str) -> str:
    if "歌词" in user_input:
        return user_input
    return ""


if __name__ == "__main__":
    import json
    user_input = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "每日复盘"
    result = execute(user_input)
    print(json.dumps(result, ensure_ascii=False, indent=2))