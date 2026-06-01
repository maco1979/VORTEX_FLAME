"""
Open Design MCP 适配器 — VORTEX FLAME × Open Design AI

Open Design 是 Claude Design 的开源替代方案（nexu-io/open-design），
提供 137 个 Skill + 150 套 DESIGN.md 系统，支持 PPT/HTML/PDF/PPTX 导出。

本适配器将 Open Design 的核心能力桥接到 VORTEX 灵魂矩阵：
  - 莫奈/梵高灵魂 → 设计生成、美学评分、色彩方案
  - 达芬奇灵魂 → 版式建议、3D设计、工程图
  - 希罗多德灵魂 → PPT生成、文档排版

三层架构：
  L1: API直连 — Open Design daemon (http://127.0.0.1:3456)
  L2: CLI封装 — pnpm tools-dev 启动本地服务
  L3: 概念生成 — 离线降级，生成设计概念描述
"""

import json
import os
import subprocess
import time
from typing import Dict, List, Optional

import requests

OPEN_DESIGN_URL = os.environ.get("OPEN_DESIGN_URL", "http://127.0.0.1:3456")
OPEN_DESIGN_TIMEOUT = int(os.environ.get("OPEN_DESIGN_TIMEOUT", "30"))


DESIGN_SYSTEMS = {
    "linear": {"name": "Linear", "style": "极简科技", "colors": ["#5E6AD2", "#1C1F26", "#F4F4F6"]},
    "vercel": {"name": "Vercel", "style": "黑白极简", "colors": ["#000000", "#FFFFFF", "#0070F3"]},
    "stripe": {"name": "Stripe", "style": "金融专业", "colors": ["#635BFF", "#0A2540", "#F6F9FC"]},
    "apple": {"name": "Apple", "style": "苹果美学", "colors": ["#0071E3", "#1D1D1F", "#F5F5F7"]},
    "cursor": {"name": "Cursor", "style": "开发者工具", "colors": ["#7C3AED", "#0F172A", "#F8FAFC"]},
    "figma": {"name": "Figma", "style": "设计工具", "colors": ["#A259FF", "#0D0D0D", "#FFFFFF"]},
    "kami": {"name": "Kami", "style": "温暖纸张", "colors": ["#1A365D", "#F5E6D3", "#8B7355"]},
    "guizang_ppt": {"name": "鬼藏PPT", "style": "杂志式演示", "colors": ["#1A1A2E", "#E94560", "#F5F5F5"]},
}

SKILL_CATALOG = {
    "generate_ppt": {"name": "PPT生成", "description": "从文本/大纲生成杂志式Web PPT", "souls": ["monet", "herodotus"]},
    "aesthetic_score": {"name": "美学评分", "description": "5维美学评估：理念·层级·执行·具体性·克制", "souls": ["monet", "vangogh"]},
    "design_system": {"name": "设计系统", "description": "150套可移植DESIGN.md系统选择", "souls": ["monet", "davinci"]},
    "layout_suggest": {"name": "版式建议", "description": "基于设计系统的版式布局推荐", "souls": ["monet", "davinci"]},
    "color_palette": {"name": "色彩方案", "description": "OKLch色彩空间调色板生成", "souls": ["monet", "vangogh"]},
    "typography": {"name": "字体排版", "description": "字体栈选择与层级设计", "souls": ["monet", "herodotus"]},
    "export_html": {"name": "HTML导出", "description": "导出为独立HTML文件", "souls": ["davinci"]},
    "export_pdf": {"name": "PDF导出", "description": "导出为PDF文档", "souls": ["herodotus"]},
    "export_pptx": {"name": "PPTX导出", "description": "导出为PowerPoint文件", "souls": ["herodotus"]},
    "visual_reasoning": {"name": "视觉推理", "description": "5维自评：理念·层级·执行·具体性·克制", "souls": ["monet", "vangogh"]},
    "prompt_choreography": {"name": "提示词编舞", "description": "交互式问题表单，30秒锁定方向", "souls": ["monet"]},
    "guizang_ppt": {"name": "鬼藏PPT", "description": "杂志式Web PPT生成，WebGL hero", "souls": ["herodotus", "monet"]},
}


class OpenDesignAdapter:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or OPEN_DESIGN_URL
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            resp = requests.get(f"{self.base_url}/api/health", timeout=5)
            self._available = resp.status_code == 200
            return self._available
        except Exception:
            self._available = False
            return False

    def reset_availability(self):
        self._available = None

    def list_skills(self, soul: str = None) -> List[dict]:
        skills = []
        for sid, info in SKILL_CATALOG.items():
            if soul and soul not in info["souls"]:
                continue
            skills.append({"id": sid, **info})
        return skills

    def list_design_systems(self) -> Dict[str, dict]:
        return DESIGN_SYSTEMS

    def generate_ppt(self, prompt: str, soul: str = "monet",
                     design_system: str = "guizang_ppt",
                     title: str = "", outline: str = "",
                     export_format: str = "html") -> dict:
        if self.is_available():
            try:
                resp = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "skill": "guizang-ppt",
                        "prompt": prompt,
                        "design_system": design_system,
                        "title": title,
                        "outline": outline,
                        "export_format": export_format,
                    },
                    timeout=OPEN_DESIGN_TIMEOUT,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "status": "ok",
                        "soul": soul,
                        "skill": "generate_ppt",
                        "design_system": design_system,
                        "output": data.get("output", ""),
                        "preview_url": data.get("preview_url", ""),
                        "export_files": data.get("export_files", []),
                    }
            except Exception as e:
                pass

        return self._concept_ppt(prompt, soul, design_system, title, outline)

    def aesthetic_score(self, content: str, soul: str = "monet",
                        dimensions: List[str] = None) -> dict:
        dims = dimensions or ["理念", "层级", "执行", "具体性", "克制"]
        if self.is_available():
            try:
                resp = requests.post(
                    f"{self.base_url}/api/score",
                    json={"content": content, "dimensions": dims},
                    timeout=OPEN_DESIGN_TIMEOUT,
                )
                if resp.status_code == 200:
                    return {"status": "ok", "soul": soul, "scores": resp.json().get("scores", {})}
            except Exception:
                pass

        scores = {}
        for d in dims:
            scores[d] = {"score": 7, "reason": f"[离线评估] 基于{soul}灵魂领域知识的启发式评估"}
        return {
            "status": "concept_only",
            "soul": soul,
            "scores": scores,
            "message": "Open Design daemon未启动，使用启发式评估",
        }

    def color_palette(self, prompt: str, soul: str = "monet",
                      style: str = "linear", n_colors: int = 5) -> dict:
        if self.is_available():
            try:
                resp = requests.post(
                    f"{self.base_url}/api/color",
                    json={"prompt": prompt, "style": style, "n_colors": n_colors},
                    timeout=OPEN_DESIGN_TIMEOUT,
                )
                if resp.status_code == 200:
                    return {"status": "ok", "soul": soul, "palette": resp.json().get("palette", [])}
            except Exception:
                pass

        ds = DESIGN_SYSTEMS.get(style, DESIGN_SYSTEMS["linear"])
        return {
            "status": "concept_only",
            "soul": soul,
            "style": style,
            "palette": ds["colors"][:n_colors],
            "message": f"基于{ds['name']}({ds['style']})设计系统的预设色彩",
        }

    def design_system_recommend(self, prompt: str, soul: str = "monet") -> dict:
        keywords_map = {
            "科技": ["linear", "vercel", "cursor"],
            "金融": ["stripe"],
            "设计": ["figma", "cursor"],
            "演示": ["guizang_ppt", "kami"],
            "文档": ["kami", "apple"],
            "温暖": ["kami"],
            "极简": ["vercel", "linear"],
            "专业": ["stripe", "apple"],
        }
        recommended = []
        for kw, systems in keywords_map.items():
            if kw in prompt:
                for s in systems:
                    if s not in recommended:
                        recommended.append(s)
        if not recommended:
            recommended = ["linear", "guizang_ppt"]

        return {
            "status": "ok",
            "soul": soul,
            "prompt": prompt,
            "recommended_systems": [{"id": sid, **DESIGN_SYSTEMS.get(sid, {})} for sid in recommended],
        }

    def _concept_ppt(self, prompt: str, soul: str, design_system: str,
                     title: str, outline: str) -> dict:
        ds = DESIGN_SYSTEMS.get(design_system, DESIGN_SYSTEMS["guizang_ppt"])
        return {
            "status": "concept_only",
            "soul": soul,
            "skill": "generate_ppt",
            "design_system": design_system,
            "title": title or f"基于{ds['name']}风格的设计概念",
            "concept": {
                "style": ds["style"],
                "color_palette": ds["colors"],
                "layout": "杂志式排版，WebGL hero区域，滚动叙事",
                "sections": [
                    "封面 — 标题 + 视觉冲击",
                    "问题 — 用户痛点与场景",
                    "方案 — 核心价值主张",
                    "细节 — 功能与架构",
                    "数据 — 关键指标与成果",
                    "结尾 — 行动号召",
                ] if not outline else outline.split("\n"),
            },
            "message": f"Open Design daemon未启动。使用{ds['name']}({ds['style']})设计系统生成概念。启动方式：pnpm tools-dev",
        }

    def status(self) -> dict:
        return {
            "daemon_connected": self.is_available(),
            "url": self.base_url,
            "skills_count": len(SKILL_CATALOG),
            "design_systems_count": len(DESIGN_SYSTEMS),
            "available_skills": list(SKILL_CATALOG.keys()),
            "available_design_systems": list(DESIGN_SYSTEMS.keys()),
        }


_adapter: Optional[OpenDesignAdapter] = None


def get_adapter() -> OpenDesignAdapter:
    global _adapter
    if _adapter is None:
        _adapter = OpenDesignAdapter()
    return _adapter


if __name__ == "__main__":
    import json
    adapter = get_adapter()
    print(json.dumps(adapter.status(), indent=2, ensure_ascii=False))
    print("\n--- Skills ---")
    print(json.dumps(adapter.list_skills(), indent=2, ensure_ascii=False))
    print("\n--- Design Systems ---")
    print(json.dumps(adapter.list_design_systems(), indent=2, ensure_ascii=False))
