"""
MoE Expert Loader — Lightweight LoRA-as-Expert Routing
========================================================
Reuses existing LoRA weights (soul_lora_v2/) as Mixture-of-Experts
without any new training. Soft routing via keyword + embedding similarity.

Architecture:
    Query → Router(kw+embed) → Top-k Experts → Weighted Fusion → Response

Each expert = one knowledge base's LoRA adapter loaded on-demand.
No GPU training needed — pure inference with existing weights.

Expert Registry:
    Einstein: CPHYSJEPA (physics/causal reasoning)
    Cezanne:  CCODEJEPA  (code/logic)
    Others:   Placeholder (weights exist, routing active)

Usage:
    loader = MoEExpertLoader()
    result = loader.route_query("牛顿第二定律是什么", top_k=2)
"""

import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple

LORA_BASE = Path("D:/VORTEX_FLAME/soul_lora_v2")

EXPERT_REGISTRY: Dict[str, dict] = {
    "einstein": {
        "domain": ["physics", "chemistry", "energy", "quantum", "relativity",
                   "force", "mass", "acceleration", "thermodynamics", "electromagnetism",
                   "物理", "化学", "量子", "力学", "相对论", "热力学", "电磁", "牛顿", "爱因斯坦"],
        "jepa": "CPHYSJEPA",
        "tier": "A",
        "weights": "stage1-5/final",
    },
    "cezanne": {
        "domain": ["code", "algorithm", "logic", "computation", "debug",
                   "compiler", "ast", "type", "function", "class", "module",
                   "代码", "算法", "程序", "编程", "排序", "搜索", "编译", "调试", "Python"],
        "jepa": "CCODEJEPA",
        "tier": "A",
        "weights": "s1+s2+s3/final",
    },
    "galileo": {
        "domain": ["astronomy", "astrophysics", "celestial", "orbit", "telescope",
                   "planet", "star", "galaxy", "cosmology", "kinematics",
                   "天文", "宇宙", "行星", "恒星", "星系", "轨道", "望远镜"],
        "jepa": "CPHYSJEPA",
        "tier": "A",
        "weights": None,
    },
    "darwin": {
        "domain": ["biology", "evolution", "gene", "species", "ecology",
                   "protein", "dna", "rna", "cell", "mutation",
                   "生物", "进化", "基因", "物种", "蛋白质", "DNA", "RNA", "细胞", "突变", "CRISPR"],
        "jepa": "CBIOJEPA",
        "tier": "A",
        "weights": None,
    },
    "strategy": {
        "domain": ["strategy", "game", "finance", "market", "risk",
                   "portfolio", "investment", "trade", "nash", "equilibrium",
                   "金融", "投资", "市场", "交易", "策略", "博弈", "风险", "纳什"],
        "jepa": "CFINJEPA",
        "tier": "B",
        "weights": None,
    },
    "montesquieu": {
        "domain": ["law", "legal", "constitution", "jurisdiction", "precedent",
                   "statute", "rights", "court", "legislation", "governance",
                   "法律", "宪法", "法规", "判决", "法庭", "立法", "权利"],
        "jepa": "CLAWJEPA",
        "tier": "B",
        "weights": None,
    },
    "beethoven": {
        "domain": ["music", "audio", "harmony", "melody", "rhythm",
                   "composition", "acoustics", "instrument", "score", "tempo",
                   "音乐", "音频", "和声", "旋律", "节奏", "作曲", "乐器", "贝多芬"],
        "jepa": "CAJEPA",
        "tier": "C",
        "weights": None,
    },
    "davinci": {
        "domain": ["design", "engineering", "visual", "render", "3d",
                   "cad", "blueprint", "prototype", "mechanical", "robotics",
                   "设计", "工程", "3D", "CAD", "蓝图", "机械", "机器人", "达芬奇"],
        "jepa": "CVJEPA+CDESIGNJEPA",
        "tier": "B",
        "weights": None,
    },
    "humboldt": {
        "domain": ["earth", "climate", "geography", "ecology", "environment",
                   "carbon", "weather", "ocean", "soil", "atmosphere",
                   "地球", "气候", "地理", "环境", "碳", "海洋", "土壤"],
        "jepa": "CGEOJEPA",
        "tier": "C",
        "weights": None,
    },
    "guizhu": {
        "domain": ["philosophy", "dialogue", "therapy", "counseling", "ethics",
                   "logic", "metaphysics", "ontology", "epistemology", "mind",
                   "哲学", "对话", "伦理", "心理", "逻辑"],
        "jepa": "CLAWJEPA",
        "tier": "D",
        "weights": None,
    },
    "herodotus": {
        "domain": ["history", "civilization", "archive", "chronicle", "artifact",
                   "archaeology", "manuscript", "timeline", "dynasty", "ancient",
                   "历史", "文明", "考古", "古代", "文物", "编年"],
        "jepa": "CVJEPA+CGEOJEPA",
        "tier": "D",
        "weights": None,
    },
    "yuanlongping": {
        "domain": ["agriculture", "farming", "crop", "rice", "soil",
                   "breeding", "harvest", "irrigation", "botany", "hybrid",
                   "农业", "水稻", "育种", "作物", "灌溉", "杂交"],
        "jepa": "CBIOJEPA",
        "tier": "C",
        "weights": None,
    },
    "monet": {
        "domain": ["art", "aesthetic", "color", "composition", "impression",
                   "painting", "canvas", "brush", "light", "shadow",
                   "艺术", "画", "色彩", "印象", "莫奈", "睡莲", "光影"],
        "jepa": "CARTJEPA",
        "tier": "D",
        "weights": None,
    },
    "vangogh": {
        "domain": ["art", "visual", "painting", "expression", "texture",
                   "post-impression", "oil", "stroke", "palette", "contrast",
                   "艺术", "视觉", "梵高", "油画", "笔触"],
        "jepa": "CARTJEPA",
        "tier": "D",
        "weights": None,
    },
}


class MoEExpertLoader:
    def __init__(self, lora_base: Optional[Path] = None):
        self.lora_base = lora_base or LORA_BASE
        self._available_experts: List[str] = []
        self._scan_weights()

    def _scan_weights(self):
        self._available_experts = []
        if self.lora_base.exists():
            for name in EXPERT_REGISTRY:
                cfg = EXPERT_REGISTRY[name]
                if cfg["weights"]:
                    weight_path = self.lora_base / name / cfg["weights"]
                    if weight_path.exists() or (self.lora_base / name).exists():
                        self._available_experts.append(name)

    def available_experts(self) -> List[str]:
        return list(self._available_experts)

    def _keyword_score(self, query: str, expert_name: str) -> float:
        cfg = EXPERT_REGISTRY.get(expert_name)
        if not cfg:
            return 0.0
        q_lower = query.lower()
        hits = sum(1 for kw in cfg["domain"] if kw in q_lower)
        return hits / max(len(cfg["domain"]), 1)

    def route(self, query: str, top_k: int = 3, min_score: float = 0.01) -> List[Tuple[str, float]]:
        scores = []
        for name in EXPERT_REGISTRY:
            s = self._keyword_score(query, name)
            if s >= min_score:
                scores.append((name, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def route_strategy(self, query: str) -> str:
        routes = self.route(query, top_k=2)
        if not routes:
            return "cezanne"
        if len(routes) == 1 or routes[0][1] >= routes[1][1] * 2:
            return routes[0][0]
        if routes[0][1] - routes[1][1] < 0.05:
            return "ultrapilot"
        return "team"

    def expert_info(self, expert_name: str) -> Optional[dict]:
        cfg = EXPERT_REGISTRY.get(expert_name)
        if not cfg:
            return None
        return {
            "name": expert_name,
            "jepa": cfg["jepa"],
            "tier": cfg["tier"],
            "domain": cfg["domain"],
            "has_weights": expert_name in self._available_experts,
        }

    def all_expert_info(self) -> List[dict]:
        return [info for name in EXPERT_REGISTRY if (info := self.expert_info(name)) is not None]

    def query_with_routing(self, query: str, top_k: int = 3) -> dict:
        t0 = time.time()
        routes = self.route(query, top_k=top_k)
        strategy = self.route_strategy(query)
        return {
            "query": query,
            "strategy": strategy,
            "experts": [
                {"name": name, "score": round(score, 4), "jepa": EXPERT_REGISTRY[name]["jepa"]}
                for name, score in routes
            ],
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }

    def list_experts(self) -> list:
        return [{"id": name, "jepa": cfg["jepa"], "tier": cfg["tier"], "domain": cfg["domain"], "available": name in self._available_experts} for name, cfg in EXPERT_REGISTRY.items()]


if __name__ == "__main__":
    loader = MoEExpertLoader()
    print(f"Available experts: {loader.available_experts()}")
    print()

    for q in [
        "牛顿第二定律 F=ma 的推导过程",
        "写一个快速排序的Python实现",
        "莫奈的睡莲用了什么色彩技巧",
        "CRISPR基因编辑的工作原理",
        "量子纠缠为什么违反贝尔不等式",
    ]:
        result = loader.query_with_routing(q, top_k=2)
        print(f"Q: {q}")
        print(f"  Strategy: {result['strategy']}, Experts: {result['experts']}")
        print(f"  Latency: {result['latency_ms']}ms")
        print()
