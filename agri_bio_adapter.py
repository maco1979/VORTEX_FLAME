"""
农业/生物适配器 — 袁隆平灵魂的农业科研工具链
=================================================
核心定位：知识型灵魂 + 即时计算能力

路径1: Python 农业计算（即时计算，MCP 包装）
  - GDD (生长度日) 模型计算
  - 作物产量预测模型
  - 土壤养分平衡计算
  - 灌溉需水量估算
  - 农时/节气计算

路径2: 生物信息学（Biopython）
  - 序列比对、基因组数据处理
  - 品种遗传分析
  - 基因频率计算 (Hardy-Weinberg)

路径3: 知识驱动（RAG 知识中心）
  - 农业知识库问答
  - 作物栽培技术
  - 病虫害识别建议

能力边界标注：
  - ✅ 可用：农业计算、生物信息学分析、知识问答
  - 🔄 开发中：遥感影像分析、气象数据实时接入
  - ⏳ 计划中：GIS 地块管理、农机调度

集成点：
  - soul_orchestrator: yuanlongping 灵魂注册 agri_* 工具
  - mcp_sandbox_server: agri_mcp 服务
  - harness_runtime: 计算端口白名单
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CropType(Enum):
    RICE = "水稻"
    WHEAT = "小麦"
    CORN = "玉米"
    SOYBEAN = "大豆"
    COTTON = "棉花"
    POTATO = "土豆"
    SWEET_POTATO = "红薯"
    RAPESEED = "油菜"
    PEANUT = "花生"
    SUGARCANE = "甘蔗"


class SoilType(Enum):
    LOAM = "壤土"
    CLAY = "黏土"
    SANDY = "沙土"
    SILT = "粉土"
    PEAT = "泥炭土"
    LATERITE = "红壤"
    BLACK_SOIL = "黑土"
    SALINE = "盐碱土"


class GrowthStage(Enum):
    SOWING = "播种期"
    SEEDLING = "苗期"
    TILLERING = "分蘖期"
    JOINTING = "拔节期"
    HEADING = "抽穗期"
    FLOWERING = "开花期"
    FILLING = "灌浆期"
    MATURITY = "成熟期"
    HARVEST = "收获期"


AGRI_CONFIG = {
    "base_temp_celsius": {
        "rice": 10.0,
        "wheat": 0.0,
        "corn": 10.0,
        "soybean": 10.0,
        "cotton": 15.0,
        "potato": 7.0,
    },
    "gdd_targets": {
        "rice": {"seedling": 200, "tillering": 600, "jointing": 900, "heading": 1200, "maturity": 1600},
        "wheat": {"seedling": 150, "tillering": 500, "jointing": 800, "heading": 1100, "maturity": 1500},
        "corn": {"seedling": 250, "jointing": 700, "tasseling": 1000, "maturity": 1500},
    },
    "kc_values": {
        "rice": {"initial": 1.05, "mid": 1.20, "late": 0.90},
        "wheat": {"initial": 0.30, "mid": 1.15, "late": 0.25},
        "corn": {"initial": 0.30, "mid": 1.20, "late": 0.35},
        "soybean": {"initial": 0.30, "mid": 1.15, "late": 0.50},
    },
    "npp_factors": {
        "rice": 0.45,
        "wheat": 0.40,
        "corn": 0.50,
        "soybean": 0.35,
    },
    "solar_term_data": [
        ("小寒", 1, 6), ("大寒", 1, 20), ("立春", 2, 4), ("雨水", 2, 19),
        ("惊蛰", 3, 6), ("春分", 3, 21), ("清明", 4, 5), ("谷雨", 4, 20),
        ("立夏", 5, 6), ("小满", 5, 21), ("芒种", 6, 6), ("夏至", 6, 21),
        ("小暑", 7, 7), ("大暑", 7, 23), ("立秋", 8, 7), ("处暑", 8, 23),
        ("白露", 9, 8), ("秋分", 9, 23), ("寒露", 10, 8), ("霜降", 10, 23),
        ("立冬", 11, 7), ("小雪", 11, 22), ("大雪", 12, 7), ("冬至", 12, 22),
    ],
}


@dataclass
class WeatherRecord:
    date: str
    temp_max: float
    temp_min: float
    temp_avg: float
    humidity: float = 0.0
    rainfall_mm: float = 0.0
    solar_radiation_mj: float = 0.0


@dataclass
class SoilNutrient:
    nitrogen_ppm: float = 0.0
    phosphorus_ppm: float = 0.0
    potassium_ppm: float = 0.0
    organic_matter_pct: float = 0.0
    ph: float = 7.0
    cec: float = 0.0


@dataclass
class CropRecommendation:
    crop: CropType
    suitable: bool
    gdd_accumulated: float
    gdd_target: float
    growth_stage: str
    irrigation_need_mm: float
    fertilizer_n_kg_ha: float
    fertilizer_p_kg_ha: float
    fertilizer_k_kg_ha: float
    notes: str = ""


@dataclass
class AgriTask:
    task_id: str
    description: str
    soul: str
    actions: List[Dict[str, Any]] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "pending"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


def calc_gdd(temp_max: float, temp_min: float, base_temp: float) -> float:
    avg = (temp_max + temp_min) / 2.0
    gdd = avg - base_temp
    return max(0.0, gdd)


def calc_gdd_accumulated(weather_records: List[WeatherRecord], base_temp: float) -> float:
    return sum(calc_gdd(r.temp_max, r.temp_min, base_temp) for r in weather_records)


def calc_et0(temperature_c: float, humidity_pct: float,
             solar_radiation_mj: float, wind_speed_ms: float = 2.0,
             altitude_m: float = 0.0) -> float:
    delta = 4098 * (0.6108 * (17.27 * temperature_c / (temperature_c + 237.3)) ** 1) / (temperature_c + 237.3) ** 2
    gamma = 0.665e-3 * 101.3 * ((293 - 0.0065 * altitude_m) / 293) ** 5.26 / 0.665e-3
    es = 0.6108 * (17.27 * temperature_c / (temperature_c + 237.3))
    ea = es * humidity_pct / 100.0
    ra = solar_radiation_mj * 0.408
    et0 = (0.408 * delta * (ra - 0) + gamma * (900 / (temperature_c + 273)) * wind_speed_ms * (es - ea)) / (delta + gamma * (1 + 0.34 * wind_speed_ms))
    return max(0.0, et0)


def get_current_solar_term() -> dict:
    from datetime import datetime
    now = datetime.now()
    month, day = now.month, now.day
    current = None
    next_term = None
    for i, (name, m, d) in enumerate(AGRI_CONFIG["solar_term_data"]):
        if (month, day) >= (m, d):
            current = (name, m, d)
        if next_term is None and (month, day) < (m, d):
            next_term = (name, m, d)
    if next_term is None:
        next_term = AGRI_CONFIG["solar_term_data"][0]
    return {
        "current": current,
        "next": next_term,
        "date": f"{month}月{day}日",
    }


class AgriBioAdapter:
    """
    农业/生物适配器 — 袁隆平灵魂专用

    核心定位：知识型 + 即时计算
    - GDD 生长度日模型
    - 作物产量预测
    - 土壤养分平衡
    - 灌溉需水量估算
    - 生物信息学分析

    使用：
        adapter = AgriBioAdapter()
        adapter.calc_gdd_report(weather_data, "rice")
        adapter.calc_irrigation("rice", et0=5.2, stage="mid")
        adapter.calc_fertilizer(soil, "rice", target_yield=9000)
    """

    def __init__(self):
        self._task_counter = 0

    def status(self) -> dict:
        return {
            "adapter": "AgriBioAdapter",
            "capability_level": "知识型+即时计算",
            "boundary": {
                "可用": ["GDD计算", "灌溉估算", "施肥推荐", "节气查询", "生物信息学"],
                "开发中": ["遥感影像分析", "气象数据实时接入"],
                "计划中": ["GIS地块管理", "农机调度"],
            },
            "supported_crops": [c.value for c in CropType],
            "tools": [
                "agri_calc_gdd", "agri_calc_gdd_report",
                "agri_calc_irrigation", "agri_calc_fertilizer",
                "agri_calc_yield_estimate", "agri_get_solar_term",
                "agri_sequence_analyze", "agri_genetic_frequency",
                "agri_screenshot",
            ],
        }

    def calc_gdd(self, temp_max: float, temp_min: float, crop: str = "rice") -> dict:
        base = AGRI_CONFIG["base_temp_celsius"].get(crop, 10.0)
        gdd = calc_gdd(temp_max, temp_min, base)
        return {"gdd": gdd, "base_temp": base, "crop": crop, "temp_avg": (temp_max + temp_min) / 2}

    def calc_gdd_report(self, weather_records: List[Dict], crop: str = "rice") -> dict:
        base = AGRI_CONFIG["base_temp_celsius"].get(crop, 10.0)
        records = [WeatherRecord(**r) for r in weather_records]
        accumulated = calc_gdd_accumulated(records, base)
        targets = AGRI_CONFIG["gdd_targets"].get(crop, {})
        current_stage = "播种前"
        for stage_name, target_gdd in targets.items():
            if accumulated >= target_gdd:
                current_stage = stage_name
        return {
            "crop": crop,
            "base_temp": base,
            "accumulated_gdd": round(accumulated, 1),
            "current_stage": current_stage,
            "gdd_targets": targets,
            "days_recorded": len(records),
        }

    def calc_irrigation(self, crop: str, et0: float, stage: str = "mid",
                        rainfall_mm: float = 0.0, efficiency: float = 0.85) -> dict:
        kc_values = AGRI_CONFIG["kc_values"].get(crop, {"initial": 0.3, "mid": 1.0, "late": 0.5})
        kc = kc_values.get(stage, 1.0)
        etc = et0 * kc
        irrigation_need = max(0.0, (etc - rainfall_mm) / efficiency)
        return {
            "crop": crop,
            "stage": stage,
            "kc": kc,
            "et0_mm": et0,
            "etc_mm": round(etc, 1),
            "rainfall_mm": rainfall_mm,
            "irrigation_need_mm": round(irrigation_need, 1),
            "efficiency": efficiency,
        }

    def calc_fertilizer(self, soil: Dict, crop: str = "rice",
                        target_yield_kg_ha: float = 9000) -> dict:
        nutrient = SoilNutrient(**soil)
        npp = AGRI_CONFIG["npp_factors"].get(crop, 0.4)
        n_uptake = target_yield_kg_ha * npp / 100.0 * 0.017
        n_soil_supply = nutrient.nitrogen_ppm * 0.3 * 2.25
        n_fertilizer = max(0.0, (n_uptake - n_soil_supply) / 0.35)

        p2o5_uptake = target_yield_kg_ha * 0.008
        p_soil_supply = nutrient.phosphorus_ppm * 0.15 * 2.25
        p2o5_fertilizer = max(0.0, (p2o5_uptake - p_soil_supply) / 0.20)

        k2o_uptake = target_yield_kg_ha * 0.020
        k_soil_supply = nutrient.potassium_ppm * 0.20 * 2.25
        k2o_fertilizer = max(0.0, (k2o_uptake - k_soil_supply) / 0.45)

        return {
            "crop": crop,
            "target_yield_kg_ha": target_yield_kg_ha,
            "soil_status": {
                "N_ppm": nutrient.nitrogen_ppm,
                "P_ppm": nutrient.phosphorus_ppm,
                "K_ppm": nutrient.potassium_ppm,
                "OM_pct": nutrient.organic_matter_pct,
                "pH": nutrient.ph,
            },
            "recommendation_kg_ha": {
                "N": round(n_fertilizer, 1),
                "P2O5": round(p2o5_fertilizer, 1),
                "K2O": round(k2o_fertilizer, 1),
            },
        }

    def calc_yield_estimate(self, crop: str, gdd_accumulated: float,
                            rainfall_total_mm: float, n_fertilizer_kg_ha: float,
                            planting_density_m2: float = 25.0) -> dict:
        base_yield = {"rice": 7500, "wheat": 6000, "corn": 9000, "soybean": 3000}.get(crop, 5000)
        gdd_factor = min(1.2, gdd_accumulated / 1500.0)
        rain_factor = min(1.1, rainfall_total_mm / 800.0) if rainfall_total_mm < 800 else 1.1 - (rainfall_total_mm - 800) / 5000.0
        n_factor = min(1.15, n_fertilizer_kg_ha / 200.0) if n_fertilizer_kg_ha > 0 else 0.7
        density_factor = min(1.1, planting_density_m2 / 25.0)
        estimated = base_yield * gdd_factor * rain_factor * n_factor * density_factor
        return {
            "crop": crop,
            "estimated_yield_kg_ha": round(estimated, 0),
            "factors": {
                "gdd": round(gdd_factor, 3),
                "rainfall": round(rain_factor, 3),
                "nitrogen": round(n_factor, 3),
                "density": round(density_factor, 3),
            },
        }

    def get_solar_term(self) -> dict:
        return get_current_solar_term()

    def sequence_analyze(self, sequence: str, analysis_type: str = "gc_content") -> dict:
        start = time.time()
        seq = sequence.upper().replace(" ", "").replace("\n", "")
        if analysis_type == "gc_content":
            gc = seq.count("G") + seq.count("C")
            total = len(seq)
            content = gc / total if total > 0 else 0
            return {"type": "gc_content", "gc_percent": round(content * 100, 2), "length": total, "compute_time_ms": round((time.time() - start) * 1000, 1)}
        elif analysis_type == "complement":
            comp_map = {"A": "T", "T": "A", "G": "C", "C": "G"}
            complement = "".join(comp_map.get(b, "N") for b in seq)
            return {"type": "complement", "complement": complement, "length": len(seq)}
        elif analysis_type == "codon_table":
            codons = [seq[i:i+3] for i in range(0, len(seq) - 2, 3)]
            from collections import Counter
            freq = dict(Counter(codons))
            return {"type": "codon_frequency", "codon_count": len(codons), "frequencies": freq}
        return {"error": f"未知分析类型: {analysis_type}"}

    def genetic_frequency(self, p: float, q: float, population: int = 1000) -> dict:
        if abs(p + q - 1.0) > 0.01:
            return {"error": f"等位基因频率之和必须为1: p={p}, q={q}, p+q={p+q}"}
        return {
            "hardy_weinberg": {
                "AA": round(p * p, 4),
                "Aa": round(2 * p * q, 4),
                "aa": round(q * q, 4),
            },
            "expected_counts": {
                "AA": round(p * p * population),
                "Aa": round(2 * p * q * population),
                "aa": round(q * q * population),
            },
            "population": population,
        }


AGRI_SKILL_DEFINITION = {
    "skill_id": "agri_bio",
    "name": "农业生物计算",
    "description": "袁隆平灵魂专用：GDD模型+灌溉估算+施肥推荐+产量预测+生物信息学",
    "soul_mapping": ["yuanlongping", "darwin"],
    "tools": [
        "agri_calc_gdd", "agri_calc_gdd_report",
        "agri_calc_irrigation", "agri_calc_fertilizer",
        "agri_calc_yield_estimate", "agri_get_solar_term",
        "agri_sequence_analyze", "agri_genetic_frequency",
        "agri_screenshot",
    ],
    "connectors": [
        {"name": "python_agri", "type": "python", "packages": ["numpy", "pandas"]},
        {"name": "biopython_mcp", "type": "python", "packages": ["biopython"], "soul": "darwin"},
        {"name": "mano_p_gui", "type": "gui_perception", "fallback": True},
    ],
    "boundary": {
        "可用": ["GDD计算", "灌溉估算", "施肥推荐", "节气查询", "生物信息学"],
        "开发中": ["遥感影像分析", "气象数据实时接入"],
        "计划中": ["GIS地块管理", "农机调度"],
    },
}

_adapter_instance: Optional[AgriBioAdapter] = None


def get_adapter() -> AgriBioAdapter:
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = AgriBioAdapter()
    return _adapter_instance
