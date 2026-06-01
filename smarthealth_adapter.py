"""
SmartHealth 健康适配器 — 圭酌灵魂的健康管理工具链
=================================================
桥接 E:\SmartHealth 项目，将卓壮健康智能营养顾问系统
接入 VORTEX FLAME 圭酌灵魂。

核心能力（来自 SmartHealth）：
  - 健康评估：10年龄段/10禁忌/36慢病/6特殊时期
  - 营养处方：17字段精确营养推荐
  - 运动处方：9字段运动方案
  - 菜谱生成：A/B/C/D套餐，25食材/周，植物蛋白30%
  - 用户记忆：per-user 隔离，FAISS 语义缓存
  - 卓壮API对接：菜品数据库、配餐系统、报价管理

路径1: SmartHealth Engine API（推荐）
  - FastAPI 服务端口 9100
  - 完整的健康评估→处方→菜谱流水线

路径2: 直接调用 SmartHealth 插件
  - health_rules.py: 业务规则引擎
  - prescription_engine.py: 处方生成
  - recipe_engine.py: 菜谱生成
  - zhuozhuang_api.py: 卓壮API客户端

路径3: GUI 感知（兜底）
  - Mano-P 操作健康管理系统界面

能力边界标注：
  - ✅ 可用：健康评估、营养处方、运动处方、菜谱生成、用户记忆
  - 🔄 开发中：中医体质辨识、药物交互检查
  - ⏳ 计划中：可穿戴设备数据接入、远程问诊

集成点：
  - soul_orchestrator: guizhu 灵魂注册 health_* 工具
  - mcp_sandbox_server: health_mcp 服务
  - harness_runtime: SmartHealth 端口白名单
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


SMARTHHEALTH_CONFIG = {
    "engine_url": "http://127.0.0.1:9100",
    "project_dir": r"E:\SmartHealth",
    "api_timeout_seconds": 30,
    "supported_endpoints": {
        "health_assess": "/api/assess",
        "nutrition_prescription": "/api/prescription/nutrition",
        "exercise_prescription": "/api/prescription/exercise",
        "recipe_daily": "/api/recipe/daily",
        "recipe_weekly": "/api/recipe/weekly",
        "user_profile": "/api/user/profile",
        "cache_query": "/api/cache/query",
    },
}


class HealthAdapter:
    """
    SmartHealth 健康适配器 — 圭酌灵魂专用

    桥接 SmartHealth 引擎，提供完整的健康管理能力：
    - 健康评估 → 营养处方 → 运动处方 → 菜谱生成
    - 用户记忆隔离
    - 卓壮API菜品数据库

    使用：
        adapter = HealthAdapter()
        adapter.assess_health(age=35, gender="男", height=175, weight=70)
        adapter.get_nutrition_prescription(user_profile)
        adapter.get_daily_recipe(user_profile, tier="B")
    """

    def __init__(self):
        self._engine_url = SMARTHHEALTH_CONFIG["engine_url"]
        self._task_counter = 0

    def status(self) -> dict:
        project_exists = Path(SMARTHHEALTH_CONFIG["project_dir"]).exists()
        engine_reachable = self._check_engine()

        return {
            "adapter": "HealthAdapter",
            "capability_level": "知识型+即时计算+业务系统",
            "project_dir": SMARTHHEALTH_CONFIG["project_dir"],
            "project_exists": project_exists,
            "engine_url": self._engine_url,
            "engine_reachable": engine_reachable,
            "boundary": {
                "可用": ["健康评估", "营养处方", "运动处方", "菜谱生成", "用户记忆", "卓壮API对接"],
                "开发中": ["中医体质辨识", "药物交互检查"],
                "计划中": ["可穿戴设备数据接入", "远程问诊"],
            },
            "tools": [
                "health_assess", "health_nutrition_prescription",
                "health_exercise_prescription", "health_daily_recipe",
                "health_weekly_recipe", "health_user_profile",
                "health_cache_query", "health_zhuozhuang_dishes",
                "health_zhuozhuang_quotation", "health_screenshot",
            ],
        }

    def _check_engine(self) -> bool:
        try:
            import urllib.request
            req = urllib.request.Request(f"{self._engine_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _api_call(self, method: str, endpoint: str, data: dict = None) -> dict:
        import urllib.request
        import urllib.error
        url = f"{self._engine_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        body = json.dumps(data, ensure_ascii=False).encode() if data else None
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=SMARTHHEALTH_CONFIG["api_timeout_seconds"]) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            return {"error": f"HTTP {e.code}", "detail": e.read().decode()[:200]}
        except Exception as e:
            return {"error": str(e)}

    def _try_import_plugin(self, module_name: str):
        project_dir = SMARTHHEALTH_CONFIG["project_dir"]
        plugins_dir = os.path.join(project_dir, "plugins")
        if plugins_dir not in os.sys.path:
            os.sys.path.insert(0, plugins_dir)
        try:
            return __import__(module_name)
        except ImportError as e:
            logger.warning(f"无法导入 SmartHealth 插件 {module_name}: {e}")
            return None

    def assess_health(self, age: int, gender: str, height: float, weight: float,
                      allergies: List[str] = None, chronic_diseases: List[str] = None,
                      special_period: str = "") -> dict:
        if self._check_engine():
            return self._api_call("POST", "/api/assess", {
                "age": age, "gender": gender, "height": height, "weight": weight,
                "allergies": allergies or [], "chronic_diseases": chronic_diseases or [],
                "special_period": special_period,
            })

        health_rules = self._try_import_plugin("health_rules")
        if health_rules:
            try:
                age_group = health_rules.get_age_group(age)
                body_type = health_rules.get_body_type(height, weight)
                return {
                    "status": "local_compute",
                    "age_group": age_group.value if hasattr(age_group, "value") else str(age_group),
                    "body_type": body_type.value if hasattr(body_type, "value") else str(body_type),
                    "bmi": round(weight / ((height / 100) ** 2), 1),
                }
            except Exception as e:
                return {"error": f"本地计算失败: {e}"}

        bmi = round(weight / ((height / 100) ** 2), 1)
        return {
            "status": "fallback_compute",
            "bmi": bmi,
            "bmi_category": "偏瘦" if bmi < 18.5 else "正常" if bmi < 24 else "超重" if bmi < 28 else "肥胖",
            "note": "SmartHealth 引擎未启动，使用基础BMI计算",
        }

    def get_nutrition_prescription(self, user_profile: dict) -> dict:
        if self._check_engine():
            return self._api_call("POST", "/api/prescription/nutrition", user_profile)

        prescription_engine = self._try_import_plugin("prescription_engine")
        if prescription_engine:
            try:
                health_rules = self._try_import_plugin("health_rules")
                if health_rules:
                    profile = health_rules.UserProfile(**user_profile)
                    result = prescription_engine.generate_full_prescription(profile)
                    return {"status": "local_compute", "prescription": result}
            except Exception as e:
                return {"error": f"本地处方生成失败: {e}"}

        return {"error": "SmartHealth 引擎未启动，且本地插件不可用"}

    def get_exercise_prescription(self, user_profile: dict) -> dict:
        if self._check_engine():
            return self._api_call("POST", "/api/prescription/exercise", user_profile)
        return {"error": "SmartHealth 引擎未启动"}

    def get_daily_recipe(self, user_profile: dict, tier: str = "B") -> dict:
        if self._check_engine():
            return self._api_call("POST", "/api/recipe/daily", {**user_profile, "tier": tier})

        recipe_engine = self._try_import_plugin("recipe_engine")
        if recipe_engine:
            try:
                health_rules = self._try_import_plugin("health_rules")
                if health_rules:
                    profile = health_rules.UserProfile(**user_profile)
                    result = recipe_engine.generate_daily_recipe(profile, tier)
                    return {"status": "local_compute", "recipe": result}
            except Exception as e:
                return {"error": f"本地菜谱生成失败: {e}"}

        return {"error": "SmartHealth 引擎未启动，且本地插件不可用"}

    def get_weekly_recipe(self, user_profile: dict) -> dict:
        if self._check_engine():
            return self._api_call("POST", "/api/recipe/weekly", user_profile)
        return {"error": "SmartHealth 引擎未启动"}

    def get_user_profile(self, user_id: str) -> dict:
        if self._check_engine():
            return self._api_call("GET", f"/api/user/profile?user_id={user_id}")
        return {"error": "SmartHealth 引擎未启动"}

    def query_cache(self, query: str, threshold: float = 0.85) -> dict:
        if self._check_engine():
            return self._api_call("POST", "/api/cache/query", {"query": query, "threshold": threshold})
        return {"error": "SmartHealth 引擎未启动"}

    def get_zhuozhuang_dishes(self, dish_type: str = None) -> dict:
        try:
            zhuozhuang = self._try_import_plugin("zhuozhuang_api")
            if zhuozhuang:
                api = zhuozhuang.ZhuozhuangAPI()
                dishes = api.get_dishes(dish_type)
                return {"status": "success", "dishes": dishes}
        except Exception as e:
            return {"error": f"卓壮API调用失败: {e}"}
        return {"error": "卓壮API插件不可用"}

    def get_zhuozhuang_quotation(self, month: str) -> dict:
        try:
            zhuozhuang = self._try_import_plugin("zhuozhuang_api")
            if zhuozhuang:
                api = zhuozhuang.ZhuozhuangAPI()
                quotations = api.get_quotations(month)
                return {"status": "success", "quotations": quotations}
        except Exception as e:
            return {"error": f"卓壮API调用失败: {e}"}
        return {"error": "卓壮API插件不可用"}


HEALTH_SKILL_DEFINITION = {
    "skill_id": "health_management",
    "name": "健康管理工具链",
    "description": "圭酌灵魂专用：健康评估+营养处方+运动处方+菜谱生成+卓壮API对接",
    "soul_mapping": ["guizhu", "darwin"],
    "tools": [
        "health_assess", "health_nutrition_prescription",
        "health_exercise_prescription", "health_daily_recipe",
        "health_weekly_recipe", "health_user_profile",
        "health_cache_query", "health_zhuozhuang_dishes",
        "health_zhuozhuang_quotation", "health_screenshot",
    ],
    "connectors": [
        {"name": "smarthealth_engine", "type": "rest_api", "base_url": "http://127.0.0.1:9100"},
        {"name": "smarthealth_plugins", "type": "python", "path": "E:\\SmartHealth\\plugins"},
        {"name": "zhuozhuang_api", "type": "rest_api", "note": "卓壮健康菜品数据库"},
        {"name": "mano_p_gui", "type": "gui_perception", "fallback": True},
    ],
    "boundary": {
        "可用": ["健康评估", "营养处方", "运动处方", "菜谱生成", "用户记忆", "卓壮API对接"],
        "开发中": ["中医体质辨识", "药物交互检查"],
        "计划中": ["可穿戴设备数据接入", "远程问诊"],
    },
}

_adapter_instance: Optional[HealthAdapter] = None


def get_adapter() -> HealthAdapter:
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = HealthAdapter()
    return _adapter_instance
