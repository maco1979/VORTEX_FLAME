"""
MCP 沙箱服务 — 统一计算工具注册与调用
=========================================
基于 Model Context Protocol (MCP) 标准，将所有专业适配器
包装为 MCP 工具，供灵魂按需调用。

架构：
  ┌──────────────────────────────────────────────────────┐
  │                  MCP Sandbox Server                   │
  │                                                       │
  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
  │  │  sympy_mcp  │  │  agri_mcp   │  │ health_mcp  │  │
  │  │  (爱因斯坦)  │  │  (袁隆平)    │  │  (圭酌)     │  │
  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │
  │         │                │                │          │
  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
  │  │ blender_mcp │  │ design_mcp  │  │ ableton_mcp │  │
  │  │  (达芬奇)    │  │  (莫奈/梵高) │  │  (贝多芬)   │  │
  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │
  │         │                │                │          │
  │         └────────────────┼────────────────┘          │
  │                          ▼                            │
  │              ┌─────────────────────┐                  │
  │              │  统一调度器          │                  │
  │              │  - 灵魂→工具映射     │                  │
  │              │  - 安全白名单检查    │                  │
  │              │  - 超时/重试控制     │                  │
  │              │  - 结果缓存         │                  │
  │              └─────────────────────┘                  │
  └──────────────────────────────────────────────────────┘

MCP 工具注册表（首批 4 个优先 + 2 个扩展）：
  P0 (本周): sympy_mcp, backtrader_mcp, biopython_mcp, geopandas_mcp
  P1 (MVP前): blender_mcp, health_mcp, ableton_mcp, design_mcp

集成点：
  - vf_api_server.py: /api/mcp/* 端点
  - soul_orchestrator.py: 灵魂调用 mcp_tool_execute()
  - harness_runtime.py: MCP 端口白名单
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MCPToolStatus(Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    PENDING = "pending"


class MCPToolCategory(Enum):
    COMPUTE = "compute"
    CREATIVE = "creative"
    HEALTH = "health"
    AGRI = "agriculture"
    DESIGN = "design"
    MUSIC = "music"
    THREE_D = "3d"
    GIS = "gis"


MCP_SERVER_CONFIG = {
    "host": "127.0.0.1",
    "port": 9500,
    "max_concurrent_tasks": 10,
    "default_timeout_seconds": 120,
    "cache_enabled": True,
    "cache_ttl_seconds": 300,
    "result_max_size_bytes": 1_000_000,
}


@dataclass
class MCPTool:
    tool_id: str
    name: str
    description: str
    category: MCPToolCategory
    soul_mapping: List[str]
    status: MCPToolStatus = MCPToolStatus.AVAILABLE
    input_schema: Dict[str, Any] = field(default_factory=dict)
    boundary: Dict[str, List[str]] = field(default_factory=dict)
    requires_network: bool = False
    requires_filesystem: bool = False
    timeout_seconds: int = 120
    call_count: int = 0
    last_called: Optional[str] = None


@dataclass
class MCPToolCall:
    call_id: str
    tool_id: str
    soul: str
    arguments: Dict[str, Any]
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    compute_time_ms: float = 0.0


MCP_TOOL_REGISTRY: Dict[str, MCPTool] = {}


def _register_tool(tool: MCPTool):
    MCP_TOOL_REGISTRY[tool.tool_id] = tool


def _init_registry():
    if MCP_TOOL_REGISTRY:
        return

    _register_tool(MCPTool(
        tool_id="sympy_mcp",
        name="SymPy 符号计算",
        description="爱因斯坦/纳什灵魂专用：符号推导、方程求解、微积分、矩阵运算",
        category=MCPToolCategory.COMPUTE,
        soul_mapping=["einstein", "strategy"],
        status=MCPToolStatus.AVAILABLE,
        input_schema={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "数学表达式"},
                "operation": {"type": "string", "enum": ["evaluate", "simplify", "diff", "integrate", "solve", "expand"], "description": "操作类型"},
                "variables": {"type": "object", "description": "变量值映射"},
            },
            "required": ["expression"],
        },
        boundary={"可用": ["符号推导", "方程求解", "微积分", "矩阵运算"], "开发中": ["LaTeX渲染"], "计划中": ["Wolfram Alpha桥接"]},
    ))

    _register_tool(MCPTool(
        tool_id="backtrader_mcp",
        name="量化回测引擎",
        description="纳什/爱因斯坦灵魂专用：策略回测、风险指标、收益分析",
        category=MCPToolCategory.COMPUTE,
        soul_mapping=["strategy", "einstein"],
        status=MCPToolStatus.AVAILABLE,
        input_schema={
            "type": "object",
            "properties": {
                "strategy_code": {"type": "string", "description": "策略Python代码"},
                "symbol": {"type": "string", "description": "标的代码"},
                "start_date": {"type": "string", "description": "回测起始日"},
                "end_date": {"type": "string", "description": "回测结束日"},
                "initial_cash": {"type": "number", "description": "初始资金"},
            },
            "required": ["strategy_code", "symbol"],
        },
        boundary={"可用": ["策略回测", "风险指标"], "开发中": ["实盘模拟"], "计划中": ["多因子模型"]},
    ))

    _register_tool(MCPTool(
        tool_id="biopython_mcp",
        name="生物信息学分析",
        description="达尔文/袁隆平灵魂专用：序列分析、基因组数据处理、遗传频率计算",
        category=MCPToolCategory.AGRI,
        soul_mapping=["darwin", "yuanlongping"],
        status=MCPToolStatus.AVAILABLE,
        input_schema={
            "type": "object",
            "properties": {
                "sequence": {"type": "string", "description": "DNA/RNA/蛋白质序列"},
                "analysis_type": {"type": "string", "enum": ["gc_content", "complement", "codon_table", "alignment"], "description": "分析类型"},
            },
        },
        boundary={"可用": ["序列分析", "遗传频率"], "开发中": ["多序列比对"], "计划中": ["系统发育树"]},
    ))

    _register_tool(MCPTool(
        tool_id="geopandas_mcp",
        name="地理空间分析",
        description="洪堡灵魂专用：GIS数据处理、地理可视化、空间统计",
        category=MCPToolCategory.GIS,
        soul_mapping=["humboldt"],
        status=MCPToolStatus.PENDING,
        input_schema={
            "type": "object",
            "properties": {
                "data_path": {"type": "string", "description": "GeoJSON/Shapefile路径"},
                "operation": {"type": "string", "enum": ["read", "plot", "buffer", "intersect", "stats"], "description": "操作类型"},
            },
        },
        boundary={"可用": [], "开发中": ["GeoPandas集成"], "计划中": ["卫星影像处理", "空间插值"]},
    ))

    _register_tool(MCPTool(
        tool_id="agri_mcp",
        name="农业计算引擎",
        description="袁隆平灵魂专用：GDD模型、灌溉估算、施肥推荐、产量预测、节气查询",
        category=MCPToolCategory.AGRI,
        soul_mapping=["yuanlongping", "darwin"],
        status=MCPToolStatus.AVAILABLE,
        input_schema={
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["gdd", "irrigation", "fertilizer", "yield", "solar_term", "genetic"], "description": "计算类型"},
                "params": {"type": "object", "description": "计算参数"},
            },
            "required": ["operation"],
        },
        boundary={"可用": ["GDD计算", "灌溉估算", "施肥推荐", "节气查询"], "开发中": ["遥感影像"], "计划中": ["GIS地块管理"]},
    ))

    _register_tool(MCPTool(
        tool_id="health_mcp",
        name="健康管理引擎",
        description="圭酌灵魂专用：健康评估、营养处方、运动处方、菜谱生成、卓壮API",
        category=MCPToolCategory.HEALTH,
        soul_mapping=["guizhu", "darwin"],
        status=MCPToolStatus.AVAILABLE,
        input_schema={
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["assess", "nutrition", "exercise", "recipe_daily", "recipe_weekly", "dishes", "quotation"], "description": "操作类型"},
                "params": {"type": "object", "description": "操作参数"},
            },
            "required": ["operation"],
        },
        boundary={"可用": ["健康评估", "营养处方", "菜谱生成", "卓壮API"], "开发中": ["中医体质辨识"], "计划中": ["可穿戴设备接入"]},
    ))

    _register_tool(MCPTool(
        tool_id="blender_mcp",
        name="Blender 3D 建模",
        description="达芬奇灵魂专用：3D建模、材质、灯光、渲染、导出（概念层+脚本生成）",
        category=MCPToolCategory.THREE_D,
        soul_mapping=["davinci"],
        status=MCPToolStatus.AVAILABLE,
        input_schema={
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["create_primitive", "import", "material", "light", "camera", "render", "export", "scene_info"], "description": "操作类型"},
                "params": {"type": "object", "description": "操作参数"},
            },
            "required": ["operation"],
        },
        boundary={"可用": ["建模脚本生成", "材质设置", "渲染"], "开发中": ["实时预览"], "计划中": ["雕刻模式GUI操作"]},
    ))

    _register_tool(MCPTool(
        tool_id="design_mcp",
        name="视觉设计工具",
        description="莫奈/梵高灵魂专用：Figma API、ImageMagick、FFmpeg（概念层+批量处理）",
        category=MCPToolCategory.DESIGN,
        soul_mapping=["monet", "vangogh"],
        status=MCPToolStatus.AVAILABLE,
        input_schema={
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["figma_file", "figma_image", "image_resize", "image_convert", "color_adjust", "extract_frames", "create_gif"], "description": "操作类型"},
                "params": {"type": "object", "description": "操作参数"},
            },
            "required": ["operation"],
        },
        boundary={"可用": ["Figma读取", "图片处理", "色彩调整"], "开发中": ["ComfyUI集成"], "计划中": ["PS/UXP插件"]},
    ))

    _register_tool(MCPTool(
        tool_id="ableton_mcp",
        name="Ableton Live 音乐制作",
        description="贝多芬灵魂专用：MIDI生成、音轨控制、混音、导出（概念层+OSC控制）",
        category=MCPToolCategory.MUSIC,
        soul_mapping=["beethoven"],
        status=MCPToolStatus.AVAILABLE,
        input_schema={
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["connect", "create_track", "set_tempo", "add_notes", "set_volume", "load_device", "fire_clip", "playback", "export"], "description": "操作类型"},
                "params": {"type": "object", "description": "操作参数"},
            },
            "required": ["operation"],
        },
        boundary={"可用": ["OSC控制", "MIDI生成"], "开发中": ["MIDIUtil本地生成"], "计划中": ["MuseScore乐谱导出"]},
    ))

    _register_tool(MCPTool(
        tool_id="science_plot_mcp",
        name="科学数据可视化",
        description="爱因斯坦/伽利略灵魂专用：函数绘图、数据可视化、统计图表",
        category=MCPToolCategory.COMPUTE,
        soul_mapping=["einstein", "galileo", "darwin", "humboldt"],
        status=MCPToolStatus.AVAILABLE,
        input_schema={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "数学表达式或数据"},
                "plot_type": {"type": "string", "enum": ["function", "scatter", "histogram", "heatmap", "contour"], "description": "图表类型"},
                "title": {"type": "string", "description": "图表标题"},
                "x_range": {"type": "array", "items": {"type": "number"}, "description": "X轴范围"},
            },
        },
        boundary={"可用": ["函数绘图", "数据可视化"], "开发中": ["3D曲面图"], "计划中": ["交互式Plotly图表"]},
    ))

    _register_tool(MCPTool(
        tool_id="aigen_txt2img",
        name="AI图像生成",
        description="莫奈/梵高/达芬奇灵魂专用：ComfyUI/DALL-E/FLUX图像生成",
        category=MCPToolCategory.CREATIVE,
        soul_mapping=["monet", "vangogh", "davinci"],
        status=MCPToolStatus.AVAILABLE,
        input_schema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "图像提示词"},
                "negative_prompt": {"type": "string", "description": "反向提示词"},
                "width": {"type": "integer", "description": "宽度"},
                "height": {"type": "integer", "description": "高度"},
                "model": {"type": "string", "description": "模型名称(flux1-dev/sd3/dalle)"},
            },
            "required": ["prompt"],
        },
        boundary={"可用": ["ComfyUI本地", "DALL-E API", "FLUX"], "开发中": [], "计划中": ["Midjourney"]},
    ))

    _register_tool(MCPTool(
        tool_id="aigen_txt2music",
        name="AI音乐生成",
        description="贝多芬灵魂专用：MIDIUtil本地作曲/Suno云端生成/MusicGen",
        category=MCPToolCategory.MUSIC,
        soul_mapping=["beethoven"],
        status=MCPToolStatus.AVAILABLE,
        input_schema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "音乐描述"},
                "duration_seconds": {"type": "integer", "description": "时长(秒)"},
                "bpm": {"type": "integer", "description": "BPM"},
                "key": {"type": "string", "description": "调性"},
                "genre": {"type": "string", "description": "风格"},
            },
            "required": ["prompt"],
        },
        boundary={"可用": ["MIDIUtil本地", "MusicGen本地"], "开发中": ["Suno API"], "计划中": ["Udio"]},
    ))

    _register_tool(MCPTool(
        tool_id="aigen_txt2video",
        name="AI视频生成",
        description="达芬奇/莫奈灵魂专用：Kling/Runway/Pika/Sora视频生成",
        category=MCPToolCategory.CREATIVE,
        soul_mapping=["davinci", "monet", "vangogh"],
        status=MCPToolStatus.PENDING,
        input_schema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "视频描述"},
                "duration_seconds": {"type": "integer", "description": "时长(秒)"},
            },
            "required": ["prompt"],
        },
        boundary={"可用": [], "开发中": ["Kling API", "Hailuo API"], "计划中": ["Runway", "Sora", "Pika"]},
    ))

    _register_tool(MCPTool(
        tool_id="aigen_txt2voice",
        name="AI语音合成",
        description="圭酌/贝多芬灵魂专用：ElevenLabs/ChatTTS语音合成",
        category=MCPToolCategory.HEALTH,
        soul_mapping=["guizhu", "beethoven"],
        status=MCPToolStatus.PENDING,
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要合成的文本"},
                "language": {"type": "string", "description": "语言(zh/en)"},
            },
            "required": ["text"],
        },
        boundary={"可用": [], "开发中": ["ElevenLabs API"], "计划中": ["ChatTTS本地"]},
    ))

    _register_tool(MCPTool(
        tool_id="aigen_txt23d",
        name="AI 3D生成",
        description="达芬奇灵魂专用：Tripo3D/Meshy文本转3D模型",
        category=MCPToolCategory.THREE_D,
        soul_mapping=["davinci"],
        status=MCPToolStatus.PENDING,
        input_schema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "3D模型描述"},
                "style": {"type": "string", "description": "风格(realistic/low-poly)"},
            },
            "required": ["prompt"],
        },
        boundary={"可用": [], "开发中": ["Tripo3D API"], "计划中": ["Meshy", "Blender桥接"]},
    ))

    _register_tool(MCPTool(
        tool_id="vf_sec_audit",
        name="VF-SEC 安全审计",
        description="Cezanne/Strategy灵魂专用：16阶段安全审计管线，commit前全面扫描",
        category=MCPToolCategory.COMPUTE,
        soul_mapping=["cezanne", "strategy", "montesquieu"],
        status=MCPToolStatus.AVAILABLE,
        input_schema={
            "type": "object",
            "properties": {
                "project_path": {"type": "string", "description": "项目路径"},
                "mode": {"type": "string", "enum": ["daily", "comprehensive"], "description": "审计模式"},
                "diff_text": {"type": "string", "description": "可选的git diff文本"},
                "staged_files": {"type": "array", "items": {"type": "string"}, "description": "可选的暂存文件列表"},
            },
            "required": ["project_path"],
        },
        boundary={"可用": ["16阶段安全审计", "秘密检测", "OWASP扫描", "STRIDE威胁模型", "数据分类"], "开发中": ["CI/CD集成"], "计划中": ["SBOM生成"]},
    ))

    _register_tool(MCPTool(
        tool_id="vf_ship_release",
        name="VF-SHIP 发布工程",
        description="Cezanne/Strategy灵魂专用：20步自动化发布管线，三层审查+版本决策",
        category=MCPToolCategory.COMPUTE,
        soul_mapping=["cezanne", "strategy", "davinci"],
        status=MCPToolStatus.AVAILABLE,
        input_schema={
            "type": "object",
            "properties": {
                "project_path": {"type": "string", "description": "项目路径"},
                "branch": {"type": "string", "description": "源分支"},
                "target": {"type": "string", "description": "目标分支(默认main)"},
                "dry_run": {"type": "boolean", "description": "试运行模式"},
            },
            "required": ["project_path"],
        },
        boundary={"可用": ["三层审查", "自动版本决策", "Changelog生成", "PR自动化"], "开发中": ["对抗性审查增强"], "计划中": ["多仓库协调发布"]},
    ))


class MCPSandboxServer:
    """
    MCP 沙箱服务 — 统一计算工具注册与调用

    核心职责：
    1. 注册所有专业适配器为 MCP 工具
    2. 按灵魂白名单控制工具访问
    3. 统一超时/重试/缓存策略
    4. 提供工具发现和状态查询

    使用：
        server = MCPSandboxServer()
        result = server.execute("sympy_mcp", "einstein", {"expression": "diff(x**3, x)"})
    """

    def __init__(self):
        self._call_counter = 0
        self._result_cache: Dict[str, Any] = {}
        _init_registry()

    def list_tools(self, soul: str = None, category: str = None) -> List[dict]:  # type: ignore[reportArgumentType]
        tools = list(MCP_TOOL_REGISTRY.values())
        if soul:
            tools = [t for t in tools if soul in t.soul_mapping]
        if category:
            tools = [t for t in tools if t.category.value == category]
        return [
            {
                "tool_id": t.tool_id,
                "name": t.name,
                "description": t.description,
                "category": t.category.value,
                "soul_mapping": t.soul_mapping,
                "status": t.status.value,
                "boundary": t.boundary,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]

    def get_tool(self, tool_id: str) -> Optional[dict]:
        tool = MCP_TOOL_REGISTRY.get(tool_id)
        if not tool:
            return None
        return {
            "tool_id": tool.tool_id,
            "name": tool.name,
            "description": tool.description,
            "category": tool.category.value,
            "soul_mapping": tool.soul_mapping,
            "status": tool.status.value,
            "boundary": tool.boundary,
            "input_schema": tool.input_schema,
            "call_count": tool.call_count,
            "last_called": tool.last_called,
        }

    def get_tools_for_soul(self, soul: str) -> List[dict]:
        return self.list_tools(soul=soul)

    def execute(self, tool_id: str, soul: str, arguments: Dict[str, Any]) -> MCPToolCall:
        self._call_counter += 1
        call_id = f"mcp_{int(time.time())}_{self._call_counter}"

        call = MCPToolCall(
            call_id=call_id,
            tool_id=tool_id,
            soul=soul,
            arguments=arguments,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        tool = MCP_TOOL_REGISTRY.get(tool_id)
        if not tool:
            call.status = "error"
            call.error = f"工具不存在: {tool_id}"
            return call

        if soul not in tool.soul_mapping:
            call.status = "forbidden"
            call.error = f"灵魂 {soul} 无权访问工具 {tool_id} (允许: {tool.soul_mapping})"
            return call

        if tool.status == MCPToolStatus.UNAVAILABLE:
            call.status = "error"
            call.error = f"工具 {tool_id} 当前不可用"
            return call

        if tool.status == MCPToolStatus.PENDING:
            call.status = "error"
            call.error = f"工具 {tool_id} 尚未实现 (状态: {tool.status.value})"
            return call

        start = time.time()
        try:
            result = self._dispatch(tool_id, arguments)
            elapsed = (time.time() - start) * 1000

            call.status = "success"
            call.result = result
            call.compute_time_ms = elapsed

            tool.call_count += 1
            tool.last_called = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        except Exception as e:
            elapsed = (time.time() - start) * 1000
            call.status = "error"
            call.error = str(e)
            call.compute_time_ms = elapsed

        call.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return call

    def _dispatch(self, tool_id: str, arguments: Dict[str, Any]) -> Any:
        if tool_id == "sympy_mcp":
            from science_adapter import get_adapter as get_science
            adapter = get_science()
            op = arguments.get("operation", "evaluate")
            expr = arguments.get("expression", "")
            if op == "symbolic" or op in ("simplify", "diff", "integrate", "solve", "expand"):
                return asdict(adapter.symbolic(expr))
            return asdict(adapter.evaluate(expr, arguments.get("variables")))  # type: ignore[reportArgumentType]

        elif tool_id == "science_plot_mcp":
            from science_adapter import get_adapter as get_science
            adapter = get_science()
            return asdict(adapter.plot(
                arguments.get("expression", "sin(x)"),
                tuple(arguments.get("x_range", [-10, 10])),
                arguments.get("title", ""),
                arguments.get("output_path", ""),
            ))

        elif tool_id == "agri_mcp":
            from agri_bio_adapter import get_adapter as get_agri
            adapter = get_agri()
            op = arguments.get("operation", "gdd")
            params = arguments.get("params", {})
            if op == "gdd":
                return adapter.calc_gdd(params.get("temp_max", 30), params.get("temp_min", 20), params.get("crop", "rice"))
            elif op == "irrigation":
                return adapter.calc_irrigation(params.get("crop", "rice"), params.get("et0", 5.0), params.get("stage", "mid"), params.get("rainfall_mm", 0))
            elif op == "fertilizer":
                return adapter.calc_fertilizer(params.get("soil", {}), params.get("crop", "rice"), params.get("target_yield_kg_ha", 9000))
            elif op == "yield":
                return adapter.calc_yield_estimate(params.get("crop", "rice"), params.get("gdd", 1200), params.get("rainfall", 800), params.get("n_fertilizer", 180))
            elif op == "solar_term":
                return adapter.get_solar_term()
            elif op == "genetic":
                return adapter.genetic_frequency(params.get("p", 0.6), params.get("q", 0.4), params.get("population", 1000))
            return {"error": f"未知操作: {op}"}

        elif tool_id == "health_mcp":
            from smarthealth_adapter import get_adapter as get_health
            adapter = get_health()
            op = arguments.get("operation", "assess")
            params = arguments.get("params", {})
            if op == "assess":
                return adapter.assess_health(**params)
            elif op == "nutrition":
                return adapter.get_nutrition_prescription(params)
            elif op == "exercise":
                return adapter.get_exercise_prescription(params)
            elif op == "recipe_daily":
                return adapter.get_daily_recipe(params, params.get("tier", "B"))
            elif op == "recipe_weekly":
                return adapter.get_weekly_recipe(params)
            elif op == "dishes":
                return adapter.get_zhuozhuang_dishes(params.get("dish_type"))
            elif op == "quotation":
                return adapter.get_zhuozhuang_quotation(params.get("month", "2026-01"))
            return {"error": f"未知操作: {op}"}

        elif tool_id == "blender_mcp":
            from blender_adapter import get_adapter as get_blender
            adapter = get_blender()
            op = arguments.get("operation", "scene_info")
            params = arguments.get("params", {})
            if op == "create_primitive":
                return adapter.create_primitive(params.get("type", "CUBE"), params.get("name", ""), tuple(params.get("location", [0, 0, 0])))
            elif op == "render":
                return adapter.render(params.get("output_path", ""), params.get("engine", "CYCLES"), params.get("samples", 128))
            elif op == "export":
                return adapter.export(params.get("filepath", ""), params.get("object_name", ""), params.get("format", "FBX"))
            elif op == "scene_info":
                return adapter.get_scene_info()
            return adapter.status()

        elif tool_id == "design_mcp":
            from design_adapter import get_adapter as get_design
            adapter = get_design()
            op = arguments.get("operation", "")
            params = arguments.get("params", {})
            if op == "figma_file":
                return adapter.figma_get_file(params.get("file_key", ""))
            elif op == "figma_image":
                return adapter.figma_get_image(params.get("file_key", ""), params.get("node_id", ""))
            elif op == "image_resize":
                return adapter.image_resize(params.get("input", ""), params.get("output", ""), params.get("width", 800), params.get("height", 600))
            elif op == "color_adjust":
                return adapter.color_adjust(params.get("input", ""), params.get("output", ""), params.get("brightness", 0), params.get("contrast", 0), params.get("saturation", 0))
            return adapter.status()

        elif tool_id == "ableton_mcp":
            from ableton_adapter import get_adapter as get_ableton
            adapter = get_ableton()
            op = arguments.get("operation", "status")
            params = arguments.get("params", {})
            if op == "connect":
                return adapter.connect()
            elif op == "create_track":
                return adapter.create_midi_track(params.get("name", ""), params.get("position", -1))
            elif op == "set_tempo":
                return adapter.set_tempo(params.get("bpm", 120))
            elif op == "add_notes":
                return adapter.add_midi_notes(params.get("track", 0), params.get("clip", 0), [])
            elif op == "playback":
                if params.get("action") == "start":
                    return adapter.start_playback()
                return adapter.stop_playback()
            return adapter.status()

        elif tool_id == "biopython_mcp":
            from agri_bio_adapter import get_adapter as get_agri
            adapter = get_agri()
            return adapter.sequence_analyze(arguments.get("sequence", ""), arguments.get("analysis_type", "gc_content"))

        elif tool_id == "geopandas_mcp":
            return {"status": "pending", "message": "GeoPandas MCP 尚未实现，计划在后续版本集成"}

        elif tool_id == "backtrader_mcp":
            return {"status": "pending", "message": "Backtrader MCP 尚未实现，计划在后续版本集成"}

        elif tool_id == "aigen_txt2img":
            from ai_gen_adapter import get_adapter as get_aigen
            adapter = get_aigen()
            return adapter.txt2img(
                arguments.get("prompt", ""),
                soul=arguments.get("soul", "monet"),
                negative_prompt=arguments.get("negative_prompt", ""),
                width=arguments.get("width", 1024),
                height=arguments.get("height", 1024),
                model=arguments.get("model", "flux1-dev"),
            )

        elif tool_id == "aigen_txt2music":
            from ai_gen_adapter import get_adapter as get_aigen
            adapter = get_aigen()
            return adapter.txt2music(
                arguments.get("prompt", ""),
                soul=arguments.get("soul", "beethoven"),
                duration_seconds=arguments.get("duration_seconds", 30),
                bpm=arguments.get("bpm", 120),
                key=arguments.get("key", ""),
                genre=arguments.get("genre", ""),
            )

        elif tool_id == "aigen_txt2video":
            from ai_gen_adapter import get_adapter as get_aigen
            adapter = get_aigen()
            return adapter.txt2video(
                arguments.get("prompt", ""),
                soul=arguments.get("soul", "davinci"),
            )

        elif tool_id == "aigen_txt2voice":
            from ai_gen_adapter import get_adapter as get_aigen
            adapter = get_aigen()
            return adapter.txt2voice(
                arguments.get("text", ""),
                soul=arguments.get("soul", "guizhu"),
            )

        elif tool_id == "aigen_txt23d":
            from ai_gen_adapter import get_adapter as get_aigen
            adapter = get_aigen()
            return adapter.txt23d(
                arguments.get("prompt", ""),
                soul=arguments.get("soul", "davinci"),
            )

        return {"error": f"未知工具: {tool_id}"}

    def status(self) -> dict:
        tools = list(MCP_TOOL_REGISTRY.values())
        by_status = {}
        for t in tools:
            by_status.setdefault(t.status.value, []).append(t.tool_id)
        by_category = {}
        for t in tools:
            by_category.setdefault(t.category.value, []).append(t.tool_id)
        return {
            "server": "MCPSandboxServer",
            "total_tools": len(tools),
            "by_status": by_status,
            "by_category": by_category,
            "config": MCP_SERVER_CONFIG,
        }


_server_instance: Optional[MCPSandboxServer] = None


def get_server() -> MCPSandboxServer:
    global _server_instance
    if _server_instance is None:
        _server_instance = MCPSandboxServer()
    return _server_instance
