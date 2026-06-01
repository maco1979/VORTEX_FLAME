"""
VORTEX FLAME — 16 MCP Servers 注册表

每个 MCP Server 对应一个外部工具/能力域，
通过标准 MCP 协议与灵魂矩阵对接。

架构：
  MCPRegistry → MCPServerConfig → 运行时由 mcp_sandbox_server.py 统一调度
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional


class MCPServerCategory(str, Enum):
    CORE = "core"
    VISUAL = "visual"
    TESTING = "testing"
    OSINT = "osint"
    KNOWLEDGE = "knowledge"
    DESIGN = "design"
    SOUL = "soul"
    PIPELINE = "pipeline"
    INTERPRETABILITY = "interpretability"
    SECURITY = "security"
    MUSIC = "music"
    VOICE = "voice"
    AUTOMATION = "automation"


class MCPServerStatus(str, Enum):
    AVAILABLE = "available"
    DEVELOPING = "developing"
    PENDING = "pending"


@dataclass
class MCPServerConfig:
    server_id: str
    name: str
    name_zh: str
    category: MCPServerCategory
    core_capability: str
    core_capability_zh: str
    status: MCPServerStatus
    soul_mapping: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    api_endpoint: str = ""
    cli_command: str = ""
    config_path: str = ""
    env_vars: List[str] = field(default_factory=list)
    boundary: Dict[str, List[str]] = field(default_factory=dict)
    adapter_module: str = ""
    description: str = ""


MCP_SERVER_REGISTRY: Dict[str, MCPServerConfig] = {
    "soul-memory": MCPServerConfig(
        server_id="soul-memory",
        name="Soul Memory",
        name_zh="灵魂记忆",
        category=MCPServerCategory.CORE,
        core_capability="Conversation, knowledge base, recall, todo",
        core_capability_zh="对话、知识库、回忆、待办",
        status=MCPServerStatus.AVAILABLE,
        soul_mapping=["all"],
        tools=["store", "recall", "search", "forget", "todo_add", "todo_list", "todo_complete"],
        adapter_module="soul_memory",
        config_path="mcp_servers/soul-memory.json",
        boundary={"可用": ["记忆存储", "语义回忆", "待办管理", "知识图谱"], "开发中": ["跨灵魂共享"], "计划中": ["长期记忆压缩"]},
        description="核心记忆系统，所有灵魂共享的对话/知识/待办存储",
    ),
    "comfyui": MCPServerConfig(
        server_id="comfyui",
        name="ComfyUI",
        name_zh="ComfyUI图像生成",
        category=MCPServerCategory.VISUAL,
        core_capability="txt2img, img2img, 7 soul style presets",
        core_capability_zh="文生图、图生图、7种灵魂风格预设",
        status=MCPServerStatus.AVAILABLE,
        soul_mapping=["monet", "vangogh", "davinci", "beethoven"],
        tools=["txt2img", "img2img", "inpainting", "controlnet", "lora", "upscale", "style_preset"],
        api_endpoint="http://127.0.0.1:8188",
        adapter_module="ai_gen_adapter",
        config_path="mcp_servers/comfyui.json",
        env_vars=["COMFYUI_URL"],
        boundary={"可用": ["txt2img", "img2img", "ControlNet", "LoRA", "风格预设"], "开发中": ["Inpainting"], "计划中": ["视频生成", "AnimateDiff"]},
        description="本地Stable Diffusion/FLUX工作流引擎，7种灵魂风格预设",
    ),
    "browse": MCPServerConfig(
        server_id="browse",
        name="Browse",
        name_zh="浏览器自动化",
        category=MCPServerCategory.TESTING,
        core_capability="Navigate, screenshot, click, fill",
        core_capability_zh="导航、截图、点击、填写",
        status=MCPServerStatus.AVAILABLE,
        soul_mapping=["cezanne", "davinci", "guizhu"],
        tools=["navigate", "screenshot", "click", "fill", "scroll", "wait", "extract"],
        adapter_module="mano_p_adapter",
        config_path="mcp_servers/browse.json",
        boundary={"可用": ["网页导航", "截图", "表单填写", "内容提取"], "开发中": ["SPA支持"], "计划中": ["PDF导出"]},
        description="浏览器自动化测试与内容提取",
    ),
    "osint": MCPServerConfig(
        server_id="osint",
        name="OSINT",
        name_zh="开源情报",
        category=MCPServerCategory.OSINT,
        core_capability="User search, profile, compliance check",
        core_capability_zh="用户搜索、画像、合规检查",
        status=MCPServerStatus.PENDING,
        soul_mapping=["herodotus", "montesquieu", "strategy"],
        tools=["search_user", "profile", "compliance_check", "risk_assess"],
        config_path="mcp_servers/osint.json",
        env_vars=["OSINT_API_KEY"],
        boundary={"可用": [], "开发中": ["用户搜索"], "计划中": ["画像分析", "合规检查", "风险评估"]},
        description="开源情报收集与合规性检查",
    ),
    "rag": MCPServerConfig(
        server_id="rag",
        name="RAG",
        name_zh="检索增强生成",
        category=MCPServerCategory.KNOWLEDGE,
        core_capability="Create KB, add documents, query",
        core_capability_zh="创建知识库、添加文档、查询",
        status=MCPServerStatus.AVAILABLE,
        soul_mapping=["einstein", "galileo", "darwin", "herodotus", "humboldt"],
        tools=["create_kb", "add_documents", "query", "list_kbs", "delete_kb"],
        adapter_module="soul_memory",
        config_path="mcp_servers/rag.json",
        boundary={"可用": ["知识库创建", "文档添加", "语义查询"], "开发中": ["PDF解析"], "计划中": ["多模态RAG"]},
        description="检索增强生成，基于灵魂记忆系统的知识库管理",
    ),
    "open-design": MCPServerConfig(
        server_id="open-design",
        name="Open Design",
        name_zh="开放设计",
        category=MCPServerCategory.DESIGN,
        core_capability="31 skills, PPT generation, aesthetic scoring",
        core_capability_zh="31项技能、PPT生成、美学评分",
        status=MCPServerStatus.AVAILABLE,
        soul_mapping=["monet", "vangogh", "davinci", "herodotus"],
        tools=["generate_ppt", "aesthetic_score", "design_system", "layout_suggest", "color_palette", "typography", "export_html", "export_pdf", "export_pptx"],
        api_endpoint="http://127.0.0.1:3456",
        cli_command="pnpm tools-dev",
        config_path="mcp_servers/open-design.json",
        env_vars=["OPEN_DESIGN_PORT"],
        boundary={"可用": ["PPT生成", "美学评分", "设计系统", "色彩方案", "版式建议"], "开发中": ["HTML导出", "PDF导出"], "计划中": ["Figma同步", "实时协作"]},
        description="Open Design AI设计工作台，137个Skill + 150套DESIGN.md系统，支持PPT/HTML/PDF导出",
    ),
    "codex-enhance": MCPServerConfig(
        server_id="codex-enhance",
        name="Codex Enhance",
        name_zh="代码增强",
        category=MCPServerCategory.SOUL,
        core_capability="Syntax check, type check, security scan",
        core_capability_zh="语法检查、类型检查、安全扫描",
        status=MCPServerStatus.AVAILABLE,
        soul_mapping=["cezanne", "einstein", "strategy"],
        tools=["syntax_check", "type_check", "security_scan", "complexity_analysis", "refactor_suggest"],
        adapter_module="code_intelligence",
        config_path="mcp_servers/codex-enhance.json",
        boundary={"可用": ["语法检查", "类型检查", "安全扫描", "复杂度分析"], "开发中": ["自动重构"], "计划中": ["性能优化建议"]},
        description="代码质量增强引擎，语法/类型/安全三重检查",
    ),
    "soul-pipeline": MCPServerConfig(
        server_id="soul-pipeline",
        name="Soul Pipeline",
        name_zh="灵魂管线",
        category=MCPServerCategory.PIPELINE,
        core_capability="Get soul config, plan training",
        core_capability_zh="获取灵魂配置、规划训练",
        status=MCPServerStatus.AVAILABLE,
        soul_mapping=["all"],
        tools=["get_config", "plan_training", "check_status", "list_souls", "validate_config"],
        adapter_module="soul_orchestrator",
        config_path="mcp_servers/soul-pipeline.json",
        boundary={"可用": ["配置获取", "训练规划", "状态查询"], "开发中": ["自动调参"], "计划中": ["分布式训练"]},
        description="灵魂训练管线管理，配置/规划/状态一体化",
    ),
    "nla": MCPServerConfig(
        server_id="nla",
        name="NLA",
        name_zh="神经激活分析",
        category=MCPServerCategory.INTERPRETABILITY,
        core_capability="Extract activations, train SAE",
        core_capability_zh="提取激活、训练稀疏自编码器",
        status=MCPServerStatus.PENDING,
        soul_mapping=["einstein", "cezanne", "guizhu"],
        tools=["extract_activations", "train_sae", "visualize_features", "feature_attribution"],
        config_path="mcp_servers/nla.json",
        boundary={"可用": [], "开发中": ["激活提取"], "计划中": ["SAE训练", "特征可视化", "归因分析"]},
        description="神经语言学分析，提取模型激活并训练稀疏自编码器进行可解释性研究",
    ),
    "blackbox-shield": MCPServerConfig(
        server_id="blackbox-shield",
        name="Blackbox Shield",
        name_zh="黑盒护盾",
        category=MCPServerCategory.SECURITY,
        core_capability="Obfuscate, checksum, scan",
        core_capability_zh="混淆、校验、扫描",
        status=MCPServerStatus.AVAILABLE,
        soul_mapping=["cezanne", "strategy"],
        tools=["obfuscate", "generate_checksums", "verify_integrity", "scan_exposure", "scan_structure", "full_check"],
        adapter_module="blackbox_shield_adapter",
        config_path="mcp_servers/blackbox-shield.json",
        boundary={"可用": ["代码混淆", "完整性校验", "泄露扫描", "结构扫描"], "开发中": [], "计划中": ["实时监控"]},
        description="代码保护与安全审计，PyArmor混淆+SHA256校验+泄露扫描",
    ),
    "ableton": MCPServerConfig(
        server_id="ableton",
        name="Ableton",
        name_zh="Ableton音乐制作",
        category=MCPServerCategory.MUSIC,
        core_capability="MIDI, track operations",
        core_capability_zh="MIDI、轨道操作",
        status=MCPServerStatus.AVAILABLE,
        soul_mapping=["beethoven"],
        tools=["midi_send", "track_create", "track_volume", "clip_launch", "transport_control", "tempo_set"],
        api_endpoint="osc://127.0.0.1:11000",
        adapter_module="ableton_adapter",
        config_path="mcp_servers/ableton.json",
        boundary={"可用": ["OSC控制", "MIDI生成", "轨道操作", "传输控制"], "开发中": ["实时录制"], "计划中": ["Live Set模板"]},
        description="Ableton Live DAW控制，OSC/MIDI/CLI三层适配",
    ),
    "delivery-audit": MCPServerConfig(
        server_id="delivery-audit",
        name="Delivery Audit",
        name_zh="交付审计",
        category=MCPServerCategory.SECURITY,
        core_capability="Delivery audit",
        core_capability_zh="交付审计",
        status=MCPServerStatus.AVAILABLE,
        soul_mapping=["cezanne", "montesquieu", "guizhu"],
        tools=["audit_code", "audit_config", "audit_dependencies", "audit_secrets", "generate_report"],
        adapter_module="guardian",
        config_path="mcp_servers/delivery-audit.json",
        boundary={"可用": ["代码审计", "配置审计", "依赖审计", "密钥扫描"], "开发中": [], "计划中": ["合规报告生成"]},
        description="交付前安全审计，代码/配置/依赖/密钥四维检查",
    ),
    "dsa": MCPServerConfig(
        server_id="dsa",
        name="DSA",
        name_zh="算法引擎",
        category=MCPServerCategory.CORE,
        core_capability="Algorithms",
        core_capability_zh="算法",
        status=MCPServerStatus.AVAILABLE,
        soul_mapping=["cezanne", "einstein", "strategy"],
        tools=["sort", "search", "graph", "dp", "greedy", "divide_conquer"],
        adapter_module="science_adapter",
        config_path="mcp_servers/dsa.json",
        boundary={"可用": ["排序", "搜索", "图算法", "动态规划", "贪心"], "开发中": [], "计划中": ["竞赛算法库"]},
        description="数据结构与算法引擎，覆盖主流算法范式",
    ),
    "animejs": MCPServerConfig(
        server_id="animejs",
        name="AnimeJS",
        name_zh="动画生成",
        category=MCPServerCategory.VISUAL,
        core_capability="Animation generation",
        core_capability_zh="动画生成",
        status=MCPServerStatus.AVAILABLE,
        soul_mapping=["monet", "vangogh", "davinci"],
        tools=["animate_element", "timeline", "svg_path", "morph", "stagger", "spring_physics"],
        config_path="mcp_servers/animejs.json",
        boundary={"可用": ["CSS动画", "SVG动画", "时间轴", "形变动画"], "开发中": ["3D动画"], "计划中": ["Lottie导出"]},
        description="Web动画生成引擎，基于Anime.js，支持CSS/SVG/Canvas动画",
    ),
    "voice": MCPServerConfig(
        server_id="voice",
        name="Voice",
        name_zh="语音引擎",
        category=MCPServerCategory.VOICE,
        core_capability="Whisper + edge-tts",
        core_capability_zh="Whisper语音识别 + edge-tts语音合成",
        status=MCPServerStatus.AVAILABLE,
        soul_mapping=["beethoven", "guizhu", "herodotus"],
        tools=["transcribe", "synthesize", "translate_speech", "voice_clone"],
        config_path="mcp_servers/voice.json",
        env_vars=["WHISPER_MODEL", "EDGE_TTS_VOICE"],
        boundary={"可用": ["Whisper语音识别", "edge-tts语音合成", "多语言支持"], "开发中": ["实时转写"], "计划中": ["声音克隆", "ElevenLabs"]},
        description="语音识别与合成，Whisper转写+edge-tts合成，支持中英日韩",
    ),
    "ui-tars": MCPServerConfig(
        server_id="ui-tars",
        name="UI-TARS",
        name_zh="UI自动化",
        category=MCPServerCategory.AUTOMATION,
        core_capability="Screenshot, analyze, execute, run_task",
        core_capability_zh="截图、分析、执行、运行任务",
        status=MCPServerStatus.AVAILABLE,
        soul_mapping=["cezanne", "davinci", "guizhu"],
        tools=["screenshot", "analyze", "execute", "run_task", "click_at", "type_text", "hotkey", "scroll"],
        adapter_module="mano_p_adapter",
        config_path="mcp_servers/ui-tars.json",
        boundary={"可用": ["截图", "UI分析", "点击", "输入", "快捷键", "滚动"], "开发中": ["多窗口"], "计划中": ["移动端"]},
        description="桌面UI自动化，基于UI-TARS的视觉感知+操作执行",
    ),
}


def get_server_config(server_id: str) -> Optional[MCPServerConfig]:
    return MCP_SERVER_REGISTRY.get(server_id)


def list_servers(category: str = None, soul: str = None, status: str = None) -> List[MCPServerConfig]:
    servers = list(MCP_SERVER_REGISTRY.values())
    if category:
        servers = [s for s in servers if s.category.value == category]
    if soul and soul != "all":
        servers = [s for s in servers if soul in s.soul_mapping or "all" in s.soul_mapping]
    if status:
        servers = [s for s in servers if s.status.value == status]
    return servers


def get_servers_for_soul(soul: str) -> List[MCPServerConfig]:
    return list_servers(soul=soul)


def generate_mcp_json(server_id: str) -> dict:
    config = get_server_config(server_id)
    if not config:
        return {}
    return {
        "mcpServers": {
            config.server_id: {
                "command": config.cli_command or "python",
                "args": [f"mcp_servers/{config.server_id}/server.py"],
                "env": {v: "" for v in config.env_vars} if config.env_vars else {},
                "tools": config.tools,
                "soul_mapping": config.soul_mapping,
                "category": config.category.value,
                "description": config.core_capability_zh,
            }
        }
    }


def generate_all_mcp_json() -> dict:
    result = {"mcpServers": {}}
    for server_id in MCP_SERVER_REGISTRY:
        config = generate_mcp_json(server_id)
        result["mcpServers"].update(config.get("mcpServers", {}))
    return result


def status_report() -> dict:
    by_category = {}
    for s in MCP_SERVER_REGISTRY.values():
        by_category.setdefault(s.category.value, []).append(s.server_id)
    by_status = {}
    for s in MCP_SERVER_REGISTRY.values():
        by_status.setdefault(s.status.value, []).append(s.server_id)
    return {
        "total": len(MCP_SERVER_REGISTRY),
        "by_category": by_category,
        "by_status": by_status,
        "servers": {sid: {"name_zh": s.name_zh, "category": s.category.value, "status": s.status.value, "tools_count": len(s.tools)} for sid, s in MCP_SERVER_REGISTRY.items()},
    }


if __name__ == "__main__":
    import json
    print(json.dumps(status_report(), indent=2, ensure_ascii=False))
    print("\n--- MCP JSON ---")
    print(json.dumps(generate_all_mcp_json(), indent=2, ensure_ascii=False))
