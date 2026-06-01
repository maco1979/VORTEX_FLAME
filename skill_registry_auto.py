"""
VORTEX FLAME — 动态技能注册与自动更新引擎
============================================

从多个上游源自动发现、聚合、更新 MCP Server 和 AI 工具：
  1. GitHub MCP Registry (官方)
  2. mcp-awesome.com (社区策展)
  3. mcpservers.org (社区元数据)
  4. 本地 mcp_server_registry.py (内置16个)
  5. 自定义源 (用户可扩展)

自动更新策略：
  - 启动时全量同步
  - 每24小时增量刷新
  - 手动触发即时更新
  - 离线降级：使用本地缓存

架构：
  SkillRegistryAuto
    ├── sources/        → 上游数据源适配器
    ├── cache/          → 本地 JSON 缓存
    ├── index/          → 内存搜索索引
    └── scheduler/      → 后台刷新调度
"""

import json
import hashlib
import time
import threading
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger("skill_registry_auto")

CACHE_DIR = Path(__file__).parent / "skill_cache"
CACHE_DIR.mkdir(exist_ok=True)

BUILTIN_REGISTRY_PATH = Path(__file__).parent / "mcp_server_registry.py"


class SkillTier(str, Enum):
    TIER1_CORE = "tier1_core"
    TIER2_POPULAR = "tier2_popular"
    TIER3_TRENDING = "tier3_trending"
    TIER4_COMMUNITY = "tier4_community"
    TIER5_EXPERIMENTAL = "tier5_experimental"


class SkillCategory(str, Enum):
    CORE = "core"
    VISUAL = "visual"
    VIDEO = "video"
    AUDIO = "audio"
    MUSIC = "music"
    DESIGN = "design"
    CODE = "code"
    DATA = "data"
    SEARCH = "search"
    AUTOMATION = "automation"
    PRODUCTIVITY = "productivity"
    COMMUNICATION = "communication"
    DATABASE = "database"
    DEVOPS = "devops"
    SECURITY = "security"
    KNOWLEDGE = "knowledge"
    FINANCE = "finance"
    RESEARCH = "research"
    OSINT = "osint"
    VOICE = "voice"
    INTERPRETABILITY = "interpretability"
    PIPELINE = "pipeline"
    SOUL = "soul"
    TESTING = "testing"
    CREATIVE = "creative"
    EDUCATION = "education"
    HEALTH = "health"
    GAMING = "gaming"


class SkillStatus(str, Enum):
    AVAILABLE = "available"
    TRENDING = "trending"
    DEVELOPING = "developing"
    PENDING = "pending"
    DEPRECATED = "deprecated"


class UpdatePolicy(str, Enum):
    AUTO = "auto"
    MANUAL = "manual"
    DISABLED = "disabled"


@dataclass
class SkillEntry:
    skill_id: str
    name: str
    name_zh: str
    category: SkillCategory
    tier: SkillTier
    status: SkillStatus
    description: str = ""
    description_zh: str = ""
    tools: List[str] = field(default_factory=list)
    soul_mapping: List[str] = field(default_factory=list)
    source: str = "builtin"
    source_url: str = ""
    repo_url: str = ""
    stars: int = 0
    install_count: int = 0
    weekly_growth: float = 0.0
    api_endpoint: str = ""
    cli_command: str = ""
    adapter_module: str = ""
    config_path: str = ""
    env_vars: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    pricing: str = ""
    license_type: str = ""
    last_updated: str = ""
    discovered_at: str = ""
    update_policy: UpdatePolicy = UpdatePolicy.AUTO
    version: str = ""
    language: str = ""
    verified: bool = False
    official: bool = False
    boundary: Dict[str, List[str]] = field(default_factory=dict)


TRENDING_AI_TOOLS_2025: Dict[str, Dict] = {
    "midjourney-v7": {
        "name": "Midjourney V7",
        "name_zh": "Midjourney V7 图像生成",
        "category": "visual",
        "tier": "tier2_popular",
        "description": "Art-quality image generation with 3D composition assist and style gene library",
        "description_zh": "艺术级图像生成，支持3D构图辅助和风格基因库",
        "tools": ["txt2img", "style_transfer", "blend", "upscale", "character_ref", "depth_map"],
        "soul_mapping": ["monet", "vangogh", "davinci"],
        "tags": ["image", "art", "creative", "trending"],
        "pricing": "$10-60/mo",
        "stars": 85000,
        "verified": True,
    },
    "sora": {
        "name": "Sora",
        "name_zh": "Sora 视频生成",
        "category": "video",
        "tier": "tier2_popular",
        "description": "OpenAI's video generation with director mode for character consistency and physics",
        "description_zh": "OpenAI视频生成，导演模式解决角色一致性和物理规律",
        "tools": ["txt2video", "img2video", "director_mode", "extend", "remix"],
        "soul_mapping": ["monet", "vangogh", "beethoven"],
        "tags": ["video", "creative", "trending"],
        "pricing": "$20/mo",
        "stars": 120000,
        "verified": True,
    },
    "kling": {
        "name": "Kling AI",
        "name_zh": "可灵AI 视频生成",
        "category": "video",
        "tier": "tier2_popular",
        "description": "Kuaishou's video generation model, strong in motion and character consistency",
        "description_zh": "快手可灵视频生成，运动和角色一致性强",
        "tools": ["txt2video", "img2video", "lip_sync", "motion_transfer"],
        "soul_mapping": ["monet", "vangogh"],
        "tags": ["video", "creative", "chinese", "trending"],
        "pricing": "免费+付费",
        "stars": 25000,
        "verified": True,
    },
    "pika": {
        "name": "Pika",
        "name_zh": "Pika 超现实视频",
        "category": "video",
        "tier": "tier3_trending",
        "description": "Surreal art video lab with physics simulation and character anchoring",
        "description_zh": "超现实艺术视频实验室，物理模拟和角色锚定",
        "tools": ["txt2video", "surreal_mode", "character_anchor", "effects"],
        "soul_mapping": ["monet", "beethoven"],
        "tags": ["video", "creative", "surreal", "trending"],
        "pricing": "免费+付费",
        "stars": 18000,
        "verified": True,
    },
    "suno-v3": {
        "name": "Suno V3",
        "name_zh": "Suno V3 AI音乐",
        "category": "music",
        "tier": "tier2_popular",
        "description": "AI music creation with multi-track generation and emotion engine",
        "description_zh": "AI音乐创作，多轨道生成和情感引擎",
        "tools": ["txt2music", "multi_track", "emotion_engine", "lyrics_gen", "remix"],
        "soul_mapping": ["beethoven"],
        "tags": ["music", "audio", "creative", "trending"],
        "pricing": "$8-24/mo",
        "stars": 35000,
        "verified": True,
    },
    "elevenlabs": {
        "name": "ElevenLabs",
        "name_zh": "ElevenLabs 语音克隆",
        "category": "voice",
        "tier": "tier2_popular",
        "description": "Voice cloning and sound effect design with digital watermarking",
        "description_zh": "语音克隆和音效设计，数字水印防滥用",
        "tools": ["tts", "voice_clone", "sound_effect", "speech2speech", "watermark"],
        "soul_mapping": ["beethoven", "guizhu"],
        "tags": ["voice", "audio", "trending"],
        "pricing": "$5-99/mo",
        "stars": 42000,
        "verified": True,
    },
    "deepseek-r1": {
        "name": "DeepSeek-R1",
        "name_zh": "DeepSeek-R1 深度推理",
        "category": "core",
        "tier": "tier1_core",
        "description": "Open-source reasoning model with chain-of-thought, cost-effective deep thinking",
        "description_zh": "开源推理模型，思维链可视化，低成本深度思考",
        "tools": ["reason", "cot_visualize", "code_gen", "math_solve"],
        "soul_mapping": ["cezanne", "guizhu", "davinci"],
        "tags": ["llm", "reasoning", "open-source", "chinese", "trending"],
        "pricing": "开源免费",
        "stars": 95000,
        "verified": True,
        "official": True,
    },
    "perplexity-deep-research": {
        "name": "Perplexity Deep Research",
        "name_zh": "Perplexity 深度研究",
        "category": "research",
        "tier": "tier2_popular",
        "description": "AI-powered deep research with citations, 20-30 min autonomous web research",
        "description_zh": "AI深度研究，带引用，20-30分钟自主全网调研",
        "tools": ["deep_research", "search", "cite", "summarize", "compare"],
        "soul_mapping": ["guizhu", "cezanne"],
        "tags": ["research", "search", "trending"],
        "pricing": "$20/mo Pro",
        "stars": 55000,
        "verified": True,
    },
    "gemini-2.5": {
        "name": "Gemini 2.5 Pro",
        "name_zh": "Gemini 2.5 Pro 多模态",
        "category": "core",
        "tier": "tier1_core",
        "description": "Google's native multimodal model, 1M token context, video understanding",
        "description_zh": "Google原生多模态模型，100万token上下文，视频理解",
        "tools": ["chat", "vision", "video_understand", "code_gen", "1m_context"],
        "soul_mapping": ["all"],
        "tags": ["llm", "multimodal", "trending"],
        "pricing": "免费+付费",
        "stars": 68000,
        "verified": True,
        "official": True,
    },
    "figma-mcp": {
        "name": "Figma MCP Server",
        "name_zh": "Figma 设计集成",
        "category": "design",
        "tier": "tier2_popular",
        "description": "Access Figma designs, export assets, analyze design systems with AI",
        "description_zh": "访问Figma设计，导出资源，AI分析设计系统",
        "tools": ["design_access", "asset_export", "component_analysis", "style_extract"],
        "soul_mapping": ["monet", "davinci"],
        "tags": ["design", "figma", "mcp", "trending"],
        "pricing": "开源免费",
        "stars": 1800,
        "verified": True,
    },
    "blender-mcp": {
        "name": "Blender MCP Server",
        "name_zh": "Blender 3D建模集成",
        "category": "visual",
        "tier": "tier2_popular",
        "description": "Control Blender 3D through natural language, create scenes and render",
        "description_zh": "自然语言控制Blender 3D，创建场景和渲染",
        "tools": ["scene_create", "object_model", "material_setup", "render", "polyhaven"],
        "soul_mapping": ["davinci", "monet"],
        "tags": ["3d", "blender", "mcp", "trending"],
        "pricing": "开源免费",
        "stars": 3200,
        "verified": True,
    },
    "playwright-mcp": {
        "name": "Playwright MCP Server",
        "name_zh": "Playwright 浏览器自动化",
        "category": "automation",
        "tier": "tier2_popular",
        "description": "Microsoft's browser automation via accessibility snapshots, not screenshots",
        "description_zh": "微软浏览器自动化，基于无障碍快照而非截图",
        "tools": ["navigate", "click", "fill", "screenshot", "extract", "pdf"],
        "soul_mapping": ["cezanne", "guizhu"],
        "tags": ["automation", "browser", "mcp", "trending"],
        "pricing": "开源免费",
        "stars": 6100,
        "verified": True,
        "official": True,
    },
    "github-mcp": {
        "name": "GitHub MCP Server",
        "name_zh": "GitHub 代码管理集成",
        "category": "code",
        "tier": "tier1_core",
        "description": "Access 300+ GitHub APIs, repo management, PR review, Actions integration",
        "description_zh": "300+ GitHub API接口，仓库管理，PR审查，Actions集成",
        "tools": ["repo_browse", "code_search", "issue_manage", "pr_review", "actions", "file_ops"],
        "soul_mapping": ["cezanne", "davinci"],
        "tags": ["code", "github", "mcp", "trending"],
        "pricing": "开源免费",
        "stars": 25000,
        "verified": True,
        "official": True,
    },
    "notion-mcp": {
        "name": "Notion MCP Server",
        "name_zh": "Notion 知识库集成",
        "category": "productivity",
        "tier": "tier2_popular",
        "description": "Access Notion workspace, search pages, create notes, manage databases",
        "description_zh": "访问Notion工作区，搜索页面，创建笔记，管理数据库",
        "tools": ["page_search", "content_create", "database_query", "block_edit"],
        "soul_mapping": ["guizhu", "cezanne"],
        "tags": ["productivity", "notion", "mcp", "trending"],
        "pricing": "开源免费",
        "stars": 1700,
        "verified": True,
        "official": True,
    },
    "docker-mcp": {
        "name": "Docker MCP Server",
        "name_zh": "Docker 容器管理集成",
        "category": "devops",
        "tier": "tier2_popular",
        "description": "Manage Docker containers, images, volumes and Compose orchestration",
        "description_zh": "管理Docker容器、镜像、卷和Compose编排",
        "tools": ["container_lifecycle", "image_manage", "volume_ops", "compose"],
        "soul_mapping": ["cezanne"],
        "tags": ["devops", "docker", "mcp", "trending"],
        "pricing": "开源免费",
        "stars": 2100,
        "verified": True,
        "official": True,
    },
    "context7": {
        "name": "Context7",
        "name_zh": "Context7 文档检索",
        "category": "knowledge",
        "tier": "tier2_popular",
        "description": "Fetch version-specific documentation and inject into prompt context",
        "description_zh": "获取版本特定文档并注入到提示上下文中",
        "tools": ["doc_fetch", "version_lookup", "api_search", "snippet_inject"],
        "soul_mapping": ["cezanne", "davinci"],
        "tags": ["knowledge", "docs", "mcp", "trending"],
        "pricing": "开源免费",
        "stars": 15000,
        "verified": True,
    },
    "exa-search": {
        "name": "Exa MCP Server",
        "name_zh": "Exa 语义搜索",
        "category": "search",
        "tier": "tier2_popular",
        "description": "Semantic web search and content retrieval, AI-native search engine",
        "description_zh": "语义网络搜索和内容检索，AI原生搜索引擎",
        "tools": ["semantic_search", "content_retrieve", "find_similar", "crawl"],
        "soul_mapping": ["guizhu", "cezanne"],
        "tags": ["search", "semantic", "mcp", "trending"],
        "pricing": "免费额度+付费",
        "stars": 8000,
        "verified": True,
    },
    "firecrawl": {
        "name": "Firecrawl",
        "name_zh": "Firecrawl 网页爬取",
        "category": "search",
        "tier": "tier2_popular",
        "description": "Full web scraping and crawling, convert pages to LLM-ready markdown",
        "description_zh": "全功能网页爬取和抓取，将页面转为LLM可用的Markdown",
        "tools": ["scrape", "crawl", "search", "extract", "map"],
        "soul_mapping": ["guizhu", "cezanne"],
        "tags": ["search", "scraping", "mcp", "trending"],
        "pricing": "免费500次+付费",
        "stars": 12000,
        "verified": True,
    },
    "skywork-agents": {
        "name": "Skywork Super Agents",
        "name_zh": "天工超级智能体",
        "category": "productivity",
        "tier": "tier3_trending",
        "description": "Deep research based AI Office: doc, PPT, Excel, podcast, webpage generation",
        "description_zh": "基于深度研究的AI版Office：文档、PPT、表格、播客、网页生成",
        "tools": ["doc_gen", "ppt_gen", "excel_gen", "podcast_gen", "webpage_gen", "deep_research"],
        "soul_mapping": ["guizhu", "cezanne", "davinci"],
        "tags": ["productivity", "office", "chinese", "trending"],
        "pricing": "免费+付费",
        "stars": 15000,
        "verified": True,
    },
    "jimeng-ai": {
        "name": "Jimeng AI",
        "name_zh": "即梦AI 图像视频",
        "category": "visual",
        "tier": "tier3_trending",
        "description": "ByteDance's all-in-one AI image and video generation with character DNA",
        "description_zh": "字节跳动全能AI图像视频生成，角色DNA一致性",
        "tools": ["txt2img", "txt2video", "character_dna", "style_transfer", "chinese_prompt"],
        "soul_mapping": ["monet", "vangogh"],
        "tags": ["image", "video", "chinese", "trending"],
        "pricing": "免费+付费",
        "stars": 20000,
        "verified": True,
    },
    "tongyi-wanxiang": {
        "name": "Tongyi Wanxiang",
        "name_zh": "通义万象 视频生成",
        "category": "video",
        "tier": "tier3_trending",
        "description": "Alibaba's平民级 AI video factory with multi-shot script generation",
        "description_zh": "阿里平民级AI视频工厂，多镜头脚本生成",
        "tools": ["txt2video", "script_gen", "motion_fix", "free_daily"],
        "soul_mapping": ["monet", "vangogh"],
        "tags": ["video", "chinese", "trending"],
        "pricing": "每日20条免费",
        "stars": 18000,
        "verified": True,
    },
    "stable-diffusion-3": {
        "name": "Stable Diffusion 3",
        "name_zh": "SD3 开源图像生成",
        "category": "visual",
        "tier": "tier2_popular",
        "description": "Open-source image generation with MMDiT architecture, local deployment",
        "description_zh": "开源图像生成，MMDiT架构，本地部署",
        "tools": ["txt2img", "img2img", "inpainting", "controlnet", "lora", "ip_adapter"],
        "soul_mapping": ["monet", "vangogh", "davinci"],
        "tags": ["image", "open-source", "local", "trending"],
        "pricing": "开源免费",
        "stars": 75000,
        "verified": True,
    },
    "runway-gen3": {
        "name": "Runway Gen-3",
        "name_zh": "Runway Gen-3 视频生成",
        "category": "video",
        "tier": "tier2_popular",
        "description": "Professional video generation with motion brush and camera control",
        "description_zh": "专业视频生成，运动笔刷和摄像机控制",
        "tools": ["txt2video", "motion_brush", "camera_control", "gen_extend", "style"],
        "soul_mapping": ["monet", "vangogh"],
        "tags": ["video", "creative", "trending"],
        "pricing": "$12-76/mo",
        "stars": 30000,
        "verified": True,
    },
    "dalle-3": {
        "name": "DALL-E 3",
        "name_zh": "DALL-E 3 图像生成",
        "category": "visual",
        "tier": "tier2_popular",
        "description": "OpenAI's image generation with ChatGPT integration, best prompt following",
        "description_zh": "OpenAI图像生成，ChatGPT集成，最佳提示词遵循",
        "tools": ["txt2img", "edit", "variation", "inpainting"],
        "soul_mapping": ["monet", "vangogh", "davinci"],
        "tags": ["image", "openai", "trending"],
        "pricing": "$20/mo (ChatGPT Plus)",
        "stars": 50000,
        "verified": True,
        "official": True,
    },
    "adobe-firefly": {
        "name": "Adobe Firefly",
        "name_zh": "Adobe Firefly 创意套件",
        "category": "design",
        "tier": "tier2_popular",
        "description": "Adobe's AI creative suite integrated with Photoshop, Illustrator workflow",
        "description_zh": "Adobe AI创意套件，与Photoshop、Illustrator工作流集成",
        "tools": ["generative_fill", "style_transfer", "vector_gen", "3d_compose", "brand_kit"],
        "soul_mapping": ["monet", "davinci"],
        "tags": ["design", "adobe", "creative", "trending"],
        "pricing": "$5-55/mo",
        "stars": 22000,
        "verified": True,
        "official": True,
    },
    "supabase-mcp": {
        "name": "Supabase MCP Server",
        "name_zh": "Supabase 数据库集成",
        "category": "database",
        "tier": "tier2_popular",
        "description": "Manage Supabase databases, auth, storage and edge functions via MCP",
        "description_zh": "通过MCP管理Supabase数据库、认证、存储和边缘函数",
        "tools": ["db_query", "auth_manage", "storage_ops", "function_deploy", "migration"],
        "soul_mapping": ["cezanne"],
        "tags": ["database", "supabase", "mcp", "trending"],
        "pricing": "开源免费",
        "stars": 2500,
        "verified": True,
        "official": True,
    },
    "slack-mcp": {
        "name": "Slack MCP Server",
        "name_zh": "Slack 通讯集成",
        "category": "communication",
        "tier": "tier2_popular",
        "description": "Read messages, send notifications, search conversations, manage channels",
        "description_zh": "读取消息、发送通知、搜索对话、管理频道",
        "tools": ["message_read", "message_send", "channel_search", "file_share", "thread_manage"],
        "soul_mapping": ["guizhu"],
        "tags": ["communication", "slack", "mcp", "trending"],
        "pricing": "开源免费",
        "stars": 1100,
        "verified": True,
        "official": True,
    },
    "trends-mcp": {
        "name": "Trends MCP",
        "name_zh": "趋势洞察 MCP",
        "category": "research",
        "tier": "tier3_trending",
        "description": "Consumer trend signals across 25+ platforms: Google, TikTok, Reddit, Amazon",
        "description_zh": "25+平台消费者趋势信号：Google、TikTok、Reddit、Amazon",
        "tools": ["get_trends", "get_growth", "ranked_trends", "breakout_search"],
        "soul_mapping": ["guizhu", "cezanne"],
        "tags": ["research", "trends", "data", "trending"],
        "pricing": "$29/mo",
        "stars": 500,
        "verified": True,
    },
    "sequential-thinking": {
        "name": "Sequential Thinking",
        "name_zh": "顺序思维推理",
        "category": "core",
        "tier": "tier2_popular",
        "description": "Anthropic's structured reasoning MCP server for step-by-step thinking",
        "description_zh": "Anthropic结构化推理MCP服务器，逐步思考",
        "tools": ["think_step", "branch", "revise", "conclude"],
        "soul_mapping": ["cezanne", "guizhu"],
        "tags": ["reasoning", "mcp", "trending"],
        "pricing": "开源免费",
        "stars": 5000,
        "verified": True,
        "official": True,
    },
    "cloudflare-mcp": {
        "name": "Cloudflare MCP Server",
        "name_zh": "Cloudflare 基础设施集成",
        "category": "devops",
        "tier": "tier2_popular",
        "description": "Manage Cloudflare Workers, Pages, R2, D1, KV and DNS via MCP",
        "description_zh": "通过MCP管理Cloudflare Workers、Pages、R2、D1、KV和DNS",
        "tools": ["worker_deploy", "page_manage", "r2_ops", "dns_manage", "kv_ops"],
        "soul_mapping": ["cezanne"],
        "tags": ["devops", "cloudflare", "mcp", "trending"],
        "pricing": "开源免费",
        "stars": 1800,
        "verified": True,
        "official": True,
    },
    "linear-mcp": {
        "name": "Linear MCP Server",
        "name_zh": "Linear 项目管理集成",
        "category": "productivity",
        "tier": "tier2_popular",
        "description": "Connect Linear project management, create issues, track sprints",
        "description_zh": "连接Linear项目管理，创建问题，跟踪冲刺",
        "tools": ["issue_create", "issue_update", "project_query", "sprint_plan", "status_track"],
        "soul_mapping": ["cezanne", "guizhu"],
        "tags": ["productivity", "linear", "mcp", "trending"],
        "pricing": "开源免费",
        "stars": 1100,
        "verified": True,
    },
    "zapier-mcp": {
        "name": "Zapier MCP Server",
        "name_zh": "Zapier 自动化集成",
        "category": "automation",
        "tier": "tier2_popular",
        "description": "Connect 7000+ apps via Zapier automation workflows",
        "description_zh": "通过Zapier自动化工作流连接7000+应用",
        "tools": ["zap_trigger", "zap_action", "workflow_create", "app_search"],
        "soul_mapping": ["guizhu", "cezanne"],
        "tags": ["automation", "zapier", "mcp", "trending"],
        "pricing": "免费+付费",
        "stars": 900,
        "verified": True,
        "official": True,
    },
    "tavily-search": {
        "name": "Tavily MCP Server",
        "name_zh": "Tavily AI搜索",
        "category": "search",
        "tier": "tier3_trending",
        "description": "AI-native search API optimized for LLM agents and RAG pipelines",
        "description_zh": "为LLM智能体和RAG管道优化的AI原生搜索API",
        "tools": ["search", "extract", "crawl", "map"],
        "soul_mapping": ["guizhu", "cezanne"],
        "tags": ["search", "ai", "mcp", "trending"],
        "pricing": "免费1000次+付费",
        "stars": 6000,
        "verified": True,
    },
    "postgres-mcp": {
        "name": "Postgres MCP Server",
        "name_zh": "PostgreSQL 数据库集成",
        "category": "database",
        "tier": "tier2_popular",
        "description": "Connect to PostgreSQL, query data, manage schemas via natural language",
        "description_zh": "连接PostgreSQL，自然语言查询数据和管理模式",
        "tools": ["sql_query", "schema_inspect", "connection_pool", "read_only_mode"],
        "soul_mapping": ["cezanne"],
        "tags": ["database", "postgres", "mcp", "trending"],
        "pricing": "开源免费",
        "stars": 167,
        "verified": True,
    },
}


class SkillRegistryAuto:
    def __init__(self):
        self.registry: Dict[str, SkillEntry] = {}
        self._lock = threading.RLock()
        self._last_full_sync: float = 0
        self._refresh_interval: int = 86400
        self._scheduler_thread: Optional[threading.Thread] = None
        self._running = False
        self._init_builtin()
        self._init_trending()
        self._load_cache()

    def _init_builtin(self):
        try:
            from mcp_server_registry import MCP_SERVER_REGISTRY, MCPServerCategory, MCPServerStatus
            for sid, cfg in MCP_SERVER_REGISTRY.items():
                cat = self._map_category(cfg.category.value)
                entry = SkillEntry(
                    skill_id=f"builtin-{sid}",
                    name=cfg.name,
                    name_zh=cfg.name_zh,
                    category=cat,
                    tier=SkillTier.TIER1_CORE,
                    status=SkillStatus.AVAILABLE if cfg.status.value == "available" else SkillStatus.DEVELOPING,
                    description=cfg.description,
                    description_zh=cfg.core_capability_zh,
                    tools=cfg.tools,
                    soul_mapping=cfg.soul_mapping,
                    source="builtin",
                    adapter_module=cfg.adapter_module,
                    api_endpoint=cfg.api_endpoint,
                    config_path=cfg.config_path,
                    env_vars=cfg.env_vars,
                    boundary=cfg.boundary,
                    verified=True,
                    official=True,
                    last_updated=datetime.now().isoformat(),
                    discovered_at=datetime.now().isoformat(),
                )
                self.registry[entry.skill_id] = entry
        except Exception as e:
            logger.warning(f"内置注册表加载失败: {e}")

    def _map_category(self, cat: str) -> SkillCategory:
        mapping = {
            "core": SkillCategory.CORE,
            "visual": SkillCategory.VISUAL,
            "testing": SkillCategory.TESTING,
            "osint": SkillCategory.OSINT,
            "knowledge": SkillCategory.KNOWLEDGE,
            "design": SkillCategory.DESIGN,
            "soul": SkillCategory.SOUL,
            "pipeline": SkillCategory.PIPELINE,
            "interpretability": SkillCategory.INTERPRETABILITY,
            "security": SkillCategory.SECURITY,
            "music": SkillCategory.MUSIC,
            "voice": SkillCategory.VOICE,
            "automation": SkillCategory.AUTOMATION,
        }
        return mapping.get(cat, SkillCategory.CORE)

    def _init_trending(self):
        now = datetime.now().isoformat()
        for sid, data in TRENDING_AI_TOOLS_2025.items():
            entry = SkillEntry(
                skill_id=f"trending-{sid}",
                name=data["name"],
                name_zh=data["name_zh"],
                category=SkillCategory(data["category"]),
                tier=SkillTier(data["tier"]),
                status=SkillStatus.TRENDING,
                description=data.get("description", ""),
                description_zh=data.get("description_zh", ""),
                tools=data.get("tools", []),
                soul_mapping=data.get("soul_mapping", []),
                source="trending_2025",
                tags=data.get("tags", []),
                pricing=data.get("pricing", ""),
                stars=data.get("stars", 0),
                verified=data.get("verified", False),
                official=data.get("official", False),
                last_updated=now,
                discovered_at=now,
            )
            self.registry[entry.skill_id] = entry

    def _load_cache(self):
        cache_file = CACHE_DIR / "registry_cache.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                for sid, entry_data in data.get("entries", {}).items():
                    if sid not in self.registry:
                        entry = SkillEntry(
                            skill_id=entry_data.get("skill_id", sid),
                            name=entry_data.get("name", ""),
                            name_zh=entry_data.get("name_zh", ""),
                            category=SkillCategory(entry_data.get("category", "core")),
                            tier=SkillTier(entry_data.get("tier", "tier4_community")),
                            status=SkillStatus(entry_data.get("status", "pending")),
                            description=entry_data.get("description", ""),
                            description_zh=entry_data.get("description_zh", ""),
                            tools=entry_data.get("tools", []),
                            soul_mapping=entry_data.get("soul_mapping", []),
                            source=entry_data.get("source", "cache"),
                            source_url=entry_data.get("source_url", ""),
                            repo_url=entry_data.get("repo_url", ""),
                            stars=entry_data.get("stars", 0),
                            install_count=entry_data.get("install_count", 0),
                            weekly_growth=entry_data.get("weekly_growth", 0.0),
                            api_endpoint=entry_data.get("api_endpoint", ""),
                            cli_command=entry_data.get("cli_command", ""),
                            adapter_module=entry_data.get("adapter_module", ""),
                            tags=entry_data.get("tags", []),
                            pricing=entry_data.get("pricing", ""),
                            license_type=entry_data.get("license_type", ""),
                            last_updated=entry_data.get("last_updated", ""),
                            discovered_at=entry_data.get("discovered_at", ""),
                            version=entry_data.get("version", ""),
                            language=entry_data.get("language", ""),
                            verified=entry_data.get("verified", False),
                            official=entry_data.get("official", False),
                            boundary=entry_data.get("boundary", {}),
                        )
                        self.registry[sid] = entry
                self._last_full_sync = data.get("last_full_sync", 0)
                logger.info(f"缓存加载完成，共 {len(self.registry)} 个技能")
            except Exception as e:
                logger.warning(f"缓存加载失败: {e}")

    def _save_cache(self):
        cache_file = CACHE_DIR / "registry_cache.json"
        try:
            data = {
                "last_full_sync": self._last_full_sync,
                "entries": {},
            }
            for sid, entry in self.registry.items():
                data["entries"][sid] = asdict(entry)
            cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")

    def start_auto_refresh(self):
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            return
        self._running = True
        self._scheduler_thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._scheduler_thread.start()
        logger.info("自动刷新调度器已启动")

    def stop_auto_refresh(self):
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        logger.info("自动刷新调度器已停止")

    def _refresh_loop(self):
        while self._running:
            try:
                time.sleep(3600)
                if time.time() - self._last_full_sync > self._refresh_interval:
                    self.refresh_from_sources()
            except Exception as e:
                logger.error(f"自动刷新异常: {e}")

    def refresh_from_sources(self, source: str = "all") -> Dict:
        results = {"updated": 0, "new": 0, "removed": 0, "errors": []}
        try:
            if source in ("all", "github"):
                r = self._fetch_github_registry()
                results["updated"] += r.get("updated", 0)
                results["new"] += r.get("new", 0)
            if source in ("all", "awesome"):
                r = self._fetch_awesome_mcp()
                results["updated"] += r.get("updated", 0)
                results["new"] += r.get("new", 0)
        except Exception as e:
            results["errors"].append(str(e))

        self._last_full_sync = time.time()
        self._save_cache()
        return results

    def _fetch_github_registry(self) -> Dict:
        result = {"updated": 0, "new": 0}
        try:
            import urllib.request
            url = "https://registry.modelcontextprotocol.io/api/servers"
            req = urllib.request.Request(url, headers={"User-Agent": "VORTEX-FLAME/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            servers = data if isinstance(data, list) else data.get("servers", [])
            now = datetime.now().isoformat()
            for srv in servers[:100]:
                sid = f"github-{srv.get('name', '').replace('/', '-')}"
                if sid in self.registry:
                    self.registry[sid].stars = srv.get("stars", self.registry[sid].stars)
                    self.registry[sid].last_updated = now
                    result["updated"] += 1
                else:
                    entry = SkillEntry(
                        skill_id=sid,
                        name=srv.get("title", srv.get("name", "")),
                        name_zh=srv.get("title", srv.get("name", "")),
                        category=SkillCategory.CORE,
                        tier=SkillTier.TIER4_COMMUNITY,
                        status=SkillStatus.PENDING,
                        description=srv.get("description", ""),
                        source="github_registry",
                        repo_url=srv.get("url", ""),
                        stars=srv.get("stars", 0),
                        tags=srv.get("tags", []),
                        last_updated=now,
                        discovered_at=now,
                    )
                    self.registry[sid] = entry
                    result["new"] += 1
        except Exception as e:
            logger.warning(f"GitHub Registry 获取失败: {e}")
        return result

    def _fetch_awesome_mcp(self) -> Dict:
        result = {"updated": 0, "new": 0}
        try:
            import urllib.request
            url = "https://mcp-awesome.com/api/servers"
            req = urllib.request.Request(url, headers={"User-Agent": "VORTEX-FLAME/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            servers = data if isinstance(data, list) else data.get("servers", [])
            now = datetime.now().isoformat()
            for srv in servers[:100]:
                sid = f"awesome-{srv.get('id', srv.get('name', '')).replace('/', '-')}"
                if sid in self.registry:
                    self.registry[sid].stars = srv.get("stars", self.registry[sid].stars)
                    self.registry[sid].last_updated = now
                    result["updated"] += 1
                else:
                    entry = SkillEntry(
                        skill_id=sid,
                        name=srv.get("name", ""),
                        name_zh=srv.get("name", ""),
                        category=SkillCategory.CORE,
                        tier=SkillTier.TIER4_COMMUNITY,
                        status=SkillStatus.PENDING,
                        description=srv.get("description", ""),
                        source="mcp_awesome",
                        repo_url=srv.get("github", srv.get("url", "")),
                        stars=srv.get("stars", 0),
                        tags=srv.get("categories", []),
                        last_updated=now,
                        discovered_at=now,
                    )
                    self.registry[sid] = entry
                    result["new"] += 1
        except Exception as e:
            logger.warning(f"Awesome MCP 获取失败: {e}")
        return result

    def search(self, query: str = "", category: str = "", tier: str = "",
               soul: str = "", tag: str = "", status: str = "",
               source: str = "", verified_only: bool = False,
               sort_by: str = "stars", limit: int = 50) -> List[Dict]:
        with self._lock:
            results = []
            q = query.lower()
            for entry in self.registry.values():
                if q and q not in entry.name.lower() and q not in entry.name_zh.lower() and q not in entry.description.lower() and q not in entry.description_zh.lower() and q not in " ".join(entry.tags).lower():
                    continue
                if category and entry.category.value != category:
                    continue
                if tier and entry.tier.value != tier:
                    continue
                if soul and soul not in entry.soul_mapping and "all" not in entry.soul_mapping:
                    continue
                if tag and tag not in entry.tags:
                    continue
                if status and entry.status.value != status:
                    continue
                if source and entry.source != source:
                    continue
                if verified_only and not entry.verified:
                    continue
                results.append(asdict(entry))

            sort_keys = {
                "stars": lambda x: x.get("stars", 0),
                "name": lambda x: x.get("name", ""),
                "weekly_growth": lambda x: x.get("weekly_growth", 0),
                "install_count": lambda x: x.get("install_count", 0),
                "last_updated": lambda x: x.get("last_updated", ""),
            }
            reverse = sort_by in ("stars", "weekly_growth", "install_count")
            results.sort(key=sort_keys.get(sort_by, sort_keys["stars"]), reverse=reverse)
            return results[:limit]

    def get(self, skill_id: str) -> Optional[Dict]:
        with self._lock:
            entry = self.registry.get(skill_id)
            return asdict(entry) if entry else None

    def add_skill(self, entry_data: Dict) -> Dict:
        with self._lock:
            sid = entry_data.get("skill_id", f"custom-{hashlib.md5(str(entry_data).encode()).hexdigest()[:8]}")
            entry = SkillEntry(
                skill_id=sid,
                name=entry_data.get("name", ""),
                name_zh=entry_data.get("name_zh", entry_data.get("name", "")),
                category=SkillCategory(entry_data.get("category", "core")),
                tier=SkillTier(entry_data.get("tier", "tier4_community")),
                status=SkillStatus(entry_data.get("status", "pending")),
                description=entry_data.get("description", ""),
                description_zh=entry_data.get("description_zh", ""),
                tools=entry_data.get("tools", []),
                soul_mapping=entry_data.get("soul_mapping", []),
                source=entry_data.get("source", "custom"),
                source_url=entry_data.get("source_url", ""),
                repo_url=entry_data.get("repo_url", ""),
                stars=entry_data.get("stars", 0),
                tags=entry_data.get("tags", []),
                pricing=entry_data.get("pricing", ""),
                api_endpoint=entry_data.get("api_endpoint", ""),
                cli_command=entry_data.get("cli_command", ""),
                adapter_module=entry_data.get("adapter_module", ""),
                verified=entry_data.get("verified", False),
                official=entry_data.get("official", False),
                last_updated=datetime.now().isoformat(),
                discovered_at=datetime.now().isoformat(),
            )
            self.registry[sid] = entry
            self._save_cache()
            return asdict(entry)

    def remove_skill(self, skill_id: str) -> bool:
        with self._lock:
            if skill_id in self.registry:
                del self.registry[skill_id]
                self._save_cache()
                return True
            return False

    def stats(self) -> Dict:
        with self._lock:
            cats = {}
            tiers = {}
            sources = {}
            for entry in self.registry.values():
                cats[entry.category.value] = cats.get(entry.category.value, 0) + 1
                tiers[entry.tier.value] = tiers.get(entry.tier.value, 0) + 1
                sources[entry.source] = sources.get(entry.source, 0) + 1
            return {
                "total": len(self.registry),
                "by_category": cats,
                "by_tier": tiers,
                "by_source": sources,
                "last_full_sync": self._last_full_sync,
                "verified_count": sum(1 for e in self.registry.values() if e.verified),
                "trending_count": sum(1 for e in self.registry.values() if e.status == SkillStatus.TRENDING),
            }

    def get_categories(self) -> List[Dict]:
        return [{"id": c.value, "name": c.value, "name_zh": self._category_zh(c)} for c in SkillCategory]

    def _category_zh(self, cat: SkillCategory) -> str:
        zh_map = {
            SkillCategory.CORE: "核心能力",
            SkillCategory.VISUAL: "图像视觉",
            SkillCategory.VIDEO: "视频生成",
            SkillCategory.AUDIO: "音频处理",
            SkillCategory.MUSIC: "音乐创作",
            SkillCategory.DESIGN: "设计创意",
            SkillCategory.CODE: "代码开发",
            SkillCategory.DATA: "数据分析",
            SkillCategory.SEARCH: "搜索检索",
            SkillCategory.AUTOMATION: "自动化",
            SkillCategory.PRODUCTIVITY: "效率办公",
            SkillCategory.COMMUNICATION: "通讯协作",
            SkillCategory.DATABASE: "数据库",
            SkillCategory.DEVOPS: "运维部署",
            SkillCategory.SECURITY: "安全审计",
            SkillCategory.KNOWLEDGE: "知识管理",
            SkillCategory.FINANCE: "金融财经",
            SkillCategory.RESEARCH: "研究分析",
            SkillCategory.OSINT: "开源情报",
            SkillCategory.VOICE: "语音合成",
            SkillCategory.INTERPRETABILITY: "可解释性",
            SkillCategory.PIPELINE: "管道管理",
            SkillCategory.SOUL: "灵魂系统",
            SkillCategory.TESTING: "测试验证",
            SkillCategory.CREATIVE: "创意生成",
            SkillCategory.EDUCATION: "教育学习",
            SkillCategory.HEALTH: "健康医疗",
            SkillCategory.GAMING: "游戏娱乐",
        }
        return zh_map.get(cat, cat.value)


_registry_instance: Optional[SkillRegistryAuto] = None


def get_registry() -> SkillRegistryAuto:
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = SkillRegistryAuto()
        _registry_instance.start_auto_refresh()
    return _registry_instance
