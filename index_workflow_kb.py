#!/usr/bin/env python3
"""
VORTEX FLAME Workflow Knowledge Base Indexer
=============================================

Indexes top-tier workflow engine repos into soul_memory for dual-pathway retrieval:
  Path A (RAG): text facts for factual retrieval
  Path B (C-JEPA): causal structure for world model reasoning

Workflow Repos:
  1. n8n:       https://github.com/n8n-io/n8n
  2. Flowise:   https://github.com/FlowiseAI/Flowise
  3. Dify:      https://github.com/langgenius/dify
  4. LangFlow:  https://github.com/langflow-ai/langflow
  5. Qdrant:    https://github.com/qdrant/qdrant
  6. Chroma:    https://github.com/chroma-core/chroma
  7. LangChain: https://github.com/langchain-ai/langchain

Also indexes structured workflow knowledge (n8n/Flowise/Dify/MCP/Financial/Doc processing)
as curated entries for both pathways.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(os.path.dirname(__file__)))

from soul_memory import SoulMemoryEngine

WF_ROOT = r"D:\VORTEX_FLAME\kb_workflow"

WORKFLOW_REPOS = [
    {
        "name": "n8n",
        "url": "https://github.com/n8n-io/n8n",
        "soul": "cezanne",
        "category": "knowledge",
        "tags": ["n8n", "workflow", "automation", "api", "excel", "word", "ecommerce", "wechat"],
        "description": "Global #1 universal workflow automation engine. 2000+ native nodes: e-commerce, WeChat, DingTalk, Feishu, Excel, Word, database, HTTP, JSON, image processing. Docker self-hosted, no wall.",
    },
    {
        "name": "flowise",
        "url": "https://github.com/FlowiseAI/Flowise",
        "soul": "guizhu",
        "category": "knowledge",
        "tags": ["flowise", "workflow", "ai", "rag", "agent", "mcp", "drawing", "sd", "comfyui"],
        "description": "Global #1 AI-native workflow. Agent+RAG+MCP+Drawing. Native DeepSeek, Qwen, Wenxin, Ollama. SD/ComfyUI/MJ drawing nodes. Outputs structured JSON for n8n execution.",
    },
    {
        "name": "dify",
        "url": "https://github.com/langgenius/dify",
        "soul": "guizhu",
        "category": "knowledge",
        "tags": ["dify", "workflow", "knowledge-base", "chinese", "rag", "wechat", "team"],
        "description": "China #1 all-in-one knowledge base + lightweight workflow. Full Chinese UI, team collaboration, permission management. One-click RAG: document upload, auto-chunking, vector DB, KB Q&A. WeChat/Enterprise WeChat integration.",
    },
    {
        "name": "langflow",
        "url": "https://github.com/langflow-ai/langflow",
        "soul": "cezanne",
        "category": "knowledge",
        "tags": ["langflow", "langchain", "workflow", "agent", "mcp", "tool-chain"],
        "description": "LangChain official visual workflow. MCP, Agent, tool chain global standard. For deep Agent development, MCP protocol integration, complex reasoning, long-term memory.",
    },
    {
        "name": "qdrant",
        "url": "https://github.com/qdrant/qdrant",
        "soul": "einstein",
        "category": "knowledge",
        "tags": ["qdrant", "vector-db", "rag", "memory", "embedding", "search"],
        "description": "Enterprise-grade #1 open-source vector DB. Fast, stable, Rust-based. Shared memory substrate for MCP+RAG. Recommended for production deployment.",
    },
    {
        "name": "chroma",
        "url": "https://github.com/chroma-core/chroma",
        "soul": "einstein",
        "category": "knowledge",
        "tags": ["chroma", "vector-db", "rag", "local", "lightweight", "embedding"],
        "description": "Lightweight local-first vector DB. Best for personal/small teams. Zero-config, Python-native, perfect for Flowise/Dify local RAG.",
    },
    {
        "name": "langchain",
        "url": "https://github.com/langchain-ai/langchain",
        "soul": "cezanne",
        "category": "knowledge",
        "tags": ["langchain", "llm", "agent", "chain", "rag", "tool", "memory"],
        "description": "Global #1 LLM application framework. Chain/Agent/Tool/Memory abstraction layer. Foundation for all workflow engines and Agent systems.",
    },
]

CURATED_KNOWLEDGE = [
    {
        "soul": "cezanne",
        "category": "knowledge",
        "topic": "[Workflow] n8n Universal Automation Knowledge Base",
        "tags": ["n8n", "workflow", "automation", "excel", "word", "financial", "ecommerce"],
        "text": """n8n Universal Automation Engine - Complete Knowledge Base

CORE POSITIONING:
Pure process-driven, all-platform API connected. Suitable for: multi-system linkage, e-commerce listing, image editing & publishing, WeChat push, form trigger, scheduled tasks, cross-platform sync.
One-liner: The automation brain that can connect to every API in the world.

CORE CAPABILITIES (Universal Modules, All Scenarios Reusable):
- Visual drag-and-drop nodes, zero-code orchestration
- Built-in 2000+ native nodes: Taobao, Pinduoduo, JD, Douyin, WeChat, Enterprise WeChat, DingTalk, Feishu, OSS, SFTP, Database, HTTP, JSON, Image Processing
- Support: AI Drawing API (SD, MJ, Jimeng, Tongyi Wanxiang), File Processing, Excel Parsing, Email, SMS
- Support: Scheduled, Webhook, Form, WeChat Message Trigger
- Support: Conditional Logic, Loop, Branch, Error Retry, Logging

EXCEL CAPABILITIES:
- Read/Write/Pivot/Formula/Batch Export, CSV, Google Sheets
- Auto-generate financial reports, profit statements, reconciliation tables
- Batch data processing with formula support

WORD CAPABILITIES:
- Generate contracts, invoices, reports
- Batch replace, mail merge
- PDF conversion

FINANCIAL CAPABILITIES:
- Connect Kingdee/Yonyou/QuickBooks/Xero/SAP
- Auto reconciliation, invoice OCR, expense workflow
- Profit statement generation, tax calculation

ECOMMERCE CAPABILITIES:
- Taobao/Pinduoduo/JD/Douyin listing, image editing, publishing
- Multi-platform sync, price monitoring, inventory management

NOTIFICATION CAPABILITIES:
- WeChat/Enterprise WeChat/DingTalk/Feishu auto-reporting
- Email/SMS alerts
- Custom webhook notifications

KNOWLEDGE BASE INTEGRATION:
- Vector DB: Qdrant/Milvus/Chroma (private product DB, script DB, listing parameter DB)
- Structured DB: MySQL/SQLite (product ID, price, specs, platform accounts, listing history)
- Document DB: Dify/Notion (product manuals, listing SOP, error handling guides)

UNIVERSAL NODE LIST:
1. Trigger: Webhook, Scheduled, WeChat Message, Form
2. AI: Call LLM, Call Drawing API, Copy Generation
3. E-commerce: Taobao Open Platform, Pinduoduo API, JD Wanxiang API, Product Listing/Delisting, Image Upload
4. Notification: Personal WeChat, Enterprise WeChat, DingTalk, Feishu
5. Knowledge Base: Vector Search, Document Read, Parameter Read
6. Storage: OSS, Local File, Database Write

TYPICAL CLOSED LOOP:
Customer WeChat trigger -> n8n reads product KB -> calls AI image edit/generate -> calls e-commerce API multi-platform listing -> writes result to KB -> WeChat auto-report

DEPLOYMENT:
- Open source free, Docker one-click self-hosted
- China mirror available, no wall issues""",
    },
    {
        "soul": "guizhu",
        "category": "knowledge",
        "topic": "[Workflow] Flowise AI-Native Workflow Knowledge Base",
        "tags": ["flowise", "workflow", "ai", "rag", "agent", "mcp", "drawing", "sd"],
        "text": """Flowise AI-Native Workflow Engine - Complete Knowledge Base

CORE POSITIONING:
AI-native workflow, designed specifically for LLMs, knowledge bases, Agents, image generation.
One-liner: A workflow for AI, understands RAG better than n8n.

CORE CAPABILITIES:
- Drag-and-drop Agent orchestration, RAG orchestration, multi-model chaining
- Native support: DeepSeek, Qwen, Wenxin, Qianfan, local Ollama models
- Native drawing nodes: Stable Diffusion, ComfyUI, MJ
- Knowledge base: Chroma, Qdrant, Milvus, local document RAG
- Can output structured data, directly feed to n8n for listing automation

RAG KNOWLEDGE BASE:
- PDF/Word/Excel/Image knowledge base Q&A
- Auto chunking, vectorization, retrieval, reranking
- Support hybrid search (semantic + keyword)

AGENT CAPABILITIES:
- Tool calling, counterfactual reasoning, planning
- Function calling, structured output
- Multi-step reasoning chains

MCP NATIVE ADAPTATION:
- Connect long-term memory, cross-session memory, slot memory
- Memory decay, conflict resolution
- Integration with VORTEX_FLAME soul memory system

DRAWING CAPABILITIES:
- SD/ComfyUI product image generation
- Image-to-image, white background generation, background removal
- Batch image processing pipeline

OUTPUT:
- JSON structured output (for n8n listing execution)
- Parameter extraction (product name, price, specs, images)
- Batch processing support

KNOWLEDGE BASE:
- Vector: Chroma (lightweight local first choice), Qdrant (enterprise)
- Document: PDF/Markdown/Excel (product materials, scripts, SOP, spec sheets)
- Memory: Short-term conversation memory + Long-term vector memory (customer service history, product conversations)

UNIVERSAL NODE LIST:
1. LLM Nodes: LLM, FunctionCall, Tool Calling
2. RAG Nodes: Document Load, Chunk, Vectorize, Retrieve, Rerank
3. Drawing Nodes: SD, MJ, Image-to-Image, White Background, Background Removal
4. Output Nodes: JSON Structured, Parameter Output (for n8n listing)

TYPICAL CLOSED LOOP:
Read product KB -> AI generates selling point copy -> AI generates/modifies product images -> outputs listing parameters -> passes to n8n for listing execution

DEPLOYMENT:
- Open source free, Docker self-hosted, lightweight""",
    },
    {
        "soul": "guizhu",
        "category": "knowledge",
        "topic": "[Workflow] Dify All-in-One Knowledge Base Knowledge Base",
        "tags": ["dify", "workflow", "knowledge-base", "chinese", "rag", "wechat"],
        "text": """Dify All-in-One Knowledge Base + Lightweight Workflow - Complete Knowledge Base

CORE POSITIONING:
China #1: All-in-one 'Knowledge Base + Lightweight Workflow + Customer Service Dialogue', no need for two systems, out of the box.
Weakness: E-commerce API nodes weak, must pair with n8n for execution.
Suitable for: Internal knowledge base accumulation, customer service, scripts, notifications.

CORE CAPABILITIES:
- Full Chinese UI, team collaboration, permission management
- One-stop RAG: Document upload, auto chunking, vector DB, knowledge base Q&A
- Built-in lightweight workflow: Conditional branching, API calling, notification
- WeChat/Enterprise WeChat one-click integration
- Support models: Qwen, Wenxin, DeepSeek, OpenAI

KNOWLEDGE BASE (Built-in, No Extra Install):
- Built-in vector DB
- Structured knowledge base, dialogue knowledge base, product knowledge base, SOP knowledge base
- Auto chunking strategies: automatic, custom, Q&A format

WORKFLOW NODES:
- LLM node, knowledge retrieval node, question classifier
- Conditional branching, variable aggregation
- HTTP request, code execution
- WeChat/Enterprise WeChat notification

TYPICAL CLOSED LOOP:
Customer question -> Search Dify product KB -> AI generates copy -> Call external drawing -> Push to WeChat

DEPLOYMENT:
- Open source self-hosted, Docker one-click deploy, China ecosystem most friendly""",
    },
    {
        "soul": "cezanne",
        "category": "knowledge",
        "topic": "[Workflow] LangFlow Agent Standard Knowledge Base",
        "tags": ["langflow", "langchain", "agent", "mcp", "tool-chain"],
        "text": """LangFlow - LangChain Official Visual Workflow - Complete Knowledge Base

CORE POSITIONING:
LangChain official visual workflow, MCP, Agent, tool chain global standard.
For deep Agent development, MCP protocol integration, complex reasoning, long-term memory.

CORE CAPABILITIES:
- Visual drag-and-drop LangChain component assembly
- Native MCP server/client integration
- Agent types: ReAct, OpenAI Functions, Plan-and-Execute
- Memory: Conversation, Summary, Vector Store
- Tool integration: Python REPL, Web Search, API calls
- Multi-model support: OpenAI, Anthropic, local models

MCP INTEGRATION:
- Native MCP server nodes
- MCP tool discovery and invocation
- Cross-agent memory sharing via MCP protocol

USE CASES:
- Complex multi-step reasoning workflows
- Agent with persistent memory
- Tool-augmented generation
- Multi-agent coordination

DEPLOYMENT:
- Open source, pip install langflow
- Docker deployment available""",
    },
    {
        "soul": "einstein",
        "category": "knowledge",
        "topic": "[Workflow] Vector DB Knowledge Base (Qdrant + Chroma)",
        "tags": ["qdrant", "chroma", "vector-db", "rag", "memory", "embedding"],
        "text": """Vector DB Knowledge Base - Shared Memory Substrate for All Workflows

QDRANT (Enterprise #1):
- Open source, Rust-based, high performance
- Supports filtering, payload, geo search
- gRPC + REST API
- Horizontal scaling, sharding
- Production-proven, recommended for enterprise deployment
- Docker: docker run -p 6333:6333 qdrant/qdrant

CHROMA (Lightweight Local):
- Zero-config, Python-native
- Perfect for Flowise/Dify local RAG
- pip install chromadb
- Supports embedding functions, metadata filtering
- Best for personal/small team use

MILVUS (Ultra-large Scale):
- Cloud-native, supports billion-scale vectors
- GPU-accelerated indexing
- Suitable for production with massive data

SHARED USAGE PATTERN:
All workflow engines share the same vector DB layer:
- n8n: HTTP call to vector search API
- Flowise: Native Chroma/Qdrant connector
- Dify: Built-in vector DB (can connect external)
- VORTEX_FLAME: Soul memory engine + World-Embedding cache

MEMORY ARCHITECTURE:
Vector DB (long-term memory) + SQLite (structured memory) + FTS5 (full-text search)
= Complete MCP-compatible memory stack""",
    },
    {
        "soul": "montesquieu",
        "category": "knowledge",
        "topic": "[Workflow] Financial Automation Knowledge Base",
        "tags": ["financial", "automation", "accounting", "invoice", "reconciliation", "tax"],
        "text": """Financial Automation Knowledge Base - Universal Financial Skills

SYSTEM INTEGRATION (n8n Native Nodes):
- China: Kingdee, Yonyou, Chanjet, Enterprise WeChat Expense, Alipay/WeChat Bills
- Global: QuickBooks, Xero, SAP

FINANCIAL AUTOMATION CLOSED LOOP:
Invoice OCR recognition -> Extract amount/tax ID -> Write to Excel -> Auto reconciliation -> Generate profit statement -> Push to finance WeChat -> Store in financial KB

FINANCIAL KB CONTENT:
- Chart of accounts, invoice templates, expense SOP
- Tax rules, contract templates, historical reports
- Reconciliation rules, approval workflows

KEY AUTOMATION SCENARIOS:
1. Auto Invoice Processing: OCR scan -> data extraction -> verification -> entry -> filing
2. Auto Reconciliation: Bank statement import -> match with ledger -> flag discrepancies -> notify
3. Expense Workflow: Submit -> approve -> payment -> record -> report
4. Financial Reporting: Auto-generate P&L, balance sheet, cash flow from transaction data
5. Tax Calculation: Auto-calculate VAT, income tax -> generate declarations -> file

TOOLS:
- pandas/openpyxl: Excel automation
- python-docx: Contract/report generation
- PyPDF2/pdfplumber: Invoice/receipt parsing
- LibreOffice: Full format conversion (Word<->PDF<->Excel<->Markdown)

DEPLOYMENT:
- n8n handles all financial workflow orchestration
- Dify provides financial knowledge Q&A
- Qdrant stores financial regulation/policy vectors""",
    },
    {
        "soul": "cezanne",
        "category": "knowledge",
        "topic": "[Workflow] Document/Excel/Word/PDF Processing Knowledge Base",
        "tags": ["document", "excel", "word", "pdf", "ocr", "pandas", "openpyxl"],
        "text": """Document Processing Knowledge Base - Universal Doc Skills

OPEN SOURCE TOOLS (Workflow Direct Call):
- Excel: pandas, openpyxl, ExcelJS, LibreOffice (auto formula, pivot, batch export)
- Word: python-docx, LibreOffice, docx-mailmerge (batch contracts, invoices, reports)
- PDF: PyPDF2, pdfplumber, Marker-PDF (OCR, table extraction, text extraction)
- Full Format Conversion: LibreOffice (Word<->PDF<->Excel<->Markdown)

CLOUD/API TOP TIER:
- Microsoft Power Automate (Office suite strongest, Excel/Word native integration)
- Alibaba Modao, Tongyi Document, Tencent Hunyuan Document (China Chinese doc OCR+table strongest)

UNIVERSAL DOCUMENT KB USAGE:
Word/Excel -> Chunk -> Vectorize to Qdrant/Chroma -> Flowise/Dify RAG retrieval -> n8n generate new documents/reports

KEY PATTERNS:
1. Template-based generation: Load Word template -> fill data -> export PDF
2. Batch processing: Read Excel -> process rows -> generate individual docs
3. Data extraction: PDF OCR -> extract tables -> write to database
4. Format conversion: Any format -> Markdown -> vectorize -> RAG

INTEGRATION WITH VORTEX_FLAME:
- CausalKnowledgeExtractor can parse document structure into ObjectGraph
- WorldEmbeddingCache stores document causal representations
- DualPathwayBridge fuses document facts (Path A) with document logic (Path B)""",
    },
    {
        "soul": "guizhu",
        "category": "knowledge",
        "topic": "[Workflow] MCP Memory Protocol Complete Knowledge Base",
        "tags": ["mcp", "memory", "protocol", "agent", "long-term", "slot", "context"],
        "text": """MCP (Memory Context Protocol) Complete Knowledge Base

WHAT IS MCP:
AI long-term memory protocol: gives LLMs long-term memory, cross-session memory, slot memory, world model memory.
This is exactly what VORTEX_FLAME's soul memory layer implements.

GLOBAL STANDARD MCP STACK:
- Protocol: MCP 0.3 (Anthropic-led, global Agent universal)
- Storage: Qdrant/Milvus (vector memory) + SQLite (structured memory)
- Access methods:
  1. Flowise/LangFlow: Native MCP nodes
  2. n8n: HTTP call to MCP memory API
  3. Dify: RAG as simple MCP memory

VORTEX_FLAME AND MCP RELATIONSHIP:
VORTEX_FLAME's Slot memory, counterfactual reasoning, planning = private MCP implementation
Global standard MCP is the industry standard, both can interoperate.

MCP MEMORY LAYERS:
1. Working Memory: Current conversation context (short-term)
2. Episodic Memory: Past interaction summaries (medium-term)
3. Semantic Memory: Knowledge and facts (long-term, vector-stored)
4. Procedural Memory: Skills and workflows (long-term, structured)

MCP IN VORTEX_FLAME:
- SoulMemoryEngine = MCP Working + Episodic + Semantic memory
- WorldEmbeddingCache = MCP World Model memory (C-JEPA slots)
- CausalKnowledgeExtractor = MCP Knowledge structuring pipeline
- DualPathwayBridge = MCP Memory fusion (text + world)

INTEGRATION PATH:
VORTEX_FLAME can expose MCP-compatible API endpoints, allowing:
- n8n to query soul memory via MCP
- Flowise to use VORTEX_FLAME as MCP memory server
- Dify to connect VORTEX_FLAME as external knowledge source""",
    },
    {
        "soul": "beethoven",
        "category": "knowledge",
        "topic": "[Workflow] Agent Skill Library - Universal Skills",
        "tags": ["agent", "skill", "document", "drawing", "ecommerce", "financial", "memory", "notification", "reasoning"],
        "text": """Agent Skill Library - Universal Skills for All Scenarios

UNIVERSAL SKILL LIST (All Scenario Reusable):

1. DOCUMENT SKILLS:
   - Word generation: contracts, invoices, reports, letters
   - Excel reports: pivot tables, charts, auto-calculation
   - PDF parsing: OCR, table extraction, text extraction
   - Invoice OCR: amount, tax ID, date, vendor extraction

2. DRAWING SKILLS:
   - Product image generation: SD/ComfyUI/MJ
   - Image editing: background removal, white background, resize
   - Style transfer: product image style consistency
   - Batch processing: multiple images in parallel

3. E-COMMERCE SKILLS:
   - Multi-platform listing: Taobao/Pinduoduo/JD/Douyin
   - Image editing for listing: main image, detail page, SKU image
   - Price monitoring: competitor price tracking
   - Inventory sync: cross-platform inventory management

4. FINANCIAL SKILLS:
   - Reconciliation: bank vs ledger matching
   - Report generation: P&L, balance sheet, cash flow
   - Expense workflow: submit -> approve -> pay -> record
   - Tax calculation: VAT, income tax auto-calculation

5. MEMORY SKILLS:
   - MCP long-term memory: store and retrieve across sessions
   - Knowledge base search: semantic + keyword hybrid
   - Slot memory: object-centric causal memory (C-JEPA)
   - World model memory: physical rules, causal chains

6. NOTIFICATION SKILLS:
   - WeChat/DingTalk auto-reporting
   - Email alerts with attachments
   - SMS notifications for critical events
   - Custom webhook integration

7. REASONING SKILLS (VORTEX_FLAME C-JEPA):
   - Counterfactual: "What if X changed?"
   - Planning: multi-step task decomposition
   - Causal reasoning: cause-effect chain analysis
   - Physical reasoning: object interaction prediction

SKILL COMPOSITION PATTERN:
Skills can be composed: Document + Financial = Auto financial report generation
Drawing + E-commerce = Auto product image listing
Memory + Reasoning = Knowledge-augmented causal analysis
Notification + All = Auto-reporting for any workflow""",
    },
    {
        "soul": "strategy",
        "category": "knowledge",
        "topic": "[Workflow] Golden Combination - Best Practice",
        "tags": ["golden-combo", "n8n", "flowise", "dify", "qdrant", "best-practice"],
        "text": """Golden Combination - Global Strongest Universal Closed Loop

FLOWISE (AI+RAG+MCP+Drawing) + n8n (Excel/Word/Financial/E-commerce/WeChat Execution) + Dify (Chinese KB) + Qdrant (Vector Memory Substrate)

ONE-LINER DIVISION:
1. Flowise = AI Brain (thinking, RAG, MCP, reasoning)
2. n8n = Universal Hands (Excel/Word/Financial/Listing/Notification)
3. Dify = Chinese Memory Library (team knowledge base)
4. Qdrant = Long-term Memory Substrate (MCP+RAG shared)

USAGE RULES (All Scenarios Route From Here):
1. AI thinking, KB, MCP, drawing -> Flowise
2. Excel, Word, financial, e-commerce, WeChat automation -> n8n
3. Chinese team KB, customer service Q&A -> Dify
4. All memory, RAG substrate -> Qdrant
5. Complex Agent development -> LangFlow + MCP

INTEGRATION WITH VORTEX_FLAME:
VORTEX_FLAME adds the CAUSAL REASONING layer on top of this stack:
- C-JEPA provides world model reasoning (counterfactual, physical, causal)
- DualPathwayBridge fuses workflow knowledge with causal knowledge
- Soul memory provides persistent, personalized memory across all engines

FULL STACK:
Workflow Orchestration (n8n/Flowise/Dify)
+ Document/Financial Processing Layer
+ MCP Protocol Layer
+ Knowledge Base Substrate (Qdrant/Chroma/SQLite)
+ LLM Agent Skill Layer
+ WeChat/E-commerce/ERP System Layer
+ VORTEX_FLAME C-JEPA Causal Reasoning Layer

SCENARIO ROUTING:
- Pure KB Q&A, customer service -> Dify
- AI drawing, product copy, RAG, Agent -> Flowise
- E-commerce listing, cross-platform, WeChat notification, multi-system automation -> n8n
- Complete closed loop (core need) -> Flowise + n8n + Dify three-way combo
- Causal reasoning, counterfactual, physical prediction -> VORTEX_FLAME C-JEPA""",
    },
]

SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", "venv", ".venv", "dist", "build",
    ".tox", "egg-info", ".mypy_cache", ".pytest_cache", "target", "vendor",
    ".next", ".nuxt", "coverage", ".cache", "public", "assets", "static",
    "cypress", "__snapshots__", "fixtures", "mocks", "stubs",
}

DOC_EXTENSIONS = {".md", ".rst", ".txt", ".adoc"}
CODE_EXTENSIONS = {".py", ".js", ".ts", ".rs", ".go", ".java"}
CONFIG_EXTENSIONS = {".yaml", ".yml", ".toml", ".json", ".cfg"}


def read_file_safe(fpath: str, max_chars: int = 5000) -> str:
    try:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(max_chars)
    except Exception:
        return ""


def index_repo_docs(repo_info: dict, repo_path: str, memory: SoulMemoryEngine) -> dict:
    soul = repo_info["soul"]
    category = repo_info["category"]
    tags = repo_info.get("tags", [])
    repo_name = repo_info["name"]

    indexed = 0
    doc_count = 0

    PRIORITY_DIRS = {"docs", "doc", "documentation", "packages", "src"}
    PRIORITY_FILES = {"README.md", "readme.md", "CHANGELOG.md", "CONTRIBUTING.md"}

    all_files = []
    for dirpath, dirnames, filenames in os.walk(repo_path):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext not in DOC_EXTENSIONS:
                continue
            fpath = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(fpath, repo_path)
            is_priority = (
                fname in PRIORITY_FILES
                or any(d in rel_path.split(os.sep) for d in PRIORITY_DIRS)
            )
            all_files.append((rel_path, fpath, is_priority))

    all_files.sort(key=lambda x: (0 if x[2] else 1, x[0]))
    all_files = all_files[:25]

    for rel_path, fpath, _ in all_files:
        content = read_file_safe(fpath, max_chars=6000)
        if len(content.strip()) < 50:
            continue

        title = Path(rel_path).stem.replace("-", " ").replace("_", " ")
        entry_content = {
            "topic": f"[{repo_name}] {title}",
            "source": repo_info["url"],
            "path": rel_path,
            "text": content[:3000],
            "tags": tags,
        }

        try:
            memory.write(soul, category, entry_content, tags=tags)
            indexed += 1
            doc_count += 1
        except Exception:
            pass

        if doc_count >= 20:
            break

    return {"indexed": indexed, "docs": doc_count}


def index_curated_knowledge(memory: SoulMemoryEngine) -> dict:
    indexed = 0
    errors = 0

    for entry in CURATED_KNOWLEDGE:
        try:
            memory.write(
                entry["soul"],
                entry["category"],
                {
                    "topic": entry["topic"],
                    "text": entry["text"],
                    "tags": entry["tags"],
                },
                tags=entry["tags"],
            )
            indexed += 1
            print(f"  OK: {entry['topic'][:60]}")
        except Exception as e:
            errors += 1
            print(f"  ERR: {e}")

    return {"indexed": indexed, "errors": errors}


def main():
    print("=" * 60)
    print("VORTEX FLAME Workflow KB Indexer")
    print("=" * 60)

    memory = SoulMemoryEngine()

    print("\n[1/3] Indexing curated workflow knowledge...")
    curated_result = index_curated_knowledge(memory)
    print(f"  Curated: {curated_result['indexed']} entries indexed")

    print("\n[2/3] Indexing workflow repos from disk...")
    repo_result = {"indexed": 0, "docs": 0, "repos_found": 0}

    for repo_info in WORKFLOW_REPOS:
        repo_path = os.path.join(WF_ROOT, repo_info["name"])
        if not os.path.exists(repo_path):
            print(f"  SKIP (not cloned yet): {repo_info['name']}")
            continue

        print(f"  Indexing: {repo_info['name']}...")
        result = index_repo_docs(repo_info, repo_path, memory)
        repo_result["indexed"] += result["indexed"]
        repo_result["docs"] += result["docs"]
        repo_result["repos_found"] += 1
        print(f"    -> {result['indexed']} entries ({result['docs']} docs)")

    print(f"\n  Repos indexed: {repo_result['repos_found']}")
    print(f"  Total entries: {repo_result['indexed']}")

    print("\n[3/3] Summary...")
    total = curated_result["indexed"] + repo_result["indexed"]
    print(f"  Curated entries: {curated_result['indexed']}")
    print(f"  Repo entries: {repo_result['indexed']}")
    print(f"  TOTAL: {total} workflow knowledge entries indexed")

    print("\nDone! Workflow KB is now available for dual-pathway retrieval.")
    print("  Path A (RAG): soul_memory text search")
    print("  Path B (C-JEPA): CausalKnowledgeExtractor -> ObjectGraph -> World Slots")


if __name__ == "__main__":
    main()

