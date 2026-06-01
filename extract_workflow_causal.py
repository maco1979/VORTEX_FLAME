#!/usr/bin/env python3
"""
VORTEX FLAME Workflow Causal Knowledge Extractor
=================================================

Extracts causal structure from workflow knowledge and stores
in WorldEmbeddingCache for C-JEPA Path B retrieval.

Converts curated workflow knowledge into ObjectGraph format:
  - Entities: workflow engines, tools, protocols, data stores
  - Relations: feeds_into, triggers, depends_on, integrates_with
  - Causal chains: trigger -> process -> output -> feedback
  - Temporal sequences: workflow execution order

This enables C-JEPA to reason about:
  - What happens if a workflow step fails?
  - What are the causal dependencies between engines?
  - Counterfactual: What if we replace n8n with Dify?
"""

import json
import os
import sys
import uuid

sys.path.insert(0, str(os.path.dirname(__file__)))

from causal_knowledge_extractor import CausalKnowledgeExtractor
from world_embedding_cache import WorldEmbeddingCache


WORKFLOW_CAUSAL_STRUCTURES = [
    {
        "graph_id": "wf_n8n_automation",
        "source_text": "n8n Universal Automation Engine",
        "entities": [
            {"name": "n8n", "type": "workflow_engine", "attributes": {"nodes": 2000, "protocol": "REST", "deployment": "Docker"}},
            {"name": "Webhook_Trigger", "type": "trigger", "attributes": {"protocol": "HTTP", "async": True}},
            {"name": "Excel_Processor", "type": "tool", "attributes": {"format": "xlsx", "operations": ["read", "write", "pivot", "formula"]}},
            {"name": "Word_Generator", "type": "tool", "attributes": {"format": "docx", "operations": ["template", "mailmerge", "pdf_convert"]}},
            {"name": "Ecommerce_API", "type": "connector", "attributes": {"platforms": ["Taobao", "Pinduoduo", "JD", "Douyin"]}},
            {"name": "WeChat_Notifier", "type": "connector", "attributes": {"type": "personal_enterprise", "async": True}},
            {"name": "Qdrant_VectorDB", "type": "storage", "attributes": {"type": "vector", "protocol": "gRPC_REST"}},
            {"name": "MySQL_StructuredDB", "type": "storage", "attributes": {"type": "relational", "protocol": "SQL"}},
        ],
        "relations": [
            {"source": "Webhook_Trigger", "target": "n8n", "type": "triggers", "strength": 1.0},
            {"source": "n8n", "target": "Excel_Processor", "type": "calls", "strength": 0.9},
            {"source": "n8n", "target": "Word_Generator", "type": "calls", "strength": 0.8},
            {"source": "n8n", "target": "Ecommerce_API", "type": "calls", "strength": 0.9},
            {"source": "n8n", "target": "WeChat_Notifier", "type": "notifies", "strength": 0.7},
            {"source": "n8n", "target": "Qdrant_VectorDB", "type": "reads_writes", "strength": 0.6},
            {"source": "n8n", "target": "MySQL_StructuredDB", "type": "reads_writes", "strength": 0.8},
            {"source": "Excel_Processor", "target": "WeChat_Notifier", "type": "feeds_into", "strength": 0.5},
            {"source": "Ecommerce_API", "target": "MySQL_StructuredDB", "type": "writes_results", "strength": 0.9},
        ],
        "causal_chains": [
            {"cause": "Webhook_Trigger fires", "effect": "n8n starts workflow execution", "counterfactual": "If no trigger, workflow stays idle"},
            {"cause": "n8n calls Ecommerce_API", "effect": "Product listed on multiple platforms", "counterfactual": "If API fails, product not listed, error logged"},
            {"cause": "n8n writes to MySQL", "effect": "Listing record persisted for audit", "counterfactual": "If DB write fails, listing exists but no audit trail"},
            {"cause": "Excel_Processor generates report", "effect": "Financial data available for analysis", "counterfactual": "If Excel fails, manual data entry required"},
        ],
        "temporal_sequence": [
            {"step_order": 1, "description": "Trigger event received (Webhook/Scheduled/Form)"},
            {"step_order": 2, "description": "n8n reads knowledge base for context"},
            {"step_order": 3, "description": "n8n calls AI/tools for processing"},
            {"step_order": 4, "description": "n8n writes results to storage"},
            {"step_order": 5, "description": "n8n sends notification via WeChat/Email"},
        ],
    },
    {
        "graph_id": "wf_flowise_ai",
        "source_text": "Flowise AI-Native Workflow Engine",
        "entities": [
            {"name": "Flowise", "type": "workflow_engine", "attributes": {"specialty": "AI_RAG_Agent", "deployment": "Docker"}},
            {"name": "LLM_Node", "type": "ai_component", "attributes": {"models": ["DeepSeek", "Qwen", "Wenxin", "Ollama"]}},
            {"name": "RAG_Pipeline", "type": "ai_component", "attributes": {"steps": ["load", "chunk", "vectorize", "retrieve", "rerank"]}},
            {"name": "Agent_Node", "type": "ai_component", "attributes": {"capabilities": ["tool_calling", "reasoning", "planning"]}},
            {"name": "Drawing_Node", "type": "tool", "attributes": {"engines": ["SD", "ComfyUI", "MJ"]}},
            {"name": "MCP_Memory", "type": "protocol", "attributes": {"type": "long_term_memory", "version": "0.3"}},
            {"name": "Chroma_VectorDB", "type": "storage", "attributes": {"type": "vector", "weight": "lightweight"}},
            {"name": "JSON_Output", "type": "output", "attributes": {"format": "structured_json", "consumer": "n8n"}},
        ],
        "relations": [
            {"source": "RAG_Pipeline", "target": "Chroma_VectorDB", "type": "stores_in", "strength": 0.9},
            {"source": "LLM_Node", "target": "Agent_Node", "type": "powers", "strength": 1.0},
            {"source": "Agent_Node", "target": "Drawing_Node", "type": "can_call", "strength": 0.7},
            {"source": "MCP_Memory", "target": "Agent_Node", "type": "provides_context", "strength": 0.8},
            {"source": "Flowise", "target": "JSON_Output", "type": "produces", "strength": 0.9},
            {"source": "JSON_Output", "target": "n8n", "type": "feeds_into", "strength": 1.0},
        ],
        "causal_chains": [
            {"cause": "RAG_Pipeline retrieves relevant docs", "effect": "LLM has factual context for generation", "counterfactual": "If retrieval fails, LLM generates from parametric knowledge only (risk of hallucination)"},
            {"cause": "Agent_Node uses MCP_Memory", "effect": "Agent maintains context across sessions", "counterfactual": "If MCP disconnected, Agent loses long-term memory"},
            {"cause": "Flowise outputs structured JSON", "effect": "n8n can execute listing automation", "counterfactual": "If output format changes, n8n workflow breaks"},
        ],
        "temporal_sequence": [
            {"step_order": 1, "description": "User query received by Flowise"},
            {"step_order": 2, "description": "RAG pipeline retrieves from knowledge base"},
            {"step_order": 3, "description": "LLM generates response with RAG context"},
            {"step_order": 4, "description": "Agent decides if tools needed (drawing, API)"},
            {"step_order": 5, "description": "Structured output produced for downstream"},
        ],
    },
    {
        "graph_id": "wf_dify_knowledge",
        "source_text": "Dify All-in-One Knowledge Base",
        "entities": [
            {"name": "Dify", "type": "workflow_engine", "attributes": {"specialty": "Chinese_KB", "deployment": "Docker"}},
            {"name": "Document_Uploader", "type": "tool", "attributes": {"formats": ["Word", "Excel", "PDF", "Markdown"]}},
            {"name": "Auto_Chunker", "type": "processor", "attributes": {"strategies": ["automatic", "custom", "QA_format"]}},
            {"name": "BuiltIn_VectorDB", "type": "storage", "attributes": {"type": "vector", "built_in": True}},
            {"name": "WeChat_Connector", "type": "connector", "attributes": {"type": "personal_enterprise"}},
            {"name": "Light_Workflow", "type": "orchestrator", "attributes": {"nodes": ["condition", "API", "notification"]}},
        ],
        "relations": [
            {"source": "Document_Uploader", "target": "Auto_Chunker", "type": "feeds_into", "strength": 1.0},
            {"source": "Auto_Chunker", "target": "BuiltIn_VectorDB", "type": "stores_in", "strength": 1.0},
            {"source": "BuiltIn_VectorDB", "target": "Light_Workflow", "type": "provides_retrieval", "strength": 0.9},
            {"source": "Light_Workflow", "target": "WeChat_Connector", "type": "notifies", "strength": 0.7},
            {"source": "Dify", "target": "n8n", "type": "delegates_execution", "strength": 0.6},
        ],
        "causal_chains": [
            {"cause": "Document uploaded to Dify", "effect": "Auto-chunked and vectorized for RAG", "counterfactual": "If chunking fails, document not searchable"},
            {"cause": "Dify receives customer question", "effect": "KB search + LLM generates answer", "counterfactual": "If KB empty, LLM answers from general knowledge only"},
            {"cause": "Dify needs e-commerce execution", "effect": "Delegates to n8n via API", "counterfactual": "If n8n unavailable, Dify cannot execute actions"},
        ],
        "temporal_sequence": [
            {"step_order": 1, "description": "Upload document (Word/Excel/PDF)"},
            {"step_order": 2, "description": "Auto-chunk and vectorize"},
            {"step_order": 3, "description": "Customer asks question"},
            {"step_order": 4, "description": "RAG retrieves + LLM generates"},
            {"step_order": 5, "description": "Push answer to WeChat or delegate to n8n"},
        ],
    },
    {
        "graph_id": "wf_golden_combo",
        "source_text": "Golden Combination: Flowise + n8n + Dify + Qdrant",
        "entities": [
            {"name": "Flowise", "type": "ai_brain", "attributes": {"role": "thinking_RAG_MCP_drawing"}},
            {"name": "n8n", "type": "execution_hands", "attributes": {"role": "Excel_Word_Financial_Ecommerce_WeChat"}},
            {"name": "Dify", "type": "chinese_memory", "attributes": {"role": "team_KB_customer_service"}},
            {"name": "Qdrant", "type": "memory_substrate", "attributes": {"role": "MCP_RAG_shared_vector_store"}},
            {"name": "VORTEX_CJEPA", "type": "causal_reasoning", "attributes": {"role": "counterfactual_physical_causal"}},
            {"name": "DualPathwayBridge", "type": "fusion_layer", "attributes": {"role": "text_world_fusion"}},
        ],
        "relations": [
            {"source": "Flowise", "target": "n8n", "type": "feeds_structured_output", "strength": 1.0},
            {"source": "Dify", "target": "n8n", "type": "delegates_execution", "strength": 0.8},
            {"source": "Qdrant", "target": "Flowise", "type": "provides_vectors", "strength": 0.9},
            {"source": "Qdrant", "target": "Dify", "type": "provides_vectors", "strength": 0.7},
            {"source": "VORTEX_CJEPA", "target": "DualPathwayBridge", "type": "provides_world_embeds", "strength": 1.0},
            {"source": "DualPathwayBridge", "target": "Flowise", "type": "enhances_reasoning", "strength": 0.9},
            {"source": "n8n", "target": "Qdrant", "type": "writes_results", "strength": 0.6},
        ],
        "causal_chains": [
            {"cause": "Flowise generates product copy + images", "effect": "n8n executes multi-platform listing", "counterfactual": "If Flowise fails, no AI-generated content for listing"},
            {"cause": "Dify accumulates team knowledge", "effect": "Flowise RAG has richer context", "counterfactual": "If Dify KB empty, RAG retrieval returns no results"},
            {"cause": "VORTEX_CJEPA provides causal reasoning", "effect": "DualPathwayBridge fuses facts + logic", "counterfactual": "Without C-JEPA, system only has factual RAG, no causal reasoning"},
            {"cause": "Qdrant stores all vectors", "effect": "All engines share same memory substrate", "counterfactual": "If Qdrant down, all RAG retrieval fails"},
        ],
        "temporal_sequence": [
            {"step_order": 1, "description": "Knowledge enters via Dify (upload) or n8n (API)"},
            {"step_order": 2, "description": "Qdrant stores vectors for all engines"},
            {"step_order": 3, "description": "Flowise uses RAG + C-JEPA for AI reasoning"},
            {"step_order": 4, "description": "Flowise outputs structured data to n8n"},
            {"step_order": 5, "description": "n8n executes automation (listing, notification, report)"},
            {"step_order": 6, "description": "Results feed back to knowledge base"},
        ],
    },
    {
        "graph_id": "wf_financial_automation",
        "source_text": "Financial Automation Closed Loop",
        "entities": [
            {"name": "Invoice_OCR", "type": "tool", "attributes": {"input": "PDF_Image", "output": "structured_data"}},
            {"name": "Data_Extractor", "type": "processor", "attributes": {"fields": ["amount", "tax_id", "date", "vendor"]}},
            {"name": "Excel_Writer", "type": "tool", "attributes": {"operations": ["write", "formula", "pivot"]}},
            {"name": "Auto_Reconciliation", "type": "processor", "attributes": {"method": "bank_vs_ledger_matching"}},
            {"name": "Profit_Statement_Generator", "type": "tool", "attributes": {"output": "PnL_BalanceSheet_CashFlow"}},
            {"name": "WeChat_Finance_Notify", "type": "connector", "attributes": {"target": "finance_team"}},
            {"name": "Financial_KB", "type": "storage", "attributes": {"content": "tax_rules_account_templates_SOP"}},
        ],
        "relations": [
            {"source": "Invoice_OCR", "target": "Data_Extractor", "type": "feeds_into", "strength": 1.0},
            {"source": "Data_Extractor", "target": "Excel_Writer", "type": "writes_to", "strength": 0.9},
            {"source": "Excel_Writer", "target": "Auto_Reconciliation", "type": "provides_data", "strength": 0.9},
            {"source": "Auto_Reconciliation", "target": "Profit_Statement_Generator", "type": "feeds_into", "strength": 0.8},
            {"source": "Profit_Statement_Generator", "target": "WeChat_Finance_Notify", "type": "triggers", "strength": 0.7},
            {"source": "Profit_Statement_Generator", "target": "Financial_KB", "type": "archives_to", "strength": 0.9},
        ],
        "causal_chains": [
            {"cause": "Invoice OCR extracts data", "effect": "Structured financial data available", "counterfactual": "If OCR fails, manual data entry required"},
            {"cause": "Auto reconciliation matches records", "effect": "Discrepancies flagged automatically", "counterfactual": "If reconciliation fails, errors go undetected"},
            {"cause": "Profit statement generated", "effect": "Finance team notified via WeChat", "counterfactual": "If notification fails, team unaware of new reports"},
        ],
        "temporal_sequence": [
            {"step_order": 1, "description": "Invoice OCR scans and extracts data"},
            {"step_order": 2, "description": "Data extracted: amount, tax ID, vendor"},
            {"step_order": 3, "description": "Data written to Excel with formulas"},
            {"step_order": 4, "description": "Auto reconciliation runs"},
            {"step_order": 5, "description": "Profit statement generated"},
            {"step_order": 6, "description": "Notification sent + archived to KB"},
        ],
    },
    {
        "graph_id": "wf_mcp_memory",
        "source_text": "MCP Memory Protocol Architecture",
        "entities": [
            {"name": "MCP_Protocol", "type": "protocol", "attributes": {"version": "0.3", "leader": "Anthropic"}},
            {"name": "Working_Memory", "type": "memory_layer", "attributes": {"scope": "current_conversation", "duration": "short_term"}},
            {"name": "Episodic_Memory", "type": "memory_layer", "attributes": {"scope": "past_interactions", "duration": "medium_term"}},
            {"name": "Semantic_Memory", "type": "memory_layer", "attributes": {"scope": "knowledge_facts", "duration": "long_term", "storage": "vector"}},
            {"name": "Procedural_Memory", "type": "memory_layer", "attributes": {"scope": "skills_workflows", "duration": "long_term", "storage": "structured"}},
            {"name": "Qdrant_VectorStore", "type": "storage", "attributes": {"type": "vector", "use": "semantic_memory"}},
            {"name": "SQLite_StructuredStore", "type": "storage", "attributes": {"type": "relational", "use": "procedural_working"}},
            {"name": "VORTEX_SoulMemory", "type": "implementation", "attributes": {"maps_to": "working+episodic+semantic"}},
            {"name": "VORTEX_WorldCache", "type": "implementation", "attributes": {"maps_to": "world_model_memory"}},
        ],
        "relations": [
            {"source": "MCP_Protocol", "target": "Working_Memory", "type": "defines", "strength": 1.0},
            {"source": "MCP_Protocol", "target": "Semantic_Memory", "type": "defines", "strength": 1.0},
            {"source": "Semantic_Memory", "target": "Qdrant_VectorStore", "type": "stored_in", "strength": 0.9},
            {"source": "Working_Memory", "target": "SQLite_StructuredStore", "type": "stored_in", "strength": 0.8},
            {"source": "VORTEX_SoulMemory", "target": "MCP_Protocol", "type": "implements", "strength": 0.9},
            {"source": "VORTEX_WorldCache", "target": "MCP_Protocol", "type": "extends", "strength": 0.8},
        ],
        "causal_chains": [
            {"cause": "MCP stores memory in vector DB", "effect": "Agent retrieves relevant context across sessions", "counterfactual": "If vector DB empty, Agent has no long-term memory"},
            {"cause": "VORTEX SoulMemory implements MCP", "effect": "Soul memories accessible via standard MCP protocol", "counterfactual": "Without MCP compatibility, VORTEX is isolated from ecosystem"},
            {"cause": "WorldCache stores C-JEPA slots", "effect": "Causal reasoning available for counterfactual queries", "counterfactual": "Without WorldCache, only factual RAG available"},
        ],
        "temporal_sequence": [
            {"step_order": 1, "description": "Agent receives input"},
            {"step_order": 2, "description": "Working memory provides current context"},
            {"step_order": 3, "description": "Semantic memory retrieves relevant facts"},
            {"step_order": 4, "description": "Procedural memory provides skill templates"},
            {"step_order": 5, "description": "Agent generates response using all memory layers"},
            {"step_order": 6, "description": "New memories stored for future use"},
        ],
    },
]


def main():
    print("=" * 60)
    print("VORTEX FLAME Workflow Causal Extraction")
    print("=" * 60)

    cache = WorldEmbeddingCache()
    extractor = CausalKnowledgeExtractor()

    total_stored = 0

    for struct in WORKFLOW_CAUSAL_STRUCTURES:
        graph_id = struct["graph_id"]
        print(f"\nProcessing: {graph_id}")

        objects = [e["name"] for e in struct["entities"]]
        attributes = {e["name"]: e.get("attributes", {}) for e in struct["entities"]}
        causal_chains = [
            {
                "cause": c["cause"],
                "effect": c["effect"],
                "counterfactual": c.get("counterfactual", ""),
            }
            for c in struct.get("causal_chains", [])
        ]
        temporal = [s["description"] for s in sorted(struct.get("temporal_sequence", []), key=lambda s: s["step_order"])]

        graph_dict = {
            "graph_id": graph_id,
            "objects": objects,
            "attributes": attributes,
            "causal_chains": causal_chains,
            "temporal_sequence": temporal,
            "entities": struct["entities"],
            "relations": struct.get("relations", []),
            "source_text": struct.get("source_text", ""),
        }

        for soul in ["cezanne", "guizhu", "strategy", "beethoven", "einstein"]:
            try:
                entry_id = cache.store_from_object_graph(
                    soul=soul,
                    graph_dict=graph_dict,
                    category="workflow_causal",
                )
                total_stored += 1
                print(f"  Stored in {soul}: {entry_id[:16]}...")
            except Exception as e:
                print(f"  ERR ({soul}): {e}")

    print(f"\n{'=' * 60}")
    print(f"Total causal entries stored: {total_stored}")
    print(f"Graphs processed: {len(WORKFLOW_CAUSAL_STRUCTURES)}")
    print(f"\nC-JEPA Path B now has workflow causal knowledge!")
    print(f"  Query example: 'What happens if n8n fails?'")
    print(f"  -> WorldCache returns causal chain: n8n fails -> no listing -> no notification")


if __name__ == "__main__":
    main()
