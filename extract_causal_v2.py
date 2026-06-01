#!/usr/bin/env python3
"""
VORTEX FLAME V2 Causal Extractor
=================================
Extracts causal ObjectGraphs from v2 knowledge (system rules, control loop,
software KB, audio KB, hearing KB, JEPA KB, business memory).

Stores in WorldEmbeddingCache for C-JEPA Path B.
"""

import os
import sys

sys.path.insert(0, str(os.path.dirname(__file__)))
from world_embedding_cache import WorldEmbeddingCache

CACHE_PATH = os.path.join(os.path.dirname(__file__), "world_embedding_cache.db")

STRUCTURES = [
    {
        "graph_id": "v2_control_loop",
        "source_text": "VORTEX-FLAME Control Loop: LLM decision -> VORTEX validation -> device execution -> JEPA feedback",
        "entities": [
            {"name": "User_Input", "type": "trigger", "attributes": {"format": "natural_language"}},
            {"name": "DeviceGateway_Entry", "type": "gateway", "attributes": {"action": "auth_log"}},
            {"name": "VORTEX_Hub", "type": "coordinator", "attributes": {"role": "central_scheduler"}},
            {"name": "KnowledgeBase", "type": "storage", "attributes": {"scope": "private_context"}},
            {"name": "MCP_Memory", "type": "storage", "attributes": {"layer": "text_upper"}},
            {"name": "JEPA_State", "type": "predictor", "attributes": {"layer": "vector_lower"}},
            {"name": "Cloud_LLM", "type": "decision", "attributes": {"scope": "private_context_only"}},
            {"name": "RuleEngine", "type": "validator", "attributes": {"check": "permission_safety"}},
            {"name": "Workflow_Executor", "type": "executor", "attributes": {"type": "pluggable"}},
            {"name": "Device_Backend", "type": "executor", "attributes": {"type": "hardware_software"}},
            {"name": "JEPA_Filter", "type": "validator", "attributes": {"check": "physical_feasibility"}},
            {"name": "CircuitBreaker", "type": "safety", "attributes": {"trigger": "3_consecutive_failures"}},
        ],
        "relations": [
            {"source": "User_Input", "target": "DeviceGateway_Entry", "type": "triggers", "strength": 1.0},
            {"source": "VORTEX_Hub", "target": "KnowledgeBase", "type": "reads", "strength": 0.9},
            {"source": "VORTEX_Hub", "target": "MCP_Memory", "type": "reads_writes", "strength": 0.8},
            {"source": "VORTEX_Hub", "target": "JEPA_State", "type": "reads_writes", "strength": 0.7},
            {"source": "Cloud_LLM", "target": "VORTEX_Hub", "type": "receives_context", "strength": 1.0},
            {"source": "RuleEngine", "target": "Cloud_LLM", "type": "validates_output", "strength": 1.0},
            {"source": "JEPA_Filter", "target": "Cloud_LLM", "type": "validates_physics", "strength": 0.9},
            {"source": "Workflow_Executor", "target": "Device_Backend", "type": "calls", "strength": 1.0},
            {"source": "CircuitBreaker", "target": "Workflow_Executor", "type": "can_block", "strength": 1.0},
            {"source": "Device_Backend", "target": "JEPA_State", "type": "feeds_back", "strength": 0.9},
        ],
        "causal_chains": [
            {"cause": "RuleEngine rejects LLM output", "effect": "Workflow execution blocked, safety maintained", "counterfactual": "If RuleEngine skipped, unsafe action would execute"},
            {"cause": "JEPA predicts action is physically impossible", "effect": "CircuitBreaker increments failure counter", "counterfactual": "If JEPA absent, system would attempt impossible action"},
            {"cause": "CircuitBreaker reaches 3 failures", "effect": "System enters degraded mode, blocks all high-risk ops for 8h", "counterfactual": "If no breaker, system would continue attempting dangerous actions"},
            {"cause": "Device returns feedback to JEPA", "effect": "World state updated, next prediction more accurate", "counterfactual": "Without feedback loop, JEPA world model would drift"},
        ],
        "temporal_sequence": [
            {"step_order": 1, "description": "User sends natural language instruction"},
            {"step_order": 2, "description": "DeviceGateway authenticates and logs request"},
            {"step_order": 3, "description": "VORTEX bundles private context (KB+MCP+JEPA)"},
            {"step_order": 4, "description": "Cloud LLM produces JSON intent"},
            {"step_order": 5, "description": "RuleEngine hard-checks permissions and safety"},
            {"step_order": 6, "description": "JEPA performs physics feasibility prediction"},
            {"step_order": 7, "description": "Workflow executes approved action via DeviceGateway"},
            {"step_order": 8, "description": "Device results flow back to JEPA for state update"},
        ],
    },
    {
        "graph_id": "v2_jepa_training_safety",
        "source_text": "JEPA training safety: guard rails, anomaly detection, anti-collapse mechanisms",
        "entities": [
            {"name": "CJEPA_Model", "type": "model", "attributes": {"slots": 8, "dim": 128}},
            {"name": "Training_Loop", "type": "process", "attributes": {"epoch": "54/100", "lr": "5e-5"}},
            {"name": "LossSpikeDetector", "type": "guard", "attributes": {"threshold": "3x_moving_avg"}},
            {"name": "GradientClipper", "type": "guard", "attributes": {"max_norm": 1.0}},
            {"name": "NaN_Guard", "type": "guard", "attributes": {"action": "skip_batch"}},
            {"name": "ParameterDriftMonitor", "type": "guard", "attributes": {"threshold": "0.1_std"}},
            {"name": "EMAScheduler", "type": "stabilizer", "attributes": {"decay": 0.996}},
            {"name": "TargetEncoder", "type": "encoder", "attributes": {"frozen": True, "ema": True}},
            {"name": "SIGReg", "type": "loss", "attributes": {"terms": 2, "hyperparams": 1}},
            {"name": "CausalVICReg", "type": "loss", "attributes": {"terms": 4, "hyperparams": 4}},
        ],
        "relations": [
            {"source": "Training_Loop", "target": "CJEPA_Model", "type": "optimizes", "strength": 1.0},
            {"source": "LossSpikeDetector", "target": "Training_Loop", "type": "guards", "strength": 0.9},
            {"source": "GradientClipper", "target": "Training_Loop", "type": "constrains", "strength": 0.8},
            {"source": "NaN_Guard", "target": "Training_Loop", "type": "filters", "strength": 0.9},
            {"source": "ParameterDriftMonitor", "target": "CJEPA_Model", "type": "monitors", "strength": 0.7},
            {"source": "EMAScheduler", "target": "TargetEncoder", "type": "updates", "strength": 1.0},
            {"source": "CausalVICReg", "target": "CJEPA_Model", "type": "computes_gradient", "strength": 1.0},
            {"source": "SIGReg", "target": "CJEPA_Model", "type": "potential_replacement", "strength": 0.5},
        ],
        "causal_chains": [
            {"cause": "Loss spike detected (3x moving average)", "effect": "Auto-rollback to previous checkpoint", "counterfactual": "Without spike detection, training would diverge"},
            {"cause": "Gradient norm exceeds max_norm", "effect": "Gradients clipped, update stable", "counterfactual": "Without clipping, large gradient step would cause instability"},
            {"cause": "NaN detected in loss", "effect": "Batch discarded, no gradient update applied", "counterfactual": "NaN gradient would corrupt model parameters"},
            {"cause": "Parameter drift exceeds 0.1 std", "effect": "Training paused for human inspection", "counterfactual": "Continued training with drift leads to unrecoverable state"},
        ],
        "temporal_sequence": [
            {"step_order": 1, "description": "Data batch loaded from disk"},
            {"step_order": 2, "description": "Forward pass: encoder->slots->predictor"},
            {"step_order": 3, "description": "CausalVICReg loss computed"},
            {"step_order": 4, "description": "Spike/NaN/drift checks run"},
            {"step_order": 5, "description": "Gradient clipping applied if needed"},
            {"step_order": 6, "description": "Backward pass and optimizer step"},
            {"step_order": 7, "description": "TargetEncoder EMA updated"},
        ],
    },
    {
        "graph_id": "v2_audio_chain",
        "source_text": "Audio production chain: DAW -> hardware -> JEPA encoding -> LLM decision -> execution",
        "entities": [
            {"name": "Ableton_Live", "type": "daw", "attributes": {"control": "MIDI_Remote_Scripts"}},
            {"name": "SoundCard", "type": "hardware", "attributes": {"driver": "ASIO", "latency": "<10ms"}},
            {"name": "Microphone", "type": "hardware", "attributes": {"type": "Condenser_U87", "power": "48V"}},
            {"name": "MIDI_Controller", "type": "hardware", "attributes": {"protocol": "MIDI_USB"}},
            {"name": "CAJEPA_Encoder", "type": "jepa", "attributes": {"slots": 5, "dim": 128}},
            {"name": "Music_Knowledge_KB", "type": "knowledge", "attributes": {"scope": "private_theory"}},
            {"name": "LLM_Decision", "type": "decision", "attributes": {"constraint": "private_only"}},
            {"name": "Mixer_State", "type": "state", "attributes": {"channels": 32, "effects": "EQ_Comp_Reverb"}},
            {"name": "Render_Queue", "type": "task", "attributes": {"type": "long_running"}},
        ],
        "relations": [
            {"source": "Microphone", "target": "SoundCard", "type": "inputs", "strength": 1.0},
            {"source": "SoundCard", "target": "CAJEPA_Encoder", "type": "streams", "strength": 1.0},
            {"source": "MIDI_Controller", "target": "Ableton_Live", "type": "controls", "strength": 0.9},
            {"source": "Ableton_Live", "target": "SoundCard", "type": "outputs", "strength": 1.0},
            {"source": "CAJEPA_Encoder", "target": "LLM_Decision", "type": "summarizes", "strength": 0.8},
            {"source": "Music_Knowledge_KB", "target": "LLM_Decision", "type": "constrains", "strength": 0.9},
            {"source": "LLM_Decision", "target": "Ableton_Live", "type": "commands", "strength": 0.9},
            {"source": "LLM_Decision", "target": "Render_Queue", "type": "dispatches", "strength": 0.7},
        ],
        "causal_chains": [
            {"cause": "SoundCard ASIO buffer underrun", "effect": "Audio glitch, CAJEPA state marked as noisy", "counterfactual": "If buffer size was larger, underrun would not occur"},
            {"cause": "CAJEPA detects key change from C to G", "effect": "LLM adjusts chord progression to G major", "counterfactual": "Without JEPA, LLM would miss real-time key change"},
            {"cause": "LLM decision contains chord outside private KB", "effect": "RuleEngine rejects, falls back to known progression", "counterfactual": "Without KB constraint, LLM may hallucinate invalid chords"},
            {"cause": "Render task exceeds 30 minutes", "effect": "Workflow switches to long-running executor", "counterfactual": "Short workflow would timeout and lose render"},
        ],
        "temporal_sequence": [
            {"step_order": 1, "description": "Microphone captures audio / MIDI controller sends notes"},
            {"step_order": 2, "description": "SoundCard digitizes and streams to CAJEPA"},
            {"step_order": 3, "description": "CAJEPA encodes: instruments detected, key, tempo, groove"},
            {"step_order": 4, "description": "LLM reads JEPA summary + private music KB"},
            {"step_order": 5, "description": "LLM outputs structured JSON music command"},
            {"step_order": 6, "description": "Ableton executes command (play/mute/add track)"},
            {"step_order": 7, "description": "Resulting audio feeds back into CAJEPA loop"},
        ],
    },
    {
        "graph_id": "v2_memory_lifecycle",
        "source_text": "Business memory lifecycle: write -> index -> retrieve -> forget",
        "entities": [
            {"name": "Event_Log", "type": "source", "attributes": {"format": "conversation_action_result"}},
            {"name": "MemoryWriter", "type": "process", "attributes": {"trigger": "auto_post_execution"}},
            {"name": "BM25_Index", "type": "index", "attributes": {"type": "keyword_sparse"}},
            {"name": "FAISS_Index", "type": "index", "attributes": {"type": "semantic_dense"}},
            {"name": "MemoryChunk_L1", "type": "storage", "attributes": {"max_len": "3k_chars"}},
            {"name": "MemorySummary_L2", "type": "storage", "attributes": {"group_by": "category"}},
            {"name": "MemoryWiki_L3", "type": "storage", "attributes": {"scope": "cross_category"}},
            {"name": "ReflectEvaluator", "type": "analyzer", "attributes": {"dimensions": 4}},
            {"name": "ForgettingAgent", "type": "maintenance", "attributes": {"threshold": "90_days"}},
            {"name": "Soul_Space", "type": "isolation", "attributes": {"count": 14}},
        ],
        "relations": [
            {"source": "Event_Log", "target": "MemoryWriter", "type": "feeds", "strength": 1.0},
            {"source": "MemoryWriter", "target": "MemoryChunk_L1", "type": "stores", "strength": 1.0},
            {"source": "MemoryChunk_L1", "target": "MemorySummary_L2", "type": "compresses_to", "strength": 0.6},
            {"source": "MemorySummary_L2", "target": "MemoryWiki_L3", "type": "merges_to", "strength": 0.3},
            {"source": "BM25_Index", "target": "MemoryChunk_L1", "type": "indexes", "strength": 0.8},
            {"source": "FAISS_Index", "target": "MemoryChunk_L1", "type": "indexes", "strength": 0.9},
            {"source": "ReflectEvaluator", "target": "MemoryChunk_L1", "type": "scores", "strength": 0.7},
            {"source": "ForgettingAgent", "target": "MemoryChunk_L1", "type": "archives", "strength": 0.5},
            {"source": "Soul_Space", "target": "MemoryChunk_L1", "type": "isolates", "strength": 1.0},
        ],
        "causal_chains": [
            {"cause": "Memory chunk not accessed for 90 days", "effect": "ForgettingAgent archives to cold storage", "counterfactual": "Without forgetting, storage grows unboundedly"},
            {"cause": "ReflectEvaluator scores coverage as low", "effect": "System triggers self-training for gap area", "counterfactual": "Without Reflect, knowledge gaps silently accumulate"},
            {"cause": "Context length exceeds LLM window", "effect": "Retriever truncates to top-K relevant chunks", "counterfactual": "Without truncation, LLM would overflow context window"},
        ],
        "temporal_sequence": [
            {"step_order": 1, "description": "Action completes, event log written"},
            {"step_order": 2, "description": "MemoryWriter chunks text into L1 (3k chars)"},
            {"step_order": 3, "description": "BM25 + FAISS dual-index updated"},
            {"step_order": 4, "description": "ReflectEvaluator scores memory coverage"},
            {"step_order": 5, "description": "L2 summary generated when L1 reaches threshold"},
            {"step_order": 6, "description": "ForgettingAgent runs periodic cleanup"},
        ],
    },
    {
        "graph_id": "v2_software_orchestration",
        "source_text": "Multi-software orchestration: Blender + OBS + Ableton + PyTorch coordinated via DeviceGateway",
        "entities": [
            {"name": "DeviceGateway", "type": "coordinator", "attributes": {"backends": 24, "mode": "abstract"}},
            {"name": "Blender_Render", "type": "task", "attributes": {"type": "long_running", "max_duration": "4h"}},
            {"name": "OBS_Stream", "type": "task", "attributes": {"type": "realtime", "fps": 30}},
            {"name": "Ableton_Session", "type": "task", "attributes": {"type": "realtime", "bpm": 128}},
            {"name": "PyTorch_Training", "type": "task", "attributes": {"type": "long_running", "gpu": True}},
            {"name": "PowerShell_Script", "type": "task", "attributes": {"type": "short", "timeout": 30}},
            {"name": "WorkflowRouter", "type": "router", "attributes": {"engines": ["n8n","long_worker","fastapi","prefect"]}},
            {"name": "ResourceMonitor", "type": "monitor", "attributes": {"metrics": ["cpu","ram","gpu_vram","disk"]}},
            {"name": "SafetyMiddleware", "type": "safety", "attributes": {"whitelist_only": True}},
        ],
        "relations": [
            {"source": "DeviceGateway", "target": "WorkflowRouter", "type": "delegates", "strength": 1.0},
            {"source": "WorkflowRouter", "target": "Blender_Render", "type": "routes_to_long", "strength": 1.0},
            {"source": "WorkflowRouter", "target": "OBS_Stream", "type": "routes_to_fastapi", "strength": 1.0},
            {"source": "WorkflowRouter", "target": "Ableton_Session", "type": "routes_to_fastapi", "strength": 1.0},
            {"source": "WorkflowRouter", "target": "PyTorch_Training", "type": "routes_to_long", "strength": 1.0},
            {"source": "WorkflowRouter", "target": "PowerShell_Script", "type": "routes_to_n8n", "strength": 1.0},
            {"source": "ResourceMonitor", "target": "WorkflowRouter", "type": "constrains", "strength": 0.8},
            {"source": "SafetyMiddleware", "target": "DeviceGateway", "type": "filters", "strength": 1.0},
        ],
        "causal_chains": [
            {"cause": "GPU VRAM exceeds 90% during Blender render", "effect": "ResourceMonitor blocks new PyTorch training tasks", "counterfactual": "Without monitor, GPU OOM would crash both tasks"},
            {"cause": "PowerShell script runs unknown command", "effect": "SafetyMiddleware blocks execution before reaching shell", "counterfactual": "Without safety filter, arbitrary code would execute"},
            {"cause": "OBS stream + Ableton both active", "effect": "ASIO shared mode prevents conflict, both work", "counterfactual": "Exclusive ASIO mode would block second app"},
        ],
        "temporal_sequence": [
            {"step_order": 1, "description": "User issues multi-step creative task"},
            {"step_order": 2, "description": "VORTEX decomposes into sub-tasks"},
            {"step_order": 3, "description": "DeviceGateway routes each sub-task to WorkflowRouter"},
            {"step_order": 4, "description": "ResourceMonitor checks capacity before dispatch"},
            {"step_order": 5, "description": "SafetyMiddleware validates each sub-command"},
            {"step_order": 6, "description": "Tasks execute in parallel where possible"},
        ],
    },
    {
        "graph_id": "v2_action_conditioned_jepa",
        "source_text": "Action-Conditioned JEPA upgrade: adding action input to CausalPredictor for device control",
        "entities": [
            {"name": "Current_CJEPA", "type": "model", "attributes": {"input": "slots_only", "output": "predicted_slots"}},
            {"name": "Upgraded_CJEPA", "type": "model", "attributes": {"input": "slots_plus_action", "output": "next_slots"}},
            {"name": "ActionEmbedding", "type": "module", "attributes": {"dim": 64, "source": "device_gateway"}},
            {"name": "CausalPredictor", "type": "module", "attributes": {"backbone": "transformer_4_layer"}},
            {"name": "DeviceActionHistory", "type": "data", "attributes": {"format": "action_result_pairs"}},
            {"name": "V_JEPA2_Reference", "type": "research", "attributes": {"hours": 62, "task": "robot_control"}},
            {"name": "SIGReg_Loss", "type": "training", "attributes": {"terms": 2, "params": "15M"}},
            {"name": "CausalVICReg_Loss", "type": "training", "attributes": {"terms": 4, "params": "current"}},
        ],
        "relations": [
            {"source": "Current_CJEPA", "target": "Upgraded_CJEPA", "type": "evolves_to", "strength": 1.0},
            {"source": "ActionEmbedding", "target": "CausalPredictor", "type": "appends_to_input", "strength": 1.0},
            {"source": "DeviceActionHistory", "target": "ActionEmbedding", "type": "encodes", "strength": 0.9},
            {"source": "V_JEPA2_Reference", "target": "Upgraded_CJEPA", "type": "inspires", "strength": 0.8},
            {"source": "SIGReg_Loss", "target": "CausalVICReg_Loss", "type": "may_replace", "strength": 0.5},
        ],
        "causal_chains": [
            {"cause": "Action embedding added to CausalPredictor input", "effect": "JEPA can predict action consequences, enabling device control", "counterfactual": "Without action input, JEPA only observes but cannot control"},
            {"cause": "SIGReg replaces CausalVICReg", "effect": "Training faster and more stable, but causal interaction term lost", "counterfactual": "With CausalVICReg, causal reasoning better but training 4-term complex"},
        ],
        "temporal_sequence": [
            {"step_order": 1, "description": "Current A-JEPA completes Phase 1 (100 epochs)"},
            {"step_order": 2, "description": "ActionEmbedding module designed (64-dim, from DeviceActionHistory)"},
            {"step_order": 3, "description": "CausalPredictor modified: input = cat([slot_tokens, action_token])"},
            {"step_order": 4, "description": "SIGReg vs CausalVICReg comparison experiment"},
            {"step_order": 5, "description": "Action-conditioned training with device action history"},
            {"step_order": 6, "description": "End-to-end device control validation"},
        ],
    },
]

SOUL_MAP = {
    "v2_control_loop": "cezanne",
    "v2_jepa_training_safety": "einstein",
    "v2_audio_chain": "beethoven",
    "v2_memory_lifecycle": "guizhu",
    "v2_software_orchestration": "cezanne",
    "v2_action_conditioned_jepa": "cezanne",
}


def main():
    cache = WorldEmbeddingCache(CACHE_PATH)
    stored = 0

    for s in STRUCTURES:
        soul = SOUL_MAP.get(s["graph_id"], "cezanne")
        try:
            object_names = [e["name"] for e in s["entities"]]
            graph_dict = {
                "graph_id": s["graph_id"],
                "source_text": s["source_text"],
                "entities": s["entities"],
                "relations": s["relations"],
                "causal_chains": s["causal_chains"],
                "temporal_sequence": s["temporal_sequence"],
                "objects": object_names,
            }
            entry_id = cache.store_from_object_graph(soul, graph_dict)
            stored += 1
            print(f"  OK: {s['graph_id']} ({len(s['entities'])} entities, {len(s['causal_chains'])} causal chains) -> {soul}")
        except Exception as e:
            print(f"  ERR: {s['graph_id']} - {e}")

    print(f"Total: {stored} causal graphs stored in WorldEmbeddingCache")


if __name__ == "__main__":
    main()
