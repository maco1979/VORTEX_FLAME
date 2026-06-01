#!/usr/bin/env python3
"""
VORTEX FLAME Extended Domain Causal Extractor
===============================================
Extracts causal structure from extended domain knowledge:
  - SciCompute, VisualVideo, DatabaseETL, NetworkOps
  - EmbeddedIoT, VerticalApps, AudioJEPA

Stores ObjectGraph in WorldEmbeddingCache for C-JEPA Path B.
"""

import os
import sys

sys.path.insert(0, str(os.path.dirname(__file__)))
from world_embedding_cache import WorldEmbeddingCache

STRUCTURES = [
    {
        "graph_id": "ext_sci_compute",
        "source_text": "SciCompute: GPU/CUDA/Docker/Simulation causal chain",
        "entities": [
            {"name": "PyTorch", "type": "framework", "attributes": {"backend": "CUDA", "precision": "fp16"}},
            {"name": "GPU_V100", "type": "hardware", "attributes": {"VRAM": "16GB", "TFLOPS_fp16": 28}},
            {"name": "Docker", "type": "container", "attributes": {"isolation": "full", "reproducibility": "high"}},
            {"name": "MATLAB", "type": "simulation", "attributes": {"domain": "multiphysics"}},
            {"name": "Training_Loop", "type": "process", "attributes": {"status": "running", "epoch": 54}},
            {"name": "Checkpoint", "type": "storage", "attributes": {"format": "pt", "location": "ajepa_checkpoints/"}},
        ],
        "relations": [
            {"source": "PyTorch", "target": "GPU_V100", "type": "uses", "strength": 1.0},
            {"source": "Docker", "target": "PyTorch", "type": "hosts", "strength": 0.8},
            {"source": "Training_Loop", "target": "GPU_V100", "type": "consumes", "strength": 1.0},
            {"source": "Training_Loop", "target": "Checkpoint", "type": "writes", "strength": 0.9},
            {"source": "MATLAB", "target": "PyTorch", "type": "exports_to", "strength": 0.5},
        ],
        "causal_chains": [
            {"cause": "GPU_V100 overheats", "effect": "Training_Loop throttles or crashes", "counterfactual": "If GPU temp stays below 85C, training continues normally"},
            {"cause": "Docker container crashes", "effect": "Training state lost unless checkpoint saved", "counterfactual": "If checkpoint was saved recently, loss is minimal"},
            {"cause": "PyTorch OOM error", "effect": "Batch fails, gradient update skipped", "counterfactual": "If batch_size was smaller, OOM would not occur"},
        ],
        "temporal_sequence": [
            {"step_order": 1, "description": "Docker starts PyTorch container"},
            {"step_order": 2, "description": "GPU initializes CUDA context"},
            {"step_order": 3, "description": "Training loop loads data batch"},
            {"step_order": 4, "description": "Forward pass + loss computation"},
            {"step_order": 5, "description": "Backward pass + gradient update"},
            {"step_order": 6, "description": "Checkpoint saved periodically"},
        ],
    },
    {
        "graph_id": "ext_visual_video",
        "source_text": "VisualVideo: Camera/OpenCV/OBS/FFmpeg causal chain",
        "entities": [
            {"name": "Camera", "type": "sensor", "attributes": {"resolution": "1080p", "fps": 30}},
            {"name": "OpenCV", "type": "framework", "attributes": {"backend": "CPU_or_CUDA"}},
            {"name": "OBS_Studio", "type": "software", "attributes": {"output": "stream_or_record"}},
            {"name": "FFmpeg", "type": "tool", "attributes": {"format": "universal"}},
            {"name": "V_JEPA", "type": "model", "attributes": {"input": "video_frames", "slots": 7}},
        ],
        "relations": [
            {"source": "Camera", "target": "OpenCV", "type": "feeds", "strength": 1.0},
            {"source": "Camera", "target": "OBS_Studio", "type": "feeds", "strength": 0.9},
            {"source": "OpenCV", "target": "V_JEPA", "type": "provides_frames", "strength": 0.8},
            {"source": "OBS_Studio", "target": "FFmpeg", "type": "encodes_via", "strength": 0.7},
            {"source": "FFmpeg", "target": "V_JEPA", "type": "preprocesses_for", "strength": 0.6},
        ],
        "causal_chains": [
            {"cause": "Camera disconnected", "effect": "OpenCV returns empty frames, V_JEPA state becomes stale", "counterfactual": "If camera is reconnected, V_JEPA can resume from last valid state"},
            {"cause": "OBS scene switch fails", "effect": "Stream shows wrong content", "counterfactual": "If scene was pre-configured correctly, switch succeeds"},
            {"cause": "FFmpeg encoding error", "effect": "Video output corrupted", "counterfactual": "If fallback codec available, encoding retries with alternate"},
        ],
        "temporal_sequence": [
            {"step_order": 1, "description": "Camera captures frame"},
            {"step_order": 2, "description": "OpenCV reads frame buffer"},
            {"step_order": 3, "description": "Frame preprocessed (resize/normalize)"},
            {"step_order": 4, "description": "V_JEPA encodes frame to visual slots"},
            {"step_order": 5, "description": "World model state updated"},
        ],
    },
    {
        "graph_id": "ext_database_etl",
        "source_text": "DatabaseETL: VectorDB/SQL/ETL causal chain",
        "entities": [
            {"name": "Qdrant", "type": "vector_db", "attributes": {"dim": 384, "engine": "Rust"}},
            {"name": "SQLite", "type": "relational_db", "attributes": {"mode": "embedded"}},
            {"name": "ETL_Pipeline", "type": "process", "attributes": {"stages": ["extract", "transform", "load"]}},
            {"name": "soul_memory", "type": "system", "attributes": {"backend": "SQLite+FAISS"}},
            {"name": "WorldCache", "type": "system", "attributes": {"backend": "SQLite+FTS5"}},
        ],
        "relations": [
            {"source": "ETL_Pipeline", "target": "soul_memory", "type": "populates", "strength": 0.9},
            {"source": "soul_memory", "target": "Qdrant", "type": "syncs_vectors", "strength": 0.8},
            {"source": "soul_memory", "target": "SQLite", "type": "stores_metadata", "strength": 1.0},
            {"source": "WorldCache", "target": "Qdrant", "type": "can_mirror", "strength": 0.3},
        ],
        "causal_chains": [
            {"cause": "Qdrant index corrupted", "effect": "Semantic search returns empty results", "counterfactual": "If SQLite FTS fallback enabled, keyword search still works"},
            {"cause": "ETL pipeline fails mid-transform", "effect": "KB entries partially indexed, inconsistency detected", "counterfactual": "If transactional writes used, rollback prevents partial state"},
            {"cause": "SQLite DB locked by concurrent write", "effect": "Write operation queued or fails", "counterfactual": "If WAL mode enabled, concurrent reads still work"},
        ],
        "temporal_sequence": [
            {"step_order": 1, "description": "ETL extracts knowledge from source"},
            {"step_order": 2, "description": "Data cleaned and normalized"},
            {"step_order": 3, "description": "Embeddings computed via all-MiniLM-L6-v2"},
            {"step_order": 4, "description": "Metadata stored in SQLite"},
            {"step_order": 5, "description": "Vectors indexed in FAISS/Qdrant"},
        ],
    },
    {
        "graph_id": "ext_network_ops",
        "source_text": "NetworkOps: SSH/RemoteDesktop/Firewall causal chain",
        "entities": [
            {"name": "SSH_Client", "type": "protocol", "attributes": {"auth": "ed25519", "port": 22}},
            {"name": "Remote_Host", "type": "endpoint", "attributes": {"os": "Linux_or_Windows"}},
            {"name": "Firewall", "type": "security", "attributes": {"mode": "allowlist"}},
            {"name": "VORTEX_API", "type": "server", "attributes": {"bind": "localhost", "port": 8080}},
            {"name": "LLM_Cloud", "type": "service", "attributes": {"provider": "GPT_or_Claude"}},
        ],
        "relations": [
            {"source": "SSH_Client", "target": "Remote_Host", "type": "connects_to", "strength": 1.0},
            {"source": "Firewall", "target": "SSH_Client", "type": "filters", "strength": 0.9},
            {"source": "VORTEX_API", "target": "LLM_Cloud", "type": "proxies_to", "strength": 0.8},
            {"source": "Firewall", "target": "VORTEX_API", "type": "protects", "strength": 0.7},
        ],
        "causal_chains": [
            {"cause": "SSH connection dropped", "effect": "Remote monitoring interrupted, training status unknown", "counterfactual": "If local log persists, status can be checked after reconnect"},
            {"cause": "Firewall blocks outbound API call", "effect": "Cloud LLM unavailable, fallback to local model", "counterfactual": "If local LLM ready, service continues with reduced quality"},
            {"cause": "VORTEX_API crashes", "effect": "All external requests fail", "counterfactual": "If health check + auto-restart configured, downtime is seconds"},
        ],
        "temporal_sequence": [
            {"step_order": 1, "description": "SSH key exchange + authentication"},
            {"step_order": 2, "description": "VPN/tunnel established if needed"},
            {"step_order": 3, "description": "Command executed on remote host"},
            {"step_order": 4, "description": "Result transferred back via SCP/SFTP"},
            {"step_order": 5, "description": "SSH session terminated or kept alive"},
        ],
    },
    {
        "graph_id": "ext_embedded_iot",
        "source_text": "EmbeddedIoT: Arduino/RPi/PLC sensor-actuator causal chain",
        "entities": [
            {"name": "Sensor", "type": "input", "attributes": {"protocol": "I2C_or_SPI", "rate": "10Hz"}},
            {"name": "Microcontroller", "type": "compute", "attributes": {"arch": "ARM_Cortex", "firmware": "Arduino_or_MicroPython"}},
            {"name": "Actuator", "type": "output", "attributes": {"type": "motor_or_relay_or_servo", "max_current": "2A"}},
            {"name": "JEPA_Edge", "type": "model", "attributes": {"location": "local", "latency": "<100ms"}},
            {"name": "Emergency_Stop", "type": "safety", "attributes": {"trigger": "hardware", "independent": True}},
        ],
        "relations": [
            {"source": "Sensor", "target": "Microcontroller", "type": "sends_data", "strength": 1.0},
            {"source": "Microcontroller", "target": "JEPA_Edge", "type": "feeds_state", "strength": 0.8},
            {"source": "JEPA_Edge", "target": "Microcontroller", "type": "suggests_action", "strength": 0.7},
            {"source": "Microcontroller", "target": "Actuator", "type": "controls", "strength": 1.0},
            {"source": "Emergency_Stop", "target": "Actuator", "type": "overrides", "strength": 1.0},
        ],
        "causal_chains": [
            {"cause": "Sensor reads value above threshold", "effect": "JEPA_Edge predicts anomaly, suggests stop", "counterfactual": "If sensor was miscalibrated, false positive trigger"},
            {"cause": "Microcontroller loses connection to JEPA_Edge", "effect": "Action decisions revert to local safety defaults", "counterfactual": "If watchdog timer active, safe state entered"},
            {"cause": "Emergency_Stop triggered", "effect": "Actuator immediately cut, regardless of JEPA decision", "counterfactual": "If emergency stop was software-only, microcontroller could ignore it"},
        ],
        "temporal_sequence": [
            {"step_order": 1, "description": "Sensor samples physical quantity"},
            {"step_order": 2, "description": "Microcontroller reads and timestamps data"},
            {"step_order": 3, "description": "JEPA_Edge encodes sensor state into world model"},
            {"step_order": 4, "description": "JEPA_Edge predicts next state, suggests action"},
            {"step_order": 5, "description": "Microcontroller validates action against safety rules"},
            {"step_order": 6, "description": "Actuator executes (or Emergency_Stop intervenes)"},
        ],
    },
    {
        "graph_id": "ext_vertical_apps",
        "source_text": "VerticalApps: GIS/Medical/ERP domain causal chain",
        "entities": [
            {"name": "GIS_Engine", "type": "domain_software", "attributes": {"format": "geotiff_shapefile", "crs": "WGS84"}},
            {"name": "DICOM_Viewer", "type": "domain_software", "attributes": {"standard": "DICOM", "modality": "CT_MRI"}},
            {"name": "ERP_System", "type": "domain_software", "attributes": {"modules": ["GL", "AP", "AR", "Tax"]}},
            {"name": "JEPA_Domain", "type": "model", "attributes": {"variant": "CGEO_or_CBIO_or_CFIN"}},
            {"name": "Rule_Engine", "type": "safety", "attributes": {"type": "domain_constraints"}},
        ],
        "relations": [
            {"source": "GIS_Engine", "target": "JEPA_Domain", "type": "provides_spatial", "strength": 0.8},
            {"source": "DICOM_Viewer", "target": "JEPA_Domain", "type": "provides_medical", "strength": 0.8},
            {"source": "ERP_System", "target": "JEPA_Domain", "type": "provides_financial", "strength": 0.8},
            {"source": "Rule_Engine", "target": "JEPA_Domain", "type": "constrains", "strength": 0.9},
        ],
        "causal_chains": [
            {"cause": "GIS spatial reference mismatch", "effect": "Analysis results geographically wrong", "counterfactual": "If CRS validated before analysis, error caught early"},
            {"cause": "DICOM PHI not de-identified", "effect": "Privacy violation, legal liability", "counterfactual": "If de-identification pipeline runs first, PHI stripped"},
            {"cause": "ERP journal entry miscalculated", "effect": "Financial report inaccurate, audit risk", "counterfactual": "If reconciliation check passes, error detected before reporting"},
        ],
        "temporal_sequence": [
            {"step_order": 1, "description": "Domain data ingested (spatial/medical/financial)"},
            {"step_order": 2, "description": "Data validated against domain rules"},
            {"step_order": 3, "description": "JEPA_Domain encodes into causal slots"},
            {"step_order": 4, "description": "Causal prediction + counterfactual analysis"},
            {"step_order": 5, "description": "Result validated by Rule_Engine before output"},
        ],
    },
    {
        "graph_id": "ext_audio_jepa",
        "source_text": "AudioJEPA: Sound card/DAW/CAJEPA music causal chain",
        "entities": [
            {"name": "Sound_Card", "type": "hardware", "attributes": {"driver": "ASIO", "sample_rate": 48000}},
            {"name": "DAW", "type": "software", "attributes": {"name": "Ableton_or_Logic", "tracks": "multi"}},
            {"name": "CAJEPA", "type": "model", "attributes": {"slots": 5, "input": "mel_spectrogram", "status": "training"}},
            {"name": "Microphone", "type": "input", "attributes": {"type": "condenser", "phantom": "+48V"}},
            {"name": "Speaker", "type": "output", "attributes": {"monitor": "studio", "spl_limit": "85dB"}},
        ],
        "relations": [
            {"source": "Microphone", "target": "Sound_Card", "type": "connects", "strength": 1.0},
            {"source": "Sound_Card", "target": "CAJEPA", "type": "streams_audio", "strength": 0.9},
            {"source": "DAW", "target": "CAJEPA", "type": "feeds_tracks", "strength": 0.8},
            {"source": "CAJEPA", "target": "DAW", "type": "returns_analysis", "strength": 0.6},
            {"source": "Sound_Card", "target": "Speaker", "type": "routes_to", "strength": 1.0},
        ],
        "causal_chains": [
            {"cause": "Microphone phantom power off with condenser mic", "effect": "No signal, CAJEPA receives silence", "counterfactual": "If phantom power verified before recording, signal present"},
            {"cause": "CAJEPA predicts next chord progression", "effect": "DAW pre-loads instrument preset for that chord", "counterfactual": "If prediction is wrong, musician overrides manually"},
            {"cause": "Sound card driver crashes (ASIO buffer underrun)", "effect": "Audio stream interrupted, CAJEPA state gaps", "counterfactual": "If larger buffer size configured, underrun less likely"},
            {"cause": "Speaker SPL exceeds 85dB", "effect": "Hearing damage risk, automatic volume reduction", "counterfactual": "If monitoring at safe levels, no intervention needed"},
        ],
        "temporal_sequence": [
            {"step_order": 1, "description": "Microphone captures audio waveform"},
            {"step_order": 2, "description": "Sound card ADC converts to digital"},
            {"step_order": 3, "description": "Mel spectrogram computed from waveform"},
            {"step_order": 4, "description": "CAJEPA encodes into 5 instrument slots"},
            {"step_order": 5, "description": "CAJEPA predicts temporal evolution of slots"},
            {"step_order": 6, "description": "DAW receives analysis, adjusts playback/recording"},
            {"step_order": 7, "description": "Speaker outputs result, feeds back to CAJEPA"},
        ],
    },
]

SOULS = ["cezanne", "beethoven", "guizhu", "strategy", "einstein", "davinci"]


def main():
    print("=" * 60)
    print("VORTEX FLAME Extended Domain Causal Extractor")
    print("=" * 60)

    cache = WorldEmbeddingCache()
    total_stored = 0

    for struct in STRUCTURES:
        graph_id = struct["graph_id"]
        print(f"\nProcessing: {graph_id}")

        objects = [e["name"] for e in struct["entities"]]
        attributes = {e["name"]: e.get("attributes", {}) for e in struct["entities"]}
        causal_chains = [
            {"cause": c["cause"], "effect": c["effect"], "counterfactual": c.get("counterfactual", "")}
            for c in struct.get("causal_chains", [])
        ]
        temporal = [s["description"] for s in sorted(
            struct.get("temporal_sequence", []), key=lambda s: s["step_order"]
        )]

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

        for soul in SOULS:
            try:
                entry_id = cache.store_from_object_graph(
                    soul=soul,
                    graph_dict=graph_dict,
                    category="extended_domain_causal",
                )
                total_stored += 1
                print(f"  Stored in {soul}: {entry_id[:16]}...")
            except Exception as e:
                print(f"  ERR ({soul}): {e}")

    print(f"\n{'=' * 60}")
    print(f"Total causal entries stored: {total_stored}")
    print(f"Graphs processed: {len(STRUCTURES)}")
    print("Done!")


if __name__ == "__main__":
    main()
