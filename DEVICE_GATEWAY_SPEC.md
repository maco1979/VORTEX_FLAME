# VORTEX_FLAME Device Gateway & Control Loop Specification

> **Version**: 1.0 | **Date**: 2026-05-30

## 1. Philosophy

The Device Gateway is a **reserved interface abstraction layer** — not a hardware implementation.
It establishes the contract between VORTEX's AI decision-making and the physical world,
so that when hardware/software integration happens, the interface is already defined.

```
Design principle:
  Today:   NullBackend placeholders (validates interface shape)
  Tomorrow: RealBackend implementations (plugs into hardware)
  Never:    LLM directly touching hardware/serial/USB/drivers
```

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  VORTEX-FLAME (Central Hub)                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │MCP Memory│ │Knowledge │ │ JEPA     │ │ Safety Harness   │ │
│  │(text)    │ │Base      │ │(vectors) │ │ (RuleEngine)     │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘ │
│                           │                                     │
│                   Dual Validation:                              │
│                   RuleEngine (hard) + JEPA (physics)            │
└────────────────────────────┬────────────────────────────────────┘
                             │  JSON Intent
┌────────────────────────────▼────────────────────────────────────┐
│  DEVICE GATEWAY                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  DeviceBackend ABC:                                      │ │
│  │    connect() / disconnect() / execute() / get_state()    │ │
│  └──────────────────────────────────────────────────────────┘ │
│                              │                                  │
│     ┌────────────────────────┼────────────────────────┐        │
│     ▼                        ▼                        ▼        │
│  NullBackend            RealBackend              RealBackend   │
│  (placeholder)          (when ready)            (when ready)   │
└─────────────────────────────────────────────────────────────────┘
```

## 3. Device Registry (24 devices, 9 categories)

### 3.1 Audio Category
| Device ID | Type | Capabilities | Soul |
|-----------|------|-------------|------|
| `sound_card` | Audio IO | Capture, playback, routing | Beethoven |
| `ableton_daw` | DAW Software | Load project, render, MIDI control | Beethoven |
| `microphone` | Input | Audio capture via ASIO/WASAPI | Beethoven |

### 3.2 Visual Category
| Device ID | Type | Capabilities | Soul |
|-----------|------|-------------|------|
| `camera_0` | Camera | Image capture, video stream, 1-30Hz | Cezanne |
| `obs_studio` | Streaming | Scene switch, record, stream | Cezanne |

### 3.3 Compute Category
| Device ID | Type | Capabilities | Soul |
|-----------|------|-------------|------|
| `python_runtime` | Runtime | Execute Python code | Cezanne |
| `jupyter_server` | Notebook | Interactive kernels | Cezanne |
| `cuda_gpu` | GPU | CUDA ops, VRAM management | Cezanne |

### 3.4 Creative Category
| Device ID | Type | Capabilities | Soul |
|-----------|------|-------------|------|
| `blender_3d` | 3D Engine | Scene, render, export | DaVinci |
| `davinci_resolve` | Video Editor | Timeline, color, render | DaVinci |
| `photoshop` | Image Editor | Layers, filters, export | Monet |

### 3.5 Network Category
| Device ID | Type | Capabilities | Soul |
|-----------|------|-------------|------|
| `ssh_client` | Remote Shell | SSH connect, exec, file transfer | Cezanne |
| `remote_desktop` | RDP/VNC | Remote GUI access | Cezanne |

### 3.6 Embedded Category
| Device ID | Type | Capabilities | Soul |
|-----------|------|-------------|------|
| `serial_port` | COM Port | Read/write serial | DaVinci |
| `arduino_board` | MCU | Upload, GPIO, sensors | DaVinci |
| `raspberry_pi` | SBC | SSH, GPIO, Linux ops | DaVinci |

### 3.7 Database Category
| Device ID | Type | Capabilities | Soul |
|-----------|------|-------------|------|
| `qdrant_db` | Vector DB | Index, search, filter | Cezanne |
| `sqlite_db` | SQL DB | Query, write, schema | Cezanne |
| `file_system` | FS | Read, write, list, delete | Cezanne |

### 3.8 Industry Category
| Device ID | Type | Capabilities | Soul |
|-----------|------|-------------|------|
| `gis_software` | GIS | Map ops, geo query | Humboldt |
| `medical_viewer` | Medical | DICOM, image analysis | Darwin |
| `erp_system` | ERP | Business ops, finance | Strategy |

### 3.9 System Category
| Device ID | Type | Capabilities | Soul |
|-----------|------|-------------|------|
| `powershell` | Shell | System commands (whitelist) | Cezanne |
| `docker_engine` | Container | Run, manage containers | Cezanne |

## 4. Standard Call Format

```json
{
  "device_id": "blender_3d",
  "action": "render_scene",
  "params": {
    "scene": "main",
    "frames": [1, 250],
    "output_path": "E:/renders/scene_001/",
    "format": "PNG",
    "resolution": "1920x1080"
  },
  "safety_threshold": {
    "max_duration_seconds": 14400,
    "max_gpu_vram_gb": 8.0,
    "max_cpu_percent": 80
  },
  "bypass_safety": false
}
```

## 5. Control Loop (7 Steps)

```
Step 1: USER INPUT → DeviceGateway entry (auth + log)
Step 2: VORTEX bundles context (MCP + Knowledge Base + JEPA state)
Step 3: LLM outputs standardized JSON intent (private context only)
Step 4: DUAL VALIDATION
        ├── RuleEngine: hard-check permissions + safety boundaries
        └── JEPA: predict action consequences (physics feasibility)
Step 5: Workflow calls DeviceGateway.execute(device_id, action, params)
Step 6: Device results feed back to JEPA → world state updated
Step 7: Results + logs written to MCP (text) + JEPA state DB (vectors)
```

## 6. Safety Model

### 6.1 Four-Layer Anti-Drift

| Layer | Type | Mechanism |
|-------|------|-----------|
| L1: Prompt | Soft | Private context + system prompt restrictions |
| L2: RuleEngine | Hard | JSON Schema validation + device permission table |
| L3: JEPA | Physics | Action consequence prediction + anomaly detection |
| L4: CircuitBreaker | Hard | 3 consecutive failures → 8h degraded mode |

### 6.2 Safety Rules

```
1. Whitelist-only: only registered devices can be called
2. All actions must have safety_threshold
3. Destructive actions require bypass_safety=False (default: True)
4. LLM NEVER touches hardware directly — always through VORTEX + DeviceGateway
5. Action history maintained (last 500 entries, ring buffer)
6. JEPA can query device state for world model updates
```

## 7. JEPA Integration

### 7.1 State Query
```python
# JEPA reads device state for world model
state = device_gateway.query_jepa_state()
# Returns: dict mapping device_id → {status, last_action, metrics}
```

### 7.2 Action Conditioned Prediction (v2)
```python
# ActionCAJEPA evaluates action before execution
action_vector = device_gateway.encode_action(device_id, action, params)
future_state = action_cajepa.predict_consequence(history_features, action_vector)
# VORTEX checks if predicted state is safe before allowing execution
```

## 8. Future: Real Backend Implementation Priority

When hardware/software is ready, implement backends in this order:

```
1. PowerShellBackend   → System automation (immediately useful)
2. PythonRuntimeBackend → Code execution (immediately useful)
3. FileSystemBackend    → Knowledge base management
4. N8NWebhookBackend   → Workflow engine integration
5. DockerBackend        → Container management
6. CameraBackend        → OpenCV integration
7. OBSWebSocketBackend → OBS control
8. SSHClientBackend     → Remote operations
```

## 9. File Reference

| File | Purpose |
|------|---------|
| `device_gateway.py` | Core gateway with NullBackend placeholders |
| `five_layer_jepa/causal_jepa_v2.py` | ActionConditionedCausalPredictor + ActionCAJEPA |
| `extended_domain_knowledge_v2.json` | Device KB, Control Loop KB, Audio KB entries |
| `gen_knowledge_v2.py` | Knowledge entry generator for device/control rules |
| `index_knowledge_v2.py` | Knowledge indexer for device/control rules |
| `extract_causal_v2.py` | Causal extraction for control loop graphs |
