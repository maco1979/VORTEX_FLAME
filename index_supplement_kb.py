"""
VORTEX FLAME — Index Supplemental Knowledge Bases
Covers: 6 supplementary categories + Audio-JEPA + Control Loop Architecture
Only indexes HIGH-VALUE knowledge NOT already in the system
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from soul_memory import SoulMemoryEngine

SUPPLEMENT_KNOWLEDGE = [
    # ========================================================================
    # MODULE 1: Scientific Computing / AI / GPU Computing
    # ========================================================================
    {
        "soul": "einstein",
        "category": "knowledge",
        "topic": "[SciCompute] PyTorch CUDA Docker GPU Training Rules",
        "text": """Scientific Computing & AI Training Hardware Rules:

1. PyTorch, CUDA, Docker, MATLAB, simulation software operations only allowed in authorized compute environments.
2. Long-running training, simulation tasks MUST route to Prefect/Temporal (never N8N short processes).
3. GPU state, VRAM usage, compute load MUST real-time stream back to JEPA for hardware world-state modeling.
4. Prohibit LLM from generating destructive Docker commands or system-level compute reset instructions.
5. All model weights, checkpoints stored in models/ or soul_lora_v2/.
6. Training scripts: train_ajepa.py (Audio JEPA). Note: train_soul.py is legacy experimental only.
7. V100-16GB constraint: SFT training ~8GB, DPO training ~11.5GB, max batch_size=8, always 4bit NF4.

JEPA integration: GPU/VRAM metrics → hardware world-state encoding → predict training stability → alert before OOM.
Software list: PyTorch, CUDA Toolkit, Docker Desktop, Jupyter, WSL2, ComfyUI, Stable Diffusion WebUI, Ansys, MATLAB, COMSOL.""",
        "tags": ["scientific-computing", "gpu", "training", "hardware", "jepa"],
    },
    {
        "soul": "cezanne",
        "category": "knowledge",
        "topic": "[SciCompute] Code Execution Sandbox Training Pipeline",
        "text": """Code Execution & Training Pipeline Rules:

1. All code generation from LLM MUST pass sandbox execution (10s timeout) before deployment.
2. extract_python_code → sandbox_exec → build_test_cases → scoring pipeline.
3. HARD_CONSTRAINTS for Algorithm dimension: no code = LOSE, Debug: no code + no bug keywords = LOSE, Complexity: no O() = LOSE.
4. Docker containers for isolated execution: docker run --rm --network=none --memory=512m --cpus=1.
5. GPU training monitoring: nvidia-smi polling every 30s, log to hermes_logs/.
6. Python scientific stack versions locked: torch 2.1.2, transformers 4.36, peft 0.7, bitsandbytes 0.41.

Automation tools: CUDA_VISIBLE_DEVICES control, batch job scheduling, checkpoint auto-save, OOM auto-recovery.
Model registry: models/ for base models, soul_lora_v2/ for LoRA weights.""",
        "tags": ["scientific-computing", "code-execution", "sandbox", "pipeline"],
    },
    {
        "soul": "strategy",
        "category": "knowledge",
        "topic": "[SciCompute] Compute Resource Strategy & Scheduling",
        "text": """Compute Resource Strategy & Scheduling Rules:

1. V100-16GB single GPU strategy: serial training only, never concurrent training processes.
2. Training priority: Einstein > Cezanne > DaVinci > Strategy > Galileo > Darwin > Beethoven > Guizhu > Monet > VanGogh > Humboldt > Herodotus > Montesquieu > YuanLongping.
3. GPU time-slicing: daytime (8h inference + light training), nighttime (dedicated training).
4. OOM prevention: monitor VRAM every 30s, auto-kill rogue processes, nvidia-smi based watchdog.
5. Future V100-32GB upgrade: FP16 7B+DPO ~26GB/32GB, retire 4bit, full precision training.
6. DGX migration plan: only 5 config constants need change, zero code changes required.

Cost optimization: local inference Qwen2.5-3B (zero cost), cloud LLM only for heavy reasoning tasks.
Software: Docker Compose for multi-service, NVIDIA Container Toolkit, WSL2 for Linux tools.""",
        "tags": ["scientific-computing", "strategy", "gpu", "scheduling"],
    },

    # ========================================================================
    # MODULE 2: Video / Streaming / Camera Vision
    # ========================================================================
    {
        "soul": "cezanne",
        "category": "knowledge",
        "topic": "[Vision] OBS FFmpeg OpenCV YOLO Video Pipeline",
        "text": """Video / Streaming / Camera Vision Software Rules:

1. Real-time video streams (OBS, RTSP cameras, webcams) feed directly into Vision-JEPA for environment/object/device state encoding.
2. FFmpeg for all video transcoding: ffmpeg -i input -c:v libx264 -preset fast -crf 23 output.mp4.
3. OpenCV pipeline: cv2.VideoCapture → frame extraction → Vision-JEPA encoding → anomaly detection → alert.
4. YOLO object detection: Ultralytics YOLOv8 pretrained, fine-tune on custom objects via JEPA-guided data curation.
5. Video rendering, transcoding, screen recording are LONG tasks → MUST route to Prefect/Temporal (not N8N).
6. Camera anomaly detection: black screen, disconnect, frame freeze → JEPA real-time state marking → VORTEX circuit breaker.
7. DaVinci Resolve/Premiere Pro: render automation via Python API/ExtendScript, output to designated directories.

JEPA integration: visual stream → Vision-JEPA encoding → world state vector → predict motion, objects, anomalies → feedback loop.
Hardware cameras: USB, RTSP IP cameras, capture cards. Standard interface: execute_action(device_id, action, params).
Software list: OBS Studio, FFmpeg, OpenCV, YOLOv8, DaVinci Resolve, Adobe Premiere, RTSP streaming SDKs.""",
        "tags": ["vision", "video", "camera", "opencv", "jepa"],
    },
    {
        "soul": "davinci",
        "category": "knowledge",
        "topic": "[Vision] Blender 3D Video Integration Pipeline",
        "text": """3D Design & Video Integration Pipeline:

1. Blender Python API (bpy) for all 3D automation: scene creation, rendering, animation, export.
2. Long 3D rendering tasks → Prefect with checkpoint/resume capability, progress tracking.
3. Blender + Vision-JEPA: render previews → JEPA quality check → iterate until pass → final render.
4. Output formats: .blend (source), .fbx/.gltf (exchange), .mp4/.png (render), all versioned with timestamp.
5. CAD software (SolidWorks, AutoCAD): STEP/IGES import, parameter modification via API, export for Blender.
6. Unity/Unreal: asset pipeline from Blender → fbx → Unity, automated build scripts.

JEPA integration: 3D scene → Vision-JEPA spatial encoding → predict lighting, shadows, collisions → design validation.
Software: Blender 4.0+, SolidWorks, AutoCAD, Unity, Unreal Engine, SketchUp, Rhino.""",
        "tags": ["vision", "3d", "blender", "rendering", "jepa"],
    },

    # ========================================================================
    # MODULE 3: Database / Data Platform / ETL
    # ========================================================================
    {
        "soul": "cezanne",
        "category": "knowledge",
        "topic": "[Database] Qdrant Chroma PostgreSQL Redis Rules",
        "text": """Database & Vector Store Management Rules:

1. Qdrant (primary vector store): qdrant_storage/, collection per soul, 384-dim vectors (all-MiniLM-L6-v2).
2. Chroma (lightweight): chroma_db/, for development and local testing.
3. PostgreSQL: structured data (orders, users, financial records, training metadata).
4. Redis: caching layer, session state, real-time JEPA state snapshots, MCP transient memory.
5. SQLite: soul_memory_store/{soul}/ for per-soul memory persistence.
6. ALL writes/deletes/updates to knowledge bases MUST pass VORTEX permission verification.
7. PROHIBIT LLM from directly deleting databases, clearing vector collections, dropping tables.
8. Knowledge base changes MUST sync to MCP + JEPA world state in real-time.

ETL pipeline: Airflow for scheduled data sync, incremental updates, data quality checks.
Backup: daily snapshots to backups/, 7-day retention, automated via Windows Task Scheduler.
Monitoring: disk usage alerts at 80%, connection pool health checks, query latency tracking.
Software: Qdrant, Chroma, PostgreSQL, Redis, Airflow, SQLite, pgAdmin, RedisInsight.""",
        "tags": ["database", "vector-store", "qdrant", "etl", "backup"],
    },
    {
        "soul": "strategy",
        "category": "knowledge",
        "topic": "[Database] NAS Cloud Storage File System Strategy",
        "text": """NAS & Cloud Storage File System Strategy:

1. E drive = training data, datasets, large models, Stable Diffusion, raw audio/video.
2. Project root = code, knowledge bases, LoRA weights, logs, MCP configs.
3. File naming: {project}_{type}_{date}_{version}.{ext}, always timestamped.
4. Dataset versioning: VORTEX_FLAME_Soul_Training/ with subdirectories by project.
5. Sync strategy: rsync/robocopy for local backup, never auto-sync to cloud (security).
6. Cleanup policy: archive >30 days unused to _archive/, delete duplicates weekly.

Storage hierarchy: Hot (SSD, active training) → Warm (HDD, recent datasets) → Cold (archive, >90 days).
Software: Windows File Server, SMB shares, Robocopy, WinSCP, rclone, NAS management tools.""",
        "tags": ["database", "storage", "nas", "filesystem", "backup"],
    },

    # ========================================================================
    # MODULE 4: Network / Ops / Remote Control
    # ========================================================================
    {
        "soul": "cezanne",
        "category": "knowledge",
        "topic": "[Network] SSH Remote Desktop Tunnel Ops Rules",
        "text": """Network & Remote Control Operations Rules:

1. SSH, Remote Desktop, intranet tunneling ONLY within security whitelist (predefined IP ranges, key-based auth).
2. Network state, port status, connection health → real-time stream to JEPA for infrastructure world-state.
3. Firewall rules: inbound restricted to localhost + whitelisted IPs, outbound limited to known API endpoints.
4. Port management: vortex_local_server.py on 8765, ComfyUI on 8188, SD WebUI on 7860, Qdrant on 6333.
5. VPN/intranet tunneling: ZeroTier/Tailscale for secure remote access, never expose directly to internet.
6. Network monitoring: continuous ping to critical services, bandwidth usage alerts, connection log audit.

Security: all remote access requires 2FA, session recording, automatic timeout after 30min idle.
Software: OpenSSH, Windows Remote Desktop, ZeroTier, Tailscale, Wireshark, nmap, WinSCP, Putty.
PowerShell remoting: Enter-PSSession, Invoke-Command with constrained endpoints only.""",
        "tags": ["network", "ssh", "remote", "security", "ops"],
    },

    # ========================================================================
    # MODULE 5: Embedded / IoT / Robotics
    # ========================================================================
    {
        "soul": "einstein",
        "category": "knowledge",
        "topic": "[Embedded] Arduino ESP32 STM32 Modbus PLC Rules",
        "text": """Embedded / IoT / Industrial Hardware Control Rules:

1. All MCU/sensor/PLC commands MUST strictly match device gateway protocol definitions.
2. High-risk hardware commands (motor start, high voltage, continuous operation) MUST pass dual verification (Rule Engine + JEPA physical prediction).
3. Serial protocols: UART 115200 8N1, Modbus RTU/TCP, OPC UA, CAN bus, I2C, SPI.
4. Microcontrollers: Arduino (ATmega328P), ESP32 (WiFi/BLE), STM32 (ARM Cortex-M), Raspberry Pi (Linux GPIO).
5. Sensor types: temperature (DS18B20), humidity (DHT22), pressure, IMU (MPU6050), ultrasonic (HC-SR04), current (ACS712).
6. Actuators: servo (PCA9685), stepper motor (A4988), relay (5V/12V), DC motor (L298N), solenoid.

JEPA integration: sensor readings → JEPA hardware state encoding → predict anomalies, maintenance needs, failure modes.
Safety: emergency stop physical button independent of software, hardware watchdog timer, current limiting fuses.
Software: Arduino IDE, PlatformIO, ESP-IDF, STM32CubeIDE, Raspberry Pi OS, Node-RED, MQTT broker.""",
        "tags": ["embedded", "iot", "arduino", "modbus", "hardware"],
    },

    # ========================================================================
    # MODULE 6: Vertical Industry Software
    # ========================================================================
    {
        "soul": "strategy",
        "category": "knowledge",
        "topic": "[Vertical] GIS Medical Finance Security Industry Rules",
        "text": """Vertical Industry Software Operation Rules:

GIS & Surveying:
- ArcGIS, QGIS, SuperMap: spatial analysis, map generation, coordinate system conversion (WGS84/CGCS2000).
- Drone photogrammetry: Pix4D, Agisoft Metashape → 3D point cloud → CAD/BIM integration.

Medical & Imaging:
- DICOM standard for medical imaging, PACS integration, HIPAA/GDPR compliance for patient data.
- Medical AI: radiology image analysis, pathology slide scanning, strict regulatory validation required.
- NEVER connect medical devices to cloud LLM, all processing local only.

Finance & Trading:
- Bloomberg Terminal API, Wind Financial Terminal, QuantConnect backtesting.
- Trading strategy: backtest → paper trade → small live → scale up, JEPA for market regime detection.
- Compliance: KYC/AML checks, audit trail, trade reconciliation, regulatory reporting.

Tax & Accounting:
- 金蝶/用友 API integration, invoice OCR → amount/tax ID/date extraction, auto reconciliation.
- Chart of accounts: standard PRC GAAP, tax calculation (VAT, income tax, surcharges).

Security & Surveillance:
- Hikvision/Dahua SDK, ONVIF protocol, RTSP stream analysis, YOLO for intrusion detection.
- Access control: facial recognition, card swipe, license plate recognition, all local processing.

ERP & CRM:
- SAP, Oracle, Salesforce, 钉钉/飞书 API for workflow automation, order-to-cash, procure-to-pay.

Rule: LLM ONLY outputs instructions within industry-specific private KB scope. Cross-industry hallucination = circuit breaker.
JEPA integration: industry-specific data → domain JEPA encoding → domain-specific anomaly detection → alert.""",
        "tags": ["vertical-industry", "gis", "medical", "finance", "security"],
    },

    # ========================================================================
    # MODULE 7: Audio-JEPA Auditory World Model (Music + DAW + Hardware)
    # ========================================================================
    {
        "soul": "beethoven",
        "category": "knowledge",
        "topic": "[Audio-JEPA] Auditory World Model Core Rules",
        "text": """Audio-JEPA Auditory World Model Core Rules (JEPA ONLY, NOT to LLM):

1. AUDIO INPUT: Microphone, sound card, Ableton tracks, real-time waveforms, MIDI signals → unified Audio-JEPA temporal encoding.
2. LEARNING TARGETS:
   - Predict next beat rhythm, chord progression, timbre changes, acoustic environment changes.
   - Identify instruments, noise, device status, human voice, environmental sounds.
   - Speaker recognition, emotion detection, speech rate, volume, command intent.
3. REAL-TIME FEEDBACK LOOP:
   Local playback/recorded audio → Audio-JEPA encoding → update world state → feedback to VORTEX for music decisions.
4. PHYSICS CONSTRAINTS:
   All music predictions MUST comply with: key, chords, rhythm, frequency, overtones, acoustic physics rules. NO hallucination.
5. Audio-JEPA encoder: mel spectrogram (n_mels=128, hop_length=512) → Conv2D projector → slot encoder → 256-dim object slots.
6. Real-time streaming: 1-30Hz configurable update frequency, buffered audio chunks, overlap-add reconstruction.
7. Source: Meta FAIR Audio-JEPA paper + pretrained weights (github.com/facebookresearch/audio-jepa).

Hardware: Allen & Heath mixer MIDI/OSC protocol, Focusrite/Universal Audio sound cards, Rode/Shure microphones.
Audio specs: 44.1kHz/48kHz sample rate, 16/24 bit depth, stereo/mono, ASIO driver (Windows), CoreAudio (Mac).""",
        "tags": ["audio-jepa", "world-model", "acoustics", "jepa-core", "beethoven"],
    },
    {
        "soul": "beethoven",
        "category": "knowledge",
        "topic": "[Audio-JEPA] DAW Music Software Workflow Routing",
        "text": """DAW & Music Software Pluggable Workflow Routing:

1. Simple MIDI trigger, short audio automation → N8N (low latency, webhook-based).
2. Long mixing, mastering, multi-track project, complex music rendering → Prefect (checkpoint/resume, progress tracking).
3. Real-time low-latency performance, hardware sync, audio closed-loop control → Custom FastAPI state machine.
4. Ableton Live control: Python API (abletonosc), MIDI CC automation, clip triggering, parameter modulation.
5. Logic Pro/Cubase/Reaper: MIDI Machine Control (MMC), Mackie Control Universal (MCU) protocol.
6. Audio separation: Demucs (vocals/drums/bass/other), UVR5 for high-quality source separation.
7. Audio effects chain: EQ → Compressor → Reverb → Delay → Limiter → Export, parameter ranges from knowledge base.

Rendering pipeline: DAW project → render stems → Demucs separation → Audio-JEPA analysis → remix/recompose → final render.
Software: Ableton Live 11/12, Logic Pro, Cubase, Reaper, Traktor Pro, Allen & Heath dLive/Avantis, Demucs, UVR5.
Plugin formats: VST3, AU, AAX. MIDI: CC 0-127, Note On/Off, Program Change, Pitch Bend, Aftertouch.""",
        "tags": ["audio-jepa", "daw", "ableton", "music-production", "workflow"],
    },
    {
        "soul": "beethoven",
        "category": "knowledge",
        "topic": "[Audio-JEPA] Music Theory LLM Constraint Rules",
        "text": """Music Theory & LLM Decision Constraints:

1. LLM can ONLY make music decisions based on: private music theory KB + Audio-JEPA representation summary + MCP music memory.
2. Prohibit LLM from using external public music knowledge. All chords, genres, BPM, instruments MUST strictly extract from private KB.
3. Music instruction output MUST be structured JSON: {action, params, constraints, safety_limits}.
4. Beyond private KB scope → prohibit music instruction generation → respond "信息不足，无法执行音乐指令".

Music Theory Reference (from private KB):
- Scales: Major (Ionian), Natural Minor (Aeolian), Harmonic Minor, Melodic Minor, Pentatonic, Blues, Chromatic.
- Chords: Major (I-III-V), Minor (I-bIII-V), Dominant 7th (I-III-V-bVII), Diminished, Augmented, Suspended.
- Chord Progressions: I-IV-V-I, ii-V-I, I-V-vi-IV, vi-IV-I-V, I-vi-IV-V, 12-bar blues.
- Rhythm: 4/4, 3/4, 6/8, swing, syncopation, polyrhythm, BPM 60-200.
- Genres: Classical, Jazz, Rock, Electronic, Hip-Hop, Ambient, Cyberpunk, Lo-fi, Funk, Soul.

Instrument knowledge:
- Piano: 88 keys (A0-C8), polyphonic, velocity-sensitive, sustain pedal.
- Guitar: 6 strings (E2-E4 standard), frets, bending, palm mute, harmonics.
- Drums: Kick, Snare, Hi-hat, Toms, Cymbals, MIDI mapping GM standard.
- Synth: Oscillator (sine/square/saw/triangle), Filter (LP/HP/BP), Envelope (ADSR), LFO, effects.

Acoustic physics:
- Frequency ratios: Octave 2:1, Perfect 5th 3:2, Perfect 4th 4:3, Major 3rd 5:4, Minor 3rd 6:5.
- Harmonic series: fundamental f, overtones at 2f, 3f, 4f, 5f... with decreasing amplitude.
- Equal temperament: 12-TET, each semitone = 2^(1/12) ≈ 1.05946 frequency ratio.
- A4 = 440Hz standard pitch reference.""",
        "tags": ["audio-jepa", "music-theory", "llm-constraints", "beethoven"],
    },

    # ========================================================================
    # MODULE 8: Auditory + Voice + Environmental Sound Complete Loop
    # ========================================================================
    {
        "soul": "beethoven",
        "category": "knowledge",
        "topic": "[Audio-JEPA] Speech Voice Environmental Sound Understanding",
        "text": """Speech / Voice / Environmental Sound Understanding Rules:

1. Original audio NEVER sent to cloud. Only Audio-JEPA auditory summary + private KB match results → packaged for cloud LLM.
2. LLM can only respond/control/create based on: private auditory KB + JEPA understanding results + MCP historical auditory memory.
3. Spoken commands auto-map to standardized action JSON. E.g., "把音量调大" → device gateway volume command.

SPEECH UNDERSTANDING (NOT just STT):
- Who is speaking: speaker ID, gender, age range.
- Emotion: happy, sad, angry, neutral, fearful, surprised, commanding, whispering, shouting.
- Speech rate: words per minute, pauses, hesitation patterns.
- Intent: command, question, statement, request, warning, greeting.
- Language: Chinese (Mandarin/Cantonese), English, Japanese, Korean, detection + code-switching.

ENVIRONMENTAL SOUND UNDERSTANDING:
- Animals: dog bark, cat meow, bird chirp, insect buzz.
- Transportation: car engine, horn, train, airplane, helicopter.
- Water: rain, thunder, ocean waves, river, dripping.
- Indoor: door open/close, footsteps, keyboard typing, phone ring, alarm.
- Industrial: motor running, fan noise, machine abnormal sound, bearing failure, gear grinding.
- Device: sound card distortion, microphone clipping, mixer overload, speaker buzz.

Datasets (for Audio-JEPA training):
- LibriSpeech: 1000h English speech (openslr.org/12).
- CommonVoice: multilingual speech corpus (commonvoice.mozilla.org).
- ESC-50: 50 environmental sound classes, 2000 clips.
- UrbanSound8K: 8732 urban sound excerpts, 10 classes.
- MIMII: industrial machine sounds (normal + abnormal), pumps/fans/valves/slide rails.
- FMA-Large: 106,574 music tracks, 161 genres.
- RAVDESS: emotional speech, 8 emotions × 2 intensities.
- SpeechCommands: 105,829 keyword utterances, 35 command words (Google).

Audio-JEPA learns from ALL these to build a unified auditory world model.""",
        "tags": ["audio-jepa", "speech", "environmental-sound", "datasets", "beethoven"],
    },
    {
        "soul": "beethoven",
        "category": "knowledge",
        "topic": "[Audio-JEPA] Audio Hardware Device Gateway",
        "text": """Audio Hardware & Sound Card Device Gateway Rules:

1. UNIFIED INTERFACE for all audio devices:
   - Microphone: condenser/dynamic, XLR/USB, phantom power 48V, gain staging.
   - Sound card: Focusrite Scarlett, Universal Audio Apollo, RME, native ASIO drivers.
   - Monitor: studio monitors, headphones, level calibration (85dB SPL reference).
   - MIDI controller: keyboard, pad, fader, knob, transport control.
   - Allen & Heath mixer: dLive/Avantis MIDI/OSC protocol, channel strip, FX send/return.

2. REAL-TIME AUDIO STREAMING:
   - Audio stream → device gateway → Audio-JEPA real-time encoding → update world state.
   - Latency requirement: <10ms for live performance, <50ms for monitoring, <200ms for analysis.

3. ANOMALY DETECTION:
   - Clipping: signal > 0dBFS → JEPA marks state anomaly → VORTEX circuit breaker.
   - Latency spike: buffer underrun → audio dropout → increase buffer size or reduce plugins.
   - Disconnect: device removed/unplugged → JEPA state → graceful degradation.
   - Noise floor: SNR < threshold → check gain staging/cable/ground loop.
   - Feedback: microphone picks up speaker output → JEPA detects pitch oscillation → auto-mute.

4. AUDIO FORMATS:
   - Recording: WAV 24-bit 48kHz (production), FLAC (archive), MP3 320kbps (distribution).
   - Streaming: Ogg Vorbis/Opus for low-latency, AAC for compatibility.
   - Multi-channel: stereo, 5.1, 7.1, Ambisonics (1st/2nd/3rd order), Dolby Atmos objects.

5. DEVICE GATEWAY STANDARD:
   execute_action(device_id, action, params)
   - device_id: "mic_1", "soundcard_main", "mixer_ch1", "midi_keyboard", "monitor_l".
   - action: "set_gain", "set_volume", "record", "play", "mute", "solo", "route".
   - params: {"value": 0.75, "channel": 1, "duration_ms": 5000}.
   - safety_threshold: {"max_gain": 0.9, "max_volume": 0.85, "timeout_ms": 30000}.""",
        "tags": ["audio-jepa", "hardware", "sound-card", "device-gateway", "beethoven"],
    },

    # ========================================================================
    # MODULE 9: System Control Loop Architecture (Safety + Anti-Drift)
    # ========================================================================
    {
        "soul": "cezanne",
        "category": "knowledge",
        "topic": "[ControlLoop] System Global Safety Rules Hard Constraints",
        "text": """System Global Safety Rules (Hard Constraints, Anti-Drift Foundation):

LLM GLOBAL CONSTRAINTS (mandatory prompt injection):
1. This system is a PRIVATE CLOSED-LOOP architecture. Cloud LLM is PROHIBITED from calling its own built-in public knowledge, connecting to internet, or hallucinating device/software/workflow commands.
2. ALL responses, decisions, action commands MUST ONLY use information from: private KB, MCP memory, JEPA real-time world state provided in current context.
3. Beyond private context scope → unified response: "信息不足，无法执行，请查询私有知识库或补充观测数据".
4. Action output MUST be standardized JSON structured commands. Prohibit natural language arbitrary commands. ALL operations must pass VORTEX Rule Engine + JEPA secondary validation.
5. PROHIBIT output of high-risk, unauthorized, undefined device/software operation commands.

VORTEX SCHEDULING RULES:
1. ALL user requests, device feedback, software results MUST first pass through VORTEX hub. Cloud LLM is PROHIBITED from directly connecting to workflow, hardware, software, JEPA, MCP, knowledge base.
2. Execution order: User Request → Pluggable Workflow Entry → VORTEX (RAG+MCP+JEPA state read) → Cloud LLM Decision → Dual Validation (Rule Engine + JEPA Physical Prediction) → Dispatch Execution.
3. Circuit Breaker: validation failure, JEPA risk prediction, LLM violation output → direct block, NO entry into execution chain.

MCP LAYERING RULES:
1. Upper Text MCP: only stores conversations, business logs, tool text results, only exposed to cloud LLM.
2. Lower JEPA Temporal State DB: stores sensor vectors, hardware time series, world representations, only exposed to JEPA & VORTEX, NOT enter LLM context.

DEVICE ABSTRACTION GATEWAY:
1. All hardware, sensors, vision devices unified through Device Abstraction Gateway. Workflow, LLM, JEPA only call standard interface, NEVER directly access serial port, USB, drivers.
2. Standard call format: {"device_id":"xxx","action":"xxx","params":{},"safety_threshold":{}}.
3. All hardware actions MUST carry safety thresholds. Exceeding threshold → auto circuit breaker.""",
        "tags": ["control-loop", "safety", "anti-drift", "vortex", "architecture"],
    },
    {
        "soul": "strategy",
        "category": "knowledge",
        "topic": "[ControlLoop] Scene Routing Pluggable Workflow Selection",
        "text": """Scene-Adaptive Pluggable Workflow Routing Rules (VORTEX Auto-Match):

SCENE ROUTING MATRIX:
1. Lightweight Webhook, simple trigger, short automation, single command → Route to N8N.
   - Examples: webhook notification, API call, form submission, simple data transform.
   - Latency: <500ms, stateless, fire-and-forget OK.

2. Long 3D rendering, batch processing, async hardware acquisition, time-consuming tasks → Route to Prefect.
   - Examples: Blender render (10min-8h), batch audio processing (1000 files), video transcoding queue.
   - Features: checkpoint/resume, progress tracking, retry with backoff, distributed workers.

3. Industrial sensor closed-loop, high-reliability device control, power-failure resume, strong transactions → Route to Temporal.
   - Examples: PLC control loop, motor sequence with safety interlocks, 24/7 monitoring pipeline.
   - Features: workflow-as-code, exactly-once execution, durable timers, saga pattern for compensation.

4. Local standalone, embedded, ultra-low latency real-time control → Route to Custom FastAPI State Machine.
   - Examples: MIDI performance loop (<10ms), GPIO real-time response, audio effect chain.
   - Features: in-process execution, zero network overhead, deterministic latency.

WORKFLOW UNIFIED
"""
    }
]
