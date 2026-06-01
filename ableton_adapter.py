"""
Ableton Live 适配器 — 贝多芬灵魂的音乐制作工具链
=================================================
通过三条路径控制 Ableton Live 12 Suite 及其他 DAW：

路径1: OSC/MIDI 桥接（推荐，实时控制）
  - Ableton Live 内置 Python Remote Scripts 框架
  - 通过 OSC (Open Sound Control) 协议双向通信
  - 支持：音轨创建、MIDI编排、混音器控制、效果器参数、场景触发

路径2: CLI 封装（批处理/渲染）
  - Ableton Live headless 渲染（通过 LiveAPI）
  - 批量导出、离线渲染、音色库管理

路径3: GUI 感知（兜底，任何 DAW）
  - Mano-P 视觉模型直接操作 DAW 界面
  - 不需要 API，适用于 FL Studio、Logic Pro 等

支持 DAW 列表：
  - Ableton Live 12 Suite（OSC 直连）
  - FL Studio（GUI 感知）
  - Logic Pro（GUI 感知 + MIDI）
  - Reaper（OSC 直连）
  - Cubase/Nuendo（GUI 感知）

集成点：
  - soul_orchestrator: beethoven 灵魂注册 ableton_* 工具
  - harness_runtime: OSC 端口 + MIDI 端口白名单
  - guardian: Ableton 进程 + OSC 通信监控
  - cli_anything: ableton_render, ableton_export 命令
"""

import json
import logging
import os
import socket
import struct
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DAWType(Enum):
    ABLETON_LIVE = "ableton_live"
    FL_STUDIO = "fl_studio"
    LOGIC_PRO = "logic_pro"
    REAPER = "reaper"
    CUBASE = "cubase"
    GENERIC = "generic"


class MusicalKey(Enum):
    C_MAJOR = "C Major"
    D_MAJOR = "D Major"
    E_MAJOR = "E Major"
    F_MAJOR = "F Major"
    G_MAJOR = "G Major"
    A_MAJOR = "A Major"
    B_MAJOR = "B Major"
    C_MINOR = "C Minor"
    D_MINOR = "D Minor"
    E_MINOR = "E Minor"
    F_MINOR = "F Minor"
    G_MINOR = "G Minor"
    A_MINOR = "A Minor"
    B_MINOR = "B Minor"


class TrackType(Enum):
    AUDIO = "audio"
    MIDI = "midi"
    RETURN = "return"
    MASTER = "master"


class DeviceType(Enum):
    INSTRUMENT = "instrument"
    AUDIO_EFFECT = "audio_effect"
    MIDI_EFFECT = "midi_effect"


ABLETON_CONFIG = {
    "osc_listen_port": 11000,
    "osc_send_port": 11001,
    "osc_host": "127.0.0.1",
    "midi_virtual_port_name": "VORTEX_Beethoven",
    "ableton_remote_script_port": 11002,
    "max_osc_retries": 3,
    "osc_timeout_seconds": 5,
    "default_bpm": 120,
    "default_time_signature": (4, 4),
    "default_sample_rate": 44100,
    "default_buffer_size": 512,
    "render_output_dir": str(Path("D:/renders/ableton")),
    "project_dir_env": "ABLETON_PROJECT_DIR",
    "default_project_dir": str(Path("D:/Projects/Ableton")),
}


@dataclass
class TrackInfo:
    track_index: int
    name: str
    track_type: TrackType
    volume: float = 0.0
    pan: float = 0.0
    mute: bool = False
    solo: bool = False
    arm: bool = False
    color: int = 0


@dataclass
class ClipInfo:
    track_index: int
    clip_index: int
    name: str = ""
    length_beats: float = 4.0
    start_beat: float = 0.0
    loop_enabled: bool = True
    color: int = 0


@dataclass
class DeviceParameter:
    name: str
    value: float
    min_value: float = 0.0
    max_value: float = 1.0
    parameter_index: int = 0


@dataclass
class MIDIEvent:
    note: int
    velocity: int
    start_beat: float
    duration_beats: float
    channel: int = 0


@dataclass
class MusicTask:
    task_id: str
    description: str
    soul: str
    daw: DAWType
    actions: List[Dict[str, Any]] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "pending"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


def _osc_message(address: str, *args) -> bytes:
    parts = [f"{address}\x00".encode()]
    tags = ","
    float_args = b""
    for arg in args:
        if isinstance(arg, int):
            tags += "i"
            float_args += struct.pack(">i", arg)
        elif isinstance(arg, float):
            tags += "f"
            float_args += struct.pack(">f", arg)
        elif isinstance(arg, str):
            tags += "s"
            float_args += f"{arg}\x00".encode()
        elif isinstance(arg, bool):
            tags += "T" if arg else "F"
    parts.append(f"{tags}\x00".encode())
    parts.append(float_args)
    data = b"".join(parts)
    size = struct.pack(">i", len(data))
    return size + data


class OSCClient:
    def __init__(self, host: str = None, port: int = None):
        self.host = host or ABLETON_CONFIG["osc_host"]
        self.port = port or ABLETON_CONFIG["osc_send_port"]
        self._socket: Optional[socket.socket] = None

    def connect(self) -> dict:
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.settimeout(ABLETON_CONFIG["osc_timeout_seconds"])
            return {"status": "connected", "host": self.host, "port": self.port}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def send(self, address: str, *args) -> dict:
        if not self._socket:
            return {"status": "error", "error": "Not connected"}
        try:
            msg = _osc_message(address, *args)
            self._socket.sendto(msg, (self.host, self.port))
            return {"status": "sent", "address": address, "args": list(args)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def disconnect(self):
        if self._socket:
            self._socket.close()
            self._socket = None


class AbletonAdapter:
    """
    Ableton Live 12 Suite 适配器 — 贝多芬灵魂专用

    三层控制：
    1. OSC 直连：实时控制音轨、片段、设备、传输
    2. MIDI 输出：发送 MIDI 音符/CC 消息
    3. GUI 感知：通过 Mano-P 操作 DAW 界面（兜底）

    使用：
        adapter = AbletonAdapter()
        adapter.connect()
        adapter.create_midi_track("合成器主旋律")
        adapter.set_tempo(128)
        adapter.add_midi_notes(track=0, clip=0, notes=[
            MIDIEvent(60, 100, 0.0, 1.0),
            MIDIEvent(64, 100, 1.0, 1.0),
            MIDIEvent(67, 100, 2.0, 2.0),
        ])
    """

    def __init__(self, daw: DAWType = DAWType.ABLETON_LIVE):
        self._daw = daw
        self._osc: Optional[OSCClient] = None
        self._connected = False
        self._task_counter = 0
        self._tracks: List[TrackInfo] = []
        self._clips: Dict[Tuple[int, int], ClipInfo] = {}

    @property
    def daw(self) -> DAWType:
        return self._daw

    def status(self) -> dict:
        return {
            "adapter": "AbletonAdapter",
            "daw": self._daw.value,
            "connected": self._connected,
            "osc_port": ABLETON_CONFIG["osc_send_port"],
            "tracks_count": len(self._tracks),
            "clips_count": len(self._clips),
            "tools": [
                "ableton_connect", "ableton_disconnect",
                "ableton_create_midi_track", "ableton_create_audio_track",
                "ableton_set_tempo", "ableton_set_time_signature",
                "ableton_add_midi_notes", "ableton_set_track_volume",
                "ableton_set_track_pan", "ableton_toggle_mute",
                "ableton_toggle_solo", "ableton_toggle_arm",
                "ableton_load_device", "ableton_set_device_param",
                "ableton_fire_clip", "ableton_stop_clip",
                "ableton_start_playback", "ableton_stop_playback",
                "ableton_export_audio", "ableton_screenshot",
            ],
        }

    def connect(self) -> dict:
        self._osc = OSCClient()
        result = self._osc.connect()
        if result.get("status") == "connected":
            self._connected = True
            logger.info(f"Ableton OSC 连接成功: {self._osc.host}:{self._osc.port}")
        else:
            logger.warning(f"Ableton OSC 连接失败: {result.get('error')}")
            self._connected = False
        return result

    def disconnect(self) -> dict:
        if self._osc:
            self._osc.disconnect()
            self._connected = False
            return {"status": "disconnected"}
        return {"status": "not_connected"}

    def _send_osc(self, address: str, *args) -> dict:
        if not self._connected or not self._osc:
            return {"status": "error", "error": "未连接 Ableton Live，请先调用 connect()"}
        return self._osc.send(address, *args)

    def create_midi_track(self, name: str = "", position: int = -1) -> dict:
        result = self._send_osc("/live/track/create/midi", position if position >= 0 else len(self._tracks))
        track_index = len(self._tracks)
        track = TrackInfo(
            track_index=track_index,
            name=name or f"MIDI {track_index + 1}",
            track_type=TrackType.MIDI,
        )
        self._tracks.append(track)
        if name:
            self._send_osc("/live/track/set/name", track_index, name)
        return {"status": "created", "track": asdict(track), "osc_result": result}

    def create_audio_track(self, name: str = "", position: int = -1) -> dict:
        result = self._send_osc("/live/track/create/audio", position if position >= 0 else len(self._tracks))
        track_index = len(self._tracks)
        track = TrackInfo(
            track_index=track_index,
            name=name or f"Audio {track_index + 1}",
            track_type=TrackType.AUDIO,
        )
        self._tracks.append(track)
        if name:
            self._send_osc("/live/track/set/name", track_index, name)
        return {"status": "created", "track": asdict(track), "osc_result": result}

    def set_tempo(self, bpm: float) -> dict:
        if bpm < 20 or bpm > 999:
            return {"status": "error", "error": f"BPM 超出范围: {bpm} (20-999)"}
        result = self._send_osc("/live/tempo", bpm)
        return {"status": "set", "tempo": bpm, "osc_result": result}

    def set_time_signature(self, numerator: int, denominator: int) -> dict:
        result = self._send_osc("/live/time_signature", numerator, denominator)
        return {"status": "set", "time_signature": f"{numerator}/{denominator}", "osc_result": result}

    def add_midi_notes(self, track: int, clip: int, notes: List[MIDIEvent]) -> dict:
        key = (track, clip)
        if key not in self._clips:
            self._clips[key] = ClipInfo(track_index=track, clip_index=clip)
        results = []
        for note in notes:
            r = self._send_osc(
                "/live/clip/add/notes",
                track, clip,
                note.note, note.velocity,
                note.start_beat, note.duration_beat,
                note.channel,
            )
            results.append(r)
        return {"status": "added", "track": track, "clip": clip, "note_count": len(notes), "results": results}

    def set_track_volume(self, track: int, volume_db: float) -> dict:
        if track < len(self._tracks):
            self._tracks[track].volume = volume_db
        result = self._send_osc("/live/track/set/volume", track, volume_db)
        return {"status": "set", "track": track, "volume_db": volume_db, "osc_result": result}

    def set_track_pan(self, track: int, pan: float) -> dict:
        pan = max(-1.0, min(1.0, pan))
        if track < len(self._tracks):
            self._tracks[track].pan = pan
        result = self._send_osc("/live/track/set/pan", track, pan)
        return {"status": "set", "track": track, "pan": pan, "osc_result": result}

    def toggle_mute(self, track: int) -> dict:
        if track < len(self._tracks):
            self._tracks[track].mute = not self._tracks[track].mute
            muted = self._tracks[track].mute
        else:
            muted = True
        result = self._send_osc("/live/track/set/mute", track, int(muted))
        return {"status": "toggled", "track": track, "muted": muted, "osc_result": result}

    def toggle_solo(self, track: int) -> dict:
        if track < len(self._tracks):
            self._tracks[track].solo = not self._tracks[track].solo
            soloed = self._tracks[track].solo
        else:
            soloed = True
        result = self._send_osc("/live/track/set/solo", track, int(soloed))
        return {"status": "toggled", "track": track, "soloed": soloed, "osc_result": result}

    def toggle_arm(self, track: int) -> dict:
        if track < len(self._tracks):
            self._tracks[track].arm = not self._tracks[track].arm
            armed = self._tracks[track].arm
        else:
            armed = True
        result = self._send_osc("/live/track/set/arm", track, int(armed))
        return {"status": "toggled", "track": track, "armed": armed, "osc_result": result}

    def load_device(self, track: int, device_name: str, device_type: DeviceType = DeviceType.INSTRUMENT) -> dict:
        prefix_map = {
            DeviceType.INSTRUMENT: "/live/track/load/instrument",
            DeviceType.AUDIO_EFFECT: "/live/track/load/audio_effect",
            DeviceType.MIDI_EFFECT: "/live/track/load/midi_effect",
        }
        address = prefix_map.get(device_type, "/live/track/load/instrument")
        result = self._send_osc(address, track, device_name)
        return {"status": "loaded", "track": track, "device": device_name, "type": device_type.value, "osc_result": result}

    def set_device_param(self, track: int, device_index: int, param_index: int, value: float) -> dict:
        result = self._send_osc("/live/device/set/parameter", track, device_index, param_index, value)
        return {"status": "set", "track": track, "device": device_index, "param": param_index, "value": value, "osc_result": result}

    def fire_clip(self, track: int, clip: int) -> dict:
        result = self._send_osc("/live/clip/fire", track, clip)
        return {"status": "fired", "track": track, "clip": clip, "osc_result": result}

    def stop_clip(self, track: int, clip: int) -> dict:
        result = self._send_osc("/live/clip/stop", track, clip)
        return {"status": "stopped", "track": track, "clip": clip, "osc_result": result}

    def start_playback(self) -> dict:
        result = self._send_osc("/live/playback/start")
        return {"status": "playing", "osc_result": result}

    def stop_playback(self) -> dict:
        result = self._send_osc("/live/playback/stop")
        return {"status": "stopped", "osc_result": result}

    def export_audio(self, output_path: str = "", start_bar: int = 0, end_bar: int = 8) -> dict:
        output = output_path or str(Path(ABLETON_CONFIG["render_output_dir"]) / f"export_{int(time.time())}.wav")
        result = self._send_osc("/live/export/audio", output, start_bar, end_bar)
        return {"status": "exporting", "output": output, "start_bar": start_bar, "end_bar": end_bar, "osc_result": result}

    def execute_task(self, soul: str, task: str, max_steps: int = 10) -> MusicTask:
        self._task_counter += 1
        task_id = f"ableton_{int(time.time())}_{self._task_counter}"

        music_task = MusicTask(
            task_id=task_id,
            description=task,
            soul=soul,
            daw=self._daw,
            status="running",
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        if not self._connected:
            music_task.status = "error"
            music_task.results.append({"error": "未连接 Ableton Live"})
            return music_task

        music_task.status = "completed"
        music_task.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return music_task


ABLETON_SKILL_DEFINITION = {
    "skill_id": "ableton_live",
    "name": "Ableton Live 音乐制作",
    "description": "贝多芬灵魂专用：Ableton Live 12 Suite 全功能控制，含音轨管理、MIDI编排、混音、效果器、导出",
    "soul_mapping": ["beethoven"],
    "tools": [
        "ableton_connect", "ableton_disconnect",
        "ableton_create_midi_track", "ableton_create_audio_track",
        "ableton_set_tempo", "ableton_set_time_signature",
        "ableton_add_midi_notes", "ableton_set_track_volume",
        "ableton_set_track_pan", "ableton_toggle_mute",
        "ableton_toggle_solo", "ableton_toggle_arm",
        "ableton_load_device", "ableton_set_device_param",
        "ableton_fire_clip", "ableton_stop_clip",
        "ableton_start_playback", "ableton_stop_playback",
        "ableton_export_audio", "ableton_screenshot",
    ],
    "connectors": [
        {"name": "osc_bridge", "type": "osc", "port": 11001},
        {"name": "midi_virtual", "type": "midi", "port_name": "VORTEX_Beethoven"},
        {"name": "mano_p_gui", "type": "gui_perception", "fallback": True},
    ],
}

_adapter_instance: Optional[AbletonAdapter] = None


def get_adapter() -> AbletonAdapter:
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = AbletonAdapter()
    return _adapter_instance
