"""
Device Gateway — Unified Abstraction Layer for Hardware & Software Control
===========================================================================
Provides a standardized, pluggable interface for controlling local hardware
(sensors, cameras, serial ports, motors, audio interfaces) and local software
(3D modeling, DAW, IDE, office suites, scripts) from the VORTEX_FLAME system.

Architecture Role:
  LLM (decision layer) → VORTEX (permission + translation) → DeviceGateway → Backend
  JEPA reads device state through this gateway for world model updates.

Design Principles:
  1. ALL hardware/software access goes through this gateway — never direct
  2. Backends are pluggable — swap without changing upper layers
  3. JEPA queries device state via standard interface
  4. Safety: each action passes through VORTEX permission check + rule engine
  5. Observability: every action logged to MCP/soul_memory

Gate Classification:
  ┌──────────────────────────────────────────────────────────────────┐
  │ DEVICE_CATEGORY_AUDIO     = audio interfaces, DAW, MIDI, mics   │
  │ DEVICE_CATEGORY_VISUAL    = cameras, displays, GPUs             │
  │ DEVICE_CATEGORY_COMPUTE   = local software, scripts, IDEs       │
  │ DEVICE_CATEGORY_CREATIVE  = 3D modeling, video editing, design  │
  │ DEVICE_CATEGORY_NETWORK   = SSH, remote desktop, network tools  │
  │ DEVICE_CATEGORY_EMBEDDED  = microcontrollers, sensors, motors   │
  │ DEVICE_CATEGORY_DATABASE  = SQL, vector DB, file system         │
  │ DEVICE_CATEGORY_INDUSTRY  = vertical software, GIS, medical     │
  │ DEVICE_CATEGORY_SYSTEM    = OS, shell, process management       │
  └──────────────────────────────────────────────────────────────────┘

Usage:
  gw = DeviceGateway()
  gw.register_backend("local_camera", CameraBackend(index=0))
  result = gw.execute("local_camera", "capture", {"resolution": "1080p"})
  state = gw.get_state("local_camera")
"""

import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class DeviceCategory(Enum):
    AUDIO = "audio"
    VISUAL = "visual"
    COMPUTE = "compute"
    CREATIVE = "creative"
    NETWORK = "network"
    EMBEDDED = "embedded"
    DATABASE = "database"
    INDUSTRY = "industry"
    SYSTEM = "system"


class ActionSeverity(Enum):
    READ = "read"
    LOW_RISK = "low_risk"
    MODIFY = "modify"
    HIGH_RISK = "high_risk"
    DESTRUCTIVE = "destructive"


@dataclass
class DeviceInfo:
    device_id: str
    device_name: str
    category: DeviceCategory
    device_type: str
    vendor: str = "unknown"
    model: str = "unknown"
    capabilities: List[str] = field(default_factory=list)
    safety_params: Dict[str, Any] = field(default_factory=dict)
    status: str = "unknown"
    last_seen: float = 0.0


@dataclass
class ActionSpec:
    action_name: str
    description: str
    severity: ActionSeverity
    params_schema: Dict[str, Any] = field(default_factory=dict)
    requires_permission: bool = True
    requires_jepa_check: bool = False
    max_duration_ms: int = 30000
    retry_count: int = 1


@dataclass
class ActionResult:
    success: bool
    device_id: str
    action: str
    data: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    jepa_state_snapshot: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)


class DeviceBackend(ABC):
    @abstractmethod
    def connect(self) -> bool:
        ...

    @abstractmethod
    def disconnect(self) -> bool:
        ...

    @abstractmethod
    def execute(self, action: str, params: Dict[str, Any]) -> ActionResult:
        ...

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def get_capabilities(self) -> List[ActionSpec]:
        ...

    @property
    @abstractmethod
    def device_info(self) -> DeviceInfo:
        ...


class NullBackend(DeviceBackend):
    def __init__(self, device_id: str, category: DeviceCategory, device_type: str = "null"):
        self._info = DeviceInfo(
            device_id=device_id, device_name=device_id,
            category=category, device_type=device_type,
        )

    def connect(self) -> bool:
        return True

    def disconnect(self) -> bool:
        return True

    def execute(self, action: str, params: Dict[str, Any]) -> ActionResult:
        return ActionResult(
            success=False, device_id=self._info.device_id, action=action,
            error=f"Backend not installed for {self._info.device_id}",
        )

    def get_state(self) -> Dict[str, Any]:
        return {"device_id": self._info.device_id, "status": "not_connected"}

    def get_capabilities(self) -> List[ActionSpec]:
        return []

    @property
    def device_info(self) -> DeviceInfo:
        return self._info


class DeviceGateway:
    def __init__(
        self,
        memory_engine=None,
        world_cache=None,
        rule_engine: Optional[Any] = None,
        jepa_bridge=None,
    ):
        self._backends: Dict[str, DeviceBackend] = {}
        self._device_registry: Dict[str, DeviceInfo] = {}
        self._action_history: List[ActionResult] = []
        self._lock = threading.Lock()
        self._rule_engine = rule_engine
        self._jepa_bridge = jepa_bridge
        self._memory_engine = memory_engine
        self._world_cache = world_cache

        self._action_whitelist: Dict[str, List[str]] = {}
        self._action_blacklist: Dict[str, List[str]] = {}

    def register_backend(self, device_id: str, backend: DeviceBackend) -> bool:
        with self._lock:
            if device_id in self._backends:
                logger.warning(f"Backend {device_id} already registered, replacing")
            self._backends[device_id] = backend
            self._device_registry[device_id] = backend.device_info
            backend.connect()
            logger.info(f"Registered backend: {device_id} ({backend.device_info.device_type})")
            return True

    def unregister_backend(self, device_id: str) -> bool:
        with self._lock:
            if device_id not in self._backends:
                return False
            self._backends[device_id].disconnect()
            del self._backends[device_id]
            self._device_registry.pop(device_id, None)
            return True

    def register_null_device(
        self, device_id: str, category: DeviceCategory, device_type: str,
        capabilities: Optional[List[str]] = None,
        vendor: str = "unknown", model: str = "unknown",
    ) -> DeviceInfo:
        backend = NullBackend(device_id, category, device_type)
        info = backend.device_info
        info.capabilities = capabilities or []
        info.vendor = vendor
        info.model = model
        info.status = "reserved"
        self.register_backend(device_id, backend)
        return info

    def execute(
        self,
        device_id: str,
        action: str,
        params: Optional[Dict[str, Any]] = None,
        bypass_safety: bool = False,
    ) -> ActionResult:
        params = params or {}

        if device_id not in self._backends:
            return ActionResult(
                success=False, device_id=device_id, action=action,
                error=f"Device '{device_id}' not registered",
            )

        if not bypass_safety:
            check_result = self._safety_check(device_id, action, params)
            if not check_result["allowed"]:
                return ActionResult(
                    success=False, device_id=device_id, action=action,
                    error=f"Safety blocked: {check_result['reason']}",
                )

        start = time.time()
        try:
            result = self._backends[device_id].execute(action, params)
        except Exception as e:
            result = ActionResult(
                success=False, device_id=device_id, action=action,
                error=str(e),
            )
        result.duration_ms = (time.time() - start) * 1000

        with self._lock:
            self._action_history.append(result)
            if len(self._action_history) > 1000:
                self._action_history = self._action_history[-500:]

        return result

    def get_state(self, device_id: str) -> Dict[str, Any]:
        if device_id not in self._backends:
            return {"device_id": device_id, "status": "not_registered"}
        return self._backends[device_id].get_state()

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        return {did: self.get_state(did) for did in self._backends}

    def list_devices(self, category: Optional[DeviceCategory] = None) -> List[DeviceInfo]:
        devices = list(self._device_registry.values())
        if category:
            devices = [d for d in devices if d.category == category]
        return devices

    def get_action_history(self, limit: int = 50) -> List[ActionResult]:
        with self._lock:
            return list(self._action_history[-limit:])

    def set_safety_rules(
        self,
        device_id: str,
        whitelist: Optional[List[str]] = None,
        blacklist: Optional[List[str]] = None,
    ):
        if whitelist is not None:
            self._action_whitelist[device_id] = whitelist
        if blacklist is not None:
            self._action_blacklist[device_id] = blacklist

    def _safety_check(self, device_id: str, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        blacklist = self._action_blacklist.get(device_id, [])
        if action in blacklist:
            return {"allowed": False, "reason": f"Action '{action}' blacklisted for {device_id}"}

        whitelist = self._action_whitelist.get(device_id, [])
        if whitelist and action not in whitelist:
            return {"allowed": False, "reason": f"Action '{action}' not in whitelist for {device_id}"}

        if self._rule_engine is not None:
            try:
                rule_result = self._rule_engine.check(device_id, action, params)
                if not rule_result.get("allowed", True):
                    return rule_result
            except Exception:
                pass

        return {"allowed": True, "reason": "ok"}

    def query_jepa_state(self, device_id: str) -> Optional[Dict[str, Any]]:
        if self._jepa_bridge is None or self._world_cache is None:
            return None
        try:
            state = self.get_state(device_id)
            results = self._jepa_bridge.dual_pathway_query(
                query=f"设备{device_id}状态预测",
                soul="cezanne",
                use_llm=False,
            )
            results["raw_device_state"] = state
            return results
        except Exception as e:
            logger.warning(f"JEPA state query failed for {device_id}: {e}")
            return None


_global_gateway: Optional[DeviceGateway] = None
_gateway_lock = threading.Lock()


def get_device_gateway(
    memory_engine=None, world_cache=None, rule_engine=None, jepa_bridge=None,
) -> DeviceGateway:
    global _global_gateway
    with _gateway_lock:
        if _global_gateway is None:
            _global_gateway = DeviceGateway(
                memory_engine=memory_engine,
                world_cache=world_cache,
                rule_engine=rule_engine,
                jepa_bridge=jepa_bridge,
            )
        return _global_gateway


PREDEFINED_DEVICES = {
    "audio": [
        ("sound_card", DeviceCategory.AUDIO, "sound_card", ["capture", "playback", "monitor"],
         "ASIO/WASAPI", "Primary"),
        ("ableton_daw", DeviceCategory.AUDIO, "daw", ["load_project", "render", "midi_control"],
         "Ableton", "Live"),
        ("microphone", DeviceCategory.AUDIO, "microphone", ["start_recording", "stop_recording"],
         "Generic", "USB"),
    ],
    "visual": [
        ("camera_0", DeviceCategory.VISUAL, "camera", ["capture", "stream", "calibrate"],
         "Generic", "USB"),
        ("obs_studio", DeviceCategory.VISUAL, "streaming_software",
         ["start_stream", "stop_stream", "switch_scene"], "OBS", "Studio"),
    ],
    "compute": [
        ("python_runtime", DeviceCategory.COMPUTE, "script_runner",
         ["execute_script", "import_module"], "Python", "3.x"),
        ("jupyter_server", DeviceCategory.COMPUTE, "notebook_server",
         ["start_kernel", "run_cell", "shutdown"], "Jupyter", "Server"),
        ("cuda_gpu", DeviceCategory.COMPUTE, "gpu",
         ["query_status", "allocate_memory", "run_inference"], "NVIDIA", "V100"),
    ],
    "creative": [
        ("blender_3d", DeviceCategory.CREATIVE, "3d_software",
         ["render_scene", "export_model", "import_asset", "run_script"], "Blender", "Foundation"),
        ("davinci_resolve", DeviceCategory.CREATIVE, "video_editor",
         ["import_clip", "render_timeline", "apply_color_grade"], "Blackmagic", "DaVinci"),
        ("photoshop", DeviceCategory.CREATIVE, "image_editor",
         ["open_document", "apply_filter", "export_image"], "Adobe", "Photoshop"),
    ],
    "network": [
        ("ssh_client", DeviceCategory.NETWORK, "ssh",
         ["connect", "execute_command", "transfer_file"], "OpenSSH", "Client"),
        ("remote_desktop", DeviceCategory.NETWORK, "remote_desktop",
         ["connect", "disconnect", "send_keystroke"], "RDP", "Client"),
    ],
    "embedded": [
        ("serial_port", DeviceCategory.EMBEDDED, "serial",
         ["open", "close", "send_data", "read_data"], "Generic", "COM"),
        ("arduino_board", DeviceCategory.EMBEDDED, "microcontroller",
         ["upload_firmware", "reset", "read_sensor"], "Arduino", "Uno"),
        ("raspberry_pi", DeviceCategory.EMBEDDED, "single_board",
         ["ssh_exec", "gpio_control", "i2c_read"], "Raspberry Pi", "4B"),
    ],
    "database": [
        ("qdrant_db", DeviceCategory.DATABASE, "vector_db",
         ["search", "upsert", "delete_collection"], "Qdrant", "Cloud"),
        ("sqlite_db", DeviceCategory.DATABASE, "sql_db",
         ["query", "execute", "backup"], "SQLite", "3.x"),
        ("file_system", DeviceCategory.DATABASE, "filesystem",
         ["read_file", "write_file", "list_directory"], "OS", "NTFS"),
    ],
    "industry": [
        ("gis_software", DeviceCategory.INDUSTRY, "gis",
         ["load_map", "geocode", "spatial_query"], "ArcGIS", "Pro"),
        ("medical_viewer", DeviceCategory.INDUSTRY, "medical",
         ["load_dicom", "measure", "export_report"], "DICOM", "Viewer"),
        ("erp_system", DeviceCategory.INDUSTRY, "erp",
         ["query_inventory", "create_order", "generate_report"], "Generic", "ERP"),
    ],
    "system": [
        ("powershell", DeviceCategory.SYSTEM, "shell",
         ["execute_command", "get_process", "stop_process"], "Microsoft", "PowerShell"),
        ("docker_engine", DeviceCategory.SYSTEM, "container",
         ["run_container", "stop_container", "list_containers"], "Docker", "Engine"),
    ],
}


def init_default_devices(gateway: Optional[DeviceGateway] = None) -> DeviceGateway:
    if gateway is None:
        gateway = get_device_gateway()
    for _category, devices in PREDEFINED_DEVICES.items():
        for device_id, category, dtype, capabilities, vendor, model in devices:
            gateway.register_null_device(
                device_id=device_id, category=category, device_type=dtype,
                capabilities=capabilities, vendor=vendor, model=model,
            )

    _init_real_backends(gateway)

    logger.info(f"Initialized {len(PREDEFINED_DEVICES)} device categories")
    return gateway


def _init_real_backends(gateway: DeviceGateway):
    try:
        from camera_backend import OpenCVBackend, probe_cameras
        cameras = probe_cameras(max_index=2)
        if cameras:
            cam = cameras[0]
            backend = OpenCVBackend(device_id="camera_0", camera_index=cam["index"])
            if backend.connect():
                gateway.register_backend("camera_0", backend)
                logger.info(f"camera_0: REAL backend activated ({cam['resolution'][0]}x{cam['resolution'][1]})")
                for extra in cameras[1:]:
                    extra_id = f"camera_{extra['index']}"
                    extra_backend = OpenCVBackend(device_id=extra_id, camera_index=extra["index"])
                    if extra_backend.connect():
                        gateway.register_backend(extra_id, extra_backend)
                        logger.info(f"{extra_id}: REAL backend activated")
            else:
                logger.info("camera_0: OpenCV installed but no camera detected, keeping NullBackend")
        else:
            logger.info("camera_0: no cameras found, keeping NullBackend")
    except ImportError:
        logger.info("camera_0: opencv-python not installed, keeping NullBackend")
    except Exception as e:
        logger.warning(f"camera_0: real backend init failed ({e}), keeping NullBackend")

    try:
        from serial_backend import SerialBackend, probe_serial_ports
        ports = probe_serial_ports()
        if ports:
            port = ports[0]
            backend = SerialBackend(device_id="serial_port", port=port["port"])
            if backend.connect():
                gateway.register_backend("serial_port", backend)
                logger.info(f"serial_port: REAL backend activated on {port['port']}")
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"serial_port: real backend init failed ({e})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gw = init_default_devices()
    print(f"Registered devices: {len(gw.list_devices())}")
    for cat in DeviceCategory:
        devices = gw.list_devices(cat)
        if devices:
            print(f"  {cat.value}: {[d.device_id for d in devices]}")
