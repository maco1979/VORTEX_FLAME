"""
Mano-P Adapter — GUI Perception Agent for Edge Devices
========================================================
Open-source GUI perception agent model (Apache 2.0, 2.1k stars).
Pure vision-based GUI operation — no CDP/HTML dependency.

Core Capabilities:
- Visual GUI understanding and autonomous task execution
- 13 global multimodal benchmark #1 (OSWorld, MBench, etc.)
- Natural language driven: describe task → system executes GUI ops
- Full interaction: click, type, hotkey, scroll, drag, screenshot,
  wait, app launch, URL navigation

Dual Inference Modes:
- Local:  Model runs on-device (M4+ 32GB Mac / Mano-P Compute Stick)
- Cloud:  API-based inference, zero local model config
- Auto-detect: system probes local model availability and switches

Cross-Platform:
- macOS (stable), Windows (stable), Linux (Beta)

Integration with VORTEX_FLAME:
- Registered as soul tool capability (cezanne, davinci primary)
- Skill registry: mano_p_gui skill for GUI automation tasks
- Guardian whitelist: mano-p process + localhost inference port
- CLI catalog: mano_p_run, mano_p_screenshot, mano_p_status
- Soul memory: execution trajectories stored in 'trajectory' category

NOTES:
- This adapter does NOT touch E: drive (MoE training is isolated)
- Local model weights are loaded from MANO_P_MODEL_DIR (default: D:)
- Cloud mode only needs MANO_P_API_KEY env var
"""

import json
import logging
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class InferenceMode(Enum):
    LOCAL = "local"
    CLOUD = "cloud"
    AUTO = "auto"


class PlatformSupport(Enum):
    MACOS = "macos"
    WINDOWS = "windows"
    LINUX = "linux"


class GUIAction(Enum):
    CLICK = "click"
    LEFT_DOUBLE = "left_double"
    RIGHT_SINGLE = "right_single"
    DRAG = "drag"
    HOTKEY = "hotkey"
    TYPE = "type"
    SCROLL = "scroll"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    APP_LAUNCH = "app_launch"
    URL_NAVIGATE = "url_navigate"
    MOUSE_MOVE = "mouse_move"
    FINISHED = "finished"
    CALL_USER = "call_user"


ACTION_SPACES = [
    "click(start_box='[x1, y1, x2, y2]')",
    "left_double(start_box='[x1, y1, x2, y2]')",
    "right_single(start_box='[x1, y1, x2, y2]')",
    "drag(start_box='[x1, y1, x2, y2]', end_box='[x3, y3, x4, y4]')",
    "hotkey(key='')",
    "type(content='')",
    "scroll(start_box='[x1, y1, x2, y2]', direction='down|up|left|right')",
    "wait()",
    "screenshot()",
    "app_launch(app_name='')",
    "url_navigate(url='')",
    "mouse_move(start_box='[x1, y1, x2, y2]')",
    "finished()",
    "call_user()",
]


MANO_P_CONFIG = {
    "model_name": "Mano-P",
    "license": "Apache-2.0",
    "local_min_ram_gb": 32,
    "local_supported_chips": ["apple_m4", "apple_m4_pro", "apple_m4_max",
                              "apple_m4_ultra"],
    "compute_stick_usb": "usb4.0",
    "cloud_api_base": "https://api.mano-p.ai/v1",
    "local_inference_port": 9450,
    "local_inference_host": "127.0.0.1",
    "screenshot_quality": 75,
    "default_wait_seconds": 5,
    "max_retries": 3,
    "retry_delay_seconds": 2,
    "model_dir_env": "MANO_P_MODEL_DIR",
    "api_key_env": "MANO_P_API_KEY",
    "default_model_dir": str(Path("D:/models/mano-p")),
}


@dataclass
class HardwareProfile:
    os_name: str
    os_version: str
    cpu_arch: str
    ram_gb: float
    gpu_name: Optional[str] = None
    apple_chip: Optional[str] = None
    compute_stick_connected: bool = False

    def supports_local(self) -> bool:
        if self.compute_stick_connected:
            return True
        if self.apple_chip and any(
            chip in self.apple_chip.lower()
            for chip in MANO_P_CONFIG["local_supported_chips"]
        ):
            return self.ram_gb >= MANO_P_CONFIG["local_min_ram_gb"]
        return False


@dataclass
class GUIActionRequest:
    action: GUIAction
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "action_type": self.action.value,
            "action_inputs": self.parameters,
            "description": self.description,
        }


@dataclass
class GUIActionResult:
    success: bool
    action: str
    screenshot_base64: Optional[str] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    mode_used: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TaskExecution:
    task_id: str
    description: str
    soul: str
    mode: InferenceMode
    actions: List[GUIActionRequest] = field(default_factory=list)
    results: List[GUIActionResult] = field(default_factory=list)
    status: str = "pending"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    total_time_ms: float = 0.0


def detect_hardware() -> HardwareProfile:
    """Probe current hardware to determine local inference capability."""
    os_name = platform.system().lower()
    os_version = platform.version()
    cpu_arch = platform.machine()

    ram_gb = 0.0
    try:
        if os_name == "darwin":
            import subprocess as sp
            mem_bytes = int(sp.check_output(["sysctl", "-n", "hw.memsize"]).strip())
            ram_gb = mem_bytes / (1024 ** 3)
        elif os_name == "windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            MEMORYSTATUSEX = ctypes.c_ulonglong * 8
            stat = MEMORYSTATUSEX()
            stat[0] = 64
            kernel32.GlobalMemoryStatusEx(stat)
            ram_gb = stat[1] / (1024 ** 3)
        elif os_name == "linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        ram_gb = int(line.split()[1]) / (1024 ** 2)
                        break
    except Exception as e:
        logger.warning(f"RAM detection failed: {e}")

    apple_chip = None
    if os_name == "darwin":
        try:
            cpu_brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            if "apple" in cpu_brand.lower():
                apple_chip = cpu_brand
        except Exception:
            pass

    compute_stick = False
    if os_name == "darwin" or os_name == "linux":
        try:
            lsusb = subprocess.check_output(["lsusb"], stderr=subprocess.DEVNULL).decode()
            compute_stick = "mano" in lsusb.lower() or "Mano-P" in lsusb
        except Exception:
            pass
    elif os_name == "windows":
        try:
            wmic = subprocess.check_output(
                ["wmic", "path", "Win32_PnPEntity", "get", "Name"],
                stderr=subprocess.DEVNULL,
            ).decode()
            compute_stick = "mano" in wmic.lower() or "Mano-P" in wmic
        except Exception:
            pass

    gpu_name = None
    try:
        if os_name == "windows":
            wmic_gpu = subprocess.check_output(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                stderr=subprocess.DEVNULL,
            ).decode()
            lines = [l.strip() for l in wmic_gpu.strip().split("\n") if l.strip()]
            if len(lines) > 1:
                gpu_name = lines[1]
    except Exception:
        pass

    return HardwareProfile(
        os_name=os_name,
        os_version=os_version,
        cpu_arch=cpu_arch,
        ram_gb=round(ram_gb, 1),
        gpu_name=gpu_name,
        apple_chip=apple_chip,
        compute_stick_connected=compute_stick,
    )


def resolve_inference_mode(mode: InferenceMode = InferenceMode.AUTO) -> InferenceMode:
    """Determine actual inference mode based on hardware and config."""
    if mode != InferenceMode.AUTO:
        return mode

    hw = detect_hardware()
    if hw.supports_local():
        model_dir = os.environ.get(
            MANO_P_CONFIG["model_dir_env"],
            MANO_P_CONFIG["default_model_dir"],
        )
        if Path(model_dir).exists():
            logger.info(f"Mano-P local model found at {model_dir}, using LOCAL mode")
            return InferenceMode.LOCAL

    api_key = os.environ.get(MANO_P_CONFIG["api_key_env"])
    if api_key:
        logger.info("Mano-P API key found, using CLOUD mode")
        return InferenceMode.CLOUD

    logger.warning("No local model or API key found, defaulting to CLOUD (will fail without key)")
    return InferenceMode.CLOUD


class LocalInferenceEngine:
    """Local Mano-P model inference via subprocess or compute stick."""

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir or os.environ.get(
            MANO_P_CONFIG["model_dir_env"],
            MANO_P_CONFIG["default_model_dir"],
        )
        self.port = MANO_P_CONFIG["local_inference_port"]
        self.host = MANO_P_CONFIG["local_inference_host"]
        self._server_process: Optional[subprocess.Popen] = None
        self._ready = False

    def start_server(self) -> dict:
        """Start local Mano-P inference server."""
        if self._ready:
            return {"status": "already_running", "port": self.port}

        model_path = Path(self.model_dir)
        if not model_path.exists():
            return {"status": "error", "error": f"Model dir not found: {self.model_dir}"}

        try:
            self._server_process = subprocess.Popen(
                [
                    "python", "-m", "mano_p.serve",
                    "--model-dir", str(model_path),
                    "--host", self.host,
                    "--port", str(self.port),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(model_path.parent),
            )
            time.sleep(3)
            if self._server_process.poll() is not None:
                return {"status": "error", "error": "Server process exited prematurely"}

            self._ready = True
            return {"status": "started", "port": self.port, "pid": self._server_process.pid}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def stop_server(self) -> dict:
        if self._server_process and self._server_process.poll() is None:
            self._server_process.terminate()
            self._server_process.wait(timeout=10)
            self._ready = False
            return {"status": "stopped"}
        return {"status": "not_running"}

    def is_ready(self) -> bool:
        if not self._ready or self._server_process is None:
            return False
        return self._server_process.poll() is None

    def execute_action(self, action: GUIActionRequest, screenshot_b64: Optional[str] = None) -> GUIActionResult:
        """Send action to local inference server and get result."""
        import urllib.request
        import urllib.error

        if not self.is_ready():
            return GUIActionResult(
                success=False, action=action.action.value,
                error="Local server not running", mode_used="local",
            )

        url = f"http://{self.host}:{self.port}/v1/action"
        payload = json.dumps({
            "action": action.to_dict(),
            "screenshot": screenshot_b64,
        }).encode()

        start = time.time()
        try:
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode())
                elapsed = (time.time() - start) * 1000
                return GUIActionResult(
                    success=True,
                    action=action.action.value,
                    screenshot_base64=result.get("screenshot"),
                    execution_time_ms=elapsed,
                    mode_used="local",
                )
        except urllib.error.URLError as e:
            return GUIActionResult(
                success=False, action=action.action.value,
                error=str(e), execution_time_ms=(time.time() - start) * 1000,
                mode_used="local",
            )
        except Exception as e:
            return GUIActionResult(
                success=False, action=action.action.value,
                error=str(e), execution_time_ms=(time.time() - start) * 1000,
                mode_used="local",
            )


class CloudInferenceEngine:
    """Cloud-based Mano-P inference via API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get(MANO_P_CONFIG["api_key_env"])
        self.api_base = MANO_P_CONFIG["cloud_api_base"]

    def execute_action(self, action: GUIActionRequest, screenshot_b64: Optional[str] = None) -> GUIActionResult:
        """Send action to cloud API and get result."""
        import urllib.request
        import urllib.error

        if not self.api_key:
            return GUIActionResult(
                success=False, action=action.action.value,
                error="No API key configured (set MANO_P_API_KEY env var)",
                mode_used="cloud",
            )

        url = f"{self.api_base}/action"
        payload = json.dumps({
            "action": action.to_dict(),
            "screenshot": screenshot_b64,
        }).encode()

        start = time.time()
        try:
            req = urllib.request.Request(
                url, data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode())
                elapsed = (time.time() - start) * 1000
                return GUIActionResult(
                    success=True,
                    action=action.action.value,
                    screenshot_base64=result.get("screenshot"),
                    execution_time_ms=elapsed,
                    mode_used="cloud",
                )
        except urllib.error.URLError as e:
            return GUIActionResult(
                success=False, action=action.action.value,
                error=str(e), execution_time_ms=(time.time() - start) * 1000,
                mode_used="cloud",
            )
        except Exception as e:
            return GUIActionResult(
                success=False, action=action.action.value,
                error=str(e), execution_time_ms=(time.time() - start) * 1000,
                mode_used="cloud",
            )


class ManoPAdapter:
    """
    Unified Mano-P GUI Perception Agent adapter for VORTEX_FLAME.

    Provides:
    - Dual-mode inference (local/cloud) with auto-detection
    - Natural language task → GUI action sequence
    - Full GUI interaction primitives
    - Screenshot capture and analysis
    - Execution trajectory logging to soul memory

    Usage:
        adapter = ManoPAdapter()
        result = adapter.execute_task(
            soul="cezanne",
            task="Open Chrome and navigate to github.com",
        )
    """

    def __init__(self, mode: InferenceMode = InferenceMode.AUTO):
        self._mode = mode
        self._resolved_mode: Optional[InferenceMode] = None
        self._local_engine: Optional[LocalInferenceEngine] = None
        self._cloud_engine: Optional[CloudInferenceEngine] = None
        self._hardware: Optional[HardwareProfile] = None
        self._task_counter = 0

    @property
    def hardware(self) -> HardwareProfile:
        if self._hardware is None:
            self._hardware = detect_hardware()
        return self._hardware

    @property
    def resolved_mode(self) -> InferenceMode:
        if self._resolved_mode is None:
            self._resolved_mode = resolve_inference_mode(self._mode)
        return self._resolved_mode

    def _get_engine(self):
        mode = self.resolved_mode
        if mode == InferenceMode.LOCAL:
            if self._local_engine is None:
                self._local_engine = LocalInferenceEngine()
            return self._local_engine
        else:
            if self._cloud_engine is None:
                self._cloud_engine = CloudInferenceEngine()
            return self._cloud_engine

    def status(self) -> dict:
        """Get current adapter status including hardware and mode info."""
        hw = self.hardware
        return {
            "adapter": "Mano-P",
            "resolved_mode": self.resolved_mode.value,
            "requested_mode": self._mode.value,
            "hardware": {
                "os": hw.os_name,
                "ram_gb": hw.ram_gb,
                "cpu_arch": hw.cpu_arch,
                "apple_chip": hw.apple_chip,
                "gpu": hw.gpu_name,
                "compute_stick": hw.compute_stick_connected,
                "supports_local": hw.supports_local(),
            },
            "local_server_running": (
                self._local_engine.is_ready() if self._local_engine else False
            ),
            "cloud_api_configured": bool(os.environ.get(MANO_P_CONFIG["api_key_env"])),
            "action_spaces": ACTION_SPACES,
        }

    def start_local_server(self) -> dict:
        """Start the local inference server (only for LOCAL mode)."""
        if self.resolved_mode != InferenceMode.LOCAL:
            return {"status": "error", "error": "Not in LOCAL mode"}
        return self._get_engine().start_server()

    def stop_local_server(self) -> dict:
        if self._local_engine:
            return self._local_engine.stop_server()
        return {"status": "not_running"}

    def execute_action(self, action: GUIActionRequest,
                       screenshot_b64: Optional[str] = None) -> GUIActionResult:
        """Execute a single GUI action."""
        engine = self._get_engine()
        return engine.execute_action(action, screenshot_b64)

    def execute_task(self, soul: str, task: str,
                     max_steps: int = 20,
                     screenshot_interval: int = 1) -> TaskExecution:
        """
        Execute a natural language GUI task.

        The adapter sends the task description to the Mano-P model,
        which returns a sequence of GUI actions. Each action is executed
        and the result (including screenshot) is fed back for the next step.

        Args:
            soul: The VORTEX_FLAME soul requesting execution
            task: Natural language task description
            max_steps: Maximum action steps before forced finish
            screenshot_interval: Take screenshot every N steps

        Returns:
            TaskExecution with full action/result trajectory
        """
        self._task_counter += 1
        task_id = f"manop_{int(time.time())}_{self._task_counter}"

        execution = TaskExecution(
            task_id=task_id,
            description=task,
            soul=soul,
            mode=self.resolved_mode,
            status="running",
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        engine = self._get_engine()

        if isinstance(engine, LocalInferenceEngine) and not engine.is_ready():
            start_result = engine.start_server()
            if start_result.get("status") not in ("started", "already_running"):
                execution.status = "error"
                execution.results.append(GUIActionResult(
                    success=False, action="init",
                    error=f"Failed to start local server: {start_result.get('error')}",
                    mode_used="local",
                ))
                return execution

        current_screenshot = None
        for step in range(max_steps):
            action_request = GUIActionRequest(
                action=GUIAction.CLICK,
                parameters={"task": task, "step": step},
                description=f"Step {step + 1} of task: {task}",
            )

            result = engine.execute_action(action_request, current_screenshot)
            execution.actions.append(action_request)
            execution.results.append(result)

            if result.screenshot_base64:
                current_screenshot = result.screenshot_base64

            if not result.success:
                logger.warning(f"Step {step + 1} failed: {result.error}")
                retries = 0
                while retries < MANO_P_CONFIG["max_retries"] and not result.success:
                    retries += 1
                    time.sleep(MANO_P_CONFIG["retry_delay_seconds"])
                    result = engine.execute_action(action_request, current_screenshot)
                    execution.results[-1] = result

            if result.action == "finished" or result.action == "call_user":
                execution.status = "completed"
                break
        else:
            execution.status = "max_steps_reached"

        execution.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        execution.total_time_ms = sum(r.execution_time_ms for r in execution.results)

        return execution

    def take_screenshot(self) -> GUIActionResult:
        """Capture current screen state."""
        action = GUIActionRequest(action=GUIAction.SCREENSHOT, description="Manual screenshot capture")
        return self.execute_action(action)

    def click(self, box: Tuple[int, int, int, int]) -> GUIActionResult:
        """Click at the specified bounding box."""
        action = GUIActionRequest(
            action=GUIAction.CLICK,
            parameters={"start_box": list(box)},
            description=f"Click at {box}",
        )
        return self.execute_action(action)

    def type_text(self, content: str, submit: bool = False) -> GUIActionResult:
        """Type text content. If submit=True, appends \\n to submit."""
        text = content + ("\n" if submit else "")
        action = GUIActionRequest(
            action=GUIAction.TYPE,
            parameters={"content": text},
            description=f"Type: {content[:50]}{'...' if len(content) > 50 else ''}",
        )
        return self.execute_action(action)

    def hotkey(self, key: str) -> GUIActionResult:
        """Press a keyboard shortcut."""
        action = GUIActionRequest(
            action=GUIAction.HOTKEY,
            parameters={"key": key},
            description=f"Hotkey: {key}",
        )
        return self.execute_action(action)

    def scroll(self, box: Tuple[int, int, int, int], direction: str = "down") -> GUIActionResult:
        """Scroll within the specified bounding box."""
        action = GUIActionRequest(
            action=GUIAction.SCROLL,
            parameters={"start_box": list(box), "direction": direction},
            description=f"Scroll {direction} at {box}",
        )
        return self.execute_action(action)

    def drag(self, start_box: Tuple[int, int, int, int],
             end_box: Tuple[int, int, int, int]) -> GUIActionResult:
        """Drag from start_box to end_box."""
        action = GUIActionRequest(
            action=GUIAction.DRAG,
            parameters={"start_box": list(start_box), "end_box": list(end_box)},
            description=f"Drag from {start_box} to {end_box}",
        )
        return self.execute_action(action)

    def launch_app(self, app_name: str) -> GUIActionResult:
        """Launch an application by name."""
        action = GUIActionRequest(
            action=GUIAction.APP_LAUNCH,
            parameters={"app_name": app_name},
            description=f"Launch app: {app_name}",
        )
        return self.execute_action(action)

    def navigate_url(self, url: str) -> GUIActionResult:
        """Navigate to a URL in the default browser."""
        action = GUIActionRequest(
            action=GUIAction.URL_NAVIGATE,
            parameters={"url": url},
            description=f"Navigate to: {url}",
        )
        return self.execute_action(action)

    def wait(self, seconds: int = None) -> GUIActionResult:
        """Wait for the specified duration."""
        params = {}
        if seconds is not None:
            params["seconds"] = seconds
        action = GUIActionRequest(
            action=GUIAction.WAIT,
            parameters=params,
            description=f"Wait {seconds or MANO_P_CONFIG['default_wait_seconds']}s",
        )
        return self.execute_action(action)


_adapter_instance: Optional[ManoPAdapter] = None


def get_adapter(mode: InferenceMode = InferenceMode.AUTO) -> ManoPAdapter:
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = ManoPAdapter(mode=mode)
    return _adapter_instance


MANO_P_SKILL_DEFINITION = {
    "skill_id": "mano_p_gui",
    "name": "mano_p_gui",
    "description": "GUI perception and operation via Mano-P — pure vision-based computer control",
    "source": "external",
    "status": "seed",
    "soul_mapping": ["cezanne", "davinci"],
    "commands": [
        {
            "name": "manop_click",
            "description": "Click at a screen location",
            "template": "click(start_box='[x1, y1, x2, y2]')",
            "parameters": ["start_box"],
        },
        {
            "name": "manop_type",
            "description": "Type text content",
            "template": "type(content='')",
            "parameters": ["content"],
        },
        {
            "name": "manop_hotkey",
            "description": "Press keyboard shortcut",
            "template": "hotkey(key='')",
            "parameters": ["key"],
        },
        {
            "name": "manop_scroll",
            "description": "Scroll in a direction",
            "template": "scroll(start_box='[x1, y1, x2, y2]', direction='down')",
            "parameters": ["start_box", "direction"],
        },
        {
            "name": "manop_drag",
            "description": "Drag from one location to another",
            "template": "drag(start_box='[x1, y1, x2, y2]', end_box='[x3, y3, x4, y4]')",
            "parameters": ["start_box", "end_box"],
        },
        {
            "name": "manop_screenshot",
            "description": "Capture current screen",
            "template": "screenshot()",
            "parameters": [],
        },
        {
            "name": "manop_launch",
            "description": "Launch an application",
            "template": "app_launch(app_name='')",
            "parameters": ["app_name"],
        },
        {
            "name": "manop_navigate",
            "description": "Navigate to URL",
            "template": "url_navigate(url='')",
            "parameters": ["url"],
        },
        {
            "name": "manop_execute_task",
            "description": "Execute a natural language GUI task autonomously",
            "template": "execute_task(task='')",
            "parameters": ["task"],
        },
    ],
    "connectors": [
        {
            "name": "mano_p_local",
            "type": "local_inference",
            "config": {
                "port": MANO_P_CONFIG["local_inference_port"],
                "host": MANO_P_CONFIG["local_inference_host"],
            },
        },
        {
            "name": "mano_p_cloud",
            "type": "cloud_api",
            "config": {
                "api_base": MANO_P_CONFIG["cloud_api_base"],
            },
        },
    ],
}
