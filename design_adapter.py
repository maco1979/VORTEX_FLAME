"""
设计软件适配器 — 莫奈/梵高灵魂的视觉创作工具链
=================================================
通过三条路径控制设计软件：

路径1: API 直连（Figma REST API / Photoshop UXP）
  - Figma: 完整 REST API，读取/修改设计文件
  - Photoshop: UXP 插件 + ExtendScript (JSX)
  - Illustrator: ExtendScript + CEP 面板

路径2: CLI 封装（ImageMagick / FFmpeg / SVG 工具链）
  - 批量图片处理、格式转换、色彩校正
  - SVG 生成/编辑、矢量图处理
  - 视频帧提取、GIF 生成

路径3: GUI 感知（兜底，任何设计软件）
  - Mano-P 视觉模型直接操作设计软件界面
  - 适用于 Sketch、CorelDRAW、Affinity Designer 等

支持设计软件列表：
  - Figma（REST API 直连）
  - Adobe Photoshop（UXP + ExtendScript）
  - Adobe Illustrator（ExtendScript）
  - Adobe After Effects（ExtendScript + JSX）
  - Sketch（GUI 感知）
  - CorelDRAW（GUI 感知）
  - Affinity Designer（GUI 感知）

集成点：
  - soul_orchestrator: monet/vangogh 灵魂注册 design_* 工具
  - harness_runtime: Figma API + 本地进程白名单
  - guardian: 设计软件进程监控
  - cli_anything: imagemagick, ffmpeg, svgexport 命令
"""

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DesignSoftware(Enum):
    FIGMA = "figma"
    PHOTOSHOP = "photoshop"
    ILLUSTRATOR = "illustrator"
    AFTER_EFFECTS = "after_effects"
    SKETCH = "sketch"
    CORELDRAW = "coreldraw"
    AFFINITY = "affinity_designer"
    GENERIC = "generic"


class ColorSpace(Enum):
    SRGB = "sRGB"
    DISPLAY_P3 = "display-p3"
    CMYK = "CMYK"
    LAB = "LAB"


class ImageFormat(Enum):
    PNG = "png"
    JPEG = "jpeg"
    SVG = "svg"
    WEBP = "webp"
    PDF = "pdf"
    PSD = "psd"
    AI = "ai"
    EPS = "eps"


DESIGN_CONFIG = {
    "figma_api_base": "https://api.figma.com/v1",
    "figma_token_env": "FIGMA_ACCESS_TOKEN",
    "photoshop_script_dir": str(Path("D:/VORTEX_FLAME/tmp/ps_scripts")),
    "imagemagick_paths": {
        "windows": [r"C:\Program Files\ImageMagick\magick.exe"],
        "darwin": ["/usr/local/bin/magick", "/opt/homebrew/bin/magick"],
        "linux": ["/usr/bin/magick", "/usr/bin/convert"],
    },
    "ffmpeg_paths": {
        "windows": [r"C:\ffmpeg\bin\ffmpeg.exe"],
        "darwin": ["/usr/local/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg"],
        "linux": ["/usr/bin/ffmpeg"],
    },
    "default_export_dir": str(Path("D:/exports/design")),
    "default_project_dir": str(Path("D:/Projects/Design")),
    "api_timeout_seconds": 30,
    "script_timeout_seconds": 120,
}


@dataclass
class ColorValue:
    r: int
    g: int
    b: int
    a: float = 1.0
    color_space: ColorSpace = ColorSpace.SRGB

    def to_hex(self) -> str:
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"

    def to_rgb_tuple(self) -> Tuple[int, int, int]:
        return (self.r, self.g, self.b)

    def to_css(self) -> str:
        if self.a < 1.0:
            return f"rgba({self.r},{self.g},{self.b},{self.a})"
        return f"rgb({self.r},{self.g},{self.b})"


@dataclass
class DesignLayer:
    name: str
    layer_type: str = "GROUP"
    visible: bool = True
    locked: bool = False
    opacity: float = 1.0
    x: float = 0.0
    y: float = 0.0
    width: float = 100.0
    height: float = 100.0
    fill_color: Optional[ColorValue] = None
    stroke_color: Optional[ColorValue] = None
    stroke_width: float = 0.0
    children: List[str] = field(default_factory=list)


@dataclass
class DesignFile:
    file_key: str
    name: str
    software: DesignSoftware
    width: float = 1920.0
    height: float = 1080.0
    layers: List[DesignLayer] = field(default_factory=list)
    last_modified: str = ""


@dataclass
class DesignTask:
    task_id: str
    description: str
    soul: str
    software: DesignSoftware
    actions: List[Dict[str, Any]] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "pending"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


def _find_executable(paths_key: str) -> Optional[str]:
    import platform
    import shutil
    os_name = platform.system().lower()
    paths = DESIGN_CONFIG[paths_key].get(os_name, [])
    for p in paths:
        if Path(p).exists():
            return p
    name = "magick" if "imagemagick" in paths_key else "ffmpeg"
    return shutil.which(name)


class FigmaClient:
    def __init__(self, token: str = None):
        self._token = token or os.environ.get(DESIGN_CONFIG["figma_token_env"], "")
        self._base = DESIGN_CONFIG["figma_api_base"]

    def _headers(self) -> dict:
        return {"X-FIGMA-TOKEN": self._token, "Content-Type": "application/json"}

    def _request(self, method: str, path: str, **kwargs) -> dict:
        import urllib.request
        import urllib.error
        url = f"{self._base}{path}"
        headers = self._headers()
        data = json.dumps(kwargs.get("json", {})).encode() if method == "POST" else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=DESIGN_CONFIG["api_timeout_seconds"]) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            return {"error": f"HTTP {e.code}: {e.read().decode()[:200]}"}
        except Exception as e:
            return {"error": str(e)}

    def get_file(self, file_key: str) -> dict:
        return self._request("GET", f"/files/{file_key}")

    def get_node(self, file_key: str, node_id: str) -> dict:
        return self._request("GET", f"/files/{file_key}/nodes?ids={node_id}")

    def get_image(self, file_key: str, node_id: str, format: str = "png", scale: float = 2.0) -> dict:
        return self._request("GET", f"/images/{file_key}?ids={node_id}&format={format}&scale={scale}")

    def get_styles(self, file_key: str) -> dict:
        return self._request("GET", f"/files/{file_key}/styles")

    def get_components(self, file_key: str) -> dict:
        return self._request("GET", f"/files/{file_key}/components")

    def is_available(self) -> bool:
        return bool(self._token)


class ImageProcessor:
    def __init__(self):
        self._magick_path: Optional[str] = None
        self._ffmpeg_path: Optional[str] = None

    @property
    def magick(self) -> Optional[str]:
        if self._magick_path is None:
            self._magick_path = _find_executable("imagemagick_paths")
        return self._magick_path

    @property
    def ffmpeg(self) -> Optional[str]:
        if self._ffmpeg_path is None:
            self._ffmpeg_path = _find_executable("ffmpeg_paths")
        return self._ffmpeg_path

    def resize(self, input_path: str, output_path: str, width: int, height: int) -> dict:
        if not self.magick:
            return {"status": "error", "error": "未找到 ImageMagick"}
        cmd = [self.magick, input_path, "-resize", f"{width}x{height}", output_path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return {"status": "success" if result.returncode == 0 else "error", "output": output_path}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def convert_format(self, input_path: str, output_path: str) -> dict:
        if not self.magick:
            return {"status": "error", "error": "未找到 ImageMagick"}
        cmd = [self.magick, input_path, output_path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return {"status": "success" if result.returncode == 0 else "error", "output": output_path}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def color_adjust(self, input_path: str, output_path: str, brightness: float = 0,
                     contrast: float = 0, saturation: float = 0) -> dict:
        if not self.magick:
            return {"status": "error", "error": "未找到 ImageMagick"}
        cmd = [self.magick, input_path]
        if brightness != 0:
            cmd.extend(["-brightness-contrast", f"{brightness}x{contrast}"])
        if saturation != 0:
            cmd.extend(["-modulate", f"100,{100 + saturation},100"])
        cmd.append(output_path)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return {"status": "success" if result.returncode == 0 else "error", "output": output_path}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def extract_frames(self, video_path: str, output_dir: str, fps: float = 1.0) -> dict:
        if not self.ffmpeg:
            return {"status": "error", "error": "未找到 FFmpeg"}
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cmd = [self.ffmpeg, "-i", video_path, "-vf", f"fps={fps}", f"{output_dir}/frame_%04d.png"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            return {"status": "success" if result.returncode == 0 else "error", "output_dir": output_dir}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def create_gif(self, input_pattern: str, output_path: str, fps: int = 10) -> dict:
        if not self.ffmpeg:
            return {"status": "error", "error": "未找到 FFmpeg"}
        cmd = [self.ffmpeg, "-framerate", str(fps), "-i", input_pattern, output_path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return {"status": "success" if result.returncode == 0 else "error", "output": output_path}
        except Exception as e:
            return {"status": "error", "error": str(e)}


class DesignAdapter:
    """
    设计软件适配器 — 莫奈/梵高灵魂专用

    三层控制：
    1. API 直连：Figma REST API 精确控制设计文件
    2. CLI 工具链：ImageMagick/FFmpeg 批处理
    3. GUI 感知：Mano-P 操作设计软件界面

    使用：
        adapter = DesignAdapter()
        adapter.figma_get_file("abc123")
        adapter.image_resize("input.png", "output.png", 800, 600)
    """

    def __init__(self, primary: DesignSoftware = DesignSoftware.FIGMA):
        self._primary = primary
        self._figma: Optional[FigmaClient] = None
        self._image_proc = ImageProcessor()
        self._task_counter = 0

    def status(self) -> dict:
        figma_available = False
        if self._figma is None:
            self._figma = FigmaClient()
        figma_available = self._figma.is_available()

        return {
            "adapter": "DesignAdapter",
            "primary_software": self._primary.value,
            "figma_available": figma_available,
            "imagemagick_available": self._image_proc.magick is not None,
            "ffmpeg_available": self._image_proc.ffmpeg is not None,
            "tools": [
                "design_figma_get_file", "design_figma_get_node",
                "design_figma_get_image", "design_figma_get_styles",
                "design_figma_get_components",
                "design_image_resize", "design_image_convert",
                "design_color_adjust", "design_extract_frames",
                "design_create_gif", "design_screenshot",
            ],
        }

    @property
    def figma(self) -> FigmaClient:
        if self._figma is None:
            self._figma = FigmaClient()
        return self._figma

    @property
    def image(self) -> ImageProcessor:
        return self._image_proc

    def figma_get_file(self, file_key: str) -> dict:
        return self.figma.get_file(file_key)

    def figma_get_node(self, file_key: str, node_id: str) -> dict:
        return self.figma.get_node(file_key, node_id)

    def figma_get_image(self, file_key: str, node_id: str, format: str = "png", scale: float = 2.0) -> dict:
        return self.figma.get_image(file_key, node_id, format, scale)

    def figma_get_styles(self, file_key: str) -> dict:
        return self.figma.get_styles(file_key)

    def figma_get_components(self, file_key: str) -> dict:
        return self.figma.get_components(file_key)

    def image_resize(self, input_path: str, output_path: str, width: int, height: int) -> dict:
        return self._image_proc.resize(input_path, output_path, width, height)

    def image_convert(self, input_path: str, output_path: str) -> dict:
        return self._image_proc.convert_format(input_path, output_path)

    def color_adjust(self, input_path: str, output_path: str, brightness: float = 0,
                     contrast: float = 0, saturation: float = 0) -> dict:
        return self._image_proc.color_adjust(input_path, output_path, brightness, contrast, saturation)

    def extract_frames(self, video_path: str, output_dir: str, fps: float = 1.0) -> dict:
        return self._image_proc.extract_frames(video_path, output_dir, fps)

    def create_gif(self, input_pattern: str, output_path: str, fps: int = 10) -> dict:
        return self._image_proc.create_gif(input_pattern, output_path, fps)


DESIGN_SKILL_DEFINITION = {
    "skill_id": "design_tools",
    "name": "视觉设计工具链",
    "description": "莫奈/梵高灵魂专用：Figma API + ImageMagick + FFmpeg + GUI感知",
    "soul_mapping": ["monet", "vangogh"],
    "tools": [
        "design_figma_get_file", "design_figma_get_node",
        "design_figma_get_image", "design_figma_get_styles",
        "design_figma_get_components",
        "design_image_resize", "design_image_convert",
        "design_color_adjust", "design_extract_frames",
        "design_create_gif", "design_screenshot",
    ],
    "connectors": [
        {"name": "figma_api", "type": "rest_api", "base_url": "https://api.figma.com/v1"},
        {"name": "imagemagick_cli", "type": "cli", "command": "magick"},
        {"name": "ffmpeg_cli", "type": "cli", "command": "ffmpeg"},
        {"name": "mano_p_gui", "type": "gui_perception", "fallback": True},
    ],
}

_adapter_instance: Optional[DesignAdapter] = None


def get_adapter() -> DesignAdapter:
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = DesignAdapter()
    return _adapter_instance
