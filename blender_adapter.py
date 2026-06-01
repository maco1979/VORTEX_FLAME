"""
Blender 适配器 — 达芬奇灵魂的3D建模/渲染工具链
=================================================
通过三条路径控制 Blender 及其他 3D 软件：

路径1: Python API 直连（推荐，精确控制）
  - Blender 内置 Python API (bpy)
  - 通过后台模式执行 Python 脚本
  - 支持：建模、材质、灯光、动画、渲染、导出

路径2: CLI 封装（批处理/渲染农场）
  - blender --background --python script.py
  - 批量渲染、格式转换、场景批处理

路径3: GUI 感知（兜底，交互式操作）
  - Mano-P 视觉模型直接操作 Blender 界面
  - 雕刻模式、权重绘制等需要视觉反馈的操作

支持 3D 软件列表：
  - Blender（Python API 直连）
  - Maya（MEL/Python + maya.cmds）
  - 3ds Max（MAXScript + Python）
  - ZBrush（GUI 感知）
  - Cinema 4D（Python API）

集成点：
  - soul_orchestrator: davinci 灵魂注册 blender_* 工具
  - harness_runtime: Blender 进程 + 端口白名单
  - guardian: Blender 进程监控
  - cli_anything: blender_render, blender_export 命令
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


class Software3D(Enum):
    BLENDER = "blender"
    MAYA = "maya"
    MAX_3DS = "3dsmax"
    ZBRUSH = "zbrush"
    CINEMA4D = "cinema4d"
    GENERIC = "generic"


class ObjectType(Enum):
    MESH = "mesh"
    CURVE = "curve"
    SURFACE = "surface"
    META = "meta"
    FONT = "font"
    ARMATURE = "armature"
    LATTICE = "lattice"
    EMPTY = "empty"
    LIGHT = "light"
    CAMERA = "camera"
    SPEAKER = "speaker"


class RenderEngine(Enum):
    EEVEE = "BLENDER_EEVEE_NEXT"
    CYCLES = "CYCLES"
    WORKBENCH = "BLENDER_WORKBENCH"


class ImageFormat(Enum):
    PNG = "PNG"
    JPEG = "JPEG"
    EXR = "OPEN_EXR"
    TIFF = "TIFF"
    HDR = "HDR"


BLENDER_CONFIG = {
    "executable_paths": {
        "windows": [
            r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe",
        ],
        "darwin": [
            "/Applications/Blender.app/Contents/MacOS/Blender",
        ],
        "linux": [
            "/usr/bin/blender",
            "/usr/local/bin/blender",
            "/snap/bin/blender",
        ],
    },
    "default_render_output": str(Path("D:/renders/blender")),
    "default_project_dir": str(Path("D:/Projects/Blender")),
    "script_temp_dir": str(Path("D:/VORTEX_FLAME/tmp/blender_scripts")),
    "render_timeout_seconds": 600,
    "python_timeout_seconds": 120,
    "default_samples": 128,
    "default_resolution": (1920, 1080),
}


@dataclass
class SceneObject:
    name: str
    object_type: ObjectType
    location: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    material_name: str = ""
    parent_name: str = ""


@dataclass
class MaterialInfo:
    name: str
    base_color: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    metallic: float = 0.0
    roughness: float = 0.5
    emission_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    emission_strength: float = 0.0


@dataclass
class LightInfo:
    name: str
    light_type: str = "POINT"
    energy: float = 1000.0
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    location: Tuple[float, float, float] = (0.0, 0.0, 5.0)


@dataclass
class RenderSettings:
    engine: RenderEngine = RenderEngine.CYCLES
    resolution_x: int = 1920
    resolution_y: int = 1080
    samples: int = 128
    output_format: ImageFormat = ImageFormat.PNG
    output_path: str = ""
    frame_start: int = 1
    frame_end: int = 1
    denoising: bool = True


@dataclass
class ModelingTask:
    task_id: str
    description: str
    soul: str
    software: Software3D
    actions: List[Dict[str, Any]] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "pending"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


def find_blender_executable() -> Optional[str]:
    import platform
    os_name = platform.system().lower()
    paths = BLENDER_CONFIG["executable_paths"].get(os_name, [])

    for path in paths:
        if Path(path).exists():
            return path

    import shutil
    found = shutil.which("blender")
    if found:
        return found

    return None


def _generate_python_script(commands: List[str]) -> str:
    header = """import bpy
import mathutils
import json
import sys

results = []

def _result(action, status, **kwargs):
    results.append({"action": action, "status": status, **kwargs})

"""
    body = "\n".join(commands)
    footer = """

print("___VORTEX_RESULT___")
print(json.dumps(results, ensure_ascii=False))
"""
    return header + body + footer


class BlenderAdapter:
    """
    Blender 适配器 — 达芬奇灵魂专用

    三层控制：
    1. Python API：通过 bpy 精确控制建模、材质、动画、渲染
    2. CLI 封装：blender --background --python 批处理
    3. GUI 感知：Mano-P 操作 Blender 界面（雕刻等）

    使用：
        adapter = BlenderAdapter()
        adapter.create_primitive("CUBE", name="方块")
        adapter.set_material("方块", base_color=(1,0,0,1))
        adapter.render(output_path="D:/renders/test.png")
    """

    def __init__(self, software: Software3D = Software3D.BLENDER):
        self._software = software
        self._blender_path: Optional[str] = None
        self._task_counter = 0
        self._script_dir = Path(BLENDER_CONFIG["script_temp_dir"])
        self._script_dir.mkdir(parents=True, exist_ok=True)

    @property
    def software(self) -> Software3D:
        return self._software

    def status(self) -> dict:
        self._blender_path = self._blender_path or find_blender_executable()
        return {
            "adapter": "BlenderAdapter",
            "software": self._software.value,
            "blender_found": self._blender_path is not None,
            "blender_path": self._blender_path,
            "script_dir": str(self._script_dir),
            "tools": [
                "blender_create_primitive", "blender_import_model",
                "blender_set_location", "blender_set_rotation", "blender_set_scale",
                "blender_create_material", "blender_assign_material",
                "blender_add_light", "blender_add_camera",
                "blender_set_render_settings", "blender_render",
                "blender_export", "blender_run_script",
                "blender_get_scene_info", "blender_screenshot",
            ],
        }

    def _run_blender_script(self, script: str, background: bool = True) -> dict:
        self._blender_path = self._blender_path or find_blender_executable()
        if not self._blender_path:
            return {"status": "error", "error": "未找到 Blender 可执行文件"}

        script_path = self._script_dir / f"vf_script_{int(time.time())}_{self._task_counter}.py"
        script_path.write_text(script, encoding="utf-8")

        cmd = [self._blender_path]
        if background:
            cmd.append("--background")
        cmd.extend(["--python", str(script_path)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=BLENDER_CONFIG["python_timeout_seconds"],
                encoding="utf-8",
                errors="replace",
            )

            output = result.stdout + result.stderr
            vortex_results = []
            if "___VORTEX_RESULT___" in output:
                json_str = output.split("___VORTEX_RESULT___")[-1].strip().split("\n")[0]
                try:
                    vortex_results = json.loads(json_str)
                except json.JSONDecodeError:
                    pass

            return {
                "status": "success" if result.returncode == 0 else "error",
                "return_code": result.returncode,
                "results": vortex_results,
                "output_preview": output[-500:] if len(output) > 500 else output,
            }
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": f"Blender 执行超时 ({BLENDER_CONFIG['python_timeout_seconds']}s)"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def create_primitive(self, primitive_type: str, name: str = "", location: Tuple[float, float, float] = (0, 0, 0)) -> dict:
        type_map = {
            "CUBE": "bpy.ops.mesh.primitive_cube_add()",
            "SPHERE": "bpy.ops.mesh.primitive_uv_sphere_add()",
            "CYLINDER": "bpy.ops.mesh.primitive_cylinder_add()",
            "CONE": "bpy.ops.mesh.primitive_cone_add()",
            "TORUS": "bpy.ops.mesh.primitive_torus_add()",
            "PLANE": "bpy.ops.mesh.primitive_plane_add()",
            "MONKEY": "bpy.ops.mesh.primitive_monkey_add()",
        }
        op = type_map.get(primitive_type.upper(), type_map["CUBE"])
        obj_name = name or primitive_type.lower()

        commands = [
            f"{op}",
            f"obj = bpy.context.active_object",
            f"obj.name = '{obj_name}'",
            f"obj.location = {list(location)}",
            f"_result('create_primitive', 'ok', name=obj_name, type='{primitive_type}')",
        ]
        return self._run_blender_script(_generate_python_script(commands))

    def import_model(self, filepath: str, name: str = "") -> dict:
        ext = Path(filepath).suffix.lower()
        import_map = {
            ".obj": f"bpy.ops.wm.obj_import(filepath=r'{filepath}')",
            ".fbx": f"bpy.ops.wm.fbx_import(filepath=r'{filepath}')",
            ".gltf": f"bpy.ops.wm.gltf_import(filepath=r'{filepath}')",
            ".glb": f"bpy.ops.wm.gltf_import(filepath=r'{filepath}')",
            ".stl": f"bpy.ops.wm.stl_import(filepath=r'{filepath}')",
            ".dae": f"bpy.ops.wm.collada_import(filepath=r'{filepath}')",
            ".abc": f"bpy.ops.wm.alembic_import(filepath=r'{filepath}')",
            ".usd": f"bpy.ops.wm.usd_import(filepath=r'{filepath}')",
            ".usda": f"bpy.ops.wm.usd_import(filepath=r'{filepath}')",
            ".usdc": f"bpy.ops.wm.usd_import(filepath=r'{filepath}')",
        }
        import_cmd = import_map.get(ext)
        if not import_cmd:
            return {"status": "error", "error": f"不支持的文件格式: {ext}"}

        commands = [
            import_cmd,
            f"obj = bpy.context.selected_objects[0] if bpy.context.selected_objects else None",
            f"if obj and '{name}': obj.name = '{name}'",
            f"_result('import_model', 'ok', filepath=r'{filepath}', name=obj.name if obj else '')",
        ]
        return self._run_blender_script(_generate_python_script(commands))

    def set_location(self, object_name: str, location: Tuple[float, float, float]) -> dict:
        commands = [
            f"obj = bpy.data.objects.get('{object_name}')",
            f"if obj: obj.location = {list(location)}; _result('set_location', 'ok', name='{object_name}')",
            f"else: _result('set_location', 'error', error='对象不存在')",
        ]
        return self._run_blender_script(_generate_python_script(commands))

    def set_rotation(self, object_name: str, rotation: Tuple[float, float, float]) -> dict:
        import math
        commands = [
            f"import math",
            f"obj = bpy.data.objects.get('{object_name}')",
            f"if obj: obj.rotation_euler = ({rotation[0]}, {rotation[1]}, {rotation[2]}); _result('set_rotation', 'ok', name='{object_name}')",
            f"else: _result('set_rotation', 'error', error='对象不存在')",
        ]
        return self._run_blender_script(_generate_python_script(commands))

    def set_scale(self, object_name: str, scale: Tuple[float, float, float]) -> dict:
        commands = [
            f"obj = bpy.data.objects.get('{object_name}')",
            f"if obj: obj.scale = {list(scale)}; _result('set_scale', 'ok', name='{object_name}')",
            f"else: _result('set_scale', 'error', error='对象不存在')",
        ]
        return self._run_blender_script(_generate_python_script(commands))

    def create_material(self, name: str, base_color: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0),
                        metallic: float = 0.0, roughness: float = 0.5) -> dict:
        commands = [
            f"mat = bpy.data.materials.new(name='{name}')",
            f"mat.use_nodes = True",
            f"bsdf = mat.node_tree.nodes.get('Principled BSDF')",
            f"if bsdf:",
            f"    bsdf.inputs['Base Color'].default_value = {list(base_color)}",
            f"    bsdf.inputs['Metallic'].default_value = {metallic}",
            f"    bsdf.inputs['Roughness'].default_value = {roughness}",
            f"_result('create_material', 'ok', name='{name}')",
        ]
        return self._run_blender_script(_generate_python_script(commands))

    def assign_material(self, object_name: str, material_name: str) -> dict:
        commands = [
            f"obj = bpy.data.objects.get('{object_name}')",
            f"mat = bpy.data.materials.get('{material_name}')",
            f"if obj and mat:",
            f"    if obj.data.materials:",
            f"        obj.data.materials[0] = mat",
            f"    else:",
            f"        obj.data.materials.append(mat)",
            f"    _result('assign_material', 'ok', object='{object_name}', material='{material_name}')",
            f"else:",
            f"    _result('assign_material', 'error', error='对象或材质不存在')",
        ]
        return self._run_blender_script(_generate_python_script(commands))

    def add_light(self, name: str = "", light_type: str = "POINT", energy: float = 1000.0,
                  color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                  location: Tuple[float, float, float] = (0, 0, 5)) -> dict:
        lt_map = {"POINT": "POINT", "SUN": "SUN", "SPOT": "SPOT", "AREA": "AREA"}
        lt = lt_map.get(light_type.upper(), "POINT")
        obj_name = name or f"Light_{lt}"

        commands = [
            f"light_data = bpy.data.lights.new(name='{obj_name}', type='{lt}')",
            f"light_data.energy = {energy}",
            f"light_data.color = {list(color)}",
            f"light_obj = bpy.data.objects.new(name='{obj_name}', object_data=light_data)",
            f"bpy.context.collection.objects.link(light_obj)",
            f"light_obj.location = {list(location)}",
            f"_result('add_light', 'ok', name='{obj_name}', type='{lt}')",
        ]
        return self._run_blender_script(_generate_python_script(commands))

    def add_camera(self, name: str = "Camera_VF", location: Tuple[float, float, float] = (7, -7, 5),
                   look_at: Tuple[float, float, float] = (0, 0, 0)) -> dict:
        commands = [
            f"cam_data = bpy.data.cameras.new(name='{name}')",
            f"cam_obj = bpy.data.objects.new(name='{name}', object_data=cam_data)",
            f"bpy.context.collection.objects.link(cam_obj)",
            f"cam_obj.location = {list(location)}",
            f"direction = mathutils.Vector({list(look_at)}) - cam_obj.location",
            f"rot_quat = direction.to_track_quat('-Z', 'Y')",
            f"cam_obj.rotation_euler = rot_quat.to_euler()",
            f"bpy.context.scene.camera = cam_obj",
            f"_result('add_camera', 'ok', name='{name}')",
        ]
        return self._run_blender_script(_generate_python_script(commands))

    def set_render_settings(self, settings: RenderSettings) -> dict:
        output = settings.output_path or str(
            Path(BLENDER_CONFIG["default_render_output"]) / f"render_{int(time.time())}.png"
        )
        commands = [
            f"scene = bpy.context.scene",
            f"scene.render.engine = '{settings.engine.value}'",
            f"scene.render.resolution_x = {settings.resolution_x}",
            f"scene.render.resolution_y = {settings.resolution_y}",
            f"scene.render.filepath = r'{output}'",
            f"scene.render.image_settings.file_format = '{settings.output_format.value}'",
            f"if '{settings.engine.value}' == 'CYCLES':",
            f"    scene.cycles.samples = {settings.samples}",
            f"    scene.cycles.use_denoising = {settings.denoising}",
            f"scene.frame_start = {settings.frame_start}",
            f"scene.frame_end = {settings.frame_end}",
            f"_result('set_render_settings', 'ok', engine='{settings.engine.value}', output=r'{output}')",
        ]
        return self._run_blender_script(_generate_python_script(commands))

    def render(self, output_path: str = "", engine: str = "CYCLES", samples: int = 128) -> dict:
        output = output_path or str(
            Path(BLENDER_CONFIG["default_render_output"]) / f"render_{int(time.time())}.png"
        )
        commands = [
            f"scene = bpy.context.scene",
            f"scene.render.engine = '{engine}'",
            f"scene.render.filepath = r'{output}'",
            f"if '{engine}' == 'CYCLES': scene.cycles.samples = {samples}",
            f"bpy.ops.render.render(write_still=True)",
            f"_result('render', 'ok', output=r'{output}')",
        ]
        return self._run_blender_script(_generate_python_script(commands))

    def export(self, filepath: str, object_name: str = "", format: str = "FBX") -> dict:
        ext_map = {
            "FBX": (".fbx", f"bpy.ops.wm.fbx_export(filepath=r'{{path}}')"),
            "OBJ": (".obj", f"bpy.ops.wm.obj_export(filepath=r'{{path}}')"),
            "GLTF": (".gltf", f"bpy.ops.wm.gltf_export(filepath=r'{{path}}')"),
            "GLB": (".glb", f"bpy.ops.wm.gltf_export(filepath=r'{{path}}')"),
            "STL": (".stl", f"bpy.ops.wm.stl_export(filepath=r'{{path}}')"),
            "USD": (".usdc", f"bpy.ops.wm.usd_export(filepath=r'{{path}}')"),
            "ABC": (".abc", f"bpy.ops.wm.alembic_export(filepath=r'{{path}}')"),
        }
        fmt = ext_map.get(format.upper())
        if not fmt:
            return {"status": "error", "error": f"不支持的导出格式: {format}"}

        if object_name:
            commands = [
                f"for obj in bpy.data.objects: obj.select_set(False)",
                f"obj = bpy.data.objects.get('{object_name}')",
                f"if obj: obj.select_set(True); bpy.context.view_layer.objects.active = obj",
            ]
        else:
            commands = [f"bpy.ops.object.select_all(action='SELECT')"]

        export_cmd = fmt[1].format(path=filepath)
        commands.append(export_cmd)
        commands.append(f"_result('export', 'ok', filepath=r'{filepath}', format='{format}')")
        return self._run_blender_script(_generate_python_script(commands))

    def run_script(self, script: str) -> dict:
        return self._run_blender_script(script)

    def get_scene_info(self) -> dict:
        commands = [
            f"objects = [{{'name': o.name, 'type': o.type}} for o in bpy.data.objects]",
            f"materials = [m.name for m in bpy.data.materials]",
            f"scene = bpy.context.scene",
            f"_result('scene_info', 'ok', objects=objects, materials=materials, "
            f"engine=scene.render.engine, "
            f"resolution=[scene.render.resolution_x, scene.render.resolution_y])",
        ]
        return self._run_blender_script(_generate_python_script(commands))


BLENDER_SKILL_DEFINITION = {
    "skill_id": "blender_3d",
    "name": "Blender 3D 建模渲染",
    "description": "达芬奇灵魂专用：Blender 全功能控制，含建模、材质、灯光、动画、渲染、导出",
    "soul_mapping": ["davinci"],
    "tools": [
        "blender_create_primitive", "blender_import_model",
        "blender_set_location", "blender_set_rotation", "blender_set_scale",
        "blender_create_material", "blender_assign_material",
        "blender_add_light", "blender_add_camera",
        "blender_set_render_settings", "blender_render",
        "blender_export", "blender_run_script",
        "blender_get_scene_info", "blender_screenshot",
    ],
    "connectors": [
        {"name": "blender_cli", "type": "cli", "command": "blender --background --python"},
        {"name": "blender_python", "type": "python_api", "module": "bpy"},
        {"name": "mano_p_gui", "type": "gui_perception", "fallback": True},
    ],
}

_adapter_instance: Optional[BlenderAdapter] = None


def get_adapter() -> BlenderAdapter:
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = BlenderAdapter()
    return _adapter_instance
