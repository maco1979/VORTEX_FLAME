"""
AI 生成工具适配器 — 对齐最流行的 AI 生成工具
================================================
将 ComfyUI、Suno、Kling、Midjourney、DALL-E、Runway、Pika 等
主流 AI 生成工具接入 VORTEX FLAME 灵魂矩阵。

核心定位：
  - 莫奈/梵高 → 图像生成（ComfyUI / DALL-E / Midjourney）
  - 贝多芬   → 音乐生成（Suno / Udio / MIDIUtil）
  - 达芬奇   → 视频/3D生成（Kling / Runway / Pika / Blender）
  - 爱因斯坦 → 数据可视化（matplotlib / Plotly）

架构：
  L1: API 直连 — ComfyUI API / Suno API / OpenAI API / Kling API
  L2: CLI 封装 — comfyui-cli / ffmpeg / midiutil
  L3: GUI 感知 — Mano-P 操作 Web 界面（兜底）

能力边界标注：
  - ✅ 可用：ComfyUI本地API、DALL-E API、MIDIUtil本地生成
  - 🔄 开发中：Suno API、Kling API、Runway API
  - ⏳ 计划中：Midjourney（需Discord）、Pika、Udio

2026年最流行AI生成工具清单：
  图像：ComfyUI(Stable Diffusion) / DALL-E 3 / Midjourney V6 / FLUX
  音乐：Suno V4 / Udio / MusicGen
  视频：Kling 2.0 / Runway Gen-3 / Pika / Sora / Hailuo
  3D：  Tripo3D / Meshy / Luma Genie / CSM
  语音：ElevenLabs / Bark / ChatTTS
  代码：Cursor / Copilot / Devin（已由塞尚覆盖）
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AIGenCategory(Enum):
    IMAGE = "image"
    MUSIC = "music"
    VIDEO = "video"
    THREE_D = "3d"
    VOICE = "voice"
    CODE = "code"


class AIGenStatus(Enum):
    AVAILABLE = "available"
    API_READY = "api_ready"
    CLI_READY = "cli_ready"
    DEVELOPING = "developing"
    PLANNED = "planned"


AIGEN_TOOL_REGISTRY = {
    "comfyui": {
        "name": "ComfyUI",
        "category": AIGenCategory.IMAGE,
        "description": "Stable Diffusion / FLUX 本地图像生成工作流引擎",
        "status": AIGenStatus.API_READY,
        "soul_mapping": ["monet", "vangogh", "davinci"],
        "api_url": "http://127.0.0.1:8188",
        "capabilities": ["txt2img", "img2img", "inpainting", "controlnet", "lora", "upscale"],
        "boundary": {"可用": ["txt2img", "img2img", "ControlNet", "LoRA"], "开发中": ["Inpainting"], "计划中": ["视频生成"]},
    },
    "dalle": {
        "name": "DALL-E 3",
        "category": AIGenCategory.IMAGE,
        "description": "OpenAI DALL-E 3 图像生成API",
        "status": AIGenStatus.API_READY,
        "soul_mapping": ["monet", "vangogh"],
        "api_url": "https://api.openai.com/v1/images/generations",
        "capabilities": ["txt2img", "variations", "edit"],
        "boundary": {"可用": ["txt2img"], "开发中": ["variations"], "计划中": []},
    },
    "flux": {
        "name": "FLUX",
        "category": AIGenCategory.IMAGE,
        "description": "Black Forest Labs FLUX 开源图像生成模型（通过ComfyUI调用）",
        "status": AIGenStatus.API_READY,
        "soul_mapping": ["monet", "vangogh"],
        "api_url": "comfyui://flux",
        "capabilities": ["txt2img", "img2img"],
        "boundary": {"可用": ["txt2img"], "开发中": [], "计划中": []},
    },
    "midjourney": {
        "name": "Midjourney V6",
        "category": AIGenCategory.IMAGE,
        "description": "Midjourney 图像生成（需Discord交互）",
        "status": AIGenStatus.PLANNED,
        "soul_mapping": ["monet", "vangogh"],
        "api_url": "",
        "capabilities": ["txt2img", "upscale", "variation"],
        "boundary": {"可用": [], "开发中": [], "计划中": ["Discord Bot集成"]},
    },
    "suno": {
        "name": "Suno V4",
        "category": AIGenCategory.MUSIC,
        "description": "Suno AI 音乐生成平台",
        "status": AIGenStatus.DEVELOPING,
        "soul_mapping": ["beethoven"],
        "api_url": "https://studio-api.suno.ai",
        "capabilities": ["txt2music", "extend", "remix", "lyrics"],
        "boundary": {"可用": [], "开发中": ["API对接"], "计划中": ["歌词生成", "风格迁移"]},
    },
    "udio": {
        "name": "Udio",
        "category": AIGenCategory.MUSIC,
        "description": "Udio AI 音乐生成平台",
        "status": AIGenStatus.PLANNED,
        "soul_mapping": ["beethoven"],
        "api_url": "",
        "capabilities": ["txt2music", "extend"],
        "boundary": {"可用": [], "开发中": [], "计划中": ["API对接"]},
    },
    "musicgen": {
        "name": "MusicGen",
        "category": AIGenCategory.MUSIC,
        "description": "Meta MusicGen 本地音乐生成（通过Audiocraft）",
        "status": AIGenStatus.CLI_READY,
        "soul_mapping": ["beethoven"],
        "api_url": "",
        "capabilities": ["txt2music", "melody_conditioning"],
        "boundary": {"可用": ["本地生成"], "开发中": [], "计划中": []},
    },
    "kling": {
        "name": "Kling 2.0",
        "category": AIGenCategory.VIDEO,
        "description": "快手可灵AI视频生成平台",
        "status": AIGenStatus.DEVELOPING,
        "soul_mapping": ["davinci", "monet", "vangogh"],
        "api_url": "https://api.klingai.com",
        "capabilities": ["txt2video", "img2video", "lip_sync", "extend"],
        "boundary": {"可用": [], "开发中": ["API对接"], "计划中": ["txt2video", "img2video"]},
    },
    "runway": {
        "name": "Runway Gen-3",
        "category": AIGenCategory.VIDEO,
        "description": "Runway ML 视频生成平台",
        "status": AIGenStatus.PLANNED,
        "soul_mapping": ["davinci", "monet"],
        "api_url": "",
        "capabilities": ["txt2video", "img2video", "motion_brush"],
        "boundary": {"可用": [], "开发中": [], "计划中": ["API对接"]},
    },
    "pika": {
        "name": "Pika",
        "category": AIGenCategory.VIDEO,
        "description": "Pika Labs 视频生成平台",
        "status": AIGenStatus.PLANNED,
        "soul_mapping": ["davinci", "monet"],
        "api_url": "",
        "capabilities": ["txt2video", "img2video"],
        "boundary": {"可用": [], "开发中": [], "计划中": ["API对接"]},
    },
    "sora": {
        "name": "Sora",
        "category": AIGenCategory.VIDEO,
        "description": "OpenAI Sora 视频生成",
        "status": AIGenStatus.PLANNED,
        "soul_mapping": ["davinci", "monet"],
        "api_url": "https://api.openai.com/v1/video/generations",
        "capabilities": ["txt2video"],
        "boundary": {"可用": [], "开发中": [], "计划中": ["API对接"]},
    },
    "hailuo": {
        "name": "Hailuo/MiniMax",
        "category": AIGenCategory.VIDEO,
        "description": "MiniMax 海螺AI视频生成",
        "status": AIGenStatus.DEVELOPING,
        "soul_mapping": ["davinci", "monet"],
        "api_url": "https://api.minimax.chat",
        "capabilities": ["txt2video", "img2video"],
        "boundary": {"可用": [], "开发中": ["API对接"], "计划中": []},
    },
    "tripo3d": {
        "name": "Tripo3D",
        "category": AIGenCategory.THREE_D,
        "description": "Tripo3D 文本/图像转3D模型",
        "status": AIGenStatus.DEVELOPING,
        "soul_mapping": ["davinci"],
        "api_url": "https://api.tripo3d.ai",
        "capabilities": ["txt2mesh", "img2mesh", "texture", "rig"],
        "boundary": {"可用": [], "开发中": ["API对接"], "计划中": ["Blender桥接"]},
    },
    "meshy": {
        "name": "Meshy",
        "category": AIGenCategory.THREE_D,
        "description": "Meshy AI 3D模型生成",
        "status": AIGenStatus.PLANNED,
        "soul_mapping": ["davinci"],
        "api_url": "",
        "capabilities": ["txt2mesh", "img2mesh", "texture"],
        "boundary": {"可用": [], "开发中": [], "计划中": ["API对接"]},
    },
    "elevenlabs": {
        "name": "ElevenLabs",
        "category": AIGenCategory.VOICE,
        "description": "ElevenLabs AI语音合成与克隆",
        "status": AIGenStatus.DEVELOPING,
        "soul_mapping": ["beethoven", "guizhu"],
        "api_url": "https://api.elevenlabs.io",
        "capabilities": ["tts", "voice_clone", "sfx"],
        "boundary": {"可用": [], "开发中": ["TTS API"], "计划中": ["语音克隆"]},
    },
    "chattts": {
        "name": "ChatTTS",
        "category": AIGenCategory.VOICE,
        "description": "ChatTTS 本地语音合成",
        "status": AIGenStatus.CLI_READY,
        "soul_mapping": ["beethoven", "guizhu"],
        "api_url": "",
        "capabilities": ["tts"],
        "boundary": {"可用": ["本地TTS"], "开发中": [], "计划中": []},
    },
}


class AIGenAdapter:
    """
    AI 生成工具适配器 — 统一调用最流行的 AI 生成工具

    核心能力：
    1. ComfyUI 本地 API — 图像生成工作流
    2. DALL-E API — OpenAI 图像生成
    3. MIDIUtil — 本地 MIDI 音乐生成
    4. Kling/Suno/Runway — 云端视频/音乐生成（开发中）

    使用：
        adapter = AIGenAdapter()
        adapter.txt2img("a beautiful sunset", soul="monet")
        adapter.txt2music("jazz piano solo", soul="beethoven")
        adapter.txt2video("ocean waves", soul="davinci")
    """

    def __init__(self):
        self._comfyui_url = "http://127.0.0.1:8188"
        self._openai_key = os.environ.get("OPENAI_API_KEY", "")
        self._kling_key = os.environ.get("KLING_API_KEY", "")
        self._suno_key = os.environ.get("SUNO_API_KEY", "")
        self._elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY", "")

    def status(self) -> dict:
        comfyui_ok = self._check_comfyui()
        dalle_ok = bool(self._openai_key)
        kling_ok = bool(self._kling_key)
        suno_ok = bool(self._suno_key)
        elevenlabs_ok = bool(self._elevenlabs_key)

        return {
            "adapter": "AIGenAdapter",
            "comfyui": {"connected": comfyui_ok, "url": self._comfyui_url},
            "dalle": {"available": dalle_ok},
            "kling": {"available": kling_ok},
            "suno": {"available": suno_ok},
            "elevenlabs": {"available": elevenlabs_ok},
            "tools_summary": {
                "可用": ["ComfyUI本地", "MIDIUtil", "ChatTTS"] + (["DALL-E"] if dalle_ok else []),
                "开发中": ["Suno", "Kling", "Hailuo", "Tripo3D", "ElevenLabs"],
                "计划中": ["Midjourney", "Runway", "Pika", "Sora", "Meshy", "Udio"],
            },
        }

    def _check_comfyui(self) -> bool:
        try:
            import urllib.request
            req = urllib.request.Request(f"{self._comfyui_url}/system_stats", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def list_tools(self, soul: str = None, category: str = None) -> List[dict]:
        tools = list(AIGEN_TOOL_REGISTRY.values())
        if soul:
            tools = [t for t in tools if soul in t["soul_mapping"]]
        if category:
            tools = [t for t in tools if t["category"].value == category]
        return [
            {
                "id": tid,
                "name": t["name"],
                "category": t["category"].value,
                "status": t["status"].value,
                "capabilities": t["capabilities"],
                "boundary": t["boundary"],
            }
            for tid, t in ((k, v) for k, v in AIGEN_TOOL_REGISTRY.items() if v in tools)
        ]

    def get_tools_for_soul(self, soul: str) -> List[dict]:
        result = []
        for tid, t in AIGEN_TOOL_REGISTRY.items():
            if soul in t["soul_mapping"]:
                result.append({
                    "id": tid,
                    "name": t["name"],
                    "category": t["category"].value,
                    "status": t["status"].value,
                    "capabilities": t["capabilities"],
                    "boundary": t["boundary"],
                })
        return result

    def txt2img(self, prompt: str, soul: str = "monet",
                negative_prompt: str = "", width: int = 1024, height: int = 1024,
                steps: int = 20, cfg_scale: float = 7.0,
                seed: int = -1, model: str = "flux1-dev") -> dict:
        if self._check_comfyui():
            return self._comfyui_txt2img(prompt, negative_prompt, width, height, steps, cfg_scale, seed, model)

        if self._openai_key:
            return self._dalle_txt2img(prompt, width, height)

        return {
            "status": "concept_only",
            "message": "ComfyUI未启动且无DALL-E API Key，仅生成概念描述",
            "prompt": prompt,
            "soul": soul,
            "concept": self._generate_concept(prompt, "image"),
            "boundary_note": "图像生成需要ComfyUI本地运行或DALL-E API Key",
        }

    def _comfyui_txt2img(self, prompt: str, negative_prompt: str,
                         width: int, height: int, steps: int,
                         cfg_scale: float, seed: int, model: str) -> dict:
        workflow = {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed if seed >= 0 else int(time.time()) % 2**32,
                    "steps": steps,
                    "cfg": cfg_scale,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0],
                },
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": f"{model}.safetensors"},
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": width, "height": height, "batch_size": 1},
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["4", 1]},
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": negative_prompt or "low quality, blurry, distorted", "clip": ["4", 1]},
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "vortex_flame", "images": ["8", 0]},
            },
        }

        try:
            import urllib.request
            data = json.dumps({"prompt": workflow}, ensure_ascii=False).encode()
            req = urllib.request.Request(
                f"{self._comfyui_url}/prompt",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode())
                return {
                    "status": "submitted",
                    "engine": "ComfyUI",
                    "prompt_id": result.get("prompt_id", ""),
                    "prompt": prompt,
                    "model": model,
                    "size": f"{width}x{height}",
                }
        except Exception as e:
            return {"status": "error", "engine": "ComfyUI", "error": str(e)}

    def _dalle_txt2img(self, prompt: str, width: int = 1024, height: int = 1024) -> dict:
        try:
            import urllib.request
            data = json.dumps({
                "model": "dall-e-3",
                "prompt": prompt,
                "n": 1,
                "size": f"{width}x{height}" if f"{width}x{height}" in ["1024x1024", "1792x1024", "1024x1792"] else "1024x1024",
                "quality": "standard",
            }).encode()
            req = urllib.request.Request(
                "https://api.openai.com/v1/images/generations",
                data=data,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._openai_key}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode())
                images = result.get("data", [])
                return {
                    "status": "ok",
                    "engine": "DALL-E 3",
                    "prompt": prompt,
                    "images": [{"url": img.get("url", ""), "revised_prompt": img.get("revised_prompt", "")} for img in images],
                }
        except Exception as e:
            return {"status": "error", "engine": "DALL-E 3", "error": str(e)}

    def txt2music(self, prompt: str, soul: str = "beethoven",
                  duration_seconds: int = 30, genre: str = "",
                  bpm: int = 120, key: str = "") -> dict:
        if self._suno_key:
            return {"status": "developing", "message": "Suno API对接开发中", "prompt": prompt}

        return self._midiutil_generate(prompt, duration_seconds, bpm, key, genre)

    def _midiutil_generate(self, prompt: str, duration: int, bpm: int, key: str, genre: str) -> dict:
        try:
            from MIDIUtil.MIDIFile import MIDIFile
        except ImportError:
            try:
                from midiutil.MIDIFile import MIDIFile
            except ImportError:
                return {
                    "status": "concept_only",
                    "message": "MIDIUtil未安装，仅生成音乐概念描述",
                    "prompt": prompt,
                    "concept": self._generate_concept(prompt, "music"),
                    "install_hint": "pip install MIDIUtil",
                }

        track = 0
        channel = 0
        time_offset = 0
        tempo = bpm
        volume = 100

        mf = MIDIFile(1)
        mf.addTrackName(track, time_offset, f"VORTEX_{genre or 'composition'}")
        mf.addTempo(track, time_offset, tempo)

        scale_intervals = [0, 2, 4, 5, 7, 9, 11]
        base_note = 60
        if key:
            key_map = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
            base_note = 60 + key_map.get(key.upper(), 0)

        import random
        random.seed(hash(prompt) % 2**32)
        beats = int(duration * tempo / 60)
        for beat in range(min(beats, 120)):
            note_idx = random.randint(0, len(scale_intervals) - 1)
            note = base_note + scale_intervals[note_idx] + (12 if random.random() > 0.7 else 0)
            dur = 1.0 if random.random() > 0.3 else 0.5
            vel = volume if random.random() > 0.2 else volume - 20
            mf.addNote(track, channel, note, beat, dur, vel)

        output_dir = Path("output/midi")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"vortex_{int(time.time())}.mid")

        with open(output_path, "wb") as f:
            mf.writeFile(f)

        return {
            "status": "ok",
            "engine": "MIDIUtil",
            "prompt": prompt,
            "output_path": output_path,
            "bpm": bpm,
            "key": key or "C",
            "duration_beats": min(beats, 120),
        }

    def txt2video(self, prompt: str, soul: str = "davinci",
                  duration_seconds: int = 5, resolution: str = "720p") -> dict:
        if self._kling_key:
            return {"status": "developing", "message": "Kling API对接开发中", "prompt": prompt}

        return {
            "status": "concept_only",
            "message": "视频生成API未配置，仅生成概念描述",
            "prompt": prompt,
            "concept": self._generate_concept(prompt, "video"),
            "available_engines": {
                "Kling": "需设置 KLING_API_KEY",
                "Runway": "计划中",
                "Sora": "计划中",
                "Hailuo": "需设置 HAILUO_API_KEY",
            },
        }

    def txt2voice(self, text: str, soul: str = "guizhu",
                  voice_id: str = "", language: str = "zh") -> dict:
        if self._elevenlabs_key:
            return {"status": "developing", "message": "ElevenLabs API对接开发中", "text": text[:200]}

        return {
            "status": "concept_only",
            "message": "语音合成API未配置，仅生成概念描述",
            "text": text[:200],
            "concept": self._generate_concept(text, "voice"),
            "available_engines": {
                "ElevenLabs": "需设置 ELEVENLABS_API_KEY",
                "ChatTTS": "本地安装可用",
            },
        }

    def txt23d(self, prompt: str, soul: str = "davinci",
               style: str = "realistic") -> dict:
        return {
            "status": "concept_only",
            "message": "3D生成API未配置，仅生成概念描述",
            "prompt": prompt,
            "concept": self._generate_concept(prompt, "3d"),
            "available_engines": {
                "Tripo3D": "需设置 TRIPO_API_KEY",
                "Meshy": "计划中",
                "Blender+Python": "已通过blender_mcp可用",
            },
        }

    def _generate_concept(self, prompt: str, gen_type: str) -> str:
        templates = {
            "image": f"[视觉概念] 基于提示词「{prompt}」生成图像。建议风格：印象派/写实/抽象。推荐引擎：ComfyUI(本地) 或 DALL-E(云端)。",
            "music": f"[音乐概念] 基于提示词「{prompt}」生成音乐。建议BPM：120，调性：C大调。推荐引擎：MIDIUtil(本地MIDI) 或 Suno(云端)。",
            "video": f"[视频概念] 基于提示词「{prompt}」生成视频。建议时长：5秒，分辨率：720p。推荐引擎：Kling(国内) 或 Runway(海外)。",
            "voice": f"[语音概念] 基于文本「{prompt}」生成语音。建议语言：中文。推荐引擎：ChatTTS(本地) 或 ElevenLabs(云端)。",
            "3d": f"[3D概念] 基于提示词「{prompt}」生成3D模型。建议风格：{style}。推荐引擎：Tripo3D(云端) 或 Blender+Python(本地)。",
        }
        return templates.get(gen_type, f"[概念] {prompt}")


AIGEN_SKILL_DEFINITION = {
    "skill_id": "ai_generation",
    "name": "AI生成工具链",
    "description": "莫奈/梵高/贝多芬/达芬奇灵魂专用：图像/音乐/视频/3D/语音AI生成",
    "soul_mapping": {
        "monet": ["comfyui", "dalle", "flux", "midjourney", "kling", "runway"],
        "vangogh": ["comfyui", "dalle", "flux", "midjourney", "kling"],
        "beethoven": ["suno", "udio", "musicgen", "elevenlabs", "chattts"],
        "davinci": ["comfyui", "kling", "runway", "pika", "sora", "hailuo", "tripo3d", "meshy"],
        "guizhu": ["elevenlabs", "chattts"],
    },
    "mcp_tools": ["aigen_txt2img", "aigen_txt2music", "aigen_txt2video", "aigen_txt2voice", "aigen_txt23d"],
    "boundary": {
        "可用": ["ComfyUI本地图像生成", "DALL-E API", "MIDIUtil本地MIDI", "ChatTTS本地语音"],
        "开发中": ["Suno API", "Kling API", "Hailuo API", "Tripo3D API", "ElevenLabs API"],
        "计划中": ["Midjourney", "Runway", "Pika", "Sora", "Meshy", "Udio"],
    },
}

_adapter_instance: Optional[AIGenAdapter] = None


def get_adapter() -> AIGenAdapter:
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = AIGenAdapter()
    return _adapter_instance
