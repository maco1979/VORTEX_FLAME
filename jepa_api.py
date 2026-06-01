#!/usr/bin/env python3
"""
VORTEX FLAME JEPA Inference API
================================

FastAPI server exposing all JEPA models + Soul Bridge as REST endpoints.

Endpoints:
  POST /encode        — Encode raw input → slot embeddings
  POST /analyze       — Full pipeline: encode → decode → knowledge → LLM
  POST /counterfactual— Counterfactual reasoning
  GET  /models        — List loaded JEPA models
  GET  /health        — Health check

Usage:
  python jepa_api.py --port 8199
  curl -X POST http://localhost:8199/analyze -H "Content-Type: application/json" \
    -d '{"modality":"audio","soul":"beethoven","query":"What key is this?"}'
"""

import argparse
import io
import os
import sys
import time
import base64
import json
from typing import Any, Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(os.path.dirname(__file__)))

from jepa_soul_bridge import (
    JEPASoulBridge, JEPAModality, SlotDescription,
    CloudBackend, LocalBackend, HybridBackend,
    MODALITY_SLOT_NAMES, MODALITY_SOUL_MAP,
)

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("FastAPI not installed. Install with: pip install fastapi uvicorn")


if HAS_FASTAPI:
    class EncodeRequest(BaseModel):
        modality: str = Field(..., description="JEPA modality: audio, visual, financial, code, etc.")
        data_b64: Optional[str] = Field(None, description="Base64-encoded numpy array")
        data_shape: Optional[List[int]] = Field(None, description="Shape of the data array")

    class AnalyzeRequest(BaseModel):
        modality: str = Field(..., description="JEPA modality")
        soul: Optional[str] = Field(None, description="Soul name (auto-selected if None)")
        query: str = Field("", description="Query for the soul")
        top_k: int = Field(5, description="Number of knowledge entries to retrieve")
        use_llm: bool = Field(True, description="Whether to call LLM backend")
        data_b64: Optional[str] = Field(None, description="Base64-encoded numpy array")
        data_shape: Optional[List[int]] = Field(None, description="Shape of the data array")

    class CounterfactualRequest(BaseModel):
        modality: str = Field(..., description="JEPA modality")
        intervene_slot_id: int = Field(..., description="Slot index to intervene on")
        intervene_b64: Optional[str] = Field(None, description="Base64-encoded intervention embedding")
        soul: Optional[str] = Field(None)
        query: str = Field("What would change?")

    class SlotInfo(BaseModel):
        slot_id: int
        name: str
        activation: float
        summary: str

    class AnalyzeResponse(BaseModel):
        modality: str
        slots: List[SlotInfo]
        knowledge_count: int
        llm_response: Optional[str] = None
        soul: str
        elapsed_ms: float


def create_app(memory_dir: str = None, llm_provider: str = "cloud",  # type: ignore[reportArgumentType]
               api_key: str = None, model: str = "gpt-4o-mini"):  # type: ignore[reportArgumentType]
    if not HAS_FASTAPI:
        raise RuntimeError("FastAPI not installed")

    app = FastAPI(
        title="VORTEX FLAME JEPA API",
        description="JEPA perception + Soul knowledge + LLM reasoning",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if llm_provider == "cloud":
        llm = CloudBackend(api_key=api_key, model=model)
    elif llm_provider == "local":
        llm = LocalBackend()
    else:
        cloud = CloudBackend(api_key=api_key, model=model)
        local = LocalBackend()
        llm = HybridBackend(cloud=cloud, local=local)

    bridge = JEPASoulBridge(memory_dir=memory_dir, llm_backend=llm)

    @app.on_event("startup")
    async def startup():
        _load_available_models(bridge)
        print(f"[JEPA API] Loaded {len(bridge._jepa_models)} JEPA models")
        print(f"[JEPA API] LLM backend: {type(llm).__name__}, available={llm.is_available()}")

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "models": list(bridge._jepa_models.keys()),
            "llm_available": bridge.llm.is_available(),
            "memory_dir": bridge.memory.memory_dir if hasattr(bridge.memory, 'memory_dir') else "default",
        }

    @app.get("/models")
    async def list_models():
        models = {}
        for modality, jepa in bridge._jepa_models.items():
            n_params = sum(p.numel() for p in jepa.parameters()) if hasattr(jepa, 'parameters') else 0
            models[modality.value] = {
                "type": type(jepa).__name__,
                "params": n_params,
                "has_projector": modality in bridge._projectors,
                "slot_names": MODALITY_SLOT_NAMES.get(modality, []),
                "souls": MODALITY_SOUL_MAP.get(modality, []),
            }
        return models

    @app.post("/encode")
    async def encode(req: EncodeRequest):
        modality = _parse_modality(req.modality)
        if modality not in bridge._jepa_models:
            raise HTTPException(404, f"No JEPA model for {req.modality}")

        data = _decode_data(req.data_b64, req.data_shape)
        try:
            slots = bridge.encode(data, modality)
            return {
                "modality": req.modality,
                "slot_shape": list(slots.shape),
                "slot_norms": [float(slots[:, :, i, :].norm().item()) for i in range(slots.shape[-2])],
            }
        except Exception as e:
            raise HTTPException(500, f"Encoding failed: {e}")

    @app.post("/analyze", response_model=AnalyzeResponse)
    async def analyze(req: AnalyzeRequest):
        modality = _parse_modality(req.modality)
        if modality not in bridge._jepa_models:
            raise HTTPException(404, f"No JEPA model for {req.modality}")

        data = _decode_data(req.data_b64, req.data_shape)
        t0 = time.time()

        try:
            result = bridge.process(
                raw_input=data,
                modality=modality,
                soul=req.soul,
                query=req.query,
                top_k=req.top_k,
                use_llm=req.use_llm,
            )
            elapsed = (time.time() - t0) * 1000

            return AnalyzeResponse(
                modality=req.modality,
                slots=[SlotInfo(
                    slot_id=d.slot_id, name=d.name,
                    activation=d.activation, summary=d.summary,
                ) for d in result.slot_descriptions],
                knowledge_count=len(result.knowledge_context),
                llm_response=result.llm_response,
                soul=result.metadata.get("soul", ""),
                elapsed_ms=elapsed,
            )
        except Exception as e:
            raise HTTPException(500, f"Analysis failed: {e}")

    @app.post("/counterfactual")
    async def counterfactual(req: CounterfactualRequest):
        modality = _parse_modality(req.modality)
        if modality not in bridge._jepa_models:
            raise HTTPException(404, f"No JEPA model for {req.modality}")

        intervene_embedding = _decode_data(req.intervene_b64, None) if req.intervene_b64 else None
        if intervene_embedding is None:
            slot_dim = 128
            intervene_embedding = torch.randn(1, 4, slot_dim) * 2.0

        try:
            result = bridge.counterfactual(
                raw_input=torch.randn(1, 4, bridge._jepa_models[modality].num_slots, 256),
                modality=modality,
                intervene_slot_id=req.intervene_slot_id,
                intervene_embedding=intervene_embedding,
                soul=req.soul,
                query=req.query,
            )
            return {
                "modality": req.modality,
                "intervened_slot": req.intervene_slot_id,
                "slots": [{"name": d.name, "activation": d.activation, "summary": d.summary}
                          for d in result.slot_descriptions],
                "llm_response": result.llm_response,
                "soul": result.metadata.get("soul", ""),
            }
        except Exception as e:
            raise HTTPException(500, f"Counterfactual failed: {e}")

    return app


def _parse_modality(modality_str: str) -> JEPAModality:
    mapping = {m.value: m for m in JEPAModality}
    if modality_str in mapping:
        return mapping[modality_str]
    raise ValueError(f"Unknown modality '{modality_str}'. Valid: {list(mapping.keys())}")


def _decode_data(b64_str: Optional[str], shape: Optional[List[int]]) -> Optional[torch.Tensor]:
    if b64_str is None:
        return None
    raw = base64.b64decode(b64_str)
    arr = np.load(io.BytesIO(raw))
    if shape:
        arr = arr.reshape(shape)
    return torch.from_numpy(arr)


def _load_available_models(bridge: JEPASoulBridge):
    checkpoint_dirs = {
        JEPAModality.AUDIO: r"D:\VORTEX_FLAME\ajepa_checkpoints",
        JEPAModality.FINANCIAL: r"D:\VORTEX_FLAME\finjepa_checkpoints",
    }

    for modality, ckpt_dir in checkpoint_dirs.items():
        best_path = os.path.join(ckpt_dir, "ajepa_best.pt" if modality == JEPAModality.AUDIO else "finjepa_best.pt")
        if not os.path.exists(best_path):
            continue

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(best_path, map_location=device, weights_only=False)

            if modality == JEPAModality.AUDIO:
                from five_layer_jepa.causal_jepa import CAJEPA
                from train_ajepa import AudioFeatureProjector
                jepa = CAJEPA(input_dim=256).to(device)
                projector = AudioFeatureProjector(output_dim=256).to(device)
                if "ajepa_state" in ckpt:
                    jepa.load_state_dict(ckpt["ajepa_state"])
                if "projector_state" in ckpt:
                    projector.load_state_dict(ckpt["projector_state"])
                jepa.eval()
                bridge.register_jepa(modality, jepa, projector)
                print(f"  Loaded A-JEPA from {best_path}")

            elif modality == JEPAModality.FINANCIAL:
                from five_layer_jepa.causal_jepa import CFINJEPA
                from train_finjepa import FinancialFeatureProjector
                jepa = CFINJEPA(input_dim=256).to(device)
                projector = FinancialFeatureProjector(output_dim=256).to(device)
                if "finjepa_state" in ckpt:
                    jepa.load_state_dict(ckpt["finjepa_state"])
                if "projector_state" in ckpt:
                    projector.load_state_dict(ckpt["projector_state"])
                jepa.eval()
                bridge.register_jepa(modality, jepa, projector)
                print(f"  Loaded FIN-JEPA from {best_path}")

        except Exception as e:
            print(f"  Failed to load {modality.value}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument("--memory-dir", type=str, default=None)
    parser.add_argument("--llm", choices=["cloud", "local", "hybrid"], default="cloud")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    if not HAS_FASTAPI:
        print("ERROR: FastAPI not installed. Run: pip install fastapi uvicorn")
        sys.exit(1)

    app = create_app(
        memory_dir=args.memory_dir,
        llm_provider=args.llm,
        api_key=args.api_key,
        model=args.model,
    )
    uvicorn.run(app, host="0.0.0.0", port=args.port)
