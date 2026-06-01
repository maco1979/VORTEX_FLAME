"""
Dual-Pathway Bridge — Cross-Attention Fusion of RAG Text + C-JEPA World Embeddings
===================================================================================
The core fusion module of the dual-pathway architecture:

  Path A (RAG):     text_chunks → text_embedding → [B, L_text, D_text]
  Path B (C-JEPA):  causal_objects → world_embedding → [B, N_slots, D_world]
  Fusion:           Cross-Attention(text, world) → [B, L_text, D_fused]
  Output:           fused representation → Transformer Decoder → response

Design Principles:
  1. Text tokens attend to world-embedding slots (text queries world)
  2. World-embedding slots attend to text tokens (world queries text)
  3. Bidirectional cross-attention captures both "what facts" (text) and "why logic" (world)
  4. Gated residual connections prevent one pathway from dominating
  5. Dimension alignment via learned projections

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │ Text Embeddings [B, L, D_text]                              │
  │     ↓ text_proj → [B, L, D_fused]                           │
  │     ↓ Cross-Attention(Q=text, K=world, V=world)             │
  │     ↓ Gate + Residual                                        │
  │     ↓ Self-Attention (text-to-text refinement)               │
  │                                                              │
  │ World Embeddings [B, N, D_world]                             │
  │     ↓ world_proj → [B, N, D_fused]                           │
  │     ↓ Cross-Attention(Q=world, K=text, V=text)              │
  │     ↓ Gate + Residual                                        │
  │                                                              │
  │ Fusion = text_fused + world_fused → [B, L+N, D_fused]       │
  │     ↓ Output projection → [B, L+N, D_output]                │
  └─────────────────────────────────────────────────────────────┘
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DEFAULT_D_TEXT = 384
DEFAULT_D_WORLD = 128
DEFAULT_D_FUSED = 256
DEFAULT_D_OUTPUT = 512
DEFAULT_NUM_HEADS = 4
DEFAULT_NUM_LAYERS = 2


class GatedResidual(nn.Module):
    """
    Gated residual connection: output = gate * new_value + (1 - gate) * residual
    Prevents one pathway from dominating the other during early training.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

    def forward(self, new_value: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if new_value.shape != residual.shape:
            min_len = min(new_value.shape[1], residual.shape[1])
            new_value = new_value[:, :min_len]
            residual = residual[:, :min_len]
        g = self.gate(torch.cat([new_value, residual], dim=-1))
        return g * new_value + (1 - g) * residual


class CrossAttentionBlock(nn.Module):
    """
    Bidirectional cross-attention between two sequences.

    Forward: Q from seq_a, K/V from seq_b → seq_a attends to seq_b
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.gate = GatedResidual(dim)

    def forward(
        self,
        query_seq: torch.Tensor,
        key_value_seq: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.norm_q(query_seq)
        kv = self.norm_kv(key_value_seq)

        attn_out, _ = self.cross_attn(q, kv, kv, key_padding_mask=kv_mask)
        gated = self.gate(attn_out, query_seq)

        ffn_out = self.ffn(self.norm_ffn(gated))
        return gated + ffn_out


class DualPathwayFusionLayer(nn.Module):
    """
    Single layer of dual-pathway fusion.

    1. Text → Cross-Attention with World (text queries world logic)
    2. World → Cross-Attention with Text (world queries text facts)
    3. Text self-attention refinement
    4. Concatenate fused text + fused world
    """

    def __init__(
        self,
        dim: int = DEFAULT_D_FUSED,
        num_heads: int = DEFAULT_NUM_HEADS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.text_queries_world = CrossAttentionBlock(dim, num_heads, dropout)
        self.world_queries_text = CrossAttentionBlock(dim, num_heads, dropout)
        self.text_self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.text_self_norm = nn.LayerNorm(dim)
        self.text_self_gate = GatedResidual(dim)

    def forward(
        self,
        text_embeds: torch.Tensor,
        world_embeds: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        world_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_fused = self.text_queries_world(text_embeds, world_embeds, query_mask=text_mask, kv_mask=world_mask)

        world_fused = self.world_queries_text(world_embeds, text_embeds, query_mask=world_mask, kv_mask=text_mask)

        text_self = self.text_self_norm(text_fused)
        text_self_out, _ = self.text_self_attn(text_self, text_self, text_self, key_padding_mask=text_mask)
        text_fused = self.text_self_gate(text_self_out, text_fused)

        return text_fused, world_fused


class DualPathwayBridge(nn.Module):
    """
    Full dual-pathway fusion bridge.

    Takes RAG text embeddings and C-JEPA world embeddings,
    fuses them via bidirectional cross-attention, and outputs
    a unified representation for the top-level Transformer.

    Usage:
      bridge = DualPathwayBridge()
      fused = bridge(text_embeds, world_embeds)
      # fused: [B, L_text + N_world, D_output] → feed to Transformer Decoder
    """

    def __init__(
        self,
        d_text: int = DEFAULT_D_TEXT,
        d_world: int = DEFAULT_D_WORLD,
        d_fused: int = DEFAULT_D_FUSED,
        d_output: int = DEFAULT_D_OUTPUT,
        num_heads: int = DEFAULT_NUM_HEADS,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = 0.1,
        max_text_len: int = 512,
        max_world_slots: int = 32,
    ):
        super().__init__()
        self.d_text = d_text
        self.d_world = d_world
        self.d_fused = d_fused
        self.d_output = d_output
        self.num_layers = num_layers

        self.text_proj = nn.Sequential(
            nn.Linear(d_text, d_fused),
            nn.LayerNorm(d_fused),
            nn.GELU(),
        )

        self.world_proj = nn.Sequential(
            nn.Linear(d_world, d_fused),
            nn.LayerNorm(d_fused),
            nn.GELU(),
        )

        self.text_pos_enc = nn.Parameter(
            torch.randn(1, max_text_len, d_fused) * 0.02
        )
        self.world_pos_enc = nn.Parameter(
            torch.randn(1, max_world_slots, d_fused) * 0.02
        )

        self.fusion_layers = nn.ModuleList([
            DualPathwayFusionLayer(d_fused, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(d_fused)
        self.output_proj = nn.Sequential(
            nn.Linear(d_fused, d_output),
            nn.LayerNorm(d_output),
            nn.GELU(),
        )

        self.pathway_gate = nn.Sequential(
            nn.Linear(d_fused * 2, d_fused),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        text_embeds: torch.Tensor,
        world_embeds: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        world_mask: Optional[torch.Tensor] = None,
        return_pathway_weights: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            text_embeds: [B, L_text, D_text] — RAG text token embeddings
            world_embeds: [B, N_slots, D_world] — C-JEPA world slot embeddings
            text_mask: [B, L_text] — True for padded positions
            world_mask: [B, N_slots] — True for padded positions
            return_pathway_weights: if True, return per-pathway gate weights

        Returns:
            Dict with:
                fused: [B, L_text + N_slots, D_output] — unified representation
                text_fused: [B, L_text, D_output] — text pathway output
                world_fused: [B, N_slots, D_output] — world pathway output
                pathway_weights: (optional) [B, 1] — gate weight between pathways
        """
        B = text_embeds.shape[0]

        text = self.text_proj(text_embeds)
        world = self.world_proj(world_embeds)

        L_text = text.shape[1]
        N_world = world.shape[1]
        text = text + self.text_pos_enc[:, :L_text, :]
        world = world + self.world_pos_enc[:, :N_world, :]

        for layer in self.fusion_layers:
            text, world = layer(text, world, text_mask=text_mask, world_mask=world_mask)

        text = self.output_norm(text)
        world = self.output_norm(world)

        text_mean = text.mean(dim=1, keepdim=True)
        world_mean = world.mean(dim=1, keepdim=True)
        gate_weight = self.pathway_gate(torch.cat([text_mean, world_mean], dim=-1))

        text_gated = text * gate_weight.expand_as(text)
        world_gated = world * (1 - gate_weight).expand_as(world)

        text_out = self.output_proj(text_gated)
        world_out = self.output_proj(world_gated)

        fused = torch.cat([text_out, world_out], dim=1)

        result = {
            "fused": fused,
            "text_fused": text_out,
            "world_fused": world_out,
        }

        if return_pathway_weights:
            result["pathway_weights"] = gate_weight.squeeze(1)

        return result

    def forward_with_rag_and_cache(
        self,
        query: str,
        soul: str,
        top_k_text: int = 3,
        top_k_world: int = 5,
    ) -> Dict[str, Any]:
        """
        Convenience method: query both RAG and World-Embedding cache,
        then fuse the results.

        This is the primary entry point for inference.
        """
        text_embeds = self._retrieve_text_embeddings(query, soul, top_k_text)
        world_embeds = self._retrieve_world_embeddings(query, soul, top_k_world)

        if text_embeds is None and world_embeds is None:
            return {"fused": None, "error": "No knowledge retrieved from either pathway"}

        if text_embeds is None:
            text_embeds = torch.zeros(1, 1, self.d_text)
        if world_embeds is None:
            world_embeds = torch.zeros(1, 1, self.d_world)

        return self.forward(text_embeds, world_embeds, return_pathway_weights=True)

    def _retrieve_text_embeddings(
        self, query: str, soul: str, top_k: int
    ) -> Optional[torch.Tensor]:
        try:
            from soul_memory import SoulMemoryEngine
            engine = SoulMemoryEngine()
            results = engine.recall(soul, query, top_k=top_k)

            if not results:
                return None

            texts = []
            for r in results:
                content = r.get("content", {})
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except Exception:
                        content = {"text": content}
                text = content.get("text", content.get("topic", ""))
                if text:
                    texts.append(text)

            if not texts:
                return None

            from soul_memory import EmbeddingProvider
            provider = EmbeddingProvider.get()
            if not provider.available:
                return None

            embeddings = []
            for t in texts:
                emb_bytes = provider.encode(t)
                if emb_bytes:
                    emb_vec = np.frombuffer(emb_bytes, dtype=np.float32)  # type: ignore[reportAttributeAccessIssue]
                    embeddings.append(emb_vec)

            if not embeddings:
                return None

            import numpy as np
            emb_tensor = torch.tensor(np.stack(embeddings), dtype=torch.float32).unsqueeze(0)
            return emb_tensor

        except Exception as e:
            logger.warning(f"Text embedding retrieval failed: {e}")
            return None

    def _retrieve_world_embeddings(
        self, query: str, soul: str, top_k: int
    ) -> Optional[torch.Tensor]:
        try:
            from world_embedding_cache import WorldEmbeddingCache
            cache = WorldEmbeddingCache()
            results = cache.search(soul, query, top_k=top_k)

            if not results:
                return None

            embeddings = []
            for r in results:
                if r.embedding is not None:
                    embeddings.append(r.embedding)

            if not embeddings:
                return None

            import numpy as np
            emb_tensor = torch.tensor(np.stack(embeddings), dtype=torch.float32).unsqueeze(0)
            return emb_tensor

        except Exception as e:
            logger.warning(f"World embedding retrieval failed: {e}")
            return None


class DualPathwayInferenceEngine:
    """
    High-level inference engine that orchestrates the full dual-pathway pipeline.

    Usage:
      engine = DualPathwayInferenceEngine()
      result = engine.query("设备故障后怎么处理？", soul="cezanne")
      # result contains: fused representation, text context, world context, LLM response
    """

    def __init__(
        self,
        d_text: int = DEFAULT_D_TEXT,
        d_world: int = DEFAULT_D_WORLD,
        d_fused: int = DEFAULT_D_FUSED,
        d_output: int = DEFAULT_D_OUTPUT,
    ):
        self.bridge = DualPathwayBridge(
            d_text=d_text,
            d_world=d_world,
            d_fused=d_fused,
            d_output=d_output,
        )
        self._llm_backend = None

    def set_llm_backend(self, backend):
        self._llm_backend = backend

    def query(
        self,
        user_query: str,
        soul: str = "cezanne",
        top_k_text: int = 3,
        top_k_world: int = 5,
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        text_context = self._fetch_rag_context(user_query, soul, top_k_text)
        world_context = self._fetch_world_context(user_query, soul, top_k_world)

        text_embeds = self._prepare_text_embeds(text_context)
        world_embeds = self._prepare_world_embeds(world_context)

        fusion_result = None
        if text_embeds is not None and world_embeds is not None:
            fusion_result = self.bridge(text_embeds, world_embeds, return_pathway_weights=True)
        elif text_embeds is not None:
            fusion_result = {"fused": text_embeds, "text_fused": text_embeds, "world_fused": None}
        elif world_embeds is not None:
            fusion_result = {"fused": world_embeds, "text_fused": None, "world_fused": world_embeds}

        llm_response = None
        if use_llm and self._llm_backend is not None:
            prompt = self._assemble_prompt(user_query, text_context, world_context)
            try:
                llm_response = self._llm_backend.generate(prompt)
            except Exception as e:
                llm_response = f"[LLM error: {e}]"

        return {
            "query": user_query,
            "soul": soul,
            "text_context": text_context,
            "world_context": world_context,
            "fusion_result": {
                "fused_shape": list(fusion_result["fused"].shape) if fusion_result and fusion_result.get("fused") is not None else None,
                "pathway_weights": fusion_result.get("pathway_weights", None) if fusion_result else None,
            } if fusion_result else None,
            "llm_response": llm_response,
        }

    def _fetch_rag_context(self, query: str, soul: str, top_k: int) -> List[Dict]:
        try:
            from soul_memory import SoulMemoryEngine
            engine = SoulMemoryEngine()
            return engine.recall(soul, query, top_k=top_k)
        except Exception:
            return []

    def _fetch_world_context(self, query: str, soul: str, top_k: int) -> List[Dict]:
        try:
            from world_embedding_cache import WorldEmbeddingCache
            cache = WorldEmbeddingCache()
            results = cache.search(soul, query, top_k=top_k)
            return [{"entry_id": r.entry_id, "objects": r.objects, "causal": r.causal_summary} for r in results]
        except Exception:
            return []

    def _prepare_text_embeds(self, context: List[Dict]) -> Optional[torch.Tensor]:
        if not context:
            return None
        try:
            from soul_memory import EmbeddingProvider
            provider = EmbeddingProvider.get()
            if not provider.available:
                return None
            embeddings = []
            for entry in context:
                content = entry.get("content", {})
                if isinstance(content, str):
                    try:
                        import json
                        content = json.loads(content)
                    except Exception:
                        content = {"text": content}
                text = content.get("text", content.get("topic", ""))
                if text:
                    emb = provider.encode(text)
                    if emb:
                        import numpy as np
                        embeddings.append(np.frombuffer(emb, dtype=np.float32))
            if embeddings:
                import numpy as np
                return torch.tensor(np.stack(embeddings), dtype=torch.float32).unsqueeze(0)
        except Exception:
            pass
        return None

    def _prepare_world_embeds(self, context: List[Dict]) -> Optional[torch.Tensor]:
        if not context:
            return None
        try:
            from world_embedding_cache import WorldEmbeddingCache
            cache = WorldEmbeddingCache()
            embeddings = []
            for entry in context:
                entry_id = entry.get("entry_id", "")
                emb = cache.get_embedding("cezanne", entry_id)
                if emb is not None:
                    embeddings.append(emb)
            if embeddings:
                import numpy as np
                return torch.tensor(np.stack(embeddings), dtype=torch.float32).unsqueeze(0)
        except Exception:
            pass
        return None

    def _assemble_prompt(self, query: str, text_context: List[Dict], world_context: List[Dict]) -> str:
        parts = [f"[User Query] {query}\n"]

        if text_context:
            parts.append("[Text Knowledge (RAG)]")
            for i, entry in enumerate(text_context[:3]):
                content = entry.get("content", {})
                if isinstance(content, str):
                    try:
                        import json
                        content = json.loads(content)
                    except Exception:
                        content = {"text": content}
                topic = content.get("topic", content.get("text", ""))[:200]
                parts.append(f"  {i+1}. {topic}")

        if world_context:
            parts.append("\n[Causal World Knowledge (C-JEPA)]")
            for i, entry in enumerate(world_context[:3]):
                objects = entry.get("objects", [])
                causal = entry.get("causal", "")
                parts.append(f"  {i+1}. Objects: {objects}")
                if causal:
                    parts.append(f"     Causal: {causal}")

        parts.append("\nPlease provide your analysis based on both factual knowledge and causal logic above.")
        return "\n".join(parts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=== Dual-Pathway Bridge Architecture Test ===\n")

    bridge = DualPathwayBridge(
        d_text=384,
        d_world=128,
        d_fused=256,
        d_output=512,
        num_heads=4,
        num_layers=2,
    )

    B, L_text, N_world = 2, 10, 7
    text_embeds = torch.randn(B, L_text, 384)
    world_embeds = torch.randn(B, N_world, 128)

    result = bridge(text_embeds, world_embeds, return_pathway_weights=True)

    print(f"Input text:  [{B}, {L_text}, 384]")
    print(f"Input world: [{B}, {N_world}, 128]")
    print(f"Fused output: {result['fused'].shape}")
    print(f"Text fused:   {result['text_fused'].shape}")
    print(f"World fused:  {result['world_fused'].shape}")
    print(f"Pathway gate: {result['pathway_weights']}")

    total_params = sum(p.numel() for p in bridge.parameters())
    trainable_params = sum(p.numel() for p in bridge.parameters() if p.requires_grad)
    print(f"\nTotal params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (fp32)")

    print("\n=== Inference Engine Test ===\n")
    engine = DualPathwayInferenceEngine()
    result = engine.query("传感器告警后如何处理", soul="cezanne", use_llm=False)
    print(f"Query: {result['query']}")
    print(f"Text context entries: {len(result['text_context'])}")
    print(f"World context entries: {len(result['world_context'])}")
    print(f"Fusion result: {result['fusion_result']}")
