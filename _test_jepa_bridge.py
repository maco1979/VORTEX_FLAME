#!/usr/bin/env python3
"""
JEPA-Soul Bridge CPU Integration Test
======================================

Validates the full pipeline WITHOUT GPU:
  1. SlotToTextDecoder — slot vectors → natural language descriptions
  2. SoulMemoryEngine — knowledge store & retrieval
  3. LLMBackend — Cloud/Local/Hybrid backends
  4. JEPASoulBridge — end-to-end: encode → decode → retrieve → generate
  5. Counterfactual — intervention-based reasoning

All tests run on CPU with mock JEPA models.
"""

import os
import sys
import json
import tempfile
import shutil

sys.path.insert(0, str(os.path.dirname(__file__)))

import torch
import numpy as np

from jepa_soul_bridge import (
    JEPASoulBridge, SlotToTextDecoder, BridgeResult,
    CloudBackend, LocalBackend, HybridBackend,
    JEPAModality, SlotDescription,
    MODALITY_SLOT_NAMES, MODALITY_SOUL_MAP,
)
from soul_memory import SoulMemoryEngine


class MockJEPA:
    def __init__(self, num_slots=6, slot_dim=128):
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.context_encoder = MockEncoder(num_slots, slot_dim)
        self.target_encoder = MockEncoder(num_slots, slot_dim)

    def counterfactual_predict(self, features, slot_id, embedding):
        slots, attn = self.context_encoder(features)
        slots[:, :, slot_id, :] = embedding
        return slots, attn


class MockEncoder:
    def __init__(self, num_slots, slot_dim):
        self.num_slots = num_slots
        self.slot_dim = slot_dim

    def __call__(self, features):
        if features.dim() == 4:
            B, T, N, D = features.shape
        elif features.dim() == 3:
            B, T, D = features.shape
            N = self.num_slots
        else:
            B = 1
            T = 1
            N = self.num_slots
        slots = torch.randn(B, T, self.num_slots, self.slot_dim) * 0.5
        attn = torch.softmax(torch.randn(B, T, self.num_slots, N), dim=-1)
        return slots, attn


class MockProjector:
    def __init__(self, output_dim=256):
        self.output_dim = output_dim
        self.linear = torch.nn.Linear(output_dim, output_dim)

    def __call__(self, x):
        return self.linear(x)


def test_slot_decoder():
    print("\n" + "=" * 60)
    print("TEST 1: SlotToTextDecoder")
    print("=" * 60)

    decoder = SlotToTextDecoder()

    for modality in [JEPAModality.AUDIO, JEPAModality.FINANCIAL, JEPAModality.CODE]:
        slot_names = MODALITY_SLOT_NAMES[modality]
        n_slots = len(slot_names)
        slots = torch.randn(1, 4, n_slots, 128) * 1.5

        descriptions = decoder.decode(slots, modality)
        assert len(descriptions) == n_slots, f"Expected {n_slots} slots, got {len(descriptions)}"
        assert descriptions[0].activation >= descriptions[-1].activation, "Not sorted by activation"

        query_text = decoder.to_query_text(descriptions, modality)
        assert modality.value in query_text, f"Modality {modality.value} not in query text"

        print(f"  {modality.value}: {n_slots} slots decoded, top='{descriptions[0].name}' "
              f"(act={descriptions[0].activation:.2f}), query='{query_text[:60]}...'")

    print("  PASS")


def test_soul_memory():
    print("\n" + "=" * 60)
    print("TEST 2: SoulMemoryEngine")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp(prefix="bridge_test_")
    try:
        engine = SoulMemoryEngine(tmpdir)

        engine.write("beethoven", "knowledge", {"topic": "harmony", "text": "Harmonic analysis of C minor"})
        engine.write("beethoven", "knowledge", {"topic": "rhythm", "text": "Rhythm patterns in sonata form"})
        engine.write("beethoven", "knowledge", {"topic": "orchestration", "text": "Orchestration techniques for strings"})
        engine.write("strategy", "knowledge", {"topic": "momentum", "text": "Momentum trading signals"})
        engine.write("strategy", "knowledge", {"topic": "volatility", "text": "Volatility regime detection"})

        results_beethoven = engine.recall("beethoven", "harmony chords", top_k=3)
        results_strategy = engine.recall("strategy", "volatility market", top_k=3)

        print(f"  beethoven recall 'harmony': {len(results_beethoven)} results")
        print(f"  strategy recall 'volatility': {len(results_strategy)} results")

        assert len(results_beethoven) > 0, "No results from beethoven memory"
        assert len(results_strategy) > 0, "No results from strategy memory"

        print("  PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_llm_backends():
    print("\n" + "=" * 60)
    print("TEST 3: LLMBackend (Cloud/Local/Hybrid)")
    print("=" * 60)

    cloud = CloudBackend(api_key="test-key", model="gpt-4o-mini")
    assert cloud.is_available(), "Cloud backend should be available with test key"
    print(f"  CloudBackend: available={cloud.is_available()}, model={cloud.model}")

    local = LocalBackend(model_path=None)
    assert not local.is_available(), "Local backend should not be available without model"
    print(f"  LocalBackend: available={local.is_available()}")

    hybrid = HybridBackend(cloud=cloud, local=local, complexity_threshold=0.5)
    assert hybrid.is_available(), "Hybrid should be available via cloud"
    print(f"  HybridBackend: available={hybrid.is_available()}")

    complexity = hybrid._estimate_complexity("Analyze the counterfactual implications of this data")
    print(f"  Complexity estimation: {complexity:.2f}")
    assert complexity > 0.3, "Complex prompt should score > 0.3"

    print("  PASS")


def test_bridge_full_pipeline():
    print("\n" + "=" * 60)
    print("TEST 4: JEPASoulBridge Full Pipeline")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp(prefix="bridge_test_")
    try:
        engine = SoulMemoryEngine(tmpdir)
        engine.write("beethoven", "knowledge", {"topic": "key_analysis", "text": "C minor is the key of fate and struggle in Beethoven's work"})
        engine.write("beethoven", "knowledge", {"topic": "form", "text": "Sonata form: exposition, development, recapitulation"})
        engine.write("strategy", "knowledge", {"topic": "momentum", "text": "Momentum indicators: RSI, MACD, Stochastic"})
        engine.write("strategy", "knowledge", {"topic": "volatility", "text": "Volatility clustering in financial time series"})

        bridge = JEPASoulBridge(memory_dir=tmpdir, llm_backend=CloudBackend(api_key="test-key"))

        mock_audio_jepa = MockJEPA(num_slots=7, slot_dim=128)
        mock_fin_jepa = MockJEPA(num_slots=6, slot_dim=128)
        mock_projector = MockProjector(output_dim=256)

        bridge.register_jepa(JEPAModality.AUDIO, mock_audio_jepa, mock_projector)
        bridge.register_jepa(JEPAModality.FINANCIAL, mock_fin_jepa, mock_projector)

        audio_input = torch.randn(1, 4, 7, 256)
        result_audio = bridge.process(
            raw_input=audio_input,
            modality=JEPAModality.AUDIO,
            soul="beethoven",
            query="What key is this in?",
            use_llm=False,
        )

        print(f"  Audio result: {len(result_audio.slot_descriptions)} slots, "
              f"{len(result_audio.knowledge_context)} knowledge entries, "
              f"top_slot='{result_audio.slot_descriptions[0].name}' "
              f"(act={result_audio.slot_descriptions[0].activation:.2f})")
        assert len(result_audio.slot_descriptions) > 0, "No slot descriptions"
        assert result_audio.modality == JEPAModality.AUDIO

        fin_input = torch.randn(1, 8, 6, 256)
        result_fin = bridge.process(
            raw_input=fin_input,
            modality=JEPAModality.FINANCIAL,
            soul="strategy",
            query="What's the market regime?",
            use_llm=False,
        )

        print(f"  Financial result: {len(result_fin.slot_descriptions)} slots, "
              f"{len(result_fin.knowledge_context)} knowledge entries, "
              f"top_slot='{result_fin.slot_descriptions[0].name}'")
        assert len(result_fin.slot_descriptions) > 0, "No slot descriptions"
        assert result_fin.modality == JEPAModality.FINANCIAL

        print("  PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_counterfactual():
    print("\n" + "=" * 60)
    print("TEST 5: Counterfactual Reasoning")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp(prefix="bridge_test_")
    try:
        engine = SoulMemoryEngine(tmpdir)
        engine.write("beethoven", "knowledge", {"topic": "tempo", "text": "Changing tempo affects emotional intensity"})

        bridge = JEPASoulBridge(memory_dir=tmpdir, llm_backend=CloudBackend(api_key="test-key"))

        mock_jepa = MockJEPA(num_slots=7, slot_dim=128)
        mock_projector = MockProjector(output_dim=256)
        bridge.register_jepa(JEPAModality.AUDIO, mock_jepa, mock_projector)

        audio_input = torch.randn(1, 4, 7, 256)
        intervene_embedding = torch.randn(1, 4, 128) * 2.0

        result = bridge.counterfactual(
            raw_input=audio_input,
            modality=JEPAModality.AUDIO,
            intervene_slot_id=2,
            intervene_embedding=intervene_embedding,
            soul="beethoven",
            query="What if we change the vocals?",
        )

        print(f"  Counterfactual result: intervened slot 2 (vocals), "
              f"{len(result.slot_descriptions)} slots predicted, "
              f"metadata={result.metadata}")
        assert result.metadata.get("counterfactual") == True
        assert result.metadata.get("intervened_slot") == 2

        print("  PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_modality_soul_mapping():
    print("\n" + "=" * 60)
    print("TEST 6: 14 Souls → 10 JEPA Modalities Mapping")
    print("=" * 60)

    all_souls = set()
    for modality, souls in MODALITY_SOUL_MAP.items():
        for s in souls:
            all_souls.add(s)
        print(f"  {modality.value:12s} → {', '.join(souls)}")

    print(f"\n  Total unique souls mapped: {len(all_souls)}")
    print(f"  Total modalities: {len(MODALITY_SOUL_MAP)}")

    expected_souls = {"beethoven", "vangogh", "monet", "strategy", "einstein",
                      "cezanne", "galileo", "davinci", "darwin", "yuanlongping",
                      "humboldt", "montesquieu"}
    missing = expected_souls - all_souls
    if missing:
        print(f"  WARNING: Souls not mapped: {missing}")

    assert len(MODALITY_SOUL_MAP) == 10, f"Expected 10 modalities, got {len(MODALITY_SOUL_MAP)}"
    print("  PASS")


def test_slot_names_coverage():
    print("\n" + "=" * 60)
    print("TEST 7: Slot Names Coverage for All Modalities")
    print("=" * 60)

    for modality, names in MODALITY_SLOT_NAMES.items():
        print(f"  {modality.value:12s}: {len(names)} slots — {', '.join(names[:3])}...")

    all_modalities = set(JEPAModality)
    covered = set(MODALITY_SLOT_NAMES.keys())
    missing = all_modalities - covered
    if missing:
        print(f"  WARNING: Modalities without slot names: {[m.value for m in missing]}")
    else:
        print(f"  All {len(all_modalities)} modalities have slot names defined")

    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("JEPA-Soul Bridge CPU Integration Test Suite")
    print("=" * 60)

    tests = [
        ("SlotToTextDecoder", test_slot_decoder),
        ("SoulMemoryEngine", test_soul_memory),
        ("LLMBackend", test_llm_backends),
        ("Full Pipeline", test_bridge_full_pipeline),
        ("Counterfactual", test_counterfactual),
        ("Soul Mapping", test_modality_soul_mapping),
        ("Slot Names", test_slot_names_coverage),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)
