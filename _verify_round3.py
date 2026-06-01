"""Verify 6 new defect fixes from Doubao's round-3 review."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  ✅ {name}")
        passed += 1
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        failed += 1


print("=" * 60)
print("VORTEX FLAME — Round-3 Defect Verification (6 items)")
print("=" * 60)


print("\n[P0-4] Knowledge base per-soul write isolation")
def test_kb_locks():
    from soul_knowledge_alignment import _get_kb_lock, _KB_LOCKS
    lock1 = _get_kb_lock("cezanne")
    lock2 = _get_kb_lock("einstein")
    assert hasattr(lock1, 'acquire') and hasattr(lock1, 'release'), "Expected Lock-like"
    assert lock1 is not lock2, "Different souls should have different locks"
    lock3 = _get_kb_lock("cezanne")
    assert lock1 is lock3, "Same soul should return same lock"
test("per-soul KB write locks", test_kb_locks)


print("\n[P1-7] Long-period dispatch statistics with trend detection")
def test_dispatch_stats():
    from soul_orchestrator import _load_dispatch_stats, _adaptive_gap_threshold
    stats = _load_dispatch_stats()
    assert "trend" in stats, "Missing trend field"
    assert "long_term_ultra_ratio" in stats, "Missing long_term_ultra_ratio"
    assert "short_term_ultra_ratio" in stats, "Missing short_term_ultra_ratio"
    assert stats["trend"] in ("stable", "rising", "declining"), f"Invalid trend: {stats['trend']}"
    threshold = _adaptive_gap_threshold()
    assert 0.05 < threshold < 0.5, f"Threshold out of range: {threshold}"
test("dispatch stats include trend + long/short term ratios", test_dispatch_stats)


print("\n[P1-8] Dynamic importance recomputation")
def test_recompute_importance():
    from soul_memory import SoulMemoryEngine
    engine = SoulMemoryEngine()
    assert hasattr(engine, 'recompute_importance'), "Missing recompute_importance method"
    result = engine.recompute_importance("cezanne", category="knowledge")
    assert isinstance(result, int), f"Expected int, got {type(result)}"
test("recompute_importance exists and returns int", test_recompute_importance)


print("\n[P1-9a] Knowledge base snapshot + rollback")
def test_kb_snapshot_rollback():
    from soul_knowledge_alignment import snapshot_knowledge_base, rollback_knowledge_base
    assert callable(snapshot_knowledge_base), "snapshot_knowledge_base not callable"
    assert callable(rollback_knowledge_base), "rollback_knowledge_base not callable"
    result = snapshot_knowledge_base("cezanne", label="test_verify")
    assert result["status"] in ("snapshot_created", "empty"), f"Unexpected status: {result['status']}"
test("snapshot_knowledge_base callable", test_kb_snapshot_rollback)


print("\n[P1-9b] Incremental update")
def test_incremental_update():
    from soul_knowledge_alignment import incremental_update
    assert callable(incremental_update), "incremental_update not callable"
    result = incremental_update("cezanne", [
        {"topic": "test_incremental_topic_001", "summary": "test"},
    ], category="domain_memory")
    assert result["status"] == "ok", f"Unexpected status: {result['status']}"
    assert "added" in result, "Missing added count"
test("incremental_update callable and returns ok", test_incremental_update)


print("\n[P1-10] Cross-soul multi-hop cycle protection")
def test_multi_hop_cycle_protection():
    from soul_memory import cross_soul_multi_hop
    import inspect
    sig = inspect.signature(cross_soul_multi_hop)
    assert "max_souls_visited" in sig.parameters, "Missing max_souls_visited parameter"
test("cross_soul_multi_hop has max_souls_visited param", test_multi_hop_cycle_protection)


print("\n[P1-11a] Auto-rollback on audit failure")
def test_auto_rollback():
    from skill_evolver import SkillEvolver
    evolver = SkillEvolver()
    assert hasattr(evolver, 'auto_rollback_on_failure'), "Missing auto_rollback_on_failure"
    result = evolver.auto_rollback_on_failure("nonexistent_skill")
    assert result["status"] == "skill_not_found", f"Expected skill_not_found, got {result['status']}"
test("auto_rollback_on_failure exists and handles missing skill", test_auto_rollback)


print("\n[P1-11b] Canary deploy / promote / rollback")
def test_canary():
    from skill_evolver import SkillEvolver
    evolver = SkillEvolver()
    assert hasattr(evolver, 'canary_deploy'), "Missing canary_deploy"
    assert hasattr(evolver, 'canary_promote'), "Missing canary_promote"
    assert hasattr(evolver, 'canary_rollback'), "Missing canary_rollback"
    result = evolver.canary_deploy("nonexistent_skill")
    assert result["status"] == "skill_not_found", f"Expected skill_not_found, got {result['status']}"
    result = evolver.canary_promote("nonexistent_skill")
    assert result["status"] == "no_active_canary", f"Expected no_active_canary, got {result['status']}"
    result = evolver.canary_rollback("nonexistent_skill")
    assert result["status"] == "no_active_canary", f"Expected no_active_canary, got {result['status']}"
test("canary deploy/promote/rollback exist and handle missing skill", test_canary)


print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
print("=" * 60)

if failed > 0:
    sys.exit(1)
