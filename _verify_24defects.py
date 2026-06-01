import sys
import traceback

passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  PASS  {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()
        failed += 1

print("=" * 60)
print("VORTEX FLAME — 24-Defect Iteration Verification")
print("=" * 60)

# P0-1: Three-stage execution engine
print("\n[P0-1] Three-stage execution engine")
def test_team_pipeline():
    from soul_orchestrator import run_team_pipeline
    result = run_team_pipeline("Write a sorting algorithm")
    assert result["status"] == "completed", f"Expected completed, got {result['status']}"
    assert len(result["results"]) >= 1, f"Expected results, got {len(result.get('results', []))}"
test("team_pipeline executes stages", test_team_pipeline)

def test_ultrapilot():
    from soul_orchestrator import run_ultrapilot
    result = run_ultrapilot("Design a REST API and implement it")
    assert result["status"] == "completed", f"Expected completed, got {result['status']}"
    assert "parallel_results" in result, "Missing parallel_results"
test("ultrapilot parallel execution", test_ultrapilot)

def test_ralph_loop():
    from soul_orchestrator import run_ralph_loop
    result = run_ralph_loop("Fix the bug in parser.py", max_iterations=2)
    assert result["status"] in ("completed", "max_iterations"), f"Unexpected status: {result['status']}"
test("ralph verify-fix loop", test_ralph_loop)

# P0-2: merge_subagent_results
print("\n[P0-2] merge_subagent_results")
def test_merge():
    from soul_orchestrator import merge_subagent_results
    results = [
        {"soul": "cezanne", "stage": "code", "output": "def sort(arr): pass"},
        {"soul": "davinci", "stage": "code", "output": "def sort(arr): return sorted(arr)"},
    ]
    merged = merge_subagent_results(results, conflict_strategy="last_wins")
    assert merged["status"] == "merged", f"Expected merged, got {merged['status']}"
    assert merged["n_results"] == 2
test("merge with last_wins strategy", test_merge)

def test_merge_empty():
    from soul_orchestrator import merge_subagent_results
    merged = merge_subagent_results([])
    assert merged["status"] == "no_results"
test("merge empty results", test_merge_empty)

# P0-4: MCP write locks
print("\n[P0-4] MCP per-soul write locks")
def test_write_locks():
    from soul_memory import SoulMemoryEngine
    engine = SoulMemoryEngine()
    assert hasattr(engine, '_write_locks'), "Missing _write_locks"
    lock = engine._get_write_lock("cezanne")
    import threading
    assert hasattr(lock, 'acquire') and hasattr(lock, 'release'), f"Expected Lock-like, got {type(lock)}"
    lock2 = engine._get_write_lock("cezanne")
    assert lock is lock2, "Same soul should return same lock"
test("per-soul write locks exist and are unique", test_write_locks)

def test_concurrent_writes():
    from soul_memory import SoulMemoryEngine
    import threading
    engine = SoulMemoryEngine()
    errors = []
    def write_fn(i):
        try:
            engine.write(f"concurrent_test_{i}", "domain_memory", {"topic": f"concurrent_{i}"}, importance=0.5)
        except Exception as e:
            errors.append(e)
    threads = [threading.Thread(target=write_fn, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(errors) == 0, f"Concurrent writes failed: {errors}"
test("concurrent writes no race condition", test_concurrent_writes)

# P1-5: Knowledge base checkpoint/resume
print("\n[P1-5] Knowledge base checkpoint/resume")
def test_checkpoint():
    from soul_knowledge_alignment import _load_checkpoint, _save_checkpoint, _CHECKPOINT_DIR
    _save_checkpoint("test_soul", {"total_indexed": 42, "offsets": {"arxiv": 100}})
    cp = _load_checkpoint("test_soul")
    assert cp["total_indexed"] == 42, f"Expected 42, got {cp.get('total_indexed')}"
    assert cp["offsets"]["arxiv"] == 100
    import os
    os.remove(str(_CHECKPOINT_DIR / "test_soul_alignment.json"))
test("checkpoint save/load round-trip", test_checkpoint)

# P1-8: Routing history closed loop
print("\n[P1-8] Routing history closed loop")
def test_dispatch_log():
    from soul_orchestrator import _log_dispatch, _load_dispatch_stats, _DISPATCH_LOG_PATH
    _log_dispatch({"mode": "team", "top1_confidence": 0.85, "gap": 0.3})
    _log_dispatch({"mode": "ultrapilot", "top1_confidence": 0.6, "gap": 0.1})
    stats = _load_dispatch_stats()
    assert stats["total"] >= 2, f"Expected >=2, got {stats['total']}"
test("dispatch log records and loads", test_dispatch_log)

def test_adaptive_threshold():
    from soul_orchestrator import _adaptive_gap_threshold, SEMANTIC_ROUTING_CONFIG
    threshold = _adaptive_gap_threshold()
    assert threshold > 0, f"Threshold should be positive, got {threshold}"
    base = SEMANTIC_ROUTING_CONFIG["ultrapilot_gap_threshold"]
    assert 0.5 * base <= threshold <= 2.0 * base, f"Threshold out of range: {threshold}"
test("adaptive gap threshold adjusts", test_adaptive_threshold)

# P1-9: Security implementation
print("\n[P1-9] Security implementation")
def test_guardian():
    from guardian import Guardian
    g = Guardian()
    result = g.start()
    assert result["status"] == "started"
    sec = g.check_security()
    assert "debug_processes" in sec
    assert "file_integrity" in sec
test("Guardian starts and checks security", test_guardian)

def test_soul_action_whitelist():
    from guardian import Guardian
    g = Guardian()
    r1 = g.check_soul_action("cezanne", "write_file")
    assert r1["allowed"] is True, f"cezanne should be allowed to write_file"
    r2 = g.check_soul_action("vangogh", "execute_code")
    assert r2["allowed"] is False, f"vangogh should NOT be allowed to execute_code"
test("Soul action whitelists enforce", test_soul_action_whitelist)

# P1-10: Skill evolution closed loop
print("\n[P1-10] Skill evolution closed loop")
def test_contrastive_update():
    from skill_evolver import SkillEvolver
    from skill_registry import SkillRegistry, RegisteredSkill, SkillStatus, SkillSource
    reg = SkillRegistry()
    skill = RegisteredSkill(
        skill_id="test_skill", name="Test", description="test skill",
        source=SkillSource.CUSTOM, status=SkillStatus.ACTIVE,
    )
    reg._skills["test_skill"] = skill
    evolver = SkillEvolver(reg)
    trajectory = {
        "successes": [{"pattern": "clean input"}],
        "failures": [{"pattern": "malformed json"}, {"pattern": "timeout"}],
    }
    result = evolver._contrastive_update(skill, trajectory)
    assert "patches" in result, "Missing patches in contrastive update"
    assert len(result["patches"]) > 0, "Should detect failure patterns"
test("contrastive update generates patches", test_contrastive_update)

# P1-11: SQLite auto backup
print("\n[P1-11] SQLite auto backup")
def test_auto_backup():
    from soul_memory import SoulMemoryEngine
    from pathlib import Path
    engine = SoulMemoryEngine()
    engine.write("backup_test", "domain_memory", {"topic": "backup_check"}, importance=0.5)
    backup_dir = engine.memory_dir / "backups"
    assert backup_dir.exists(), "Backup directory should exist"
test("auto backup creates backup dir", test_auto_backup)

# P1-13: Cross-soul multi-hop
print("\n[P1-13] Cross-soul multi-hop reasoning")
def test_multi_hop():
    from soul_memory import cross_soul_multi_hop
    result = cross_soul_multi_hop("cezanne", "sorting algorithm", max_hops=2, top_k=2)
    assert "hops" in result, "Missing hops in result"
    assert result["status"] in ("ok", "no_permissions"), f"Unexpected status: {result['status']}"
test("multi-hop cross-soul reasoning", test_multi_hop)

# P1-14: Skill Errata rollback
print("\n[P1-14] Skill Errata version rollback")
def test_errata_rollback():
    from skill_evolver import SkillEvolver
    from skill_registry import SkillRegistry, RegisteredSkill, SkillStatus, SkillSource
    reg = SkillRegistry()
    skill = RegisteredSkill(
        skill_id="rollback_test", name="Rollback", description="test",
        source=SkillSource.CUSTOM, status=SkillStatus.ACTIVE,
        errata=[{"dimension": "coverage_gap", "fix": "Add rule X"}],
    )
    reg._skills["rollback_test"] = skill
    evolver = SkillEvolver(reg)
    evolver._snapshot_errata("rollback_test", skill.errata)
    skill.errata.append({"dimension": "ambiguity", "fix": "Clarify Y"})
    evolver._snapshot_errata("rollback_test", skill.errata)
    result = evolver.rollback_errata("rollback_test", target_version=1)
    assert result["status"] == "rolled_back", f"Expected rolled_back, got {result['status']}"
    assert len(skill.errata) == 1, f"Expected 1 errata after rollback, got {len(skill.errata)}"
test("errata rollback to version 1", test_errata_rollback)

# P1-15: Arbitration provenance
print("\n[P1-15] Arbitration provenance")
def test_arbitration_provenance():
    from soul_orchestrator import arbitrate_results
    results = [
        {"soul": "cezanne", "output": "def foo(): pass"},
        {"soul": "davinci", "output": "def bar(): return 1"},
    ]
    arb = arbitrate_results(results, task_description="write code", method="confidence")
    assert "provenance" in arb, "Missing provenance in arbitration result"
    prov = arb["provenance"]
    assert "souls_involved" in prov, "Missing souls_involved"
    assert "reason" in prov, "Missing reason"
    assert len(prov["souls_involved"]) == 2
test("arbitration includes provenance", test_arbitration_provenance)

print("\n" + "=" * 60)
print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)
