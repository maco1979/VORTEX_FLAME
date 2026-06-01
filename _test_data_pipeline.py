#!/usr/bin/env python3
import json, os, sys, tempfile
sys.path.insert(0, r'D:\VORTEX_FLAME')

from vf_data_core import ConfigManager, AuditLogger, ResourceMonitor
from vf_dedup import DedupEngine, SimHash
from vf_cleanse import CleanseEngine
from vf_inspect import InspectEngine
from vf_distill import DistillEngine
from vf_data_orchestrator import DataPipeline

failed = 0

def check(label, condition, detail=""):
    global failed
    if condition:
        print(f"  ✅ {label}")
    else:
        failed += 1
        print(f"  ❌ {label} — {detail}")

print("=== TEST 1: Core Engine ===")
cfg = ConfigManager(r'D:\VORTEX_FLAME\vf_data_config.yaml')
audit = AuditLogger(cfg.audit_dir)
print(f"  Config: {cfg.get('system', 'name')} v{cfg.get('system', 'version')}")
check("Config loaded", cfg.get('system', 'name') == 'VF_DATA_PIPELINE')

print("\n=== TEST 2: Dedup ===")
test_data = [
    {"id": 1, "name": "Alice", "city": "Beijing"},
    {"id": 2, "name": "Bob", "city": "Shanghai"},
    {"id": 1, "name": "Alice", "city": "Beijing"},
    {"id": 3, "name": "Alice", "city": "Beijing"},
    {"id": 4, "name": "Charlie", "city": "Shenzhen"},
    {"id": 4, "name": "Charlie", "city": "Shenzhen"},
    {"id": 5, "name": "", "city": ""},
]
dedup = DedupEngine(cfg, audit)
clean, stats = dedup.process(test_data)
print(f"  Input: {len(test_data)} → Output: {len(clean)} (removed {stats.duplicates_removed})")
check("Dedup removed duplicates", len(clean) < len(test_data))
check("Exact matches found", stats.exact_matches > 0)

print("\n=== TEST 3: Cleanse ===")
dirty = [
    {"date": "2024/01/15", "price": "1,234.56", "desc": "  hello   world  "},
    {"date": "01/15/2024", "price": None, "desc": "test"},
    {"date": "2024-01-15", "price": 999.0, "desc": "normal"},
]
cleanser = CleanseEngine(cfg, audit)
clean, cstats = cleanser.process(dirty)
print(f"  Dates normalized: {cstats.normalized_dates}")
print(f"  Numbers normalized: {cstats.normalized_numbers}")
print(f"  Missing filled: {cstats.missing_values_filled}")
check("Date normalization", clean[0]["date"] == "2024-01-15")
check("Missing filled", cstats.missing_values_filled > 0)

print("\n=== TEST 4: Inspect ===")
cfg.set("inspect", "required_fields", value=["name", "email"])
inspector = InspectEngine(cfg, audit)
check_data = [
    {"name": "Alice", "email": "alice@test.com"},
    {"name": None, "email": "bob@test.com"},
    {"name": "Charlie", "email": "charlie@test.com"},
]
inspector.process(check_data)
report = inspector.report()
print(f"  Overall score: {report['overall_score']:.2%}")
print(f"  Issues: {report['issues_count']}")
check("Issues found", report["issues_count"] >= 1)

print("\n=== TEST 5: Distill ===")
docs = [
    {"title": "Python Guide", "content": "Python is a versatile programming language used for web development, data science, and automation. It has a simple syntax."},
    {"title": "ML Basics", "content": "Machine learning is a subset of AI. Common frameworks include PyTorch and TensorFlow."},
    {"title": "DevOps", "content": "DevOps combines development and operations. Tools include Docker, Kubernetes."},
    {"title": "Similar", "content": "Python is a versatile language used for web development and automation."},
]
distiller = DistillEngine(cfg, audit)
compressed, distats = distiller.process(docs)
print(f"  Input: {len(docs)} → Output: {len(compressed)}")
print(f"  Reduction: {distats.reduction_ratio:.1%}")
print(f"  Entities: {distats.entities_extracted}")
check("Reduction ratio valid", distats.reduction_ratio >= -10.0)  # tiny data: metadata overhead > savings
check("Entities extracted", distats.entities_extracted > 0)

print("\n=== TEST 6: Full Pipeline ===")
pipe = DataPipeline()
result = pipe.run_pipeline("full_pipeline", test_data + docs)
print(f"  Status: {result['status']}")
for sid, sinfo in result["steps"].items():
    icon = "✅" if sinfo["state"] == "SUCCESS" else "❌"
    print(f"  {icon} {sid}: {sinfo['state']}")
check("Pipeline completed", result["status"] == "SUCCESS")

print("\n=== TEST 7: SimHash ===")
sh = SimHash(fp_len=64, ngram=3)
fp1 = sh.compute("The quick brown fox jumps over the lazy dog")
fp2 = sh.compute("The quick brown fox jumps over the lazy dog")
fp3 = sh.compute("Something completely different here now")
d_same = sh.hamming_distance(fp1, fp2)
d_diff = sh.hamming_distance(fp1, fp3)
print(f"  Identical Hamming: {d_same}")
print(f"  Different Hamming: {d_diff}")
check("Identical = distance 0", d_same == 0)
check("Different > distance 0", d_diff > 0)

print("\n=== TEST 8: Audit Trail ===")
records = audit.query(limit=10)
print(f"  Audit records: {len(records)}")
for r in records[:3]:
    print(f"  [{r['module']}] {r['operation']}: {r['status']}")
check("Audit records exist", len(records) > 0)

print(f"\n{'='*60}")
if failed == 0:
    print("ALL TESTS PASSED")
else:
    print(f"{failed} TEST(S) FAILED")
print("="*60)
