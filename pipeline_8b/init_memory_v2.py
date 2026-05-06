import os,sys
sys.path.insert(0, r"D:\VORTEX_FLAME")
os.chdir(r"D:\VORTEX_FLAME")

log = open(r"D:\VORTEX_FLAME\hermes_logs\mem_v2_init.log", "w", encoding="utf-8")
sys.stdout = log
sys.stderr = log

import long_memory as lm

print("[1/4] Init default memory...")
for soul in ["cezanne", "einstein", "global"]:
    added = lm.init_default_memory(soul)
    stats = lm.get_stats(soul)
    print(f"  {soul}: added={added}, total={stats['total']}")

print("\n[2/4] Index pipeline code...")
result = lm.index_directory(r"D:\VORTEX_FLAME\pipeline_8b", soul="cezanne")
print(f"  pipeline_8b: added={result['added']} chunks")

print("\n[3/4] Record training logs...")
lm.log_training("cezanne", "stage1", 0.5658, 1.0, 60, 10.0, 229911)
lm.log_training("cezanne", "stage2", 0.5148, 1.0, 145, 11.9, 3338)
lm.log_training("cezanne", "stage3a", 0.7793, 1.0, 201, 11.9, 7963)
lm.log_benchmark("cezanne", "S1", 0.76, 1.0, "Math baseline")
lm.log_benchmark("cezanne", "S2", 0.74, 1.0, "Debug degraded 40->20%")
lm.log_benchmark("cezanne", "S3", 0.82, 1.0, "Debug to 80%, Sort/Derivative fixed")
print("  Done")

print("\n[4/4] Final stats...")
for soul in ["cezanne", "einstein", "global"]:
    stats = lm.get_stats(soul)
    print(f"  {soul}: {stats['total']} entries: {dict(stats['categories'])}")

print("\nDone!")
log.close()
