#!/usr/bin/env python3
import sys
sys.path.insert(0, "/mnt/d/VORTEX_FLAME")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from long_memory import build_knowledge_index, get_stats

soul = "global"
try:
    count = build_knowledge_index(soul)
    stats = get_stats(soul)
    print(f"global: index={count}, knowledge_total={stats.get('knowledge_total',0)}, main_total={stats.get('total',0)}")
except Exception as e:
    print(f"global: FAILED - {e}")
    import traceback
    traceback.print_exc()
