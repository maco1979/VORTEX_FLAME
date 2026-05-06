#!/usr/bin/env python3
import sys; sys.stdout.reconfigure(encoding='utf-8', errors='replace')
f=open(r'C:\Users\42235\Desktop\开发，训练重要资料\VORTEX_FLAME_项目记忆_v2_补充.txt','r',encoding='utf-8')
c=f.read(); f.close()

old = """  步骤4: 7B Stage3训练 (等8B释放GPU后训) ← 数据准备中

  7B/8B Stage3 数据最终对齐清单 (2026-05-03, 去重bug已修复):
    Stage1 (数学): 7B=229911  8B=229911 [OK]
    Stage2 (逻辑): 7B=3338    8B=3338   [OK]
    Stage3a (代码): 7B=7963   8B=7963   [OK]
    Stage3b (补充): 7B=1456   8B=1456   [完全对齐]
      DebugSupplement:  7B=12  8B=12
      LogicSupplement:  7B=11  8B=11
      MathSupplement:   7B=11  8B=11
      SortSupplement:   7B=2   8B=2
      GraphSupplement:  7B=1   8B=1
    修复: 去重指纹instruction[:50] -> instruction[:30]+input[:50]
          之前模板相同但INPUT不同的题被误判重复,7B丢了27条补充
          现在7B和8B共用37条精准补充, 1234阶段数据完全对标

  7B Stage3执行计划:
    [已完成] 拆分+补充+去重修复, 7B/8B全阶段数据对齐
    [等GPU] 跑7B Stage1+Stage2回测, 找7B特有短板
    [GPU释放后] 若有新短板补数据 -> 启动7B Stage3a
    原因: 7B和8B底座不同, 回测见真章不做预设"""

new = """  步骤4: 7B Stage3训练 — S3a训练中, S3b数据已就绪

  7B/8B Stage3 数据最终对齐清单 (2026-05-03):
    Stage1 (数学): 7B=229911  8B=229911 [OK]
    Stage2 (逻辑): 7B=3338    8B=3338   [OK]
    Stage3a (代码): 7B=7963   8B=7963   [OK] 7B训练中
    Stage3b (补充): 7B=1538   8B=1456   [7B多82条] 含70条Stage4预备
      FormalProof:   7B=20   8B=0    ← 形式证明链(真值表/自然演绎/Hoare)
      KeywordPatch:  7B=50   8B=0    ← 关键词强制(infinite/shortest/rate/prepared)
      DebugSupplement: 7B=20  8B=12
      LogicSupplement: 7B=13  8B=11
      MathSupplement:  7B=13  8B=11
    设计: Stage3a学代码→Stage3b学证明+关键词→Stage4无缝衔接
    预期: logic 25%→40-50%, Dijkstra 80%→100%, 导数80%→100%, 安全80%→100%

  7B Stage3执行计划:
    [✅] 拆分+补充+去重修复, 数据全对齐
    [✅] S1+S2双回测, 找出7B特有短板(debug=0%致命)
    [✅] S3b加了8条Debug+12条再次补充, 现DebugSupplement=20
    [✅] S3b加70条FormalProof+KeywordPatch, 目标反超8B
    [⏳] S3a训练中(7963条CodeAlpaca)→S3b接力(1538条)
    [ ] S3完成→回测→Arena双向迭代→Stage4"""

if old in c:
    c = c.replace(old, new)
    f=open(r'C:\Users\42235\Desktop\开发，训练重要资料\VORTEX_FLAME_项目记忆_v2_补充.txt','w',encoding='utf-8')
    f.write(c); f.close()
    print('OK')
else:
    print('NOT FOUND - trying line-based match')
    # Find the block
    idx = c.find('步骤4: 7B Stage3训练 (等8B释放GPU后训)')
    if idx > 0:
        print(f'Found at {idx}')
        # Print surrounding context
        print(repr(c[idx:idx+100]))
