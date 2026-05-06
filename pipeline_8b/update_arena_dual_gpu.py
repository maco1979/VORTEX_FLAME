#!/usr/bin/env python3
import sys; sys.stdout.reconfigure(encoding='utf-8', errors='replace')
f=open(r'C:\Users\42235\Desktop\开发，训练重要资料\VORTEX_FLAME_项目记忆_v2_补充.txt','r',encoding='utf-8')
c=f.read(); f.close()

old = '  方案A: 串行加载(先7B跑完,释放,再8B跑)'

new = '''  方案A(优化版): 双卡串行 — 3060加载8B + 1060加载7B
    3060(12GB): 加载8B+LoRA ~8GB, 主训卡
    1060(6GB): 加载7B+LoRA ~4.5GB, 推理专用
    串行执行: 先7B在1060跑完→结果存在内存→3060加载8B跑
    优势: 省一次3-5分钟模型重载时间
    1060绝不做训练, 只推理, 零污染风险'''

if old in c:
    c = c.replace(old, new)
    f=open(r'C:\Users\42235\Desktop\开发，训练重要资料\VORTEX_FLAME_项目记忆_v2_补充.txt','w',encoding='utf-8')
    f.write(c); f.close()
    print('OK')
else:
    print('NOT FOUND, trying broader')
    old2='当前脚本用方案A(串行,每次只加载一个模型)'
    if old2 in c:
        c=c.replace(old2, new)
        f=open(r'C:\Users\42235\Desktop\开发，训练重要资料\VORTEX_FLAME_项目记忆_v2_补充.txt','w',encoding='utf-8')
        f.write(c); f.close()
        print('OK (broader)')
    else:
        print('NOT FOUND either')
