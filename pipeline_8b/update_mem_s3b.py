#!/usr/bin/env python3
import sys; sys.stdout.reconfigure(encoding='utf-8', errors='replace')
f=open(r'C:\Users\42235\Desktop\开发，训练重要资料\VORTEX_FLAME_项目记忆_v2_补充.txt','r',encoding='utf-8')
c=f.read(); f.close()
old = '7B Stage3a: 7963条CodeAlpaca (已备好)'
new = ('7B Stage3a: 7963条CodeAlpaca (已备好)\n'
       '7B Stage3b: 1538条 = 1468原有 + 70条Stage4预备(FormalProof 20 + KeywordPatch 50)\n'
       '  FormalProof: 真值表/自然演绎/数学归纳/反证法/逆否/Hoare逻辑/集合论证明\n'
       '  KeywordPatch: shortest/infinite/rate/prepared 各10-15条强制关键词\n'
       '  设计: Stage3a学代码→Stage3b学证明+关键词→直接接Stage4\n'
       '  预期: 假言25%→40-50%, Dijkstra 80%→100%, 导数80%→100%, 安全80%→100%')
if old in c:
    c = c.replace(old, new)
    f=open(r'C:\Users\42235\Desktop\开发，训练重要资料\VORTEX_FLAME_项目记忆_v2_补充.txt','w',encoding='utf-8')
    f.write(c); f.close()
    print('OK')
else:
    print('NOT FOUND')
