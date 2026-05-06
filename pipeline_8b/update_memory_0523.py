#!/usr/bin/env python3
"""Update memory with all Session 2026-05-03 findings"""
import sys; sys.stdout.reconfigure(encoding='utf-8', errors='replace')

f = open(r'C:\Users\42235\Desktop\开发，训练重要资料\VORTEX_FLAME_项目记忆_v2_补充.txt', 'r', encoding='utf-8')
c = f.read(); f.close()

section = """

  二十、2026-05-03 全记录：7B S3分析/退步/Stage4急救路线

  【7B Stage3a (CodeAlpaca 7963条)】
  内容: 7963条CodeAlpaca通用编程题(反转整数/词频统计/字典操作/排序/Java)
  Debug相关330条(4.1%), 安全51条(0.6%), CS底层0条
  Result: Loss=0.489, 190min, VRAM=8.0GB

  【7B Stage3b (1538条补充)】
  内容: MathReview506 + LogicCode454 + LogicReview317 + Security52
        + LowLevel46 + Debug55 + FormalProof20 + KeywordPatch50
        + LogicSupplement13 + MathSupplement13 + Agent9 + Sort2 + Graph1
  Result: Loss=0.303, 37min, VRAM=8.0GB
  BUT: 回测S3=67%, 比S2=74%退步7pp!
  根因: Mistral-7B-Instruct容量不足, 1538条过载导致灾难性遗忘
  丢失: graph(shortest), math_even(derivation), security(prepared)

  【7B Stage4 v1 失败】
  从S3b出发: S3b(67%)+74条=Loss 1.66 → 废
  策略调整: 退回S2(74%最佳状态)+74条关键词急救

  【7B Stage4 v2 清洁版 (训练中)】
  底座: Mistral-7B-Instruct-v0.1-Cezanne (原始不动)
  S2备份: stage4/base_s2 (S2=74% LoRA副本)
  数据: 74条 = 50条关键词(shortest/infinite/prepared/rate/limit)
        + 12条Debug教学 + 12条关键词补丁
  配置: 1轮, 37步, LR=1e-4, 预计2mins

  【8B Stage4 待训】
  数据: 84条 = 40条关键词急救 + 20条形式证明链 + 24条CS底层
        CS底层: 编译原理/OS进程/虚存/Hoare逻辑/TCP拥塞/词法分析
  策略: 7B跑完→8B接, 8B容量大可吞CS知识

  【Arena对抗】
  尝试3次均失败(OOM/AutoModel不兼容), 改串行+GPU释放轮询检查
  结论: 跳过Arena, 用已知回测短板直接生成Stage4数据

  【工具链】
  Huashu Design: SKILL.md安装到.trae/skills/huashu-design/
    20种设计哲学×5流派, 可生成仪表盘/雷达图/PPT/动画
  DESIGN_FIRST_RULE: 写入.trae/rules/, HARD-GATE: 无设计不写代码
  HF镜像: HF_ENDPOINT=https://hf-mirror.com 已永久配置

  【关键发现】
  1. CodeAlpaca 7963条全是通用题, 缺CS底层和系统调试方法
  2. 7B S3b过载证明: Mistral-7B-Instruct最多承受~3K条精准数据
  3. S2(74%)是7B最佳状态, S3a+S3b总计9.5K条严重过载
  4. 7B的debug=0%需8B教学(8B debug=80%)
  5. 7B和8B底座差异: 7B是通用, 8B是推理专精, 容量差一倍

  【知识图谱审计】
  Cezanne绑定: 嵌入式系统+物联网与工控
  汽车电子绑给DaVinci, 建议加Cezanne交叉(ECU/CAN本质是嵌入式)
  14灵魂基础科学全覆盖, 94%学科无遗漏
"""

end = 'END OF MEMORY SUPPLEMENT v2.0'
if end in c:
    c = c.replace(end, section + '\n' + end)
    f = open(r'C:\Users\42235\Desktop\开发，训练重要资料\VORTEX_FLAME_项目记忆_v2_补充.txt', 'w', encoding='utf-8')
    f.write(c); f.close()
    print('OK: Section 二十 added')
else:
    print('NOT FOUND')
