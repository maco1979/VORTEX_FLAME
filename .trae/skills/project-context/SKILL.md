# VORTEX FLAME 项目上下文文档 v4.0

## 项目概述
宇宙魔方AI (Cosmic Cube AI) - 多灵魂AI系统，3×3×3=27灵魂架构
- 当前阶段：Batch1 = 9表层灵魂
- 通用模型：SoulModel (24L 768H 12heads, 178M params)
- 训练脚本：D:\VORTEX_FLAME\train_v81.py (v8.3 模型并行)
- 通用训练脚本：D:\VORTEX_FLAME\train_universal.py --soul {name}

## 9灵魂配置与训练状态
| 灵魂 | 名称 | 领域 | 数据量 | 状态 |
|------|------|------|--------|------|
| einstein | 爱因斯坦 | 量子物理与创新思维 | 1,228 | E95训练中 loss=22.6 |
| beethoven | 贝多芬 | 声学与语言创作 | - | 待训练 |
| cezanne | 塞尚 | 编程与逻辑 | - | 待训练 |
| davinci | 达芬奇 | 工程架构设计 | - | 待训练 |
| fkj | FKJ | 即兴音乐创作 | - | 待训练 |
| guizhu | 硅酌 | 对话与哲学 | - | 待训练 |
| monet | 莫奈 | 美学与创意 | - | 待训练 |
| strategy | 策略 | 博弈论与决策 | - | 待训练 |
| vangogh | 梵高 | 情绪美学创作 | - | 待训练 |

## 文件结构 (D:\VORTEX_FLAME)
```
D:\VORTEX_FLAME\
├── .trae/skills/               # 技能目录
├── checkpoints_v80/            # 爱因斯坦检查点(Pipeline格式)
├── soul_training_data/         # 灵魂训练数据(D盘)
│   └── einstein/               # 1,228条(853核心+375组合)
├── einstein_slice_training/    # BPE分词模型
│   └── einstein_bpe_200k.json  # 10000 vocab
├── train_v81.py                # v8.3 模型并行训练
├── train_universal.py          # 通用训练脚本(9灵魂)
├── expand_einstein_final.py    # 爱因斯坦数据生成(853条)
├── concat_einstein_v2.py       # 数据拼接(短->长序列)
├── ableton_mcp_remote_script/  # AbletonMCP Remote Script源码
├── UNIVERSAL_TRAINING_METHOD.md # 通用训练方法文档
└── v81_train_log.txt           # 训练日志
```

## F盘数据丢失说明
- F:\VORTEX_FLAME 目录已完全丢失
- 所有F盘训练数据和BPE模型已重建到D盘
- 爱因斯坦数据从10,088条降至1,228条(高质量重建)
- BPE分词器重新训练(10000 vocab, 兼容旧checkpoint)

## 课程学习5级体系
- 1a 基础初级: 核心概念定义 (Epoch 0-3)
- 1b 基础中级: 公式推导计算 (Epoch 4-7)
- 1c 基础高级: 理论框架证明 (Epoch 8-11)
- 2 关联深化: 跨学科联系 (Epoch 12-15)
- 3 行业知识: 产业应用前沿 (Epoch 16+)

## 分层冻结策略
- Phase 1 (E1-3): 冻结Soul层(0-11)，只训练Expert层(12-23)
- Phase 2 (E4+): 解冻Soul层，全模型训练178M参数

## 训练超参数 (v8.3)
- EPOCHS=150, BATCH_SIZE=6, GRAD_ACCUM=2, LR=5e-5
- WARMUP=2000, CosineAnnealingWarmRestarts T0=4000
- 混合精度AMP, LABEL_SMOOTHING=0.1
- MAX_SEQ_LEN=128, TARGET_LOSS=1.5, FREEZE_EPOCHS=3
- PIPELINE_SPLIT=18 (3060扛前18层, 1060扛后6层)

## GPU资源调度 (模型并行)
- GPU0 (RTX 3060 12GB): token_emb + pos_emb + layers[0-17] + norm_f = 72%参数
- GPU1 (GTX 1060 6GB): layers[18-23] + head = 27%参数
- 数据流: GPU0前18层 -> GPU1后6层 -> loss计算
- 训练速度: ~4000 tok/s (单卡1016 tok/s的3.9倍)
- 每epoch: 0.1-0.2分钟
- 训练顺序: einstein→beethoven→cezanne→davinci→fkj→guizhu→monet→strategy→vangogh

## 模型并行 vs DataParallel 对比
| 方案 | 速度 | GPU0负载 | GPU1负载 | 问题 |
|------|------|----------|----------|------|
| 单卡(3060) | 1016 tok/s | 100% | 0% | 1060闲置 |
| DataParallel | ~300 tok/s | 50%(等1060) | 50%(拖后腿) | 木桶效应 |
| 模型并行 | 4000 tok/s | 72%(主力) | 27%(辅助) | 最优方案 |

## Loss预测 (基于当前训练速度)
- 从E95(22.6)到随机基线(9.2): 约6-7个epoch
- 从随机基线到best(2.5): 约10个epoch
- 从best(2.5)到目标(1.5): 约15-20个epoch
- 预计总时间: 约30-40个epoch * 0.2min = 6-8分钟

## Ableton Live 12 集成
- 安装路径: C:\ProgramData\Ableton\Live 12 Suite\
- 版本: 12.3.7 (已解锁)
- 内置AI: musicai.dll + onnxruntime_abl.dll + DirectML.dll
- Remote Script: User Remote Scripts/AbletonMCP/ (已安装)
- ableton-mcp pip包: 1.0.4 (已安装)
- 音源库: F:\BaiduNetdiskDownload\原厂音源43G（待安装到Ableton）

## 关键技术决策
1. 模型并行替代DataParallel(解决木桶效应)
2. 3060扛72%计算(前18层), 1060扛27%(后6层+head)
3. Pipeline checkpoint格式: 分片保存(token_emb/main_layers/aux_layers/norm_f/head)
4. 兼容旧checkpoint: load_base_model_from_checkpoint自动转换格式
5. 每5个epoch保存一次checkpoint(避免保存卡住训练)
6. 取消权重共享(head和token_emb独立, 跨GPU不能共享)
7. 所有灵魂共用einstein_bpe_200k.json分词器(10000 vocab)
8. 训练锁文件.train_lock防止多实例冲突
9. SHA256去重保证数据唯一性
10. 禁止模板套用，所有数据基于真实学科知识

## 技能清单 (37个)
核心训练：soul-learning-system, mvp-cube-training, slice-training-engine, einstein-slice-training
灵魂技能：einstein-soul-skill, beethoven-soul-skill, fkj-soul-skill, vangogh-soul-skill, davinci-soul-skill, cezanne-soul-skill, strategy-soul-skill, monet-soul-skill, guizhu-soul-skill
音乐创作：suno-automation, suno-music-generator, sunoapi-sdk, original-music-training, fkj-audio-fine-tuning, audio-analysis-learning
系统架构：vortex-flame-architecture, model-slice, slice-origin-tracker, cosmic-cube-ai
工具技能：skill-creator, skill-self-generator, skill-orchestrator, self-learning, long-memory, persistent-agent, task-continuity-system
商业技能：blockchain-ledger, digital-wallet-manager, soul-code-generator, industry-block-mapper, industry-knowledge-graph
审查技能：quantum-reality-audit, scientific-rigor-enforcer, node-dispatcher-quantum-reasoning
决策技能：decision-navigation-system, multidimensional-thinking
工程技能：project-delivery-system, frontend-integration-bridge, mobile-mvp-deploy, system-auto-init-fix
知识技能：soul-knowledge-base, soul-skill-dispatcher
