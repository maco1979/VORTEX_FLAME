# VORTEX FLAME — 听觉预测优先理论

理论基础目录。包含 UPP（单模态物理先验）假设的完整论述、生物听觉原始模型、以及 JEPA 框架下的实验验证路径。

## 文件

| 文件 | 语言 | 内容 |
|------|------|------|
| [AUDITORY_PREDICTION_PRIMACY_EN.md](AUDITORY_PREDICTION_PRIMACY_EN.md) | English | 完整学术论文：UPP假设、Slot Attention、51类实验、核心公式 |
| [AUDITORY_PREDICTION_PRIMACY_CN.md](AUDITORY_PREDICTION_PRIMACY_CN.md) | 中文 | 同上中文版，包含原文中未展开的生物学细节 |

## 核心概念

- **JEPA**: Joint Embedding Predictive Architecture（联合嵌入预测架构），以预测替代重构的自监督范式
- **UPP假设**: Unimodal Physical Prior — 跨模态对齐质量的上限由单模态物理建模的深度决定
- **SIGReg**: 简化不变高斯正则化，2项损失替代4项损失，训练更稳定
- **CAJEPA**: Causal Audio JEPA — 带 Slot Attention 的声学世界模型
- **51类分类**: ESC-50(50类环境声) + 音乐(1类锚定)，验证单模态预训练的物理知识迁移

## 关键引用

```
耳朵不是传声器。它是戴着滤波器组的预测引擎。
The ear is not a microphone. It is a prediction engine wearing a filterbank.
```

```
跨模态对齐无法弥补单模态物理建模的缺陷。
Cross-modal alignment cannot compensate for deficits in unimodal physical modeling.
```
