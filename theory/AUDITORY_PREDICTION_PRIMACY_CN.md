# 听觉预测优先原理

## 为什么耳朵必须比眼睛先理解世界

**——论联合嵌入预测架构中单模态物理建模的逻辑优先性**

---

### 摘要

我们认为，联合嵌入预测架构（JEPA）中的跨模态对齐质量，从根本上受限于单模态预训练阶段所获得的表征深度。以音频为主要研究对象，我们论证：一个纯声学编码器——在使用时间预测目标、槽位对象解耦和对比微调进行训练后——能够独立习得编码物理属性的表征，包括距离、速度、空间遮挡和多声源分离。我们形式化一个核心原则：**对齐是对表征空间的拉力，不是知识的转移；其上限由每个模态从原始感官输入中独立建模物理规律的能力决定。**

---

### 1. 引言：U型听力曲线作为生物学先验

作者的听力特征呈现非典型的U型灵敏度曲线：3000 Hz以下超敏、3000–8000 Hz频段相对不敏感、20000 Hz以上仍保留感知能力。大多数人听觉灵敏度在3000–8000 Hz范围内达到峰值——对应共振峰辨别、擦音感知和警报信号检测——而作者的通路偏向低频共振（空间混响、声源接近）和高频泛音瞬态（空气吸收、纹理线索）。

这种偏离群体均值的现象是一次自然实验：**不同的耳蜗传递函数从相同的物理压力波中提取不同的语义原语。** 因此，听觉感知不是频谱包络的忠实再现，而是面向任务的有损压缩——将声能压缩为行为相关的隐变量：

$$\text{Perception}(x) = f_{\text{cochlea}} \circ f_{\text{attention}} \circ f_{\text{prediction}}(x)$$

其中 $f_{\text{cochlea}}$ 是生物学/Mel尺度的滤波器组，$f_{\text{attention}}$ 是习得的任务相关特征选择，$f_{\text{prediction}}$ 是生成未来感官状态预期的前向模型。只有三者合成才能产生理解。

---

### 2. 预测编码作为JEPA的生物学基础

LeCun的联合嵌入预测架构认为：**预测即理解。** 我们补充其生物学推论：**理解以选择为前提——只有具有预测价值的信号才能进入表征空间。**

考虑先天性失明者：一个从未见过房间的人，却能通过纯粹听觉判断走廊长度、门距和房间人数。听觉系统丢弃了与物理交互无关的频谱细节——墙壁颜色、地板纹理——只保留了与空间结构共变的声学线索：混响时间、耳间声级差、频谱质心偏移。

这正是JEPA在架构上所做的。预测并非在原始信号空间（像素、波形）中发生，而是在习得的隐空间中发生，不相关的变化已被压缩。编码器——类似耳蜗加皮层过滤——选择什么进入表征。预测器——类似听觉皮层中的前向模型回路——生成预期。**不匹配驱动学习。无需重构损失。**

系统通过**学习什么在变化**来**学习什么重要**。

---

### 3. Slot Attention 作为对象中心物理建模

自然声学场景是复调的。单个麦克风捕获来自独立物理声源的叠合信号：雨声、火车轮响、脚步声、风声。单体编码器将所有声源坍缩为单一纠缠的隐向量，抹去了物理推理所需的结构。

我们采用**带对象级时间掩码的 Slot Attention（C-JEPA）。** 每个时间步给定 $K = 5$ 个可学习槽位查询，每个槽位接受共享音频特征向量的独立可学习投影，模型通过迭代注意力将时间结构竞争性地分配给不同槽位：

$$\text{Slot}_k = \frac{\sum_{i} A_{k,i} \cdot v_i}{\sum_{i} A_{k,i}}, \quad A_{k,i} = \text{softmax}_k\left(\frac{q_k^\top k_i}{\sqrt{d}}\right)$$

驱动槽位差异化的训练信号纯粹是自监督的：**遮住整个对象历史，预测其未来状态。** 每个训练样本随机选取 $K$ 个槽位中的若干，将其全部时间历史（所有6个输入帧）替换为可学习的掩码 token。CausalPredictor 必须从可观察的槽位中恢复被遮槽位的状态——强制跨对象因果推理。

$$\mathcal{L}_{\text{JEPA}} = \mathcal{L}_{\text{SIGReg}}\big(\text{Predictor}(\text{MaskedSlots}), \text{TargetEncoder}(\text{AllSlots})\big)$$

没有任何外部标签标识哪个槽位对应哪个声源。模型发现分解结构，是因为**只有独立追踪每个声源，才能正确预测每个声源的未来。** 这是从预测压力中涌现的对象中心物理建模。

---

### 4. 单模态物理先验（UPP）假设

**场景A（对齐优先）：** 训练一个仅以51类声音辨识为目标的音频编码器，然后用对比损失与视觉编码器对齐。音频表征编码了*什么*但不知道*如何*——知道"直升机"但不知道这个直升机是在靠近还是远离。与视觉的对齐无法注入这种知识；视觉编码器知道空间运动，但没有机制能让视觉"教会"音频——只能把表征拉近到共享空间。

**场景B（预测优先，我们的路径）：** 音频编码器先学习一个时间世界模型：音量上升 ≈ 距离缩小，多普勒下移 ≈ 速度接近，混响增强 ≈ 封闭空间，低频衰减 ≈ 遮挡。然后才进行跨模态对齐。此时，对齐是确认，不是教导：

> 音频："我检测到一个物体以约15 m/s的速度向我移动。"
> 视觉："我确认：这是一辆摩托车，蓝色，方位30°。"

我们将其形式化为**单模态物理先验（Unimodal Physical Prior, UPP）假设：**

$$\text{AlignmentQuality}(M_A, M_V) \leq \min\big(\text{PhysicalUnderstanding}(M_A), \text{PhysicalUnderstanding}(M_V)\big)$$

跨模态对齐无法弥补单模态物理建模的缺陷。天花板由较弱的模态决定。训练投入应该前置：在每个模态独立习得物理动力学之后，再引入任何跨模态目标。

---

### 5. 实证实例：51类声学事件分类

我们在51类声学分类任务上实例化此框架：

**数据：** ESC-50（2000条标注片段，50类环境声音，每条5秒）+ Deep House音乐（从1,582首中10%分层抽样的177首）作为单一"非事件"锚定类别。

**预处理：** 128-bin Mel频谱，22,050 Hz采样率，2,048点FFT，512采样步长。所有频谱预计算并缓存到磁盘（3,776个 `.pt` 文件），比实时解码快约110倍。

**架构：** AudioFeatureProjector (Conv2D → BatchNorm → GELU) → CAJEPA（5个对象槽位，每个槽位有独立可学习投影，全帧时间遮罩器，因果Transformer预测器）。

**训练目标：**

$$\mathcal{L} = \mathcal{L}_{\text{SIGReg}}(\hat{z}_{\text{future}}, z_{\text{future}}) + \lambda \cdot \mathcal{L}_{\text{InfoNCE}}(z_{\text{clip}}, y)$$

其中 $\lambda = 0.3$ 平衡自监督时序预测和监督对比分类。

**评估：** 每5个epoch在冻结的投影器特征上进行5折线性探针，测量51类准确率。训练在单GPU上一小时内完成。

---

### 6. 设计原则与核心公式

```text
Perception(x) = f_cochlea ∘ f_attention ∘ f_prediction(x)

其中：
  f_cochlea    : Mel尺度滤波器组 → 生物学压缩
  f_attention  : 习得的预测性特征选择
  f_prediction : 生成未来状态预期的前向模型

AlignmentQuality(M_A, M_V) ≤ min(PhysicalUnderstanding(M_A), PhysicalUnderstanding(M_V))
```

**好的听觉表征**
**= 知道保留什么 + 知道丢弃什么 + 知道接下来发生什么**
**（Mel/耳蜗）           （编码器学习）           （JEPA预测任务）**

---

### 7. 结论

我们论证、形式化并实证了以下原则：

> *跨模态对齐是对表征空间的拉力，不是物理知识的转移。对齐质量的天花板由单模态预训练的深度决定，而非对齐机制的复杂度。对单模态物理建模的投入必须先于跨模态融合的投入。*

对听觉智能而言：耳朵必须先学会——音量增强意味着物体靠近，音调下移意味着声源远离，混响增厚意味着空间封闭，多个声源可以被独立追踪。只有到那时，与视觉的对齐才成为语义确认，而非浅层特征缝合。

一个有力的实证检验来自人类听觉本身：**先天失明者仅凭听觉即可构建完整的物理世界模型——距离、空间布局、运动、因果——无需依赖视觉。** 他们能通过听觉导航房间、估计人群数量、操作计算机。同理，单模态音频编码器无需视觉监督即可习得完备的物理规律。这直接导向一个工程结论：*听觉优先的架构天然无障碍，而视觉优先的架构隐式地将视觉输入作为理解的前提——这对视障用户构成障碍。* 在我们的框架中，跨模态对齐既不是感官智能的充分条件，也不是必要条件。它是确认，不是教导。

**耳朵不是传声器。它是戴着滤波器组的预测引擎。**

---

### 参考文献

1. LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. *OpenReview.*
2. Assran, M. et al. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. *CVPR.*
3. Bardes, A. et al. (2024). Revisiting Feature Prediction for Learning Visual Representations from Video. *arXiv:2404.08471.*
4. Locatello, F. et al. (2020). Object-Centric Learning with Slot Attention. *NeurIPS.*
5. Piczak, K. (2015). ESC: Dataset for Environmental Sound Classification. *ACM Multimedia.*
6. Terver, B. et al. (2026). EB-JEPA: A Lightweight Library for Energy-Based Joint Embedding Predictive Architectures. *ICLR Workshop on World Models.*
7. Nam, H. et al. (2026). Causal-JEPA: Learning World Models through Object-Level Latent Interventions. *arXiv:2602.11389.*
8. Maes, L. et al. (2026). LeWorldModel: End-to-End JEPA World Models from Pixels. *arXiv:2603.19312.*

---

*VORTEX FLAME 项目 — CAJEPA 音频流水线*
*代码仓库: github.com/maco1979/VORTEX_FLAME*
