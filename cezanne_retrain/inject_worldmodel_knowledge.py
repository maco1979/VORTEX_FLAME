#!/usr/bin/env python3
import sys, os
if os.path.exists("/mnt/d/VORTEX_FLAME"):
    sys.path.insert(0, "/mnt/d/VORTEX_FLAME")
else:
    sys.path.insert(0, r"D:\VORTEX_FLAME")
from long_memory import write_knowledge, build_knowledge_index

soul = "cezanne"

knowledge_entries = [
    {
        "category": "knowledge_WorldModel_architecture",
        "content": "世界模型核心架构：编码器-动力学模型-解码器三阶段。编码器负责将高维观察(如RGB图像)压缩到低维潜在空间(15-32 tokens)；动力学模型在潜在空间中预测未来状态转移；解码器将潜在状态重建为可观察输出。RTX 3060 12GB推荐：CNN编码器(EfficientNet-Lite/MobileNetV3, 50-80M参数) + LSTM/GRU动力学(256-512维隐藏, 2-3层) + 转置卷积解码器(50-80M参数)，总参数200-300M，FP16下batch_size=2-4可稳定运行。"
    },
    {
        "category": "knowledge_WorldModel_architecture",
        "content": "世界模型三种主流架构对比：RNN(循环神经网络)优势是时序依赖捕捉、计算开销低、适合短时域预测和低算力场景，缺点是长期依赖建模有限、梯度消失/爆炸；Transformer优势是长程依赖建模、并行计算、适合多模态融合(如Sora视频生成)，缺点是内存成本高；扩散模型优势是高保真生成、视觉细节捕捉(如DIAMOND在ATari100K人类标准化均值1.459)，缺点是推理速度慢。RTX 3060推荐RNN+CNN混合架构。"
    },
    {
        "category": "knowledge_WorldModel_VRAM",
        "content": "RTX 3060显存优化关键技术：1)梯度检查点(Gradient Checkpointing)将显存从O(L)降到O(√L)，PyTorch用torch.utils.checkpoint.checkpoint_sequential实现，只保留关键节点激活值，反向传播时重计算，计算量增加约1倍；2)混合精度训练(FP16/BF16)利用Tensor Core减少40-50%显存，用torch.cuda.amp的autocast+GradScaler；3)4-bit量化(bitsandbytes QLoRA)让7B模型在12GB显存运行，模型大小减少75%；4)torch.compile max-autotune模式JIT编译优化内核；5)批量大小1-4，序列长度512 tokens安全阈值。"
    },
    {
        "category": "knowledge_WorldModel_VRAM",
        "content": "RTX 3060显存管理实战策略：使用torch.cuda.empty_cache()手动清理缓存；del语句释放不需要的张量；torch.no_grad()上下文避免推理时梯度计算；内存池技术重用已分配显存；动态调整批量大小根据训练阶段；优化张量并行从4减到2；注意torch.compile在正向传播时创建反向图，autocast激活时反向图也用autocast。监控工具：nvidia-smi定时查询，torch.cuda.max_memory_allocated()和max_memory_cached()记录峰值。"
    },
    {
        "category": "knowledge_WorldModel_compression",
        "content": "世界模型潜在空间压缩策略：传统方法每帧需数百token，最新研究Lucid v1将Minecraft帧压缩到仅15个token(600倍计算降低)，CompACT离散分词器压缩到最少8个token。实现方法：VAE架构通过激进潜在空间压缩+GAN感知损失，保持基本机制同时丢弃冗余视觉信息。编码器输出维度256-512维，输入分辨率128x128或256x256。成功案例：JEPA图像世界模型510M参数(编码器113M+动作条件43M+预测213M+扩散Transformer 141M)在RTX 3060 13GB训练两周。"
    },
    {
        "category": "knowledge_WorldModel_rendering",
        "content": "3D渲染模块选择：NeRF vs 3D高斯泼溅(3D Gaussian Splatting)。NeRF将场景表示为连续5维函数，MLP学习(x,y,z,θ,φ)的颜色和密度，优势是隐式连续表示、固定内存成本，但计算密集，FastNeRF平均每场景54GB内存(优化后16GB)，RTX 3060 8GB难以实时。3D高斯泼溅更高效：简单场景50万高斯45fps，每高斯128字节，500万高斯约600MB显存，远低于NeRF。RTX 3060推荐3D高斯泼溅，控制高斯数量50万-500万，训练用低分辨率128x128/256x256。"
    },
    {
        "category": "knowledge_WorldModel_training",
        "content": "世界模型训练策略：分预训练+微调两阶段。预训练：无动作条件下视频预测，学习视觉动力学和时间依赖，128x128分辨率，参数<100M，batch_size=4，序列长度16-32，100K-200K迭代。微调：添加动作条件，逐步增加难度(单一方向→旋转加速)，序列长度16→32→64帧。优化器AdamW(权重衰减解耦)，学习率余弦退火(2e-5→1e-6或3e-4→3e-5)，权重衰减余弦调度(0.04→0.4)，betas=(0.9,0.95)。RTX 3060上200-300M模型训练3-4周。"
    },
    {
        "category": "knowledge_WorldModel_training",
        "content": "世界模型训练监控与评估：监控指标包括重建损失、动力学损失、KL散度、学习率、显存使用。评估体系四层次：1)像素级误差MSE(基础)；2)感知质量LPIPS(接近人类感知)；3)多样性FVD弗雷歇视频距离+熵；4)下游任务RL得分(最终指标)。评估协议用WorldTest框架，指标包括FID/FVD/LPIPS/mIoU/SPL/控制成功率。定期(每10K迭代)验证集评估，长期预测评估时间一致性。PyTorch监控：torch.cuda.memory_allocated()/memory_cached()/max_memory_allocated()。"
    },
    {
        "category": "knowledge_WorldModel_dataset",
        "content": "世界模型数据集：Kinetics-400(400类>100万片段10秒)和Kinetics-700-2020(700类)用于行为识别；Something-Something V2(220847片段)注重动作因果和物体交互，仅在此预训练可接近甚至超越机器人数据集监督预训练；ScanNet(1513场景250万视图，RGB-D+3D姿态+语义分割)；Matterport3D(1000个3D空间)；DeepMind Control Suite(基于MuJoCo物理引擎，潜在动作世界模型比纯动作条件基线少一个数量级标注样本)；OmniWorld(96K视频1850万帧214小时720P，含深度图/相机姿态/光流/前景掩码)。"
    },
    {
        "category": "knowledge_WorldModel_dataset",
        "content": "世界模型数据预处理策略：数据清洗去除运动模糊/特征点不足帧，长视频分割成可管理片段；多模态融合对齐视觉/文本/深度/相机姿态，用Qwen2-VL-72B-Instruct生成文本描述，视频嵌入语义去重；数据增强包括传统旋转缩放裁剪+动力学增强(潜在空间添加噪声/扰动增强动作多样性)；时序数据处理确保相邻帧时间间隔一致，滑动窗口采样不同时间尺度动力学模式。数据格式：视频转序列图像PNG/JPEG，3D用HDF5/Zarr分块读取，标注用Protocol Buffers/Apache Arrow。"
    },
    {
        "category": "knowledge_WorldModel_DreamerV3",
        "content": "DreamerV3架构详解：编码器+动力学+价值/策略训练管线。编码器将观察压缩到潜在空间；动力学模型(RSSM)在潜在空间预测未来状态，包含先验网络(从上一状态和动作预测)和后验网络(结合当前观察)；价值网络估计状态价值；策略网络输出动作分布。训练目标：重建损失(观察重建)+KL散度(先验后验对齐)+价值损失+策略损失。PyTorch移植版本可用。在DMC基准上，潜在动作世界模型方法比纯动作条件基线少一个数量级标注样本仍取得强性能。"
    },
    {
        "category": "knowledge_WorldModel_quantization",
        "content": "世界模型量化与压缩技术：后训练量化PTQ是最快路径，FP16/BF16/FP8模型用校准数据集压缩到FP8/NVFP4/INT8/INT4，无需修改训练循环。4-bit量化bitsandbytes减少75%模型大小保持较好精度；8-bit动态量化运行时调整参数。结构化剪枝删除整个卷积核/全连接层(比非结构化剪枝推理加速更明显)，剪枝率10%→40-50%逐步增加。知识蒸馏：大模型(教师)→小模型(学生)，学生用MobileNet编码器+LSTM动力学。RTX 3060推荐4-bit QLoRA微调+梯度检查点。"
    },
    {
        "category": "knowledge_WorldModel_TensorCore",
        "content": "RTX 3060 Tensor Core加速：Tensor Core专门加速矩阵乘法和卷积，FP16混合精度可达8倍性能提升。关键要求：1)使用FP16/BF16数据类型；2)矩阵维度是8的倍数(Tensor Core以8x8块工作)；3)使用nn.Conv2d/nn.Linear等支持操作。卷积优化：Winograd算法适合3x3小核，分组卷积分组数8的倍数，深度可分离卷积分别优化。矩阵优化：torch.baddbmm/torch.addmm批量运算，m/n/k维度8的倍数，NHWC内存布局，大矩阵分块策略防显存溢出。"
    },
    {
        "category": "knowledge_WorldModel_project",
        "content": "世界模型项目结构：configs/(dataset.yaml, model.yaml, train.yaml) + data/(datasets/, transforms/, loaders/) + models/(backbones/, dynamics/, decoders/, modules/) + scripts/(train.py, evaluate.py, inference.py) + utils/(logger.py, metrics.py, visualization.py) + logs/ + checkpoints/。配置管理用YAML，PyYAML解析支持命令行覆盖。代码规范PEP8+类型提示+docstring+单元测试+Git版本控制。开发迭代：原型开发(最简编码器-解码器+随机数据)→组件验证(预训练网络+玩具数据集)→集成测试(完整数据集+显存监控)→优化迭代。"
    },
    {
        "category": "knowledge_WorldModel_risk",
        "content": "RTX 3060世界模型风险与应对：显存不足→紧急降低batch_size到1+torch.cuda.empty_cache()；中期启用梯度检查点+4-bit量化；长期重新设计架构用轻量CNN替代大型Transformer。计算资源限制→分阶段训练+预训练起点+课程学习+梯度累积模拟大批量。数据不足→合成数据+数据增强+迁移学习。架构风险→混合架构或渐进式方法，先简单后复杂。训练时间：200-300M模型3-4周，建议夜间训练+断点续训+早停策略。硬件升级路径：短期加内存32-64GB+NVMe SSD；中期RTX 4060 Ti/4070 16GB+云GPU；长期RTX 4090/A100+本地集群。"
    },
    {
        "category": "knowledge_WorldModel_learning_path",
        "content": "世界模型学习路径四阶段：第一阶段建立直觉(1-2月)读Ha&Schmidhuber《World Models》论文+Yann LeCun《A Path Towards Autonomous Machine Intelligence》演讲，理解潜在状态/预测/想象力概念；第二阶段学习核心循环(2-3月)UC Berkeley CS285深度RL课程+DreamerV3实现，识别编码器/潜在动力学/展开/重建预测损失；第三阶段实践入门项目(3-4月)《World Models from Scratch》小型自包含项目或DreamerV3 PyTorch移植版；第四阶段选择专业方向：通用MBRL(DreamerV3深入)/自动驾驶(CARLA+占用率视频生成)/具身AI机器人(CALVIN/LIBERO+VLA/WAM)。"
    },
]

added = 0
for entry in knowledge_entries:
    eid = write_knowledge(soul, entry["category"], entry["content"],
                          metadata={"source": "RTX3060世界模型开发方案"})
    added += 1
    print(f"  [{added}/{len(knowledge_entries)}] {entry['category'][:40]}... -> {eid}")

print(f"\n写入完成: {added} 条知识")
print("重建FAISS索引...")
total = build_knowledge_index(soul)
print(f"知识库索引重建完成: {total} 条总索引")
