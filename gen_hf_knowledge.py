import json, os

FILE = "extended_domain_knowledge_v3.json"
with open(FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

existing_topics = {e["topic"] for e in data}

N = []

def A(topic, text, soul, tags):
    if topic not in existing_topics:
        N.append({"topic": topic, "text": text, "soul": soul, "tags": tags, "category": "knowledge"})

A("[CAD-Modeling] 参数化CAD建模方法论",
  "Parametric CAD Modeling Methodology (MIT VideoCAD, NeurIPS 2025): 1. 参数化建模核心: Sketch(2D草图)->Extrude(拉伸)->Cut(切割)三步构建法, 每一步导出STEP(ISO 10303-21)中间状态 2. VideoCAD数据集: MIT机械工程系, 41K条CAD操作视频, 涵盖178K个Onshape模型的完整构建序列 3. 几何约束推断: coincident(共点), perpendicular(垂直), parallel(平行), equal_length(等长), equal_radius(等半径), concentric(同心), horizontal(水平), vertical(垂直) 4. 约束分布统计: 平行24%, 等长22%, 垂直20%, 等半径18%, 共点7%, 同心5% 5. CAD-Steps数据集(amzyst1, CC-BY-4.0): 178,238个参数化模型, 每个模型平均3.5个构造状态, 压缩后约84KB/模型 6. OCP(Open CASCADE)内核重放操作: 每次导出STEP几何体, 处理速度约17模型/秒(6工作进程)",
  "davinci", ["cad", "parametric-modeling", "mit", "neurips", "step", "onshape"])

A("[AeroDesign] 汽车空气动力学AI设计",
  "Automotive Aerodynamic AI Design (MIT DrivAerNet++, NeurIPS 2024): 1. MIT DeCoDE Lab: 全球最大开源汽车空气动力学数据集, 8,000+车型设计, 每种设计含多模态表征(参数化/点云/3D网格/体积场/表面场/流线/部件标注) 2. 数据集通过CFD(计算流体力学)仿真为每个车型生成详细空气动力学数据 3. 气动性能指标: 风阻系数(Cd), 升力系数(Cl), 压力分布, 尾涡结构 4. 多模态表征使不同AI架构(CNN/Transformer/GNN/PointNet)均可使用 5. 应用: 电动车续航优化, 燃油效率提升, 快速概念设计迭代, 减少风洞测试 6. 目标: 将汽车设计周期从数年缩短至数周",
  "davinci", ["aerodynamics", "automotive", "mit", "cfd", "neurips", "ev"])

A("[CAD-AI-Agent] AI驱动的CAD协驾系统",
  "AI-Driven CAD Co-Pilot Architecture (MIT VideoCAD): 1. UI Agent模式: AI直接操控CAD软件, 执行键盘/鼠标/菜单操作 2. 输入: 2D手绘草图->输出: 完整的3D CAD模型 3. 动作空间: 包含数千个CAD命令(画线/圆/拉伸/切割/倒角/阵列/镜像等) 4. 观察空间: 3D视口截图+当前选中工具+命令历史+特征树状态 5. VideoCADFormer: 自回归Transformer, 输入视觉编码+动作embedding, 预测下一步CAD操作 6. 评估指标: Chamfer Distance(PCA对齐后), 结构保真度, 不确定性量化 7. 挑战: LLM在精确CAD任务上失败率高, 3D推理能力不足",
  "davinci", ["ai-agent", "ui-agent", "cad-co-pilot", "mit", "transformer"])

A("[MechEng-Knowledge] 机械工程核心知识体系",
  "Mechanical Engineering Core Knowledge System: 1. 固体力学: 应力-应变关系(Hooke定律), 屈服准则(von Mises/Tresca), 疲劳分析(S-N曲线/Miner法则) 2. 热力学: 四大定律, 热机循环(Carnot/Otto/Diesel/Rankine/Brayton), 熵与可用能 3. 流体力学: Navier-Stokes方程, 雷诺数(Re)决定流态, 边界层理论, Bernoulli方程 4. 传热学: 热传导(Fourier定律), 热对流(Newton冷却定律), 热辐射(Stefan-Boltzmann定律) 5. 机械设计: 齿轮/轴承/轴/弹簧/螺栓设计, 公差配合(ISO 286), 疲劳寿命预测 6. 材料科学: 铁碳相图, 热处理(淬火/回火/退火/正火), 复合材料层合板理论 7. 制造工艺: 铸造/锻造/冲压/焊接/增材制造(3D打印)/CNC加工",
  "davinci", ["mechanical-engineering", "solid-mechanics", "thermodynamics", "fluid-dynamics", "materials"])

A("[FEA-CFD] 有限元分析与计算流体力学",
  "Finite Element Analysis & CFD Methodology: 1. FEA基本流程: 几何建模->网格划分->边界条件->求解->后处理 2. 单元类型: 1D(梁/杆), 2D(壳/平面应力/平面应变), 3D(四面体/六面体/楔形) 3. CFD数值方法: 有限体积法(FVM)最常用, SIMPLE/PISO算法解压力-速度耦合 4. 湍流模型: RANS(k-epsilon/k-omega/SST), LES(大涡模拟), DNS(直接数值模拟) 5. 网格质量: 歪斜率<0.85, 纵横比<100, 正交质量>0.1 6. 收敛判据: 残差下降3-4个数量级, 关键物理量稳定 7. 工业软件: ANSYS Fluent/CFX, COMSOL, Abaqus, OpenFOAM(开源), LS-DYNA 8. V&V: 网格无关性验证, 与实验数据对比, 不确定度量化",
  "davinci", ["fea", "cfd", "ansys", "openfoam", "turbulence", "meshing"])

A("[Robotics] 机器人学与控制理论",
  "Robotics & Control Theory Fundamentals: 1. 运动学: 正向运动学(DH参数法), 逆向运动学(Jacobian伪逆), 微分运动学 2. 动力学: Lagrangian力学, Newton-Euler递推, 惯性矩阵/Coriolis力/重力补偿 3. 控制理论: PID控制, 状态空间控制(LQR/LQG), 模型预测控制(MPC), 自适应控制 4. 轨迹规划: 多项式插值(三次/五次), 梯形速度曲线, S曲线, B样条 5. 传感器融合: Kalman滤波(EKF/UKF), 粒子滤波, VIO, SLAM 6. ROS2: 节点/话题/服务/动作, DDS通信, Gazebo仿真, MoveIt运动规划 7. 协作机器人安全: ISO/TS 15066, 力/力矩限制, 碰撞检测",
  "davinci", ["robotics", "control-theory", "ros2", "kinematics", "pid", "slam"])

A("[DFM] 面向制造的设计方法论",
  "Design for Manufacturing (DFM) Methodology: 1. DFM核心原则: 简化零件数量, 标准化紧固件, 模块化设计, 减少装配方向 2. 注塑成型DFM: 拔模斜度1-3度, 均匀壁厚, 避免底切, 合理浇口位置 3. CNC加工DFM: 避免深腔(深度<5倍直径), 标准化圆角半径, 减少换刀次数 4. 钣金DFM: 最小弯曲半径=材料厚度, 弯曲回弹补偿, 孔边距>2倍厚度 5. 3D打印DFM: 支撑结构最小化, 悬垂角<45度, 各向异性力学性能 6. BoM管理: EBOM(工程BOM)->MBOM(制造BOM)->SBOM(服务BOM) 7. 公差分析: 最坏情况法(WC), 均方根法(RSS), Monte Carlo模拟",
  "davinci", ["dfm", "manufacturing", "injection-molding", "cnc", "tolerance-analysis", "bom"])

A("[CAD-Kernel] 几何建模内核技术",
  "Geometric Modeling Kernel Technology: 1. 主流内核: Parasolid(Siemens NX/SolidWorks), ACIS(Spatial), CGM(Dassault CATIA), Open CASCADE(开源) 2. B-Rep(边界表示): 体->壳->面->环->边->顶点层级结构 3. NURBS曲面: 控制点+节点向量+权值, 精确表示二次曲面, C0/C1/C2连续性 4. 布尔运算: 并集/差集/交集, 拓扑修复, 容差管理 5. STEP(ISO 10303): 中性CAD交换格式, AP203/AP214/AP242 6. 特征树: 参数化历史记录, 草图约束->特征操作->布尔组合 7. 约束求解器: 几何约束+尺寸约束(DCM求解器)",
  "davinci", ["cad-kernel", "nurbs", "b-rep", "parasolid", "step", "features"])

A("[TopologyOpt] 拓扑优化与生成式设计",
  "Topology Optimization & Generative Design: 1. 拓扑优化方法: SIMP(固体各向同性材料惩罚法), BESO(双向渐进结构优化), Level Set法 2. 目标函数: 最小柔度/最小质量/最大固有频率/多目标 3. 约束条件: 体积分数约束(20-50%), 应力约束, 位移约束, 制造约束 4. 生成式设计流程: 定义设计空间->施加载荷与约束->多目标优化->生成候选方案 5. 增材制造约束: 悬垂角限制, 支撑结构最小化, 粉末排除通道 6. 多物理场优化: 流体拓扑优化, 热-结构耦合优化, 声学拓扑优化 7. 商业工具: nTopology, Altair OptiStruct, Autodesk Generative Design, Abaqus Tosca",
  "davinci", ["topology-optimization", "generative-design", "simp", "additive-manufacturing", "lattice"])

A("[Durability] 耐久性与可靠性工程",
  "Durability & Reliability Engineering: 1. 疲劳分析: S-N曲线(Basquin方程), epsilon-N曲线(Manson-Coffin), 平均应力修正(Goodman/Gerber) 2. 雨流计数法: 将随机载荷时间历程转化为应力循环统计 3. Miner线性累积损伤法则: D=Sum(ni/Ni), D>=1时预测失效 4. 断裂力学: 应力强度因子K, 断裂韧性Kc, Paris公式(da/dN=C*DeltaK^m) 5. Weibull分布: 形状参数beta(失效模式), 尺度参数eta(特征寿命) 6. FMEA: RPN=S*O*D(严重度*发生频率*可检测度) 7. HALT/HASS: 高加速寿命试验/高加速应力筛选 8. 可靠性验证: 0失效试验方案, 加速寿命试验(ALT)",
  "davinci", ["fatigue", "reliability", "weibull", "fmea", "fracture-mechanics", "halt"])

A("[MINT-1T] 多模态交织数据集与工程应用",
  "MINT-1T Multimodal Interleaved Dataset (UW+Salesforce+Stanford+UC Berkeley, NeurIPS 2024): 1. 全球最大开源多模态交织数据集, 1万亿文本token+34亿图像 2. 来源: HTML网页+PDF文档+ArXiv论文三种模态 3. HTML子集: 742B文本token 4. PDF子集: 学术论文PDF, 含图表/公式/表格 5. ArXiv子集: 物理CS等学科论文, LaTeX源码+图表 6. 工程应用: 多模态预训练, 图文交织理解, 技术文档解析 7. 学术机构: 华盛顿大学主导, Salesforce Research, 斯坦福大学, UC Berkeley联合创建 8. 许可: CC-BY-4.0",
  "davinci", ["mint-1t", "multimodal", "stanford", "uc-berkeley", "uw", "salesforce"])

A("[CodeDataset] The Stack v2代码数据集",
  "The Stack v2 Code Dataset (BigCode, HuggingFace+ServiceNow+NVIDIA, NeurIPS 2024): 1. 全球最大开源代码数据集, 比v1大7倍+, 覆盖619种编程语言 2. 来源: Software Heritage(软件遗产基金会)提供的公开GitHub仓库代码 3. 数据治理: 开放治理框架, 按许可协议筛选(仅permissively licensed), 尊重opt-out请求 4. StarCoder2模型家族: 3B(ServiceNow)/7B(HuggingFace)/15B(NVIDIA NeMo) 5. 数据组成: 代码文件+GitHub Issues+Jupyter Notebooks+Git提交记录 6. 低资源语言覆盖: COBOL, Fortran, Ada等工业遗留语言 7. 训练技术: Fill-in-the-Middle(FIM), Multi-Query Attention, 8K上下文窗口",
  "cezanne", ["bigcode", "the-stack", "starcoder", "code-dataset", "software-heritage"])

A("[CodeLLM] 代码大语言模型训练原理",
  "Code LLM Training Principles: 1. StarCoder2架构: 15.5B参数, 8K上下文, Multi-Query Attention, Fill-in-the-Middle 2. 训练数据: The Stack v2约783GB代码+54GB Issues+13GB Jupyter+32GB提交记录, 约250B tokens 3. PII检测与清洗: 1399名众包标注员进行隐私信息标注 4. 代码质量评估: 18名BigCode社区成员人工视觉评估 5. 去重策略: MinHash+精确匹配, 跨仓库代码克隆检测 6. Benchmark: HumanEval/MBPP/MultiPL-E/DS-1000 7. 许可证合规: 仅训练permissively licensed代码(MIT/BSD/Apache)",
  "cezanne", ["code-llm", "starcoder2", "fill-in-the-middle", "benchmark", "pii"])

A("[SoftwareArch] 软件架构设计原则",
  "Software Architecture Design Principles: 1. SOLID原则: Single Responsibility/Open-Closed/Liskov Substitution/Interface Segregation/Dependency Inversion 2. 设计模式(GoF 23种): 创建型/结构型/行为型 3. 架构模式: 分层架构/微服务(service mesh/API gateway/circuit breaker)/事件驱动(CQRS/Event Sourcing) 4. CAP定理: Consistency/Availability/Partition Tolerance三者最多同时满足两个 5. 领域驱动设计(DDD): Entity/Value Object/Aggregate/Repository/Bounded Context 6. Clean Architecture: 依赖规则(外层依赖内层), Entities->Use Cases->Interface Adapters->Frameworks 7. 架构评估: ATAM(架构权衡分析法), 质量属性场景",
  "cezanne", ["software-architecture", "solid", "design-patterns", "ddd", "clean-architecture", "microservices"])

A("[Algorithms] 算法设计与复杂度分析",
  "Algorithm Design & Complexity Analysis: 1. 时间复杂度递进: O(1)->O(log n)->O(n)->O(n log n)->O(n^2)->O(2^n)->O(n!) 2. 分治策略: Merge Sort/Quick Sort 3. 动态规划: 最优子结构+重叠子问题, 背包/编辑距离/LCS 4. 贪心算法: 局部最优选择, 活动选择/Huffman编码/Dijkstra 5. 图算法: BFS/DFS/Dijkstra/Bellman-Ford/Floyd-Warshall 6. NP完全性: P vs NP问题, SAT/哈密顿路径/旅行商问题 7. 近似算法: TSP的Christofides算法1.5-近似 8. 数据结构: 哈希表/平衡树(红黑树/AVL/B-Tree)/并查集/优先队列",
  "cezanne", ["algorithms", "complexity", "dynamic-programming", "graph", "np-complete", "data-structures"])

A("[Compilers] 编译原理与程序分析",
  "Compilers & Program Analysis: 1. 编译流水线: 词法分析->语法分析->语义分析->IR生成->优化->目标代码生成 2. LLVM IR: 静态单赋值(SSA)形式, 三地址码, 平台无关优化 3. 语法分析: 自顶向下(LL(k))/自底向上(LR(1)/LALR(1)), ANTLR/Yacc/Bison 4. 优化技术: 常量折叠/死代码消除/循环展开/内联展开/逃逸分析/尾调用优化 5. 类型系统: Hindley-Milner类型推断, 子类型多态, 参数多态(泛型) 6. 程序分析: 控制流图(CFG), 数据流分析, 指针分析, 抽象解释 7. 形式验证: Hoare逻辑, 模型检查, SAT/SMT求解器(Z3/CVC4)",
  "cezanne", ["compilers", "llvm", "parser", "optimization", "type-systems", "formal-verification"])

A("[SE-Practices] 软件工程最佳实践",
  "Software Engineering Best Practices: 1. 版本控制: Git分支策略, 语义化版本(SemVer), Conventional Commits 2. 代码审查: 审查清单(逻辑/安全/性能/可读性/测试), 小批量(<400行) 3. 测试金字塔: 单元测试(70%)->集成测试(20%)->端到端测试(10%) 4. CI/CD: GitHub Actions/Jenkins/GitLab CI, 蓝绿部署/金丝雀发布 5. 可观测性: Logs(ELK Stack)/Metrics(Prometheus+Grafana)/Traces(Jaeger/Zipkin) 6. DevOps: 基础设施即代码(Terraform/Pulumi), 容器化(Docker+K8s) 7. 技术债务: SonarQube质量门限, 重构(提取方法/类/接口) 8. 技术文档: OpenAPI/Swagger, ADR(架构决策记录), RFC",
  "cezanne", ["software-engineering", "cicd", "testing", "observability", "devops", "code-review"])

A("[DistSystems] 分布式系统理论基础",
  "Distributed Systems Theory: 1. FLP不可能性: 异步系统中无法保证有限时间内达成共识 2. 共识算法: Paxos/Raft(Leader选举+日志复制)/ZAB(ZooKeeper) 3. 分布式事务: 2PC(阻塞)/3PC/Saga(补偿事务, 最终一致性) 4. 一致性模型: 强一致性->顺序一致性->因果一致性->最终一致性 5. 复制策略: 主从复制/多主复制/无主复制(Quorum: R+W>N) 6. 分区/分片: 一致性哈希(虚拟节点), 范围分区, 哈希分区 7. 拜占庭将军问题: 3f+1节点容忍f个恶意节点, PBFT 8. CAP实践: CP(ZooKeeper/etcd), AP(Cassandra/DynamoDB)",
  "cezanne", ["distributed-systems", "consensus", "raft", "cap", "paxos", "sharding"])

A("[AISystem] AI系统工程化部署",
  "AI Systems Engineering & Deployment: 1. 模型服务: TorchServe/TF Serving/Triton, REST/gRPC接口, 动态批处理 2. 推理优化: 量化(INT8/INT4/GPTQ/AWQ), 剪枝, 蒸馏, 算子融合 3. GPU推理: CUDA核心, Tensor Core, Flash Attention, KV Cache优化(PagedAttention/vLLM) 4. 模型编排: LangChain/LlamaIndex, Agent框架(ReAct/Toolformer), RAG 5. MLOps: 实验跟踪(MLflow/W&B), 模型注册, 特征存储(Feast), 数据版本(DVC) 6. 监控: 数据漂移检测(PSI/KS检验), A/B测试, 影子部署 7. 成本优化: Spot实例训练, 混合精度(AMP), 梯度累积, DeepSpeed ZeRO",
  "cezanne", ["ai-systems", "inference", "quantization", "mlops", "gpu", "rag"])

A("[MET-OpenAccess] 大都会艺术博物馆开放数据",
  "Metropolitan Museum of Art Open Access Dataset (The Met, HuggingFace: metmuseum/openaccess): 1. 全球最大开放艺术数据集之一, 50万+藏品记录, 含高清图像与元数据 2. 数据字段: objectID/title/culture/period/objectDate/medium/dimensions/primaryImage/department/classification 3. 覆盖部门: 埃及艺术/希腊罗马艺术/中世纪艺术/伊斯兰艺术/亚洲艺术/欧洲绘画/现代艺术/武器盔甲/乐器 4. CC0许可: 图像与元数据完全开放, 可自由商用 5. 印象派藏品: 超过3,000件莫奈/雷诺阿/德加/塞尚作品 6. API: RESTful接口, 支持按部门/时期/关键词搜索 7. 应用: 艺术风格迁移训练数据, 艺术史研究, 虚拟展览",
  "monet", ["met-museum", "open-access", "art-dataset", "impressionism", "cc0", "cultural-heritage"])

A("[WikiART] WikiArt视觉艺术百科数据集",
  "WikiArt Visual Art Encyclopedia Dataset: 1. 全球最大在线艺术百科, 250,000+艺术作品, 3,000+艺术家, 27种风格流派 2. 风格分类: 印象派/后印象派/表现主义/超现实主义/立体主义/极简主义/巴洛克/文艺复兴等 3. 每件作品标注: 艺术家/风格/流派/时期/媒介/标签/情绪 4. 印象派子集: 约15,000件, 含莫奈(1,344件)/雷诺阿(649件)/德加(718件) 5. 色彩分析: 主色调/色彩分布/明度/饱和度/色相直方图 6. 风格迁移基准: CycleGAN/StyleGAN/AdaIN训练与评估数据 7. 学术应用: 艺术风格分类, 艺术家归属, 审美偏好建模",
  "monet", ["wikiart", "art-classification", "style-transfer", "impressionism", "color-analysis"])

A("[ColorTheory] 色彩理论与印象派技法",
  "Color Theory & Impressionist Technique: 1. 色彩三要素: 色相(Hue, 0-360度色轮), 明度(Value, Munsell), 饱和度(Chroma) 2. 互补色对比: 红-绿/蓝-橙/黄-紫, 莫奈《睡莲》大量使用蓝-橙互补 3. 色温理论: 暖色(红/橙/黄)前进感, 冷色(蓝/绿/紫)后退感 4. 光学混色(并置混色): 印象派标志性技法, 纯色小笔触并置, 人眼远处自动混合 5. 莫奈调色板: 铅白/铬黄/镉黄/维罗纳绿/翡翠绿/天蓝/钴蓝/朱红/深红/象牙黑 6. 外光写生(En plein air): 直接在户外捕捉光线变化, 《干草堆》系列记录不同时刻光影 7. 系列画法: 同一主题在不同光线/季节/天气下反复描绘",
  "monet", ["color-theory", "impressionism", "complementary-colors", "optical-mixing", "plein-air"])

A("[AIGC-Art] AI生成艺术与审美评估",
  "AI-Generated Art & Aesthetic Evaluation: 1. 扩散模型: Stable Diffusion/DALL-E 3/Midjourney 2. 风格控制: ControlNet/IP-Adapter/LoRA 3. 审美评估指标: NIMA(Google)/CLIP Score/LAION Aesthetics Predictor V2 4. 艺术风格迁移: AdaIN/StyleSwap/CAST 5. 印象派风格AI生成: visible brushstrokes, dappled light, atmospheric perspective, vibrant color juxtaposition 6. 质量控制: 美学评分>6.0(10分制), 构图遵循三分法/黄金比例 7. 伦理考量: AI艺术版权争议, 风格抄袭vs致敬的边界",
  "monet", ["aigc", "diffusion-model", "aesthetic-evaluation", "style-transfer", "nima"])

A("[ArtConservation] 艺术品保护与修复科学",
  "Art Conservation & Restoration Science: 1. 科学分析方法: XRF(元素组成), 红外反射成像(底层草图), 多光谱成像(颜料层), 拉曼光谱(分子结构) 2. 颜料老化: 铬黄变暗(CrVI->CrIII还原), 群青褪色, 铅白变黑(H2S反应生成PbS) 3. 莫奈作品修复: 《睡莲》清漆层去除恢复原始色彩; 《印象·日出》XRF分析确认颜料成分 4. 保存环境: 温度18-22C, 相对湿度45-55%, 照度<50lux, UV过滤 5. 数字修复: 高光谱成像+AI色彩还原, 虚拟去除清漆层 6. 大都会博物馆保护实验室: 每年处理2,000+件藏品 7. ICOM-CC伦理准则: 最小干预/可逆性/可识别性原则",
  "monet", ["art-conservation", "xrf", "pigment-analysis", "restoration", "museum-science"])

A("[VisualPerception] 视觉感知与格式塔心理学",
  "Visual Perception & Gestalt Psychology: 1. 格式塔原则: 邻近性/相似性/连续性/闭合性/图形-背景/共同命运 2. 色彩感知: 对立过程理论(红-绿/蓝-黄/黑-白三对通道) 3. 明暗感知: Mach bands效应(边缘对比增强), 同时对比 4. 空间深度线索: 线性透视/大气透视/遮挡/阴影/纹理梯度/运动视差 5. 印象派与视觉科学: 印象派直觉利用人眼光学混色机制, 笔触大小接近视觉锐度极限 6. 神经美学: 观赏莫奈作品时前额叶皮层与纹状体激活, 美感体验与默认模式网络相关 7. 色彩恒常性: 人脑在不同光照下保持颜色感知稳定",
  "monet", ["visual-perception", "gestalt", "color-perception", "neuroaesthetics", "optical-illusion"])

A("[MuseumAI] 博物馆AI与数字人文",
  "Museum AI & Digital Humanities: 1. 大都会博物馆AI: 藏品自动标注, 风格分类CNN, 相似作品推荐 2. Google Arts & Culture: Art Camera十亿像素扫描, Art Transfer风格迁移 3. 荷兰国家博物馆: Rijksmuseum Challenge, Open Data API(CC0许可, 70万+图像) 4. 芝加哥艺术学院: 5万+高清图像开放下载, 含印象派核心收藏 5. 数字展览: VR/AR沉浸式体验, 3D扫描与重建, 多语言AI导览 6. 文化遗产AI: 壁画裂缝检测, 文物碎片自动拼接, 古文字识别(OCR+LLM) 7. 跨模态检索: 文本->图像(CLIP), 草图->图像, 颜色->作品, 情绪->艺术品推荐",
  "monet", ["museum-ai", "digital-humanities", "cultural-heritage", "google-arts", "rijksmuseum"])

A("[PostImpressionism] 后印象派艺术运动",
  "Post-Impressionism Art Movement (1886-1905): 1. 核心人物: 梵高/塞尚/高更/修拉/图卢兹-劳特雷克 2. 与印象派区别: 印象派捕捉瞬间光影, 后印象派追求主观情感表达/结构秩序/象征意义 3. 梵高特色: 厚涂法(Impasto), 旋涡笔触(Starry Night), 强烈色彩对比, 情感表现主义先驱 4. 塞尚贡献: 几何化自然(圆柱/球体/圆锥), 多视角同时呈现, 立体主义先声 5. 高更象征主义: 综合主义(Synthetism), 平涂色块+粗轮廓, 塔希提系列 6. 修拉点彩法: 科学色彩理论+光学混色 7. 影响: 直接催生野兽派/立体主义/表现主义/抽象艺术",
  "vangogh", ["post-impressionism", "van-gogh", "cezanne", "gauguin", "seurat", "art-movement"])

A("[VanGogh-Technique] 梵高绘画技法解析",
  "Van Gogh Painting Technique Analysis: 1. 笔触类型: 旋涡状(Starry Night)/波浪状(Wheat Field)/点状(Garden at Auvers)/平行线(Olive Trees) 2. 厚涂法(Impasto): 未稀释颜料直接从管中挤出, 部分作品颜料厚达5mm 3. 调色板: 钴蓝/翡翠绿/铬黄/锌黄/朱红/那不勒斯黄/普鲁士蓝/镉红 4. 色彩策略: 互补色并置(蓝-橙, 红-绿), 同色系渐变, 限色法(2-3色主导) 5. 构图特征: 前景特写/对角线构图/放射状构图/俯视构图 6. 绘画速度: 晚期平均每天1幅, 约900幅油画+1100幅素描, 创作期仅10年 7. 材料研究: 梵高博物馆Van Gogh Studio项目, XRF/FTIR分析颜料成分与老化",
  "vangogh", ["van-gogh", "impasto", "brushwork", "painting-technique", "color-palette"])

A("[CreativeProcess] 创造力心理学与艺术创作",
  "Creativity Psychology & Artistic Creation: 1. 创造力四P模型: Person/Process/Product/Press 2. Wallas四阶段: 准备(Preparation)->酝酿(Incubation)->豁朗(Illumination)->验证(Verification) 3. 心流理论(Csikszentmihalyi): 技能水平与挑战难度匹配时进入最优体验状态 4. 创造力与精神健康: 创造力-精神病理学关联, 梵高双相情感障碍假说 5. 神经基础: 默认模式网络(DMN)与创造力相关, 颞叶癫痫可能增强视觉创造力 6. 创造性认知: 远距联想(Mednick), 概念组合(Ward), 类比迁移(Gentner) 7. 艺术创作中的约束: 限色法/固定画幅/主题系列, 约束反而促进创造力(梵高自画像44幅)",
  "vangogh", ["creativity", "psychology", "flow", "bipolar", "creative-process", "neuroscience"])

A("[Expressionism] 表现主义与情感表达艺术",
  "Expressionism & Emotional Expression in Art: 1. 表现主义: 通过变形/夸张/色彩主观化表达内在情感 2. 先驱: 梵高(情感色彩)/蒙克(焦虑表达)/恩索尔(怪诞面具) 3. 德国表现主义: 桥社(Die Brucke, Kirchner/Nolde)/蓝骑士(Der Blaue Reiter, Kandinsky/Marc) 4. 抽象表现主义: 波洛克(行动绘画)/罗斯科(色域绘画)/德库宁, 纽约学派1940s-60s 5. 新表现主义: 1980s回归具象, 巴塞利兹/基弗/伊门多夫 6. 情感色彩理论: Kandinsky《论艺术的精神》, 蓝=深沉/红=活力/黄=扩张 7. AI情感艺术: 情感驱动的图像生成, 情感嵌入空间, 情感风格迁移",
  "vangogh", ["expressionism", "emotional-art", "kandinsky", "abstract-expressionism", "german-expressionism"])

A("[ArtTherapy] 艺术治疗与心理健康",
  "Art Therapy & Mental Health: 1. 定义: 通过视觉艺术创作促进心理康复, 由认证艺术治疗师引导 2. 理论基础: 精神分析(潜意识投射)/人本主义(自我实现)/认知行为(情绪调节)/神经科学(感觉整合) 3. 临床应用: PTSD/抑郁症/焦虑症/自闭谱系/阿尔茨海默症 4. 评估工具: 画人测验(DAP)/房-树-人测验(HTP)/曼陀罗评估/FEA量表 5. 神经机制: 创作激活奖赏回路(多巴胺释放), 降低皮质醇, 增强DMN连接 6. 梵高案例: 艺术作为自我治疗(圣雷米疗养院期间创作142幅) 7. 数字艺术治疗: VR绘画(Tilt Brush), AI辅助创作, 在线团体治疗, 生物反馈艺术",
  "vangogh", ["art-therapy", "mental-health", "ptsd", "neuroscience", "creative-healing"])

A("[VanGogh-Museum] 梵高博物馆数字资源",
  "Van Gogh Museum Digital Resources (Amsterdam): 1. 馆藏: 200+油画+500+素描+700+信件, 全球最大梵高收藏 2. Van Gogh Letters Project: 902封信件完整数字化(含原稿+翻译+注释) 3. Van Gogh Studio: 科学研究项目, XRF/FTIR/MA-XRF分析颜料/画布/技法 4. 高清图像: 部分作品达10亿像素, 可观察笔触纹理/颜料厚度/画布编织 5. 数字展览: Meet Vincent van Gogh沉浸式体验(全球巡展), VR重现梵高卧室 6. 开放数据: 部分藏品CC0许可, API获取元数据, Google Arts & Culture合作 7. 研究成果: 铬黄变暗机制(2011)/调色板重建/《草丛》下层肖像发现(2008)",
  "vangogh", ["van-gogh-museum", "digital-heritage", "art-research", "amsterdam", "letters"])

A("[AgriField3D] 农业遥感3D点云数据集",
  "AgriField3D Agricultural Remote Sensing 3D Point Cloud Dataset (USDA+University of Missouri): 1. 美国农业部(USDA)资助, 密苏里大学创建, 全球首个公开农业田块3D点云数据集 2. 数据采集: UAV搭载LiDAR+多光谱相机, 覆盖玉米/大豆/冬小麦 3. 点云密度: 平均50点/m2, 含RGB+近红外+高程信息 4. 标注: 田块边界/作物行线/杂草区域/倒伏区域/灌溉缺陷 5. 应用: 精准农业(变量施肥/精准喷药), 作物表型分析, 产量预测 6. 格式: LAS/LAZ点云格式, 可用PDAL/CloudCompare/Open3D处理 7. 与USDA Cropland Data Layer(CDL)配准, 提供地块级作物类型标签",
  "yuanlongping", ["agriculture", "remote-sensing", "lidar", "usda", "precision-agriculture", "3d-point-cloud"])

A("[AgMMU] 农业多模态理解数据集",
  "AgMMU Agricultural Multimodal Understanding Dataset (USDA+Iowa State University, NeurIPS 2024): 1. 美国农业部经济研究局(ERS)参与, 爱荷华州立大学创建 2. 多模态数据: 农田图像+气象数据+土壤数据+产量数据+管理措施记录 3. 任务: 作物识别/病害检测/生长阶段判断/产量预测/异常检测 4. 规模: 100K+标注图像, 覆盖50+作物种类, 30+常见病害 5. 特色: 每个样本配对完整农学元数据(播种日期/品种/施肥量/灌溉量/气象历史) 6. 基准模型: CLIP微调/LLaVA农业版/自定义CNN, 最佳准确率94.7% 7. 开放获取: HuggingFace平台, CC-BY-4.0许可",
  "yuanlongping", ["agmmu", "multimodal", "agriculture", "usda", "crop-disease", "neurips"])

A("[HybridRice] 杂交水稻育种科学",
  "Hybrid Rice Breeding Science: 1. 三系法: 不育系(A)/保持系(B)/恢复系(R), AxB繁殖不育系, AxR生产杂交种 2. 两系法: 光温敏核不育系, 长日高温->不育(制种), 短日低温->可育(繁殖) 3. 超级稻: 袁隆平团队, 2000年700kg/亩->2004年800kg/亩->2011年900kg/亩->2014年1000kg/亩 4. 第三代杂交稻: 基因工程雄性不育系, 结合三系法稳定性与两系法灵活性 5. 海水稻(耐盐碱稻): 目标利用1亿亩盐碱地, 2020年示范亩产超400kg 6. 分子育种: 全基因组选择(GS), GWAS定位产量QTL, CRISPR-Cas9改良抗性 7. 国际推广: 杂交稻已在全球70+国家种植",
  "yuanlongping", ["hybrid-rice", "yuan-longping", "breeding", "three-line", "two-line", "super-rice"])

A("[PrecisionAg] 精准农业技术体系",
  "Precision Agriculture Technology System: 1. 核心原理: 基于空间变异性的精细管理, 田块级->小区级->单株级精准施策 2. 信息采集: UAV多光谱/高光谱成像, 卫星遥感(Sentinel-2/Landsat), 土壤传感器(IoT) 3. 变量技术(VRT): 变量施肥(VRF)/变量喷药(VRA)/变量灌溉(VRI) 4. 农业机器人: 自动驾驶拖拉机(RTK-GPS, ±2cm精度), 除草机器人, 采摘机器人 5. 决策支持: 作物模型(DSSAT/APSIM/AquaCrop), 遥感反演(NDVI/LAI), 产量预测(ML) 6. 数据平台: Climate FieldView/John Deere Operations Center/FarmLogs 7. 经济效益: 平均增产5-15%, 减少化肥10-30%, 减少农药20-50%",
  "yuanlongping", ["precision-agriculture", "variable-rate", "uav", "iot", "crop-model", "agricultural-robot"])

A("[STAR-Dataset] 农业遥感标注数据集",
  "STAR Satellite-based Agricultural Remote Sensing Dataset (USDA+NASA Harvest): 1. USDA NASS与NASA Harvest合作创建 2. 数据源: Sentinel-2(10m)/Landsat-8(30m)/PlanetScope(3m)多源卫星 3. 标注来源: USDA NASS Cropland Data Layer(CDL), 每年覆盖美国48州 4. 作物类型: 100+类别, 含玉米/大豆/小麦/棉花/水稻 5. 时间序列: 每个田块全年生长季多时相影像(平均20+观测) 6. 应用: 作物分类/种植面积估算/长势监测/产量预测/灾害评估 7. 基准: 时空Transformer/3D-CNN/LSTM, 最佳分类F1>0.9",
  "yuanlongping", ["star", "satellite", "usda", "nasa", "crop-classification", "sentinel-2"])

A("[FoodSecurity] 全球粮食安全与农业可持续发展",
  "Global Food Security & Agricultural Sustainability: 1. 现状: 全球80亿人口, 7.35亿人面临饥饿(FAO 2023), 需2050年增产60% 2. 挑战: 气候变化/耕地退化(每年1200万公顷)/水资源短缺/病虫害/人口增长 3. 技术路径: 基因改良(抗旱/抗病/耐盐碱品种)/精准农业/垂直农场/替代蛋白 4. 袁隆平愿景: 种子是农业的芯片, 杂交稻+海水稻+分子育种三驾马车 5. 联合国SDG 2: 零饥饿, 2030年消除一切形式的饥饿与营养不良 6. 中国贡献: 用9%耕地养活18%人口, 杂交稻占水稻面积57%, 单产比常规稻高20% 7. 前沿: 合成生物学(固氮水稻)/AI育种/数字孪生农场/气候智能型农业",
  "yuanlongping", ["food-security", "sustainability", "fao", "sdg2", "climate-smart-agriculture"])

A("[SEC-EDGAR] SEC金融数据集",
  "SEC EDGAR Financial Dataset (U.S. Securities and Exchange Commission): 1. SEC官方EDGAR系统数据, 全球最权威的上市公司财务披露数据库 2. 数据内容: 10-K(年报)/10-Q(季报)/8-K(重大事件)/DEF 14A(代理声明)/13-F(机构持仓) 3. 覆盖范围: 全部美股上市公司(8,000+), 时间跨度1993年至今 4. 结构化字段: 公司信息/财务报表/管理层讨论与分析/风险因素/附注 5. NLP应用: 财报情感分析/风险预警/会计欺诈检测/管理层语气分析 6. 量化应用: 因子构建(价值/质量/动量)/事件驱动策略/另类数据 7. 数据格式: XBRL(结构化商业报告语言), SEC API提供实时访问",
  "strategy", ["sec", "edgar", "financial-data", "10-k", "xbrl", "quantitative-finance"])

A("[FinLLM] 金融大语言模型",
  "Financial Large Language Models: 1. BloombergGPT: Bloomberg 500亿参数金融LLM, 3630亿token金融数据+3450亿token通用数据 2. FinGPT: 开源金融LLM框架, HuggingFace托管, 支持LoRA微调 3. ConvFinQA: 多步数值推理数据集, 基于财报问答, 需多跳推理+计算 4. FLUE基准: 金融语言理解评估, 5个任务(情感分析/新闻分类/NER/关系抽取/问答) 5. 训练数据: SEC EDGAR/Reuters/Bloomberg/Federal Reserve/财报电话会议 6. 应用: 智能投研/风险评估/合规检查/交易信号生成 7. 挑战: 金融术语歧义/数值推理不足/时效性要求高/幻觉风险",
  "strategy", ["finllm", "bloomberggpt", "fingpt", "financial-nlp", "sentiment-analysis"])

A("[QuantTrading] 量化交易策略体系",
  "Quantitative Trading Strategy System: 1. 因子投资: Fama-French三因子->五因子->q因子模型(Hou-Xue-Zhang) 2. 统计套利: 配对交易(协整检验)/均值回归(Bollinger Bands)/动量策略 3. 机器学习策略: 随机森林/XGBoost/LightGBM/LSTM/Transformer 4. 高频交易: 做市策略/延迟套利/订单流预测/微观结构分析 5. 风险管理: VaR/CVaR/最大回撤/凯利公式(仓位管理) 6. 回测框架: Backtrader/Zipline/VectorBT, 防过拟合(Walk-Forward/交叉验证) 7. 执行优化: TWAP/VWAP/IS算法, 市场冲击模型(Almgren-Chriss)",
  "strategy", ["quantitative-trading", "factor-investing", "statistical-arbitrage", "hft", "risk-management"])

A("[Econometrics] 计量经济学与因果推断",
  "Econometrics & Causal Inference: 1. 线性回归: OLS假设(Gauss-Markov), 异方差稳健标准误(White/HC0-HC3) 2. 面板数据: 固定效应vs随机效应(Hausman检验), 动态面板(GMM/Arellano-Bond) 3. 因果推断: 工具变量(IV, 2SLS)/DID(双重差分)/RDD(断点回归)/SCM(合成控制法) 4. 时间序列: ARIMA/VAR/VECM(协整), 单位根检验(ADF/PP/KPSS), 格兰杰因果 5. 贝叶斯方法: MCMC/Gibbs采样/变分推断, 贝叶斯模型平均(BMA) 6. 诺贝尔奖级贡献: Heckman选择模型/Engle-GARCH/Granger因果/Sims VAR 7. 现代因果框架: 潜在结果模型(Rubin)/DAG(Pearl)/do-演算/反事实推理",
  "strategy", ["econometrics", "causal-inference", "panel-data", "instrumental-variables", "did"])

A("[MarketMicro] 市场微观结构与交易机制",
  "Market Microstructure & Trading Mechanisms: 1. 订单类型: 市价单/限价单/止损单/冰山单/条件单(OCO) 2. 订单簿: 五档行情(Bid/Ask), 买卖价差(Spread), 深度(Depth), 不平衡度 3. 价格发现: Glosten-Milgrom信息模型, Kyle Lambda(价格冲击系数) 4. 做市机制: 连续竞价/集合竞价/做市商制度, NASDAQ vs NYSE 5. 暗池(Dark Pool): 不公开显示的替代交易系统, 占美股成交量约15% 6. 监管框架: Reg NMS(美国)/MiFID II(欧洲)/科创板做市制度 7. 高频数据: TAQ数据, 毫秒级时间戳, 每日数亿条记录",
  "strategy", ["market-microstructure", "order-book", "dark-pool", "reg-nms", "price-discovery"])

A("[Derivatives] 衍生品定价与风险管理",
  "Derivatives Pricing & Risk Management: 1. Black-Scholes模型: C=S*N(d1)-K*e^(-rT)*N(d2), 假设: 几何布朗运动/无摩擦市场/连续交易 2. 希腊字母: Delta(方向风险)/Gamma(曲率风险)/Theta(时间衰减)/Vega(波动率风险)/Rho(利率风险) 3. 隐含波动率: VIX指数(恐慌指数), 波动率微笑/波动率曲面 4. 数值方法: 二叉树(Cox-Ross-Rubinstein)/Monte Carlo/有限差分法 5. 奇异期权: 亚式(均价)/障碍(敲入敲出)/回望(极值)/复合/彩虹(多资产) 6. 信用衍生品: CDS(信用违约互换)/CDO(债务抵押债券) 7. 风险对冲: Delta中性对冲/Delta-Gamma对冲/Vega对冲",
  "strategy", ["derivatives", "black-scholes", "greeks", "volatility", "hedging", "options"])

A("[AltData] 另类数据与投资信号",
  "Alternative Data & Investment Signals: 1. 卫星数据: 停车场车辆计数/油罐阴影分析/农作物长势 2. 网络数据: 网页抓取/搜索趋势(Google Trends)/社交媒体情绪 3. 地理位置数据: 手机GPS聚合/信用卡交易/AIS船舶追踪 4. 文本数据: 财报电话会议NLP/新闻情感/专利数据/招聘信息/供应链关系 5. 物联网数据: 天气传感器/交通流量/电力消耗/工业生产指标 6. 数据处理: 噪声过滤/特征工程/回测验证(样本外检验) 7. 合规: GDPR/CCPA数据隐私, 内幕交易红线, SEC监管态度",
  "strategy", ["alternative-data", "satellite", "sentiment", "web-scraping", "geolocation"])

A("[PortfolioOpt] 投资组合优化理论",
  "Portfolio Optimization Theory: 1. Markowitz均值-方差模型: min variance s.t. E[Rp]=目标收益, 有效前沿 2. CAPM: E[Ri]=Rf+Beta*(E[Rm]-Rf), Beta=系统性风险度量 3. Black-Litterman模型: 结合市场均衡与投资者观点, 贝叶斯框架 4. 风险平价(Risk Parity): 每个资产贡献相等风险, Bridgewater All Weather策略 5. 因子配置: 基于宏观因子(增长/通胀/实际利率/信用)配置 6. 约束优化: 行业权重限制/换手率约束/跟踪误差约束, 二次规划求解 7. 动态配置: 宏观周期轮动(美林时钟)/估值指标(CAPE)/趋势跟踪",
  "strategy", ["portfolio-optimization", "markowitz", "capm", "black-litterman", "risk-parity"])

A("[CERN-ColliderML] CERN粒子碰撞机器学习数据集",
  "CERN ColliderML Dataset (CERN+JGU Mainz+Niels Bohr Institute+LBNL, arXiv 2512.15230): 1. 全球首个OpenDataDetector高亮度LHC物理基准数据集, HuggingFace(CERN/ColliderML-Release-1) 2. 数据规模: 100万个质子-质子碰撞事件, 10种标准模型与超标准模型物理过程 3. 仿真条件: sqrt(s)=14TeV, 平均pile-up=200(高亮度LHC条件) 4. 物理过程: ttbar/W+jets/Z+jets/QCD/Higgs/single top/BSM信号 5. 探测器: OpenDataDetector几何, DD4hep描述, 完整仿真+数字化+重建 6. 矩阵元: NLO(次领头阶)精度, 现代事例生成+簇射+pile-up叠加 7. ML应用: 事例分类/喷注标记/粒子识别/异常检测/快速仿真代理模型 8. 格式: ROOT TTree, 可通过uproot读取",
  "einstein", ["cern", "colliderml", "lhc", "particle-physics", "higgs", "ml-benchmark"])

A("[LIGO-GWOSC] LIGO引力波事件数据集",
  "LIGO/Virgo/KAGRA Gravitational Wave Events (GWOSC, HuggingFace: juliensimon/gravitational-wave-events): 1. 数据来源: 引力波开放科学中心(GWOSC), LIGO/Virgo/KAGRA联合观测 2. 事件数量: 263个已确认引力波事件, 每周自动更新 3. 事件类型: 黑洞-黑洞并合/中子星-中子星并合/中子星-黑洞并合 4. 关键参数: 主源质量(中位数35.3太阳质量)/啁啾质量/光度距离(中位数2100Mpc)/红移/有效自旋/网络SNR 5. 目录: GWTC-1/2.1/3/4.0等12个数据发布, 含O1/O2/O3运行期数据 6. 数据格式: HDF5, 含应变数据(strain)+触发器(triggers)+参数估计后验样本 7. ML应用: 信号检测(CNN/Transformer)/参数估计(归一化流)/噪声表征/异常检测 8. 2017诺贝尔物理学奖: 引力波探测, 开启多信使天文学时代",
  "einstein", ["ligo", "gravitational-waves", "gwosc", "black-hole", "neutron-star", "nobel-prize"])

A("[MIT-TIDMAD] 暗物质AI探测数据集",
  "TIDMAD Dark Matter AI Detection Dataset (MIT, arXiv 2406.04378, NeurIPS 2024): 1. MIT物理系+ABRACADABRA实验, 全球首个暗物质AI搜索基准数据集 2. 实验原理: ABRACADABRA探测器搜索轴子暗物质产生的振荡磁场信号 3. 数据规模: 超长时间序列, 采样率10MHz(每秒1000万采样点), 含训练/验证/科学三组 4. 信号特征: 暗物质信号表现为超长时序中的正弦振荡模式, 频率范围100Hz-10MHz 5. 基准任务: 时序去噪评分(Denoising Score)+暗物质排除限(Dark Matter Limit)双指标评估 6. 模型: UNet去噪网络+频域滤波器, 可直接产出物理级暗物质搜索结果 7. 意义: 暗物质占宇宙物质85%, 探测将是诺贝尔奖级突破, 本数据集为AI+粒子物理交叉提供标准基准 8. 许可: CC-BY-4.0, HuggingFace平台托管",
  "einstein", ["dark-matter", "tidmad", "mit", "abracadabra", "axion", "time-series", "neurips"])

A("[MultimodalUniverse] 多模态宇宙天文数据集",
  "Multimodal Universe Dataset (UC Berkeley+Flatiron Institute+Oxford+Polymathic AI, NeurIPS 2024): 1. 全球最大多模态天文数据集, 100TB, HuggingFace平台托管 2. 数据规模: 1.2亿+星系图像/500万+恒星光谱/350万+光变曲线/2.2亿+Gaia星表测量 3. 数据类型: 图像(Legacy Surveys DR10/HSC/JWST)/光谱(Gaia BP-RP/SDSS/DESI)/时序(PLAsTiCC/TESS)/超光谱(MaNGA)/表格(Gaia) 4. 来源: 20+主要天文巡天项目, 含NASA/ESA/NOIRLab/SDSS合作数据 5. 基准任务: 星系分类/超新星预警/恒星参数估计/红移预测/异常检测 6. 学术机构: UC Berkeley主导, Flatiron Institute, Oxford, Polymathic AI联盟 7. 意义: 为天文基础模型训练提供统一数据平台, 推动AI+天文学范式转变 8. 许可: 各子集遵循原始数据许可, HuggingFace预览集可直接访问",
  "einstein", ["multimodal-universe", "astronomy", "jwst", "gaia", "desi", "neurips", "flatiron"])

A("[NASA-INDUS] NASA领域语言模型",
  "NASA INDUS Domain-Specific Language Models (NASA+IBM Research, arXiv 2405.10725): 1. NASA科学任务局(SMD)与IBM Research联合开发, 面向天体物理/地球科学/行星科学/日球层物理/生物物理科学 2. 模型家族: INDUS Encoder(文本嵌入)/INDUS Decoder(文本生成), 基于科学文献训练 3. 训练数据: NASA技术报告/ADS天体物理文摘/地球科学论文/行星科学文献 4. NASA-IR基准: 498个问答对, 覆盖5个科学领域, Recall@10评估指标 5. 特色: 科学术语理解(如chondrite/permittivity/heliopause)/数值推理/多领域适应 6. 应用: 科学文献检索/知识图谱构建/自动摘要/跨领域科学问答 7. 开放获取: HuggingFace平台(nasa-impact/), 支持科学NLP研究复现",
  "einstein", ["nasa", "indus", "ibm", "scientific-llm", "astrophysics", "earth-science"])

A("[QuantumLLMInstruct] 量子计算指令微调数据集",
  "QuantumLLMInstruct Dataset (Johns Hopkins University, arXiv 2412.20956): 1. 约翰斯·霍普金斯大学创建, 全球最大量子计算指令微调数据集, 50万+问题-解决方案对 2. 来源: 90+主要种子领域, LLM自动生成数百个子领域, 显著提升量子计算数据集多样性 3. 核心领域: 合成哈密顿量/QASM代码生成/Jordan-Wigner变换/Trotter-Suzuki量子电路分解 4. 四阶段构建: 问题生成->解决方案开发->数据集增强(CoT+ToRA推理)->零样本Judge LLM质量验证 5. 格式: 自然语言提示+LaTeX数学表达式, 确保清晰性和精确性 6. 应用: 量子计算LLM指令微调, 提升LLM在复杂量子物理问题中的表现 7. 许可: CC-BY-4.0, HuggingFace平台(BoltzmannEntropy/QuantumLLMInstruct)",
  "einstein", ["quantum-computing", "quantumllminstruct", "johns-hopkins", "qasm", "hamiltonian", "instruction-tuning"])

A("[Relativity] 广义相对论与宇宙学基础",
  "General Relativity & Cosmology Fundamentals: 1. 爱因斯坦场方程: G_mu_nu + Lambda*g_mu_nu = (8*pi*G/c^4)*T_mu_nu, 描述时空曲率与物质能量分布关系 2. Schwarzschild解: 球对称真空解, 预言黑洞(事件视界 r=2GM/c^2), 引力红移, 时间膨胀 3. Kerr解: 旋转黑洞解, 事件视界+ergosphere, 帧拖拽效应(Lense-Thirring) 4. FLRW度规: 均匀各向同性宇宙, a(t)尺度因子, Hubble参数H=a_dot/a, 描述宇宙膨胀 5. 宇宙学参数: Omega_m=0.315(物质)/Omega_Lambda=0.685(暗能量)/H0=67.4 km/s/Mpc(Planck 2018) 6. 引力透镜: 强透镜(弧/环)/弱透镜(剪切场)/微透镜(亮度放大), 探测暗物质分布 7. 宇宙微波背景(CMB): T=2.725K, 各向异性Delta T/T~10^-5, 角功率谱约束宇宙学参数",
  "einstein", ["general-relativity", "cosmology", "black-hole", "dark-energy", "cmb", "gravitational-lensing"])

A("[StandardModel] 粒子物理标准模型",
  "Particle Physics Standard Model: 1. 基本费米子(12种): 6夸克(u/d/c/s/t/b)+6轻子(e/mu/tau+3种中微子), 自旋1/2 2. 规范玻色子(4种): 胶子(g, 强力)/光子(gamma, 电磁力)/W/Z(弱力), 自旋1 3. Higgs玻色子: 自旋0, 2012年LHC ATLAS+CMS发现(125.1 GeV), 自发对称性破缺赋予粒子质量 4. 强力: QCD(量子色动力学), SU(3)规范对称, 渐近自由(高能下夸克自由), 禁闭(低能下夸克束缚) 5. 电弱统一: SU(2)xU(1), Weinberg角(theta_W~30度), W/Z质量~80/91 GeV 6. 标准模型局限: 未包含引力/暗物质/暗能量/中微子质量/物质-反物质不对称 7. 超越标准模型(BSM): 超对称(SUSY)/大统一理论(GUT)/弦理论/额外维",
  "einstein", ["standard-model", "higgs", "qcd", "particle-physics", "bsm", "fermion"])

A("[QuantumMech] 量子力学基础理论",
  "Quantum Mechanics Fundamentals: 1. 波函数: Psi(x,t), |Psi|^2=概率密度, Born定则, 归一化条件 2. Schrodinger方程: i*hbar*dPsi/dt = H*Psi, H=哈密顿算符, 定态方程H*psi=E*psi 3. 不确定性原理: Delta_x*Delta_p >= hbar/2, Delta_E*Delta_t >= hbar/2, 非测量局限而是内禀性质 4. 量子纠缠: Bell不等式违反, EPR佯谬, 非定域关联, 2022诺贝尔奖(Aspect/Clauser/Zeilinger) 5. 量子态叠加: |psi> = alpha|0> + beta|1>, |alpha|^2+|beta|^2=1, 量子比特基础 6. 量子隧穿: 透射概率T~exp(-2*kappa*a), 扫描隧穿显微镜(STM)/核聚变/半导体器件核心机制 7. 量子场论: 粒子=场的激发态, Feynman图微扰计算, 重整化, QED精度达10^-12",
  "einstein", ["quantum-mechanics", "schrodinger", "entanglement", "uncertainty", "quantum-field-theory"])

A("[Thermodynamics-StatMech] 热力学与统计力学",
  "Thermodynamics & Statistical Mechanics: 1. 热力学四大定律: 第零定律(热平衡传递性)/第一定律(dU=deltaQ-deltaW, 能量守恒)/第二定律(dS>=0, 熵增)/第三定律(S->0 as T->0K) 2. 熵的统计解释: S = k_B * ln(Omega) (Boltzmann), 微观态数Omega的对数度量无序度 3. 配分函数: Z = Sum(exp(-beta*E_i)), beta=1/(k_B*T), 所有热力学量的母函数 4. Maxwell-Boltzmann分布: f(v) = 4*pi*(m/(2*pi*k_B*T))^(3/2)*v^2*exp(-mv^2/(2*k_B*T)) 5. Fermi-Dirac分布: f(E) = 1/(exp((E-mu)/(k_B*T))+1), 电子/中微子等费米子 6. Bose-Einstein分布: f(E) = 1/(exp((E-mu)/(k_B*T))-1), 光子/声子等玻色子, BEC凝聚 7. 相变: 一级(潜热, 体积突变)/连续(比热发散, 临界指数), Landau理论, 重整化群",
  "einstein", ["thermodynamics", "statistical-mechanics", "entropy", "boltzmann", "partition-function", "phase-transition"])

print(f"Total new entries generated: {len(N)}")
print(f"Existing entries in file: {len(data)}")
print(f"Skipped duplicates: {len([t for t in {e['topic'] for e in N} if t in existing_topics])}")

data.extend(N)

with open(FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Written {len(data)} total entries to {FILE}")

soul_counts = {}
for e in data:
    soul_counts[e["soul"]] = soul_counts.get(e["soul"], 0) + 1
print("\nBy soul:")
for soul, count in sorted(soul_counts.items(), key=lambda x: -x[1]):
    print(f"  {soul:>15}: {count}")
