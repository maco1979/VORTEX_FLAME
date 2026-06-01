#!/usr/bin/env python
"""V3 Knowledge Generator — Complete all 14 souls + Logic/Chemistry KBs."""
import json

entries = []

def add(topic, text, soul, tags, cat='knowledge'):
    entries.append({'topic': topic, 'text': text, 'soul': soul, 'tags': tags, 'category': cat})

# ========================================================================
# PART A: 5 MISSING SOULS — Structured Domain Rules
# ========================================================================

# --- Galileo (Physics + IoT + Astronomy + Math) ---
add('[PhysicsSim] 物理仿真与数字孪生规范',
'Physical Simulation & Digital Twin Rules:'
' 1. CPHYSJEPA世界模型为物理仿真核心引擎, 负责刚体动力学/流体力学/电磁场/热力学的实时推演'
' 2. 物理仿真参数必须基于真实物理常数(NIST/CODATA), 禁止编造物理参数'
' 3. 数字孪生三层架构: 物理层(传感器实时数据)→模型层(CPHYSJEPA状态表征)→决策层(VORTEX规则+LLM)'
' 4. 仿真误差超过阈值时自动触发CPHYSJEPA在线微调, 调整物理参数匹配实测'
' 5. 实时性要求: 简单物理(<100体)<10ms, 流体仿真<100ms, 复杂多物理场<1s'
' 6. 物理单位体系: SI国际单位制, 角度用弧度, 温度用开尔文(K)',
'galileo', ['physics-sim', 'digital-twin', 'c-phys-jepa', 'si-units'])

add('[IoTEmbedded] 物联网传感器嵌入式规则',
'IoT Embedded Sensor Rules:'
' 1. 传感器数据通过serial_port/arduino_board/raspberry_pi三通道经设备网关统一采集'
' 2. 采样频率: 低频传感器(温湿度/气压)1Hz, 中频(加速度/陀螺仪)100Hz, 高频(麦克风/振动)最高44.1kHz'
' 3. 传感器原始数据先经CPHYSJEPA编码为时序表征向量(128-dim), 再入MCP记忆库存储'
' 4. 数据质量检测: 断线检测(心跳包>/=1Hz), 异常值检测(3σ法则), 漂移检测(滑动窗口Kalman滤波)'
' 5. 嵌入式设备上运行的轻量JEPA推理模型(JEPA-Tiny, <1M参数)负责本地预处理与异常预警'
' 6. 串口通信: 115200bps默认, 8N1格式, Modbus RTU/ASCII工业协议优先',
'galileo', ['iot', 'sensor', 'embedded', 'arduino', 'raspberry-pi'])

add('[AstroSpace] 天文空间物理仿真规则',
'Astronomy & Space Physics Simulation Rules:'
' 1. 天体力学: Kepler轨道要素, N体问题数值积分(RK4/RK45), 摄动力计算'
' 2. 空间环境模型: 太阳辐射压, 大气阻力模型(NRLMSISE-00), 地球磁场(IGRF/WMM)'
' 3. 天文观测模拟: 星表(GAIA DR3/Hipparcos), 视星等计算, 大气消光修正'
' 4. 空间任务规划: Lambert转移轨道, Hohmann转移, 引力辅助轨道设计'
' 5. CPHYSJEPA空间物理变体: 太阳风/磁层/电离层耦合建模, 空间天气预测',
'galileo', ['astronomy', 'space-physics', 'orbital-mechanics', 'astrodynamics'])

add('[MathModel] 数学模型符号计算约束',
'Mathematical Modeling & Symbolic Computation Rules:'
' 1. 符号计算引擎: SymPy代数运算, Wolfram/Maple风格公式推导, LaTeX渲染'
' 2. 微分方程: ODE初值问题(显式RK/隐式BDF), PDE有限差分/有限元/谱方法'
' 3. 优化问题: 线性规划(Simplex), 非线性优化(SLSQP/L-BFGS), 全局优化(差分进化/贝叶斯优化)'
' 4. 统计学: 假设检验, 贝叶斯推断(MCMC), 时间序列分析(ARIMA/GARCH), 因果推断(do-calculus)'
' 5. 所有数学推导必须展示中间步骤, 结果用LaTeX格式输出, 符号定义明确'
' 6. JEPA世界模型中的数学约束: 物理守恒律自动校验, 量纲分析自检',
'galileo', ['math', 'symbolic', 'optimization', 'statistics', 'differential-equations'])

# --- VanGogh (Visual Art + Color + Style + History) ---
add('[ArtTheory] 视觉艺术理论与风格约束',
'Visual Art Theory & Style Constraints:'
' 1. CVJEPA视觉世界模型负责图像/视频/视觉场景的编码与因果推理'
' 2. 艺术风格理论: 写实主义/印象派/表现主义/立体主义/抽象/超现实主义/极简主义等形式分析'
' 3. 视觉元素分析: 线条(方向/粗细/质感), 形状(几何/有机), 空间(透视/景深/层次)'
' 4. 构图法则: 三分法/黄金比例(1:1.618)/对称与非对称/引导线/框架构图/留白法则'
' 5. 光源分析: 自然光(日光/月光/逆光/侧光), 人工光(点光/面光/聚光), 明暗对比(Chiaroscuro)'
' 6. 视觉叙事: 视觉隐喻/象征符号/情绪传递/视觉节奏与韵律',
'vangogh', ['art-theory', 'composition', 'lighting', 'visual-narrative', 'c-v-jepa'])

add('[ColorComposition] 色彩构成与视觉设计规则',
'Color Composition & Visual Design Rules:'
' 1. 色彩理论: 色相环(12色), 三原色(RGB/CMYK), 补色/类似色/三角色/分裂补色配色方案'
' 2. 色彩心理学: 暖色(红橙黄)情绪激活/冷色(蓝绿紫)冷静专注/中性色(黑白灰)平衡与留白'
' 3. 色彩空间: sRGB/AdobeRGB/DCI-P3/Rec.2020, L*a*b*均匀感知空间, HSV/HSL直观调色'
' 4. 色彩和谐度量化: 色彩对比度(WCAG 2.1 AA>=4.5:1), 色彩平衡度, 色温(Kelvin)'
' 5. 数字色彩管理: ICC色彩配置文件, Gamma校正, HDR/EOTF(ST.2084/HLG)'
' 6. 视觉设计工具: Figma/Sketch/Adobe XD的统一设计Token与Design System规范',
'vangogh', ['color-theory', 'color-psychology', 'color-space', 'design-system'])

add('[VisualStyle] 绘画风格流派知识库',
'Painting Styles & Art Movements Knowledge Base:'
' 1. 文艺复兴(14-16C): 透视法发明, 解剖学准确, 明暗法, 代表: Leonardo/Michelangelo/Raphael'
' 2. 巴洛克(17C): 戏剧性光影, 动态构图, 情感张力, 代表: Caravaggio/Rembrandt/Rubens'
' 3. 印象派(19C): 外光写生, 色彩分解, 瞬间印象, 代表: Monet/Renoir/Degas'
' 4. 后印象派: 主观表达, 结构强化, 代表: VanGogh(笔触表现)/Cezanne(几何结构)/Gauguin(象征色彩)'
' 5. 现代主义(20C): 抽象/表现/立体/超现实/极简/波普艺术, 打破传统透视与具象'
' 6. 数字艺术(21C): AI生成艺术(Stable Diffusion/Midjourney/Sora), 参数化设计, 生成艺术(Processing/p5.js)',
'vangogh', ['art-history', 'painting', 'style', 'movement', 'digital-art'])

add('[ArtHistory] 艺术史与视觉文化分析',
'Art History & Visual Culture Analysis:'
' 1. 艺术史方法论: 形式分析(Fry/Bell), 图像学(Panofsky三层), 社会艺术史(Hauser/Clark), 视觉文化研究'
' 2. 东西方艺术比较: 中国山水画(散点透视/笔墨/留白)vs西方风景画(焦点透视/光影/色彩)'
' 3. 艺术品鉴定: 风格分析/颜料年代/画布材质/X射线/红外线/笔触分析/碳14定年'
' 4. 艺术市场: 拍卖记录, 画廊体系, 艺术基金, 区块链数字艺术(NFT)确权'
' 5. 视觉文化数据库: WikiArt(250K+图像), Google Arts&Cultures, MET博物馆开放数据, Rijksmuseum API',
'vangogh', ['art-history', 'visual-culture', 'authentication', 'art-market'])

# --- Montesquieu (Law + Logic + Compliance) ---
add('[LegalCompliance] 法律合规与合规审查规则',
'Legal Compliance & Regulatory Review Rules:'
' 1. CLAWJEPA法律世界模型负责法律条文/判例/法理的因果推理与一致性和合规性校验'
' 2. 法律检索: 法条数据库, 判例数据库, 法学期刊, 立法解释与司法解释'
' 3. 合规审查框架: GDPR个人数据保护, ISO合规标准(27001安全/9001质量/14001环境/45001职业健康)'
' 4. 合同审核: 条款完整性检查, 权利义务对等分析, 风险条款标注, 法律冲突检测'
' 5. 知识产权: 专利检索(IPC分类), 商标检索(Nice分类), 著作权(Fair Use四要素/CC协议)'
' 6. 跨境合规: 出口管制(EAR/ITAR), 经济制裁(OFAC), 反洗钱(AML), 反贿赂(FCPA/UK Bribery Act)',
'montesquieu', ['legal', 'compliance', 'gdpr', 'contract', 'intellectual-property'])

add('[RuleEngine] 规则引擎与合规自动化',
'Rule Engine & Compliance Automation:'
' 1. VORTEX规则引擎: 白名单/黑名单双通道, 支持AND/OR/NOT逻辑组合, 正则表达式匹配'
' 2. 规则语言: JSON Schema定义, Drools风格的when→then规则, 支持规则优先级与冲突解决'
' 3. 法律逻辑验证: 三段论(大前提→小前提→结论), 法律推理的演绎/归纳/类比模式'
' 4. 合规模块: 自动化合同审查, 法规变动监控(Gazette/RSS), 合规差距分析(Gap Analysis)'
' 5. 审计追踪: 所有合规决策记录完整推理链(Chain of Custody), 可回溯可复现',
'montesquieu', ['rule-engine', 'compliance-automation', 'legal-logic', 'audit-trail'])

# --- YuanLongping (Agriculture + Food + Genetics + Safety) ---
add('[AgriScience] 农业科学与作物育种规则',
'Agricultural Science & Crop Breeding Rules:'
' 1. CBIOJEPA生物农业变体: 基因组→表型预测, 环境因素影响建模, 产量预测'
' 2. 作物育种: 杂交育种(杂种优势利用), 分子标记辅助选择(MAS), 基因编辑(CRISPR-Cas9), 全基因组选择(GS)'
' 3. 精准农业: GPS导航自动驾驶, 变量施肥(VRT), 无人机遥感(NDVI/多光谱), 土壤传感器网络'
' 4. 水稻种植: 籼稻/粳稻/杂交稻, 栽培技术(育秧/插秧/水肥管理/病虫害防治), 超级稻品种(900-1200kg/亩)'
' 5. 农业气象: 积温计算, 有效降水, 霜冻预警, 干旱指数(SPI/SPEI), 作物模型(DSSAT/APSIM)'
' 6. 盐碱地改良: 耐盐品种筛选, 水利排盐, 化学改良(石膏/硫磺), 生物改良(耐盐微生物)',
'yuanlongping', ['agriculture', 'breeding', 'rice', 'precision-ag', 'c-bio-jepa'])

add('[FoodTech] 食品科学与加工技术规范',
'Food Science & Processing Technology:'
' 1. 食品化学: 美拉德反应(褐变), 脂质氧化(酸败), 蛋白质变性, 酶促反应, 淀粉糊化/老化'
' 2. 食品加工: 热加工(巴氏/超高温/蒸煮), 冷加工(冷冻/冷藏/冻干), 发酵(乳酸菌/酵母/霉菌)'
' 3. 食品添加剂: GB2760标准, 防腐剂/抗氧化剂/乳化剂/增稠剂/甜味剂/色素/香精'
' 4. 食品安全: HACCP体系(危害分析关键控制点), ISO22000, GMP良好生产规范, 可追溯体系'
' 5. 食品检测: 色谱(HPLC/GC), 质谱(LC-MS/GC-MS), 光谱(NIR/FTIR), PCR微生物检测'
' 6. 粮食储藏: 温湿度控制/防虫防霉/气调储藏/低温储藏, 储备粮管理制度',
'yuanlongping', ['food-science', 'processing', 'additives', 'haccp', 'food-detection'])

add('[CropGenetics] 作物遗传与生物技术规则',
'Crop Genetics & Biotechnology Rules:'
' 1. 分子生物学基础: DNA/RNA/蛋白质中心法则, PCR/qPCR/RT-PCR, 凝胶电泳, 测序(NGS三代)'
' 2. 基因组学: 全基因组测序, GWAS全基因组关联分析, QTL数量性状位点定位, 泛基因组(Pan-genome)'
' 3. 转基因技术: 农杆菌介导转化, 基因枪法, 启动子选择(组成型CaMV35S/组织特异性), 筛选标记(Kan/Hyg/Bar)'
' 4. 基因编辑: CRISPR-Cas9/Cas12a/Cas13, Prime Editing, Base Editing, 基因驱动(Gene Drive)'
' 5. 生物安全: 转基因作物环境释放评估, 食品安全性评价(实质等同性原则), 基因漂移风险评估'
' 6. 种质资源: 种质库(国家作物种质资源库), 核心种质构建, 遗传多样性评价(SNP/SSR标记)',
'yuanlongping', ['genetics', 'biotechnology', 'crispr', 'gmo', 'germplasm'])

add('[FoodSafety] 食品安全与质量控制规范',
'Food Safety & Quality Control Standards:'
' 1. 中国国标(GB): 食品安全国家标准体系, GB2760(添加剂), GB2761(真菌毒素), GB2762(污染物), GB2763(农药残留)'
' 2. 国际标准: Codex Alimentarius(食品法典), FDA(美国食品药品管理局), EFSA(欧洲食品安全局)'
' 3. 食品污染物: 重金属(Pb/Cd/Hg/As), 农药残留(有机磷/有机氯/拟除虫菊酯), 兽药残留(抗生素/激素)'
' 4. 微生物安全: 菌落总数, 大肠菌群, 沙门氏菌, 金黄色葡萄球菌, 黄曲霉毒素B1/M1'
' 5. 食品掺假检测: DNA条形码(物种鉴定), 稳定同位素比率(产地溯源), 近红外光谱(成分快速检测)'
' 6. 应急响应: 食品安全事故分级(I-IV级), 召回程序, 溯源追查, 舆情应对',
'yuanlongping', ['food-safety', 'gb-standards', 'contaminants', 'microbiology', 'recall'])

# --- Herodotus (History + Chronology + Archives + Culture) ---
add('[ChronologyAnalysis] 历史时间线分析方法论',
'Historical Chronology Analysis Methodology:'
' 1. CGEOJEPA历史地理变体: 时空耦合建模, 文明兴衰的因果分析, 历史事件链推理'
' 2. 时间线构建: 绝对年代(碳14/树木年轮/冰芯/古地磁), 相对年代(地层学/类型学/交叉定年)'
' 3. 历史分期: 史前/古代/中世纪/近代/现代, 各地区独立分期(中国朝代/欧洲断代/中东王朝)'
' 4. 历史地理信息系统(HGIS): 时空数据可视化, 历史地图叠加, 人口迁徙路线还原'
' 5. 历史因果分析: 直接原因/根本原因/触发事件, 多因素交叉分析(经济/政治/气候/技术)'
' 6. 反事实历史推理: JEPA假设推演(如关键战役不同结果), 概率因果建模, 保持逻辑严谨',
'herodotus', ['chronology', 'timeline', 'dating-methods', 'hgis', 'counterfactual'])

add('[HistoricalMethod] 历史方法论与考证规范',
'Historical Methodology & Textual Criticism:'
' 1. 史料分级: 一手史料(档案/日记/考古实物), 二手史料(史书/传记/回忆录), 三手史料(研究论文/综述)'
' 2. 文本考证: 内证(语言风格/内容一致性), 外证(同时代其他记载/考古证据), 版本校勘(对校/本校/他校/理校)'
' 3. 口述历史: 访谈方法论, 记忆偏差校正, 交叉验证(多方印证), 伦理规范(知情同意)'
' 4. 历史统计: 量化历史(Cliometrics), 历史人口学, 经济史数据分析, 长时段(Longue Duree)视角'
' 5. 历史比较法: 横向比较(同期不同文明), 纵向比较(同文明不同时期), 模式识别与规律发现'
' 6. 历史解释流派: 兰克史学派(实证主义), 年鉴学派(结构史), 新文化史(符号与意义), 全球史(跨文明互动)',
'herodotus', ['historical-method', 'textual-criticism', 'oral-history', 'cliometrics'])

add('[ArchivalDoc] 档案文献记录与分析规则',
'Archival Documentation & Record Analysis:'
' 1. 档案分类: 政府档案(行政/司法/外交), 私人档案(书信/日记/手稿), 机构档案(企业/教会/学校)'
' 2. 数字化: OCR文字识别(古籍/手写体), 扫描标准(300-600DPI), 元数据标准(Dublin Core/EAD)'
' 3. 档案保护: 温湿度控制(18±2℃/50±5%RH), 防虫防霉, 酸性纸张脱酸处理, 数字化备份'
' 4. 古籍版本: 抄本/刻本/活字本/石印本, 版本鉴定(版式/字体/避讳/纸墨/装帧)'
' 5. 文献数据库: CNKI/万方/WOS/Scopus文献检索, Google Scholar, JSTOR, HathiTrust数字图书馆'
' 6. 档案情报分析: 来源可信度评估(Admiralty Scale 1-5), 信息交叉验证, 情报周期(OSINT流程)',
'herodotus', ['archives', 'digitization', 'conservation', 'rare-books', 'osint'])

add('[CulturalPattern] 文化模式与文明演化规律',
'Cultural Patterns & Civilization Evolution:'
' 1. 文明起源理论: 大河文明(灌溉农业→中央集权), 海洋文明(贸易→城邦民主), 游牧文明(迁徙→军事征服)'
' 2. 文化传播: 扩散模型(人口迁徙/贸易/传教/征服), 文化适应(本地化改造), 文化融合(Syncretism)'
' 3. 语言谱系: 印欧语系/汉藏语系/亚非语系/尼日尔-刚果语系, 历史语言学(比较法/内部重建)'
' 4. 宗教比较: 亚伯拉罕宗教(犹太/基督/伊斯兰), 印度宗教(印度教/佛教/耆那教), 东亚宗教(儒/道/神道)'
' 5. 文化记忆: 纪念仪式/纪念碑/博物馆/文化遗产(UNESCO世界遗产), 集体记忆与身份认同'
' 6. 技术史: 农业革命→青铜器→铁器→工业革命→信息革命→AI革命, 技术传播的S曲线模型',
'herodotus', ['civilization', 'culture', 'language-family', 'religion', 'technology-history'])

# ========================================================================
# PART B: 4 WEAK SOULS — Beef Up Structured Rules
# ========================================================================

# --- Darwin (Biology + Ecology + Genomics) ---
add('[BioLab] 生物实验室规范与分子生物学',
'Biology Laboratory & Molecular Biology Rules:'
' 1. CBIOJEPA生物世界模型: 分子生物学中心法则(DNA→RNA→蛋白质)的因果建模, 基因调控网络推理'
' 2. 实验技术: PCR/qPCR, 凝胶电泳(SDS-PAGE/琼脂糖), Western/Northern/Southern Blot, ELISA, 流式细胞术'
' 3. 测序技术: Sanger测序(一代), NGS(二代Illumina/MGI), 三代测序(PacBio HiFi/ONT纳米孔), 单细胞测序(scRNA-seq)'
' 4. 显微成像: 光学显微镜(明场/相差/DIC/荧光), 共聚焦显微镜, 电子显微镜(TEM/SEM), 超分辨率显微镜(STED/PALM/STORM)'
' 5. 实验设计: 对照组设置(阳性/阴性/空白), 生物学重复(n≥3), 随机化与盲法, 统计检验(t-test/ANOVA)'
' 6. 实验室安全: BSL-1/2/3/4生物安全等级, 化学试剂MSDS, 废弃物处理(生物危害/化学/放射性)',
'darwin', ['biology-lab', 'molecular-biology', 'sequencing', 'microscopy', 'c-bio-jepa'])

add('[EcologyModel] 生态建模与环境科学规则',
'Ecological Modeling & Environmental Science:'
' 1. 种群生态: Lotka-Volterra捕食-猎物模型, 逻辑斯蒂增长, 集合种群(Metapopulation), Allee效应'
' 2. 群落生态: 物种多样性(Shannon/H/Simpson指数), 种间竞争/捕食/共生/寄生关系网络'
' 3. 生态系统: 能量流动(初级生产力GPP/NPP), 物质循环(碳/氮/磷循环), 食物链/食物网动态'
' 4. 生物多样性: IUCN红色名录, 保护生物学, 栖息地破碎化评估, 生态廊道设计, 物种分布模型(SDM/MaxEnt)'
' 5. 气候变化生态: 物种迁徙/物候变化/珊瑚白化/极地冰融, 碳汇评估(森林/海洋/湿地)'
' 6. 环境DNA(eDNA): 水样/土壤/空气DNA采集, 宏条形码(Metabarcoding), 生物监测与入侵物种预警',
'darwin', ['ecology', 'biodiversity', 'climate-change', 'eDNA', 'ecosystem'])

add('[GenomeAnalysis] 基因组分析与生物信息学',
'Genome Analysis & Bioinformatics:'
' 1. 序列比对: BWA/Bowtie2短序列比对, Minimap2长序列比对, 多序列比对(ClustalW/MAFFT/MUSCLE)'
' 2. 变异检测: SNP/InDel(Samtools/GATK), 结构变异SV(Delly/Manta), CNV拷贝数变异'
' 3. 系统发育: 距离法(NJ/UPGMA), 最大简约法(MP), 最大似然法(ML/RAxML/IQ-TREE), 贝叶斯法(MrBayes/BEAST)'
' 4. 功能注释: GO/KEGG/COG/Pfam/InterPro数据库, 差异表达分析(DESeq2/edgeR/limma)'
' 5. 蛋白质结构: AlphaFold2/3预测, PDB数据库, 分子动力学模拟(GROMACS/AMBER), 分子对接(AutoDock Vina)'
' 6. 生信工具链: Bioconda/Bioconductor, Snakemake/Nextflow工作流, Jupyter/RStudio交互分析',
'darwin', ['bioinformatics', 'genomics', 'phylogenetics', 'protein-structure', 'alphafold'])

# --- Strategy (Finance + Trading + Economics) ---
add('[TradingRule] 量化交易与风险管理规则',
'Quantitative Trading & Risk Management:'
' 1. CFINJEPA金融世界模型: 多因子/多时间尺度的因果发现, 市场微观结构推理, 反事实情景推演'
' 2. 因子体系: 价值(B/P,E/P), 动量(12-1月收益), 质量(ROE/ROA), 波动率(已实现/隐含), 流动性(Amihud)'
' 3. 策略回测: Walk-forward优化, 样本外测试(OOS), 过拟合检测(CSCV/PBO), 交易成本建模(佣金+滑点+冲击)'
' 4. 风险管理: VaR(历史/参数/Monte Carlo), CVaR(Expected Shortfall), 压力测试(历史情景/假设情景)'
' 5. 组合优化: Markowitz均值方差, Black-Litterman, 风险平价(Risk Parity), 最大分散度(Max Diversification)'
' 6. 风控铁律: 单笔最大亏损≤2%净值, 连续回撤≥10%暂停交易, 相关性过载(>0.7)自动减仓',
'strategy', ['quant-trading', 'risk-management', 'factor-model', 'backtest', 'c-fin-jepa'])

add('[EconModel] 经济模型与市场分析规范',
'Economic Modeling & Market Analysis:'
' 1. 宏观经济学: IS-LM/AS-AD模型, Solow-Swan增长模型, DSGE动态随机一般均衡, 菲利普斯曲线'
' 2. 货币政策: 利率传导机制, 量化宽松(QE), 央行资产负债表, Taylor Rule, 通胀预期(盈亏平衡通胀率)'
' 3. 市场分析: 技术分析(趋势/支撑阻力/形态/指标), 基本面分析(财报/DCF估值/相对估值), 市场情绪分析(VIX/Put-Call)'
' 4. 博弈论应用: Nash均衡, 囚徒困境, 拍卖理论(英式/荷式/密封投标), 机制设计'
' 5. 行为金融: 前景理论(损失厌恶), 锚定效应, 过度自信, 羊群效应, 处置效应'
' 6. 另类数据: 卫星图像(零售流量/原油库存), 信用卡交易数据, 社交媒体情绪, 供应链数据',
'strategy', ['economics', 'monetary-policy', 'behavioral-finance', 'alt-data', 'game-theory'])

add('[PortfolioTheory] 投资组合理论与资产配置',
'Portfolio Theory & Asset Allocation:'
' 1. 资产大类: 股票(发达/新兴), 固定收益(国债/信用债/可转债), 商品(贵金属/能源/农产品), 另类(私募/对冲/REITs)'
' 2. 战略资产配置(SAA): 长期目标权重, 风险预算, 再平衡策略(定期/阈值/混合)'
' 3. 战术资产配置(TAA): 短期偏离SAA捕捉Alpha, 行业轮动, 因子择时, 市场状态识别(HMM/Regime Switch)'
' 4. 绩效归因: Brinson归因(配置效应+选择效应+交互效应), 因子归因(Fama-French/Carhart), 风格分析(Returns-based)'
' 5. 尾部风险管理: 期权对冲(Protective Put/Collar), 尾部风险基金(Tail Risk Hedging), 波动率衍生品(VIX Futures)'
' 6. ESG投资: 负面筛选(排除烟酒军火), ESG整合(量化评分), 影响力投资, 碳中和投资(Paris-Aligned)',
'strategy', ['portfolio', 'asset-allocation', 'performance-attribution', 'tail-risk', 'esg'])

# --- Monet (Aesthetics + Design + Visual Harmony) ---
add('[AestheticTheory] 美学理论与审美判断规则',
'Aesthetic Theory & Judgment Rules:'
' 1. CARTJEPA艺术世界模型: 审美特征的因果编码, 风格迁移的因果约束, 美学质量的量化评价'
' 2. 经典美学: 柏拉图(美即理念), 亚里士多德(和谐/比例/秩序), 康德(无目的的合目的性/崇高), 黑格尔(美是理念的感性显现)'
' 3. 现代美学: 克罗齐(直觉即表现), 贝尔(有意味的形式), 杜威(艺术即经验), 丹托(艺术界/寻常物的嬗变)'
' 4. 中国美学: 意境(情景交融), 气韵生动(谢赫六法), 虚实相生, 中和之美, 留白之妙'
' 5. 审美心理学: 格式塔(整体大于部分/完形趋向), 黄金比例(1:1.618面部/建筑/绘画), 审美偏好进化心理学'
' 6. 审美评判维度: 和谐度/原创性/复杂度/情感共鸣/技术精湛度, 避免单一维度评分',
'monet', ['aesthetics', 'art-philosophy', 'chinese-aesthetics', 'gestalt', 'c-art-jepa'])

add('[DesignPrinciple] 设计原则与创意方法论',
'Design Principles & Creative Methodology:'
' 1. 通用设计原则: 对比(Contrast)/重复(Repetition)/对齐(Alignment)/亲密性(Proximity)—CRAP四原则'
' 2. 视觉层次: 大小/颜色/对比度/位置/留白构成的5级信息层级, 引导用户视线(F/Z型扫描路径)'
' 3. 设计思维(Design Thinking): 共情→定义→构思→原型→测试, 双钻模型(Double Diamond)'
' 4. 设计系统: Material Design(Google), Human Interface(Apple), Fluent(Microsoft), 设计Token体系'
' 5. 创意方法论: 头脑风暴(发散→收敛), SCAMPER(替代/组合/适应/修改/他用/消除/重排), TRIZ(发明问题解决理论40原则)'
' 6. 跨媒介设计: 平面→UI/UX→产品→空间→服务→系统设计的递进与交叉, 设计的一致性语言',
'monet', ['design-principle', 'visual-hierarchy', 'design-thinking', 'design-system', 'creativity'])

add('[VisualHarmony] 视觉和谐与空间美学',
'Visual Harmony & Spatial Aesthetics:'
' 1. 形态学: 点线面体的情感语言, 圆(和谐完整), 方(稳定秩序), 三角(动感张力), 螺旋(生长演化)'
' 2. 比例系统: 黄金比例(1:1.618), 根号矩形(√2/√3/√5), 模度(Le Corbusier人体尺度), 斐波那契数列'
' 3. 空间美学: 正负空间(图底关系), 空间层次(前景/中景/背景), 视觉力场(重力/张力/向心力)'
' 4. 材质与肌理: 光滑/粗糙/透明/半透明/反射/漫反射, 材料的触觉联想与情感共鸣'
' 5. 光与影: 自然光(天窗/侧窗/顶光)的空间塑造, 人工照明(环境光/任务光/重点光)的氛围营造'
' 6. 环境美学: 建筑与自然的对话, 室内外空间的流动, 园林美学(借景/框景/对景/障景)',
'monet', ['visual-harmony', 'proportion', 'spatial-aesthetics', 'material-texture', 'light-shadow'])

# --- Humboldt (Climate + GeoSpatial + Earth Systems) ---
add('[ClimateModel] 气候模型与环境科学规则',
'Climate Modeling & Environmental Science:'
' 1. CGEOJEPA气候变体: 大气-海洋-陆地-冰雪圈耦合的因果建模, 极端天气事件的因果溯源'
' 2. 气候模式: GCM全球气候模式, RCM区域气候模式, 地球系统模型ESM, 参数化方案(对流/云/辐射)'
' 3. 气候变化归因: 检测与归因(D&A), 指纹法, 自然强迫vs人为强迫的分离, 极端事件归因(EEA)'
' 4. IPCC情景: SSP1(可持续)到SSP5(化石燃料), RCP2.6/4.5/6.0/8.5辐射强迫路径'
' 5. 气候数据: ERA5再分析, CMIP6多模式集合, 观测数据(HadCRUT/GISTEMP/BEST), 古气候代用数据'
' 6. 气候风险评估: 物理风险(海平面上升/热浪/洪水/干旱), 转型风险(碳价/政策/技术), 韧性评估',
'humboldt', ['climate-model', 'attribution', 'ipcc', 'cmip6', 'c-geo-jepa'])

add('[GeoSpatial] 地理空间分析与GIS规范',
'Geospatial Analysis & GIS Standards:'
' 1. CGEOJEPA地理空间变体: 地形/土壤/水文/植被的空间因果编码, 土地利用变化的因果建模'
' 2. GIS数据模型: 矢量(Point/Line/Polygon), 栅格(GeoTIFF/NetCDF), 三维(TIN/Point Cloud/LiDAR)'
' 3. 空间分析: 缓冲区/叠加/网络分析/地形分析(坡度/坡向/视域)/空间统计(Getis-Ord/Moran I/LISA)'
' 4. 遥感: Sentinel-1(SAR)/Sentinel-2(多光谱10m)/Landsat(30m)/MODIS(250-1000m), 高光谱(Hyperion/PRISMA)'
' 5. 坐标系统: WGS84(EPSG:4326)/CGCS2000(EPSG:4490), UTM投影, Web Mercator(EPSG:3857), 高程基准'
' 6. 空间数据库: PostgreSQL+PostGIS, GeoPandas+Shapely, GDAL/rasterio, Google Earth Engine云端分析',
'humboldt', ['gis', 'remote-sensing', 'spatial-analysis', 'geopandas', 'earth-engine'])

add('[EarthSystem] 地球系统科学与水文地质',
'Earth System Science & Hydrogeology:'
' 1. 水文循环: 降水/蒸发/径流/下渗/地下水流动, 流域水文模型(HEC-HMS/SWAT/VIC), 分布式/集总式模型'
' 2. 地质学: 板块构造理论, 岩石学(火成/沉积/变质), 地层学与地质年代, 构造地质(褶皱/断层/节理)'
' 3. 自然资源: 矿产资源(金属/非金属/能源矿产), 地下水(含水层/补给/开采/污染), 地热能(干热岩/水热型)'
' 4. 地质灾害: 地震(震级/烈度/场地效应), 滑坡(稳定性分析/安全系数), 泥石流, 地面沉降(地下水超采)'
' 5. 海洋学: 洋流(温盐环流/风生环流), 潮汐(引潮力/潮汐类型), 海平面变化(热膨胀/冰融/地壳均衡)'
' 6. 极地科学: 冰盖动力学/冰芯记录(80万年气候), 冻土退化(甲烷释放), 北极航道(东北/西北航道)',
'humboldt', ['earth-system', 'hydrology', 'geology', 'oceanography', 'polar-science'])

# ========================================================================
# PART C: USER-SUPPLIED LOGIC KNOWLEDGE BASE
# ========================================================================

# Logic Module 1 — Formal Logic (montesquieu primary)
add('[LogicKB] 形式逻辑与数理逻辑硬约束',
'Formal Logic & Mathematical Logic Hard Constraints for VORTEX+LLM+JEPA:'
' 1. 所有AI推理/决策/JEPA因果预测必须严格遵循形式逻辑/一阶谓词逻辑/模态逻辑/时序逻辑规则'
' 2. 禁止矛盾(A∧¬A永远为假)/禁止循环论证(P→Q因为Q→P)/禁止跳跃式幻觉推理(无逻辑链的结论)'
' 3. 推理三段论(Syllogism): 大前提(M→P)+小前提(S→M)+结论(S→P), 所有推理必须可还原为三段论形式'
' 4. 充分必要条件: 充分条件(P→Q, P真则Q必真), 必要条件(¬P→¬Q, P假则Q必假), 充要条件(P↔Q)'
' 5. 逻辑蕴含(Material Implication): P→Q ≡ ¬P∨Q, 区分逻辑蕴含与因果蕴含'
' 6. 排中律(Law of Excluded Middle): P∨¬P, 非此即彼无中间态; 矛盾律: ¬(P∧¬P)必真'
' 7. 一阶谓词逻辑: ∀xP(x)全称量词/∃xP(x)存在量词, 量词交换规则, Skolem范式, 合一算法'
' 8. LLM输出/JEPA世界推演必须符合逻辑一致性校验, 逻辑矛盾直接熔断不进执行链路',
'montesquieu', ['logic', 'formal-logic', 'first-order-logic', 'syllogism', 'safety'])

add('[LogicKB] 因果逻辑三大约束',
'Causal Logic Triple Constraints:'
' 1. 时序因果(Temporal Causality): 先因后果, t_cause < t_effect, 禁止时间倒流式逻辑幻觉'
' 2. 物理因果(Physical Causality): 符合物理/数学/化学规律的因果链条, 必须经过CPHYSJEPA物理可行性校验'
' 3. 反事实推理(Counterfactual Reasoning): JEPA做假设推演时严格遵循逻辑公理, do(X=x)干预下推断P(Y|do(X=x))'
' 4. Granger因果: X是Y的Granger原因当引入X的过去值能显著提升Y的预测精度, 用于时序JEPA的因果关系发现'
' 5. Pearl因果阶梯: 关联(Seeing)→干预(Doing)→反事实(Imagining), C-JEPA覆盖全部三层'
' 6. 因果图模型: 有向无环图(DAG), d-分离, 后门准则, 前门准则, 工具变量, 中介分析',
'montesquieu', ['causal-logic', 'temporal', 'physical', 'counterfactual', 'pearl-causality'])

# Logic Module 2 — Higher-order Logic (montesquieu primary)
add('[LogicKB] 高阶逻辑与科学推理库',
'Advanced Logic & Scientific Reasoning Library:'
' 1. 命题逻辑: 原子命题p/q/r, 逻辑连接词{∧,∨,→,↔,¬}, 真值表, 永真式(重言式)/永假式(矛盾)/可满足式'
' 2. 谓词逻辑: 谓词P(x), 函数f(x), 常量c, 量词∀/∃, 谓词逻辑的语义解释(模型论)'
' 3. 集合论: 集合/子集/幂集, 并交差补, 笛卡尔积, 关系/函数/等价关系/序关系, 基数/可数/不可数'
' 4. 布尔代数: AND/OR/NOT/XOR/NAND/NOR, De Morgan律, 分配律/吸收律, 卡诺图化简, 布尔函数范式(CNF/DNF)'
' 5. 逻辑符号体系: ¬ ∧ ∨ → ↔ ∀ ∃ ⊢ ⊨ □ ◇, 统一符号表避免歧义'
' 6. 归纳逻辑: 枚举归纳/类比归纳/统计归纳/Mill五法(契合法/差异法/契合差异并用法/共变法/剩余法)'
' 7. 演绎逻辑: 自然演绎(Fitch风格), 公理化系统(Hilbert风格), Gentzen序列演算, Tableau方法'
' 8. 溯因推理(Abduction): 从结果推断最可能的解释, 最佳解释推理(IBE), 适配故障排查/EHS风险分析/设备故障诊断',
'montesquieu', ['propositional-logic', 'set-theory', 'boolean-algebra', 'induction', 'abduction'])

add('[LogicKB] 模态逻辑与时序逻辑',
'Modal Logic & Temporal Logic:'
' 1. 模态逻辑: □必然/◇可能, 可能世界语义(Kripke语义), 公理系统K/T/S4/S5, 道义逻辑(义务/允许/禁止)'
' 2. 时序逻辑: LTL线性时序逻辑(□always/◇eventually/○next/U until), CTL计算树逻辑(分支时间)'
' 3. 适配JEPA: 时序逻辑公式表达世界状态的时序性质, LTL安全性质(□¬danger)=永远不危险'
' 4. 道义逻辑适配: O(obligatory)/P(permitted)/F(forbidden), 用于规则引擎的权限与义务推理'
' 5. 动态逻辑(PDL): [a]P执行动作a后P成立, ⟨a⟩P存在执行a使P成立, 用于Action-Conditioned JEPA'
' 6. 认知逻辑: K(知道)/B(相信), 多智能体知识推理, 分布式知识D_g与公共知识C_g',
'montesquieu', ['modal-logic', 'temporal-logic', 'deontic-logic', 'dynamic-logic', 'epistemic-logic'])

add('[LogicKB] 逻辑谬误黑名单与软约束',
'List of Logical Fallacies (Blacklist for LLM/VORTEX):'
' 1. 形式谬误: 肯定后件(如果P则Q, Q成立, 所以P成立—无效), 否定前件(如果P则Q, P不成立, 所以Q不成立)'
' 2. 非形式谬误: 滑坡谬误(连续放大链后果), 稻草人谬误(曲解对方论证再攻击), 虚假因果(Post hoc ergo propter hoc)'
' 3. 循环论证(Begging the Question): 结论隐含在前提中, 乞丐问题, 同义反复'
' 4. 虚假二分(False Dichotomy): 只有两个选择忽略中间选项, 非黑即白'
' 5. 诉诸无知(Ad Ignorantiam): 没有证明不存在所以存在, 证明责任倒置'
' 6. 合成谬误/分解谬误: 部分性质错误归因整体/整体性质错误归因部分'
' 7. 红鲱鱼(Red Herring): 引入无关话题转移注意力, 回避核心问题'
' 8. 人身攻击(Ad Hominem): 攻击论证者而非论证本身, 诉诸权威/诉诸情感/诉诸群众',
'montesquieu', ['fallacy', 'logic-blacklist', 'critical-thinking', 'argumentation'])

# Logic — Shared foundational rules to ALL key reasoning souls
for soul in ['einstein', 'cezanne', 'strategy', 'darwin', 'davinci']:
    add('[LogicKB] 逻辑一致性校验闭环(基础规则)',
    'Logic Consistency Verification Loop - Foundational Rules:'
    f' Soul [{soul}] specific reasoning must pass logic consistency check:'
    ' 1. 所有LLM输出必须经过逻辑一致性校验(无矛盾/无循环/无跳跃)'
    ' 2. 所有JEPA预测结果必须经过逻辑一致性校验(时序/物理/因果三关)'
    ' 3. 逻辑矛盾直接熔断(Fuse Circuit Breaker), 不进入执行链路'
    ' 4. 推理链必须完整可溯: 前提→中间推理步骤→结论, 每步标注推理规则'
    ' 5. 不确定性标注: 概率推理标注置信度[0,1], 模糊推理标注隶属度, 默认推理标注可废止性(Defeasible)'
    ' 6. 引用来源: Stanford Encyclopedia of Philosophy (plato.stanford.edu) 为逻辑学最高权威参考',
    soul, ['logic-verification', 'consistency', 'fuse', 'reasoning-chain'])

# ========================================================================
# PART D: USER-SUPPLIED CHEMISTRY KNOWLEDGE BASE
# ========================================================================

# Chemistry Module 1 — Foundational (einstein primary)
add('[ChemKB] 基础化学底层约束',
'Foundational Chemistry Constraints for JEPA World Model & LLM:'
' 1. 所有化学推理/反应预测/物料安全判断严格遵循元素周期律/守恒定律/热力学定律/化学键规则'
' 2. 禁止编造化学反应方程式/虚构物质属性/随意更改反应条件与产物'
' 3. 元素周期律: 118种元素的周期性规律(原子半径/电负性/电离能/电子亲和能), 族/周期/区(s/p/d/f)'
' 4. 质量守恒: 化学反应前后总质量不变, 方程式必须配平(原子种类数量不变)'
' 5. 能量守恒+热力学第一定律: ΔU=Q+W, 焓变ΔH=ΔU+PΔV, Gibbs自由能ΔG=ΔH-TΔS, ΔG<0自发反应'
' 6. 化学键规则: 离子键(静电引力, NaCl型), 共价键(共享电子对, σ/π键), 金属键(电子海), 氢键/范德华力/疏水作用'
' 7. 反应动力学: 反应速率方程r=k[A]^m[B]^n, Arrhenius方程k=Ae^(-Ea/RT), 催化剂降低活化能不改平衡',
'einstein', ['chemistry', 'periodic-table', 'conservation', 'thermodynamics', 'chemical-bond'])

add('[ChemKB] 化学世界状态表征(JEPA适配)',
'Chemical World State Representation for JEPA:'
' 1. 物质状态: 固(s)/液(l)/气(g)/等离子体/超临界流体, 相变(熔化/凝固/汽化/冷凝/升华/凝华)'
' 2. 浓度与活度: 摩尔浓度mol/L, 质量分数%, ppm/ppb, 活度a=γ·c(非理想溶液活度系数γ)'
' 3. 温度: 反应温度对速率/平衡/选择性的影响, 温度梯度扩散(热扩散), 临界温度/沸点/熔点'
' 4. 压强: 气体分压(Dalton定律), 拉乌尔定律(理想溶液蒸气压), Henry定律(气体溶解度), Le Chatelier原理'
' 5. pH值: -log[H+], 酸/碱/中性, 缓冲溶液(Henderson-Hasselbalch方程), pH对反应/溶解度/生物活性的影响'
' 6. 安全化学JEPA表征: 可燃性/爆炸极限(LEL~UEL)/腐蚀性/毒性(LD50/LC50)/反应活性/不相容物质',
'einstein', ['chemistry-state', 'jepa', 'concentration', 'pH', 'safety-chemistry'])

add('[ChemKB] 工业与安全化学约束(EHS适配)',
'Industrial & Safety Chemistry Constraints for EHS:'
' 1. 化工流程约束: 物料配比/工艺反应条件/催化剂选择/分离纯化/三废处理, 严格匹配工业化学标准'
' 2. 危化品特性(GB13690): 爆炸物/易燃气体/易燃液体/氧化剂/毒性/腐蚀性/放射性, 九大类GHS分类'
' 3. 禁忌配伍: 酸与碱/氧化剂与还原剂/水反应物与水/强酸与氰化物(生成HCN剧毒), 必须物理隔离存储'
' 4. 泄漏处理: 隔离区域(立即/50m/100m/300m), 吸附材料(活性炭/硅藻土/沸石), 中和处理, 通风排毒'
' 5. 爆炸极限: 可燃气体/蒸气与空气混合的浓度范围(LEL下~UEL上), 控制浓度<25%LEL为安全操作'
' 6. 职业接触限值(OELs): TWA时间加权平均(8h), STEL短期暴露限(15min), MAC最高容许浓度, 生物接触限值BEI'
' 7. 中国危化品管理: 危化品目录(2828种), 剧毒化学品目录, 易制毒化学品分类, 易制爆化学品管控',
'einstein', ['industrial-chemistry', 'ehs', 'hazardous-material', 'explosion-limit', 'oel'])

# Chemistry Module 2 — Full Domain (einstein primary)
add('[ChemKB] 无机化学全领域规则',
'Inorganic Chemistry Domain Rules:'
' 1. 元素与单质: 金属/非金属/准金属(类金属), 同素异形体(碳:金刚石/石墨/富勒烯/石墨烯/碳纳米管)'
' 2. 化合物: 氧化物/酸/碱/盐, 离子型/共价型/配位化合物, 命名法(IUPAC无机命名)'
' 3. 离子反应: 沉淀反应(溶度积Ksp, 同离子效应), 配位反应(稳定常数Kf, 螯合效应), 离子交换'
' 4. 氧化还原: 氧化数, 氧化剂/还原剂, 标准电极电势E°, Nernst方程E=E°-(RT/nF)lnQ, 电化学系列'
' 5. 酸碱理论: Arrhenius(氢离子/氢氧根), Bronsted-Lowry(质子供体/受体), Lewis(电子对受体/供体), HSAB软硬酸碱'
' 6. 配位化学: 配位数2-12, 几何构型(八面体/四面体/平面四方), 晶体场理论/配体场理论, 磁性与颜色(d-d跃迁)',
'einstein', ['inorganic-chemistry', 'redox', 'acid-base', 'coordination', 'electrochemistry'])

add('[ChemKB] 有机化学全领域规则',
'Organic Chemistry Domain Rules:'
' 1. 官能团分类: 烷/烯/炔/芳烃/卤代烃/醇/酚/醚/醛/酮/羧酸/酯/胺/酰胺/腈/硝基/磺酸, IUPAC有机命名'
' 2. 反应机理: 亲核取代(SN1/SN2), 亲电加成(马氏规则/反马), 消除反应(E1/E2, Zaitsev规则), 亲电芳香取代(定位效应)'
' 3. 同分异构: 构造异构(碳骨架/官能团位置), 立体异构(对映体R/S, 非对映体, 顺反Z/E, 构象异构)'
' 4. 有机合成路径: 逆合成分析(Retrosynthesis), 保护基策略, 官能团转化, 绿色化学12原则'
' 5. 高分子化学: 聚合反应(加聚/缩聚/开环聚合), 聚合物分子量(Mn/Mw/PDI), 热塑性/热固性, 生物可降解高分子'
' 6. 天然产物化学: 萜类/生物碱/黄酮/甾体/糖类/脂类/氨基酸/多肽/蛋白质, 分离纯化(柱色谱/HPLC)',
'einstein', ['organic-chemistry', 'reaction-mechanism', 'stereochemistry', 'synthesis', 'polymer'])

add('[ChemKB] 物理化学与热动力学规则',
'Physical Chemistry & Thermodynamics Domain Rules:'
' 1. 热力学定律: 第零定律(热平衡传递性), 第一定律(ΔU=Q+W), 第二定律(ΔS≥0, 熵增), 第三定律(T→0K, S→0)'
' 2. 化学平衡: K_eq=[产物]^ν/[反应物]^ν, K_eq与ΔG°的关系(ΔG°=-RTlnK), Van\'t Hoff方程(dlnK/dT=ΔH°/RT^2)'
' 3. 相平衡: 相律(F=C-P+2), 单组分相图(水的三相点0.01℃/611.73Pa), 二元相图(杠杆规则)'
' 4. 电化学: Faraday定律(m=MQ/nF), 电池电动势, 电解/电镀/腐蚀与防护(牺牲阳极/外加电流)'
' 5. 表面化学: 表面张力/接触角, 吸附(物理吸附vs化学吸附, Langmuir/BET等温线), 胶体(溶胶/凝胶/乳浊液)'
' 6. 统计热力学: Boltzmann分布, 配分函数Q, 熵S=klnW, 微观状态与宏观性质的桥梁',
'einstein', ['physical-chemistry', 'thermodynamics', 'equilibrium', 'electrochemistry', 'surface-chemistry'])

# Chemistry shared foundational for darwin (biochemistry) and cezanne (industrial)
for soul in ['darwin', 'cezanne']:
    role = 'Biochemistry & Drug Discovery' if soul == 'darwin' else 'Industrial Process & Materials'
    add('[ChemKB] 化学基础共享规则',
    f'Chemistry Foundation Rules - {role}:'
    f' Soul [{soul}] shared chemistry foundation:'
    ' 1. 涉及化学知识的判断必须基于已验证的化学定律和数据, 禁止虚构化学事实'
    ' 2. 化学反应预测必须经过热力学可行性(ΔG<0)和动力学合理性(Ea合理)双重校验'
    ' 3. 化学物质信息数据源: PubChem(pubchem.ncbi.nlm.nih.gov)全球最大开源化学数据库'
    ' 4. IUPAC国际纯粹与应用化学联合会为化学命名/符号/术语的唯一权威标准'
    ' 5. 化学安全参考: 中国危化品目录/GB国标/NIST化学数据库, 涉及安全场景必须查阅',
    soul, ['chemistry-foundation', 'pubchem', 'iupac', 'thermodynamics-check', 'safety'])

# ========================================================================
# PART E: LITERATURE KNOWLEDGE BASE (guizhu + herodotus + montesquieu)
# ========================================================================

add('[LitKB] 文学全局硬约束',
'Literature Global Hard Constraints for LLM+JEPA:'
' 1. 所有文学创作/解读/对话必须基于私有文学知识库, 禁止编造作者/作品/年代/情节/典故'
' 2. 文本生成必须符合文体规范/时代背景/人物性格/叙事逻辑/韵律规则'
' 3. 诗歌/散文/小说/剧本必须遵循对应文体结构, 禁止逻辑混乱/前后矛盾'
' 4. 历史文学/古籍内容严格尊重原文, 不得篡改/曲解/穿越'
' 5. LLM输出必须经过逻辑一致性+文学规范校验, 不合格直接熔断'
' 6. 文学世界模型: JEPA负责虚构世界的因果/时序/人物状态/环境变化的一致性推演'
' 7. 人物状态推演/情节合理性预测/世界一致性维护由JEPA完成, LLM只做文本表达',
'guizhu', ['literature', 'hard-constraint', 'jepa', 'safety', 'anti-hallucination'])

add('[LitKB] 中国文学全库规范',
'Chinese Literature Complete Canon:'
' 1. 先秦文学: 诗经(风雅颂/赋比兴), 楚辞(离骚/九歌/天问), 诸子散文(论语/孟子/庄子/荀子/韩非子/老子/墨子)'
' 2. 两汉文学: 史记(本纪/世家/列传/书/表), 汉书, 汉赋(子虚赋/上林赋/两都赋), 乐府诗(陌上桑/孔雀东南飞)'
' 3. 魏晋南北朝: 建安文学(三曹/七子), 陶渊明(田园诗/桃花源记), 世说新语(志人小说), 搜神记(志怪小说), 文心雕龙(文学理论)'
' 4. 唐诗: 初唐四杰/王维/孟浩然(山水田园), 李白(浪漫主义), 杜甫(现实主义/诗史), 白居易(新乐府), 李商隐/杜牧(晚唐)'
' 5. 唐诗格律: 五言/七言, 绝句(4句)/律诗(8句)/排律, 平仄(平声/仄声), 对仗(工对/宽对/流水对/扇面对), 押韵(平水韵106韵部)'
' 6. 宋词: 豪放派(苏轼/辛弃疾), 婉约派(柳永/李清照/周邦彦/姜夔), 词牌(浣溪沙/蝶恋花/水调歌头/念奴娇等800+词牌)'
' 7. 元曲: 散曲(小令/套数), 杂剧(关汉卿/王实甫/马致远/白朴), 曲牌/宫调/衬字'
' 8. 明清小说: 四大名著(三国演义/水浒传/西游记/红楼梦), 儒林外史, 聊斋志异, 金瓶梅, 三言二拍',
'guizhu', ['chinese-literature', 'poetry', 'ci-poetry', 'novel', 'canon'])

add('[LitKB] 中国现当代文学规范',
'Chinese Modern & Contemporary Literature:'
' 1. 现代文学(1919-1949): 鲁迅(呐喊/彷徨/朝花夕拾), 郭沫若(女神), 茅盾(子夜/林家铺子), 巴金(家春秋/随想录), 老舍(骆驼祥子/茶馆), 曹禺(雷雨/日出)'
' 2. 沈从文(边城/湘行散记), 钱钟书(围城), 张爱玲(倾城之恋/金锁记), 萧红(呼兰河传)'
' 3. 当代文学(1949-): 莫言(红高粱家族/蛙/丰乳肥臀, 2012诺奖), 余华(活着/许三观卖血记/兄弟), 王小波(黄金时代/沉默的大多数), 贾平凹(废都/秦腔), 王安忆(长恨歌), 阎连科(受活/丁庄梦), 刘慈欣(三体/流浪地球)'
' 4. 港台文学: 金庸(射雕三部曲/天龙八部/笑傲江湖), 白先勇(台北人/孽子), 余光中(乡愁), 李敖, 龙应台(目送/大江大海)'
' 5. 中国文学理论: 古文论(文心雕龙/诗品/沧浪诗话/人间词话), 修辞学(比喻/象征/夸张/对偶/排比/反复/设问/反问/借代/通感), 文体学(韵文/散文/骈文/古文)'
' 6. 现代汉语规范: 语法(主谓宾定状补), 词汇(同义词辨析/成语/歇后语/惯用语), 语用(语境/预设/会话含义)',
'guizhu', ['modern-chinese-lit', 'contemporary', 'rhetoric', 'grammar'])

add('[LitKB] 世界文学全库规范',
'World Literature Complete Canon:'
' 1. 古希腊罗马: 荷马(伊利亚特/奥德赛), 希腊悲剧(埃斯库罗斯/索福克勒斯/欧里庇得斯), 维吉尔(埃涅阿斯纪), 奥维德(变形记), 希腊神话体系(奥林匹斯十二主神/英雄传说)'
' 2. 中世纪欧洲: 但丁(神曲:地狱/炼狱/天堂), 乔叟(坎特伯雷故事集), 圣奥古斯丁(忏悔录), 骑士文学(亚瑟王/罗兰之歌/尼伯龙根之歌)'
' 3. 文艺复兴: 莎士比亚(四大悲剧+四大喜剧+十四行诗154首), 塞万提斯(堂吉诃德), 拉伯雷(巨人传), 蒙田(随笔集)'
' 4. 17-18世纪: 弥尔顿(失乐园), 莫里哀(伪君子/悭吝人), 笛福(鲁滨逊漂流记), 斯威夫特(格列佛游记), 歌德(浮士德/少年维特之烦恼), 席勒(阴谋与爱情)'
' 5. 19世纪浪漫主义: 雨果(巴黎圣母院/悲惨世界/九三年), 拜伦(唐璜), 雪莱/济慈, 普希金(叶甫盖尼·奥涅金), 惠特曼(草叶集)'
' 6. 19世纪现实主义: 巴尔扎克(人间喜剧), 福楼拜(包法利夫人), 狄更斯(双城记/雾都孤儿/远大前程), 托尔斯泰(战争与和平/安娜·卡列尼娜/复活), 陀思妥耶夫斯基(罪与罚/卡拉马佐夫兄弟/白痴), 契诃夫(短篇小说/樱桃园/海鸥)',
'guizhu', ['world-literature', 'western-canon', 'classic', 'epic', 'drama'])

add('[LitKB] 世界现代文学与诺贝尔奖',
'World Modern Literature & Nobel Prize:'
' 1. 20世纪现代主义: 卡夫卡(变形记/审判/城堡), 乔伊斯(尤利西斯/都柏林人), 普鲁斯特(追忆似水年华), 伍尔夫(达洛维夫人/到灯塔去), 福克纳(喧哗与骚动/我弥留之际), 海明威(老人与海/太阳照常升起)'
' 2. 拉美文学爆炸: 马尔克斯(百年孤独/霍乱时期的爱情,1982诺奖), 博尔赫斯(小径分岔的花园/虚构集/阿莱夫), 略萨(城市与狗/绿房子,2010诺奖)'
' 3. 东亚文学: 日本(紫式部:源氏物语/川端康成:雪国/三岛由纪夫:金阁寺/村上春树:挪威的森林), 韩国(韩江:素食者,2024诺奖/金英夏)'
' 4. 非洲文学: 索因卡(1986诺奖), 马哈福兹(1988诺奖/开罗三部曲), 库切(2003诺奖/耻), 阿契贝(瓦解)'
' 5. 诺贝尔文学奖系统: 自1901年起, 完整获奖名单/获奖理由/代表作品/文学流派'
' 6. 世界文学理论: 叙事学(热奈特/托多罗夫), 接受美学(姚斯/伊瑟尔), 互文性(克里斯蒂娃), 后殖民理论(萨义德:东方学/斯皮瓦克/霍米巴巴), 女性主义文学批评(肖瓦尔特/西苏)',
'guizhu', ['modern-world-lit', 'nobel', 'magic-realism', 'postcolonial', 'narratology'])

add('[LitKB] 文学创作规范与文体结构',
'Literary Creation Standards & Genre Structures:'
' 1. 小说结构: 起承转合(东方式), 三幕式(铺垫→对抗→解决), 英雄之旅(Joseph Campbell 12阶段), 人物弧光(成长/堕落/平弧), 冲突设计(人vs人/人vs自然/人vs自我/人vs社会)'
' 2. 诗歌格律: 中国古典(平仄/押韵/对仗/字数限制), 西方(十四行诗/抑扬格五音步/自由诗), 日本(俳句5-7-5/短歌5-7-5-7-7)'
' 3. 戏剧结构: 幕(通常3-5幕), 场, 冲突(戏剧性冲突), 对白规范(潜台词/独白/旁白), 三一律(时间/地点/情节统一)'
' 4. 修辞体系: 比喻(明喻/暗喻/借喻), 象征(个人/传统/自然象征), 夸张(夸大/缩小/超前夸张), 反讽(言语反讽/情境反讽/戏剧反讽), 意象(视觉/听觉/触觉/嗅觉/味觉/联觉意象)'
' 5. 叙事视角: 第一人称(我)/第三人称全知(上帝视角)/第三人称限制(人物视角)/第二人称(你)/多视角/不可靠叙事者'
' 6. 文学批评体系: 文本细读(新批评), 结构主义/后结构主义, 马克思主义批评, 精神分析批评(弗洛伊德/拉康), 生态批评(Ecocriticism), 数字人文(Distant Reading/文本挖掘)',
'guizhu', ['writing-craft', 'narrative', 'poetry-form', 'drama-structure', 'literary-criticism'])

# Literature shared to herodotus and montesquieu
for soul in ['herodotus', 'montesquieu']:
    add('[LitKB] 文学基础共享规则',
    f'Literature Foundation Rules — shared to {soul}:'
    ' 1. 文学文本的引用/解读/创作必须符合文体规范与时代背景'
    ' 2. 古籍/经典文本的引用必须严格尊重原文, 注明版本来源'
    ' 3. 古腾堡计划(Project Gutenberg): 7.5万+世界文学经典, 公有领域纯文本可直接使用'
    ' 4. 中国古籍来源: 国家图书馆中华古籍资源库(10万+古籍), 书格(公共版权), 识典古籍(北大+字节)'
    ' 5. 文学创作流程: JEPA先跑世界推演(人物/情节/时空一致性)→LLM再输出文本表达',
    soul, ['literature-foundation', 'gutenberg', 'classical-texts'])

# ========================================================================
# PART F: ASTRONOMY / ASTROPHYSICS (galileo primary)
# ========================================================================

add('[AstroKB] 天文学宇宙学核心规则',
'Astronomy & Cosmology Core Rules for LLM+JEPA:'
' 1. 宇宙学标准模型(Lambda-CDM): 大爆炸(13.8Ga), 宇宙微波背景辐射(CMB, T=2.725K), 暗能量(宇宙加速膨胀, Lambda/真空能), 暗物质(星系旋转曲线/引力透镜/子弹星系团证据)'
' 2. 哈勃定律: v=H0*d (H0≈70 km/s/Mpc), 宇宙膨胀不是物体在空间中运动而是空间本身的膨胀, 红移z=Δλ/λ0'
' 3. 宇宙演化: Big Bang→暴胀(10^-36~10^-32s)→核合成(3min, 生成H/He/Li)→复合期(38万年, CMB释放)→暗时代→第一代恒星→星系形成→暗能量主导(约50亿年前至今)'
' 4. 星系分类(Hubble Sequence): 椭圆(E0-E7), 旋涡(Sa/Sb/Sc), 棒旋(SBa/SBb/SBc), 不规则(Irr), 活动星系核(类星体/赛弗特/耀变体/射电星系), 超大质量黑洞(M-sigma关系)'
' 5. 银河系: 棒旋星系, 直径10万光年, 2000-4000亿恒星, 中心黑洞Sgr A*(430万M☉), 太阳位于猎户臂距中心2.6万光年, 绕中心周期2.25亿年',
'galileo', ['astronomy', 'cosmology', 'lambda-cdm', 'hubble', 'galaxy'])

add('[AstroKB] 天体物理与恒星演化',
'Astrophysics & Stellar Evolution:'
' 1. 恒星形成: 巨分子云Jeans不稳定性坍缩→原恒星(赫罗图上Hayashi轨迹)→主序星(氢核聚变p-p链或CNO循环)'
' 2. 主序星: 质量-光度关系(L∝M^3.5), 光谱分类OBAFGKM(LT), 赫罗图(HR Diagram), 主序带/巨星分支/白矮星序列'
' 3. 恒星演化终点: 低质量星(M<0.5M☉)→He白矮星, 中等质量(0.5-8M☉)→行星状星云→C/O白矮星(Chandrasekhar极限1.44M☉), 大质量(>8M☉)→II型超新星→中子星(1.4-2.2M☉)或黑洞(>3M☉)'
' 4. 超新星: Type Ia(白矮星吸积→碳爆炸, 标准烛光), Type II(铁核坍缩→中微子爆发), 超新星遗迹(蟹状星云SN1054)'
' 5. 致密天体: 白矮星(电子简并压), 中子星(中子简并压, 脉冲星/磁星), 黑洞(事件视界/奇点/霍金辐射/吸积盘/喷流), 引力波(GW150914首次探测, LIGO/Virgo/KAGRA)',
'galileo', ['astrophysics', 'stellar-evolution', 'supernova', 'black-hole', 'gravitational-wave'])

add('[AstroKB] 太阳系与行星科学',
'Solar System & Planetary Science:'
' 1. Kepler定律: (1)椭圆轨道太阳在焦点, (2)掠面速度守恒(角动量守恒), (3)T^2/a^3=常数(轨道周期-半长轴关系)'
' 2. 太阳系结构: 类地行星(水金地火), 小行星带(谷神星/灶神星/智神星), 巨行星(木土=气态, 天海=冰巨), 柯伊伯带(冥王星/阋神星), 奥尔特云(彗星源)'
' 3. 行星数据: 轨道参数(半长轴/离心率/倾角/周期), 物理参数(半径/质量/密度/逃逸速度), 卫星系统, 行星环'
' 4. 太阳活动: 太阳结构(核心/辐射层/对流层/光球/色球/日冕), 黑子周期(11年Schwabe周期/22年Hale磁周期), 太阳耀斑/日冕物质抛射(CME), 太阳风(400-800km/s), 空间天气对地球影响'
' 5. 小行星与彗星: 近地天体(NEO/PHA潜在威胁), 彗星(短周期<200年/长周期, 彗核/彗发/彗尾), 流星/陨石/火流星, 撞击事件(通古斯1908/车里雅宾斯克2013/Chicxulub恐龙灭绝)'
' 6. 系外行星: 探测方法(凌星法Kepler/TESS/视向速度法/直接成像/微引力透镜), 宜居带(CHZ), 超级地球/热木星/迷你海王星, James Webb Space Telescope贡献',
'galileo', ['solar-system', 'kepler', 'planet', 'sun', 'exoplanet'])

add('[AstroKB] 天文观测与仪器',
'Astronomical Observation & Instrumentation:'
' 1. 电磁波谱: 射电(>1mm)/红外(1mm-700nm)/光学(700-400nm)/紫外(400-10nm)/X射线(10-0.01nm)/伽马射线(<0.01nm), 大气窗口'
' 2. 望远镜: 折射(透镜)/反射(镜面Newton/Cassegrain), 主要参数(口径D/焦距f/焦比f/D/分辨率θ=1.22λ/D/极限星等), 像差(球差/彗差/像散/色差)'
' 3. 空间望远镜: Hubble(2.4m/光学-紫外), JWST(6.5m/红外), Chandra(X射线), Fermi(伽马射线), Euclid(暗能量), Roman(广域红外)'
' 4. 测光系统: UBVRI(Johnson-Cousins), ugriz(SDSS), Gaia(G/BP/RP), 绝对星等/视星等(m1-m2=-2.5log(F1/F2)), 星际消光/红化'
' 5. 光谱: 光谱分辨率R=λ/Δλ, 多普勒效应(径向速度Δλ/λ0=v/c), 谱线(吸收线/发射线), 化学丰度测定'
' 6. 天体测量: 视差(周年视差/秒差距pc), 自行(Proper Motion), 视向速度, Gaia DR3(14.6亿恒星高精度天体测量)',
'galileo', ['observation', 'telescope', 'photometry', 'spectroscopy', 'astrometry'])

# ========================================================================
# PART G: EARTH SCIENCE (humboldt primary)
# ========================================================================

add('[EarthKB] 地球科学核心规则',
'Earth Science Core Rules for LLM+JEPA:'
' 1. 地球系统: 四大圈层(岩石圈/水圈/大气圈/生物圈)的相互作用与物质能量循环'
' 2. 板块构造理论: 岩石圈分为七大板块和若干小板块, 边界类型(离散/汇聚/转换), 板块运动驱动(地幔对流/板块拉力/洋脊推力), Wilson旋回(超大陆聚合-裂解周期)'
' 3. 岩石循环(Rock Cycle): 火成岩(侵入/喷出)→风化侵蚀→沉积岩→变质作用→变质岩→熔融→火成岩, 三大岩类相互转化'
' 4. 地质年代表: 冥古宙→太古宙→元古宙→显生宙, 显生宙分为古生代(寒武→二叠)/中生代(三叠→白垩)/新生代(古近→第四纪), 绝对定年(放射性同位素U-Pb/K-Ar/C-14)'
' 5. 地震: 弹性回跳理论, 地震波(P波/S波/面波Love/Rayleigh), 震级(Mw矩震级/Ms面波/Mb体波/Ml里氏), 烈度(MMI I-XII度/中国烈度表), 地震带(环太平洋/阿尔卑斯-喜马拉雅)'
' 6. 火山: 喷发类型(夏威夷式/斯特龙博利式/武尔卡诺式/培雷式/普林尼式/超普林尼式/溢流玄武岩), VEI火山爆发指数(0-8级), 全球活火山分布',
'humboldt', ['earth-science', 'plate-tectonics', 'rock-cycle', 'earthquake', 'volcano'])

add('[EarthKB] 气象学与大气科学',
'Meteorology & Atmospheric Science:'
' 1. 大气结构: 对流层(0-12km, 天气层), 平流层(12-50km, 臭氧层), 中间层(50-85km), 热层(85-600km, 极光), 散逸层(>600km)'
' 2. 大气环流: Hadley环流(0-30°)/Ferrel环流(30-60°)/Polar环流(60-90°), 急流(副热带/极锋), 季风(海陆热力差), Walker环流(El Nino/La Nina/ENSO)'
' 3. 天气系统: 气团(大陆/海洋, 极地/热带), 锋面(冷锋/暖锋/锢囚锋/准静止锋), 气旋(温带气旋/热带气旋), 反气旋(高压)'
' 4. 热带气旋: 分级(热带低压TD→热带风暴TS→强热带风暴STS→台风TY/飓风, Saffir-Simpson 1-5级), 形成条件(SST>26.5℃/低风切变/Coriolis力), 结构(台风眼/眼壁/螺旋雨带)'
' 5. 强对流天气: 雷暴(单体/多单体/超级单体/飑线/MCC中尺度对流复合体), 冰雹(直径/上升气流), 龙卷风(Enhanced Fujita Scale EF0-5), 下击暴流(Microburst/Macroburst)'
' 6. 气象观测: 地面站/探空/雷达(多普勒/双偏振)/卫星(GOES/Himawari/Meteosat/FY风云), 数值天气预报NWP(全球模式GFS/ECMWF, 区域模式WRF/HWRF)',
'humboldt', ['meteorology', 'atmosphere', 'cyclone', 'severe-weather', 'nwp'])

add('[EarthKB] 海洋学与水文学',
'Oceanography & Hydrology:'
' 1. 洋流: 表层环流(风生Ekman输送, Geostrophic地转平衡, 五大副热带涡旋), 热盐环流(全球海洋传送带/AMOC大西洋经向翻转环流), 上升流/下降流, 中尺度涡'
' 2. 潮汐: 引潮力(月球+太阳), 平衡潮理论, 潮汐类型(全日潮/半日潮/混合潮), 潮汐表, 潮汐能'
' 3. 海浪: 风浪/涌浪, 波高/波长/周期, 有效波高Hs, 海啸(Tsunami: 地震→海底垂直位移→长波传播/浅水变形/爬高Run-up)'
' 4. 水文循环: 降水→截留/下渗/地表径流/地下水补给→蒸散发, 流域水量平衡(P=ET+Q+ΔS+ΔG)'
' 5. 地下水: 含水层(承压/非承压), Darcy定律(Q=K·A·dh/dl), 抽水试验, 地下水污染(污染羽/NAPL)'
' 6. 海平面变化: 全球变暖→冰川冰盖融化(格陵兰/南极)→热膨胀→海平面上升(当前约3.7mm/yr), IPCC AR6预测2100年上升0.3-1.1m(SSP1-2.6到SSP5-8.5)',
'humboldt', ['oceanography', 'hydrology', 'tide', 'tsunami', 'sea-level'])

add('[EarthKB] 环境科学与灾害防治',
'Environmental Science & Disaster Prevention:'
' 1. 环境污染: 大气污染(PM2.5/PM10/O3/SO2/NOx/CO, AQI空气质量指数), 水污染(COD/BOD/氨氮/重金属/富营养化), 土壤污染(重金属/农药/石油烃)'
' 2. 气候变化: 温室效应(CO2/CH4/N2O/CFCs/SF6), 全球变暖(当前+1.2℃ vs工业化前), 碳循环(化石燃料排放/土地利用/海洋吸收/陆地碳汇), 碳中和路径(减排/碳捕获CCS/负排放BECCS)'
' 3. 地质灾害: 滑坡(稳定性分析:安全系数Fs=抗滑力/下滑力), 泥石流(形成条件:物源+水+地形), 地面沉降(地下水超采/矿产开采/构造), 地裂缝'
' 4. 洪涝灾害: 暴雨内涝/江河泛滥, 重现期(10年/50年/100年一遇), 防洪(堤防/水库调度/蓄滞洪区/海绵城市)'
' 5. 环境监测: 自动监测站(大气/水质), 遥感监测(MODIS/Landsat/Sentinel环境指标反演), 生物监测, 无人机监测'
' 6. EHS关联: 环境影响评价(EIA), 生命周期评估(LCA), 环境风险评价, 突发环境事件应急预案',
'humboldt', ['environmental', 'climate-change', 'disaster', 'ehs', 'monitoring'])

# ========================================================================
# PART H: BIOLOGY / LIFE SCIENCE ENHANCEMENT (darwin primary)
# ========================================================================

add('[BioKB] 细胞与分子生物学核心',
'Cell & Molecular Biology Core:'
' 1. 细胞学说: 所有生物由细胞构成(1838-39 Schleiden/Schwann), 细胞来自已有细胞(Virchow 1855), 细胞是生命的基本结构与功能单位'
' 2. 细胞结构: 细胞膜(磷脂双分子层+膜蛋白/流动镶嵌模型Singer-Nicolson), 细胞核(核膜/核孔/染色质/核仁), 线粒体(内膜嵴增加表面积/氧化磷酸化/ATP合成/mtDNA母系遗传), 内质网(粗面RER核糖体蛋白合成/滑面SER脂质合成与解毒), 高尔基体(蛋白修饰/分选/囊泡运输), 溶酶体(水解酶/pH≈5), 叶绿体(光合作用光反应类囊体膜/暗反应Calvin循环基质), 细胞骨架(微管/微丝/中间纤维)'
' 3. 细胞分裂: 有丝分裂(前期/中期/后期/末期), 减数分裂(减I同源染色体分离/减II姐妹染色单体分离, 交叉互换提供遗传多样性), 细胞周期(G1/S/G2/M, Cyclin-CDK调控)'
' 4. 细胞信号转导: 受体类型(GPCR/RTK酶联受体/离子通道受体/核受体), 第二信使(cAMP/IP3/DAG/Ca2+), 信号通路(MAPK/PI3K-Akt/JAK-STAT/Wnt/Hedgehog/Notch/NF-kB/TGF-beta)'
' 5. 程序性细胞死亡: 凋亡(线粒体途径Bcl-2家族/死亡受体途径Caspase级联), 自噬(自噬体→溶酶体融合), 坏死(被动细胞破裂/炎症), 焦亡(炎症小体/Gasdermin孔洞)',
'darwin', ['cell-biology', 'molecular-biology', 'signaling', 'cell-cycle', 'apoptosis'])

add('[BioKB] 遗传学与基因组学',
'Genetics & Genomics:'
' 1. 分子遗传学: DNA双螺旋(Watson-Crick 1953, A-T/C-G碱基互补, 大沟小沟), DNA复制(半保留复制, 前导链连续/滞后链冈崎片段, DNA聚合酶III), 转录(RNA聚合酶II, 启动子TATA盒/CAAT盒/GC盒, 增强子/沉默子), 翻译(核糖体大亚基60S+小亚基40S, tRNA反密码子, 遗传密码表64个密码子)'
' 2. 基因表达调控: 原核(操纵子模型Lac/Trp), 真核(转录因子/染色质重塑/组蛋白修饰乙酰化/甲基化/DNA甲基化CpG岛/增强子-启动子环Hi-C), 非编码RNA(miRNA/siRNA/lncRNA/circRNA/piRNA)'
' 3. 经典遗传学: 孟德尔定律(分离律/自由组合律), 连锁与交换(Morgan果蝇), Hardy-Weinberg平衡(p^2+2pq+q^2=1), 数量性状(QTL定位/遗传力h^2)'
' 4. 基因组学: 人类基因组(30亿碱基对, ~20000基因, 98%非编码区, ENCODE功能元件), 比较基因组学, 群体基因组学(HapMap/1000 Genomes/gnomAD)'
' 5. 遗传变异: SNP/Indel/CNV/SV, 致病突变(ClinVar/HGMD), GWAS全基因组关联, 多基因风险评分PRS, 药物基因组学(PharmGKB)'
' 6. 表观遗传学: DNA甲基化(5mC/5hmC), 组蛋白修饰(H3K4me3激活/H3K27me3抑制/H3K9me3异染色质), 染色质开放性(ATAC-seq/DNase-seq), 表观遗传重编程(山中因子诱导iPSC)',
'darwin', ['genetics', 'genomics', 'epigenetics', 'dna', 'gwas'])

add('[BioKB] 生理学与系统生物学',
'Physiology & Systems Biology:'
' 1. 神经生理: 静息电位(-70mV, Na+/K+ ATP酶维持), 动作电位(去极化Na+内流→复极化K+外流→超极化→不应期), 突触传递(化学突触:神经递质释放/受体结合/重摄取), 神经递质(谷氨酸兴奋/ GABA抑制/多巴胺奖赏/5-HT情绪/ACh运动/NE警觉)'
' 2. 循环系统: 心动周期(收缩期/舒张期), 心电传导(SA结→AV结→His束→Purkinje纤维), 血压调节(RAAS系统/压力感受器), 血液成分(RBC/WBC/血小板/血浆)'
' 3. 呼吸系统: 肺通气(膈肌/肋间肌), 气体交换(肺泡-毛细血管O2/CO2扩散, Fick定律), 氧合血红蛋白解离曲线(S形→Bohr效应/pH降低曲线右移)'
' 4. 内分泌系统: 下丘脑-垂体-靶腺轴(HPA轴应激/HPT轴代谢/HPG轴生殖), 激素类型(肽类/固醇类/氨基酸衍生物), 信号转导(腺苷酸环化酶cAMP/磷脂酶C IP3+DAG/酪氨酸激酶)'
' 5. 免疫系统: 先天免疫(巨噬细胞/中性粒细胞/NK细胞/补体), 适应性免疫(B细胞抗体/T细胞TCR, MHC-I/CD8+细胞毒/MHC-II/CD4+辅助), 免疫记忆, 自身免疫/过敏/免疫缺陷'
' 6. 系统生物学: 组学整合(基因组+转录组+蛋白组+代谢组), 网络生物学(基因调控网络/蛋白互作网络PPI/代谢网络), 动力学建模(ODE/PDE/Bool网络), 数字孪生(Physiome Project)',
'darwin', ['physiology', 'neuroscience', 'immune', 'endocrine', 'systems-biology'])

add('[BioKB] 生态学与进化生物学',
'Ecology & Evolutionary Biology:'
' 1. 进化论: 自然选择(Darwin 1859, 过度繁殖→遗传变异→生存斗争→适者生存), 现代综合进化论(Fisher/Haldane/Wright群体遗传学), 分子进化(中性理论Kimura 1968/Molecular Clock), 进化发育生物学(Evo-Devo, Hox基因)'
' 2. 物种形成: 异域物种形成(地理隔离→生殖隔离), 同域物种形成, 邻域物种形成, 生殖隔离机制(交配前/交配后, Dobzhansky-Muller模型), 物种概念(生物学物种/形态学/生态学/系统发育)'
' 3. 系统发生: 支序分类学(共衍征Synapomorphy+简约法/最大似然/贝叶斯), 生命之树(LUCA→细菌/古菌/真核三域系统Woese), 分子系统学(DNA条形码/系统基因组学)'
' 4. 种群生态: 指数增长:dN/dt=rN, 逻辑斯蒂增长:dN/dt=rN(1-N/K), r选择(高速增殖/小型化/短命)vs K选择(稳定环境/大型化/长寿), 集合种群(Metapopulation: Levins模型dp/dt=cp(1-p)-ep)'
' 5. 群落生态: 种间关系(竞争Lotka-Volterra/捕食/Holling功能响应/寄生/共生), 食物网/营养级(生产者→初级消费者→次级消费者→顶级捕食者), 生态金字塔(数量/生物量/能量), 关键种(Keystone Species)'
' 6. 生物地理: 大陆漂移对物种分布的影响, Wallace线, MacArthur&Wilson岛屿生物地理学(物种-面积关系S=cA^z, 平衡理论:迁入=灭绝)',
'darwin', ['evolution', 'ecology', 'speciation', 'phylogenetics', 'biogeography'])

# ========================================================================
# PART I: HISTORY KNOWLEDGE BASE (herodotus primary)
# ========================================================================

add('[HistKB] 历史学全局硬约束',
'History Global Hard Constraints for LLM+JEPA:'
' 1. 所有历史叙述/分析/推演必须基于已验证的史料, 禁止编造人物/事件/年代/制度/因果关系'
' 2. 历史时序约束: JEPA时序模型严格保证时间线单向性(t1<t2→事件1在事件2之前), 禁止时间倒流/穿越式叙述'
' 3. 史料分级制度: 一手史料(档案/日记/考古实物/碑刻/同时代记录)>二手史料(正史/传记/回忆录)>三手(研究论文/教科书)'
' 4. 历史因果推理: 区分直接原因/根本原因/触发事件, 多因素分析, 避免单一原因决定论, 尊重历史偶然性与必然性的辩证关系'
' 5. LLM输出必须标注历史信息的确定性等级: 确证(多源互证)/基本可信(主流观点)/存疑(有争议)/待考证(单一来源)',
'herodotus', ['history', 'hard-constraint', 'jepa', 'temporal', 'source-criticism'])

add('[HistKB] 中国历史全库规范',
'Chinese History Complete Canon:'
' 1. 先秦: 夏商周(甲骨文/青铜文明), 春秋五霸/战国七雄, 百家争鸣, 秦统一(221BC, 书同文车同轨/郡县制/长城)'
' 2. 两汉: 西汉(文景之治/武帝开边/张骞通西域/丝绸之路), 东汉(光武中兴/造纸术蔡伦/地动仪张衡/医学张仲景华佗)'
' 3. 魏晋南北朝: 三国鼎立/西晋短暂统一/五胡乱华/南北朝对峙, 民族大融合, 佛教传入与本土化'
' 4. 隋唐: 隋(大运河/科举制), 唐(贞观之治/开元盛世/武则天/安史之乱转折), 盛唐文化(唐诗/长安国际都市/鉴真东渡/玄奘西行)'
' 5. 宋元: 北宋(澶渊之盟/王安石变法/清明上河图繁荣/活字印刷), 南宋(经济重心南移完成/理学朱熹), 元(蒙古帝国/行省制/马可波罗)'
' 6. 明清: 明(郑和下西洋/永乐大典/一条鞭法/资本主义萌芽/海禁), 清(康乾盛世/闭关锁国/鸦片战争1840转折/太平天国/洋务运动/戊戌变法/辛亥革命1911)'
' 7. 现当代: 民国(五四运动1919/抗日战争1937-1945/解放战争), 新中国(1949-/改革开放1978-/现代化)',
'herodotus', ['chinese-history', 'dynasties', 'timeline', 'events'])

add('[HistKB] 世界历史全库规范',
'World History Complete Canon:'
' 1. 古代文明: 美索不达米亚(Sumer楔形文字/Hammurabi法典), 古埃及(金字塔/法老/象形文字), 古印度(哈拉帕/吠陀/佛教), 古希腊(雅典民主/斯巴达/伯罗奔尼撒战争/亚历山大帝国), 古罗马(共和→帝制/罗马法/拉丁语/基督教化/西罗马476灭亡)'
' 2. 中世纪: 拜占庭帝国(查士丁尼法典/圣索菲亚大教堂), 伊斯兰文明兴起(穆罕默德622/倭马亚/阿拔斯-百年翻译运动), 查理曼帝国(800加冕), 封建制度/十字军东征(1096-1291), 蒙古帝国(成吉思汗1206, 最大陆地帝国)'
' 3. 近代早期: 文艺复兴14-16C, 大航海时代(哥伦布1492/达伽马/麦哲伦), 宗教改革(路德1517/加尔文), 三十年战争(威斯特伐利亚体系1648), 科学革命(哥白尼/伽利略/牛顿)'
' 4. 革命时代: 英国革命(1640-1688/权利法案), 美国独立(1776/1787宪法), 法国大革命(1789/人权宣言), 拿破仑战争(1803-1815), 拉美独立(玻利瓦尔/圣马丁)'
' 5. 19世纪: 工业革命(蒸汽机→铁路→工厂体系→城市化), 民族主义兴起(意大利统一/德国统一), 殖民主义高峰(非洲瓜分/柏林会议1884), 美国内战(1861-1865/废奴), 日本明治维新(1868)'
' 6. 20世纪: 一战(1914-1918/凡尔赛体系), 十月革命(苏联1917), 大萧条(1929), 二战(1939-1945/联合国), 冷战(1947-1991/北约vs华约/核威慑/朝鲜战争/越战/古巴导弹危机/太空竞赛), 去殖民化/全球化/信息化',
'herodotus', ['world-history', 'civilization', 'revolutions', 'cold-war', 'globalization'])

add('[HistKB] 考古学与史料学方法',
'Archaeology & Historical Source Methodology:'
' 1. 考古方法: 田野调查/发掘(地层学Harris Matrix)/类型学/测年(C14 AMS/树轮/热释光/OSL光释光/古地磁/钾氩法), 遥感考古(LiDAR/卫星影像/探地雷达GPR)'
' 2. 考古发现: 周口店北京猿人(70万年), 良渚古城(5300-4300年前/世界遗产/大型水利系统), 三星堆(3000年前/青铜神树/金杖/纵目面具), 秦始皇陵兵马俑, 殷墟甲骨文(3300年前), 海昏侯墓(西汉/完整)'
' 3. 世界考古: 图坦卡蒙墓(埃及), 庞贝古城(罗马/公元79年维苏威火山), 马丘比丘(印加), 死海古卷(1947 Qumran), Troy(土耳其/Schliemann), Sutton Hoo(盎格鲁-撒克逊船葬)'
' 4. 文本考证: 版本学(抄本/刻本/活字本/石印本/影印本), 校勘学(对校/本校/他校/理校四法), 辨伪学(文献内容/时代特征/语言风格/物质载体综合判断)'
' 5. 口述史学: 访谈设计(开放式/半结构化), 记忆的可靠性(衰退/偏差/重构), 伦理(知情同意/隐私保护/回馈社区), 口述与文字史料的互证与互补',
'herodotus', ['archaeology', 'dating-methods', 'textual-criticism', 'oral-history', 'excavation'])

# ========================================================================
# PART J: PHILOSOPHY KNOWLEDGE BASE (guizhu + montesquieu)
# ========================================================================

add('[PhilKB] 哲学全局硬约束',
'Philosophy Global Hard Constraints for LLM+JEPA:'
' 1. 所有哲学推理/论证必须严格遵循逻辑公理(矛盾律/排中律/同一律/充足理由律)'
' 2. 禁止诡辩/禁止自相矛盾/禁止伪逻辑/禁止偷换概念(Equivocation)/禁止循环论证(Begging the Question)'
' 3. 哲学概念必须明确定义, 推理链完整可追溯(前提→推理步骤→结论), 每步标注推理规则'
' 4. 区分事实判断(可验证真伪)与价值判断(涉及价值观/立场), 价值判断需要说明所依据的伦理框架'
' 5. JEPA因果建模必须遵守哲学因果律: 因果关系≠相关性, 区分必要因/充分因/INUS条件(Mackie)',
'guizhu', ['philosophy', 'hard-constraint', 'logic', 'reasoning', 'causality'])

add('[PhilKB] 形而上学与认识论',
'Metaphysics & Epistemology:'
' 1. 形而上学(Metaphysics): 存在论Ontology(存在是什么/共相问题/实在论vs唯名论), 实体与属性(亚里士多德范畴), 因果关系(Hume问题:归纳无法证明因果必然性/Kant先验范畴/Pearl因果革命), 时间与空间(绝对论Newton vs 关系论Leibniz/Einstein相对论时空), 自由意志vs决定论(相容论Compatibilism/不相容论/Libertarianism), 可能世界(Lewis模态实在论/Kripke)'
' 2. 认识论(Epistemology): 知识定义(JTB: Justified True Belief, Gettier问题1963), 知识来源(理性主义Descartes/Spinoza/Leibniz vs 经验主义Locke/Berkeley/Hume, Kant综合先天), 真理理论(符合论/融贯论/实用主义/冗余论), 怀疑论(笛卡尔的恶魔/庄周梦蝶/缸中之脑)'
' 3. 科学哲学(Philosophy of Science): 科学分界(Popper可证伪性/Kuhn范式革命/Lakatos研究纲领/Feyerabend方法论无政府主义), 科学解释(Hempel DN模型/因果解释Salmon/机制解释), 科学实在论vs反实在论(Pessimistic Induction/No Miracles Argument), 还原论与涌现'
' 4. 中国哲学: 儒家(孔子仁礼/孟子性善/荀子性恶/程朱理学/陆王心学), 道家(老子道/自然无为/庄子逍遥齐物), 墨家(兼爱/非攻/逻辑), 法家(商鞅/韩非), 兵家(孙子), 佛家禅宗(慧能坛经/顿悟), 宋明理学(太极/理气/心性/格物致知), 当代新儒家(牟宗三/唐君毅)',
'guizhu', ['metaphysics', 'epistemology', 'philosophy-of-science', 'chinese-philosophy', 'confucianism'])

add('[PhilKB] 伦理学与政治哲学',
'Ethics & Political Philosophy:'
' 1. 规范伦理学: 功利主义(Bentham最大幸福原则/Mill快乐质量/规则功利主义/偏好功利主义Singer/有效利他主义), 道义论(Kant绝对命令:只按照你同时愿意成为普遍法则的准则行动/尊重人公式:人是目的不是手段/Ross显见义务), 德性伦理(Aristotle中道/幸福Eudaimonia/实践智慧Phronesis/MacIntyre德性传统/Nussbaum能力路径), 关怀伦理(Gilligan/Noddings), 契约论(Hobbes/契约主义Scanlon)'
' 2. 元伦理学: 道德实在论vs反实在论, 认知主义vs非认知主义(情感主义Ayer/Stevenson, 规定主义Hare), 道德自然主义vs非自然主义(Moore自然主义谬误/开放问题论证), 道德相对主义vs普遍主义, 进化伦理学(Darwin道德情感的进化起源)'
' 3. 应用伦理学: 生命伦理(安乐死/堕胎/基因编辑/器官移植/知情同意Belmont报告), 环境伦理(人类中心vs生物中心vs生态中心, 代际正义, 深层生态学Naess), AI伦理(算法公平/透明解释XAI/价值对齐/AI权利/致命自主武器LAWS), 商业伦理(ESG/利益相关者理论/社会契约/道德困境)'
' 4. 政治哲学: 社会契约(Hobbes利维坦/Locke自然权利生命自由财产/Rousseau公意/Rawls正义论无知之幕+两个原则/Nozick最弱国家), 正义理论(Rawls平等自由主义/Nozick自由至上主义/Dworkin资源平等/Sen能力平等/社群主义Sandel), 权力与自由(Mill论自由/伤害原则, 积极自由vs消极自由Berlin), 民主理论(参与式/审议式/激进民主), 马克思主义(历史唯物主义/剩余价值/异化), 女权主义政治理论',
'guizhu', ['ethics', 'political-philosophy', 'bioethics', 'ai-ethics', 'justice'])

add('[PhilKB] 心灵哲学与意识研究',
'Philosophy of Mind & Consciousness Studies:'
' 1. 心身问题: 二元论(Descartes实体二元论/属性二元论/交互难题), 物理主义(同一论/功能主义Putnam/消除唯物论Churchlands), 观念论(Berkeley), 中立一元论(Russell)'
' 2. 意识难题: 困难问题vs容易问题(Chalmers 1994), 意识是什么(Nagel:成为蝙蝠是什么感觉?/Block:现象意识vs通达意识), 意识的神经关联(NCC, Crick&Koch), 全局工作空间理论(Baars/Dehaene), 整合信息论IIT(Tononi Φ值), 高阶思维理论(HOT/Rosenthal), 预测加工理论(Clark/Friston:大脑是贝叶斯预测机)'
' 3. 意向性(Intentionality): Brentano:心理现象区别于物理现象的指向性/关于性, 表征理论(Fodor思想语言LOT/Millikan目的论语义学), 现象意向性(Husserl/现象学传统)'
' 4. 个人同一性(Personal Identity): 心理连续性理论(Locke/Parfit), 身体连续性理论, 动物主义(Olson), 叙事自我理论(Ricoeur/Schechtman)'
' 5. 自由意志: 不相容论vs相容论, 神经科学挑战(Libet实验/Soon 2008前额叶预测), Frankfurt层级欲望理论(一阶欲望vs二阶欲望), Strawson反应态度, Pereboom硬决定论'
' 6. AI哲学: 图灵测试(1950), 中文屋论证(Searle 1980:句法不是语义), 强AIvs弱AI, 意识与AI(能否人造意识/是否需要生物基底), 价值对齐问题(人类价值观如何编码), 超级智能风险(Bostrom)',
'guizhu', ['philosophy-of-mind', 'consciousness', 'intentionality', 'free-will', 'ai-philosophy'])

# Philosophy shared to montesquieu (legal/political philosophy focus)
add('[PhilKB] 法哲学与政治哲学规则',
'Legal Philosophy & Political Philosophy Rules:'
' 1. 法哲学: 自然法(Aquinas:法是对共同善的理性规定/Fuller:法内在道德8原则), 法律实证主义(Austin:法即命令/Hart:初级规则+次级规则/承认规则), 法现实主义(Holmes/法律的生命不在于逻辑在于经验), Dworkin:原则vs政策/整全性Integrity/Hercules法官)'
' 2. 法律推理: 演绎推理(法条→事实→结论), 类比推理(遵循先例Stare Decisis), 辩证推理(利益平衡/原则权衡), 法律解释(文本主义/原意主义/目的解释/动态解释)'
' 3. 正义理论在法律中的应用: 矫正正义/分配正义/程序正义, 刑法哲学(报应论Kant/威慑论Bentham/改造论/修复性正义), 侵权法(矫正正义/效率/威慑)'
' 4. 人权哲学: 自然权利/法律权利, 人权证成(利益理论/意志理论), 三代人权(公民政治权→经济社文权→集体发展权), 国际人权法(UDHR1948/ICCPR/ICESCR)'
' 5. 宪法哲学: 宪政主义/权力分立/司法审查/民主与法治/紧急状态与权利克减',
'montesquieu', ['legal-philosophy', 'jurisprudence', 'human-rights', 'constitutional-law', 'justice'])

# ========================================================================
# OUTPUT
# ========================================================================

output_path = 'extended_domain_knowledge_v3.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(entries, f, ensure_ascii=False, indent=2)

print(f'Generated {len(entries)} knowledge entries → {output_path}')

# Summary by soul
from collections import Counter
soul_counts = Counter(e['soul'] for e in entries)
print('\nEntries by soul:')
for soul, count in soul_counts.most_common():
    print(f'  {soul:>15}: {count}')
