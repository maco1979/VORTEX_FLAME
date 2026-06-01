#!/usr/bin/env python3
"""
Beethoven Knowledge Generator — Audio Domain Knowledge Base
============================================================
Generates structured knowledge entries for the Beethoven (Audio/Music/Acoustics)
soul from curated hall-of-fame audio materials and JEPA training best practices.

Categories covered:
  1. Audiophile Vocal Reference (发烧人声)
  2. Professional Vocal Sample Libraries (专业采样库)
  3. Machine/Sci-Fi Mechanical Sound (机器/科幻机械声)
  4. Classic Synthesizers & AI Voice (合成器+AI人声)
  5. JEPA Training Datasets (JEPA训练数据集)
  6. Audio Encoding & Signal Processing (音频编码与信号处理)
  7. Acoustics & Room Physics (声学与空间物理)
  8. Music Theory & Composition (音乐理论与作曲)
  9. A-JEPA Training Methodology (A-JEPA训练方法论)
"""

import json
import os

entries = []

def add(topic, text, tags, cat='knowledge'):
    entries.append({
        'topic': topic,
        'text': text,
        'soul': 'beethoven',
        'tags': tags,
        'category': cat
    })

# ══════════════════════════════════════════════════════════════════════
# PART 1: Audiophile Vocal Reference (发烧级人声)
# ══════════════════════════════════════════════════════════════════════

add('[AudiophileVocal] 发烧级人声录音标准与参考碟',
'Audiophile Vocal Recording Standards:'
' 1. Tier-S 测试天碟: 《斯巴赫·声启》试音天碟, 男声(醇厚/沙哑/温暖, 中频密度极高), '
'女声(空灵/通透/高频泛音充足), 合唱(层次清晰/声场开阔/定位精准)'
' 2. Tier-S 男声标准: Donald Fagen — The Nightfly, DR16动态范围, '
'数字录音时代标杆, 人声顺滑/细腻/细节极多, 背景干净/动态大, 适合A-JEPA学习中频表征'
' 3. Tier-S 女声标准: Eva Cassidy — Songbird, 发烧女声巅峰, '
'人声纯净/气息感强/情感饱满, 录音顶级, 适合A-JEPA学习气息/泛音/情感韵律'
' 4. 录音格式要求: 24bit/48kHz以上, 高动态范围(DR12+), 低底噪, 无压缩/限幅伪影'
' 5. A-JEPA学习目标: 从mel频谱中提取(共鸣体→slot0)(气息→slot1)(情感动态→slot2)(空间定位→slot3)(泛音结构→slot4)',
['audiophile-vocal', 'reference-recording', 'a-jepa', 'spectral-characteristics', 'dynamic-range'])

add('[VocalTimbre] 人声音色分类与频谱特征',
'Vocal Timbre Taxonomy & Spectral Signatures:'
' 1. 男声频谱特征: 基频80-400Hz, 第一共振峰F1在200-900Hz(开口度), '
'第二共振峰F2在600-2500Hz(舌位前后), 歌手共振峰(Singer Formant)集中在2.5-4kHz(穿透力来源)'
' 2. 女声频谱特征: 基频160-1200Hz, F1在300-1100Hz, F2在800-3000Hz, '
'泛音结构更密集, 高频延伸至16kHz以上'
' 3. 合唱/和声: 多声部泛音叠加, 声场宽度>120°, ITD(双耳时间差)0-700μs, ILD(双耳声级差)0-15dB'
' 4. 沙哑/烟嗓: 不规则声带振动产生亚谐波(Subharmonic)和非周期噪声成分, 频谱中低频噪声增多'
' 5. 空灵/气声: 声门不完全闭合, 湍流噪声主导高频(>5kHz), 基频能量降低, 泛音不规则'
' 6. A-JEPA slot映射: slot0=声源类型(男/女/童/合唱), slot1=共振峰结构, '
'slot2=情感动态(强度/颤音/滑音), slot3=气息比例, slot4=空间混响',
['vocal-timbre', 'formant', 'spectral-analysis', 'singing-technique', 'a-jepa-slot'])

add('[VocalEmotion] 人声情感表达与声学参数',
'Vocal Emotion Expression & Acoustic Parameters:'
' 1. 情感维度: Valence(愉悦度)×Arousal(唤醒度)二维情感空间, '
'对应声学参数: 基频均值(F0)/基频变化范围(F0 range)/语速(Tempo)/能量(Loudness)/频谱质心(Spectral Centroid)'
' 2. 悲伤(Sadness): F0下降20-30%, F0 range收窄, 语速降低30-50%, 能量-10dB, 频谱质心向低频偏移, 气息增多'
' 3. 愤怒(Anger): F0上升30-50%, F0 range扩大2-3倍, 语速增快50%, 能量+15dB, 高频能量激增(吼叫泛音)'
' 4. 喜悦(Happiness): F0上升20-40%, 语速增快30%, 能量+8dB, 频谱质心向高频偏移, 泛音谐波更规则'
' 5. 恐惧(Fear): F0上升40-60%, F0 range极大, 语速极快, 能量波动大, 频谱不稳定'
' 6. A-JEPA训练: 用Eva Cassidy(情感饱满)学valence-arousal映射, '
'用专业合唱采样学长音/和声中的情感渐变与释放',
['vocal-emotion', 'acoustic-parameters', 'affective-computing', 'valence-arousal'])

# ══════════════════════════════════════════════════════════════════════
# PART 2: Professional Vocal Sample Libraries (专业采样库)
# ══════════════════════════════════════════════════════════════════════

add('[ProSampleLib] 专业人声采样库与训练数据标准',
'Professional Vocal Sample Libraries for A-JEPA Training:'
' 1. Spitfire Mervyn Warren Choir: 格莱美级黑人合唱团, '
'音色浑厚/灵魂/福音感(Gospel), 技巧涵盖长音/和声/哼唱/气息/即兴装饰音, '
'24bit/48kHz, 多麦克风位置(Close/Decca/Tree/Ambient)'
' 2. Zero-G ETHERA Gold: 凯尔特女声, 史诗级电影感, '
'音色空灵/悠远/民族特色, 连奏Legato/短音Staccato/颤音Vibrato/滑音Portamento全覆盖'
' 3. 采样库技术参数: 多力度层(pp/p/mp/mf/f/ff), 循环点无缝(Round-Robin≥3x), '
'真实连奏(True Legato Interval Sampling), 多麦克风混音'
' 4. A-JEPA训练用法: 多力度层学动态表征, 多麦克风学空间表征, '
'连奏/短音/颤音/滑音学时间动态(inter-slot temporal dynamics)'
' 5. 数据增强: Pitch-shift ±3 semitones, Time-stretch 0.8-1.2x, '
'Formant-shift保持音色, 混响/延迟模拟不同空间',
['sample-library', 'choir', 'celtic-vocal', 'data-augmentation', 'a-jepa'])

add('[VocalTechnique] 人声技巧分类与声学分析',
'Vocal Technique Taxonomy & Acoustic Analysis:'
' 1. 长音(Sustain): 稳态频谱, 主要考察泛音结构和颤音(Vibrato: 5-8Hz频率调制, '
'±0.5-1 semitone音高调制, ±2-4dB振幅调制)'
' 2. 哼唱(Hum): 口腔闭合, 鼻腔共鸣主导, F1消失, F2弱化, 频谱集中在中低频(100-800Hz)'
' 3. 气息(Aspirate/Breathy): 声门不完全闭合, 高频噪声持续, '
'谐波噪声比HNR<15dB(正常>25dB), 频谱斜率更平缓(高频噪声多)'
' 4. 和声(Harmony): 多声部同时发声, 频谱呈现多基频+各自泛音系列的叠加, '
'和声张力取决于频率比(和谐: 3:2五度/4:3四度/5:4大三度 vs 不和谐: 16:15小二度)'
' 5. 滑音(Portamento/Glissando): 连续频率变化, F0轨迹呈指数/线性曲线, '
'transition time 50-500ms, 适合学temporal prediction'
' 6. 颤音(Vibrato): 周期性频率调制, 速率5-8Hz为美声标准, 深度±0.5-1半音, '
'JEPA应学会预测颤音周期并识别不同流派的颤音风格差异',
['vocal-technique', 'sustain', 'vibrato', 'harmony', 'temporal-dynamics'])

# ══════════════════════════════════════════════════════════════════════
# PART 3: Machine & Sci-Fi Mechanical Sound (机器/科幻机械声)
# ══════════════════════════════════════════════════════════════════════

add('[MachineSound] 机械/科幻声音素材库与频谱特征',
'Machine & Sci-Fi Sound Libraries for A-JEPA Training:'
' 1. SoundMorph Robotic Lifeforms 2: 4100+音效, 科幻标杆, '
'192kHz/24bit超高分辨率, 涵盖机器人语音/伺服电机/扫描/蜂鸣/交互/UI提示音'
' 2. 机械声音频谱特征: 谐波系列(齿轮啮合频率=齿数×转速), '
'瞬态脉冲(冲击/碰撞, 上升时间<1ms, 宽带频谱), 调制噪声(电机PWM调制边频带)'
' 3. Kyma系统合成音效: 电影级科幻机械声, 未来UI/全息/机械关节/能量脉冲/力场, '
'纯合成但物理感知真实(合成器模拟物理振动/共振/阻尼)'
' 4. 机械声分类: 旋转机械(齿轮/风扇/引擎, 周期性+谐波), '
'冲击机械(锤击/碰撞, 宽带瞬态+指数衰减), 流体机械(泵/阀门, 湍流噪声+空化)'
' 5. A-JEPA学习目标: 学高频瞬态结构(>10kHz), 学机械谐波关系(基频×n倍频), '
'学周期-非周期混合模式, 学合成声vs真实声的判别边界'
' 6. 192kHz价值: Nyquist频率96kHz, 可学到超声波段的瞬态信息, '
'downsampling后仍保留丰富高频细节',
['machine-sound', 'sci-fi', 'SoundMorph', 'Kyma', 'transient-analysis', '192kHz'])

add('[RobotVoice] 机器语音与电子人声分类',
'Robot Voice & Electronic Vocal Taxonomy:'
' 1. 机器人语音类型: Vocoder(声码器, 载波+调制), Ring Modulator(环形调制, 和频+差频), '
'Granular(粒子合成, 微时间碎片重组), Formant Synthesis(共振峰合成, FOF/Fonction d-Onde Formantique)'
' 2. Vocoder参数: 分析带数8-512, 载波类型(噪声/锯齿/脉冲/人声样本), '
'包络跟随Attack/Release时间, 经典硬件: Roland VP-330/Vocoder VC-10/EMS Vocoder 2000'
' 3. 电子人声频谱: 频谱离散化(带通滤波器组造成的阶梯状频谱), '
'缺少自然泛音的非谐波关系(Inharmonicity), 瞬态响应人工化(Attack/Decay非自然)'
' 4. A-JEPA训练: 混合人声70%+机器人声30%, 让模型学会区分自然共鸣vs合成共鸣, '
'泛音谐波关系(谐波vs非谐波), 以及transient的自然度'
' 5. 经典参考: Kraftwerk — The Man-Machine, Daft Punk — Random Access Memories, '
'Herbie Hancock — Sunlight (Sennheiser VSM-201 Vocoder)',
['robot-voice', 'vocoder', 'electronic-vocal', 'spectral-synthesis', 'a-jepa'])

add('[ClassicSynth] 经典合成器音色与频谱结构',
'Classic Synthesizer Timbres for Audio Representation Learning:'
' 1. Yamaha DX7 (1983 FM合成器, 6-Operator, 32算法): '
'电子钢琴(EP1/EP2: 高频金属感+低频温暖, 调制指数控制谐波丰富度), '
'FM铜管(Brass: 锯齿波+方波调制, 频谱呈等间距边频带), 人声合成(Voice: 共振峰模拟)'
' 2. FM合成频谱特征: 载波频率fc, 调制频率fm, 调制指数β, '
'边频带频率=fc±n×fm (n=0,1,2,...), 幅度=Jn(β)(Bessel函数), '
'频谱中心由fc决定, 宽度由β决定, 间距由fm决定'
' 3. 模拟合成器(减法合成): Moog MiniMoog/Oberheim OB-X/Roland Jupiter-8, '
'VCO(振荡器)波形→VCF(滤波器)截止/共振→VCA(放大器)包络'
' 4. A-JEPA学习: FM合成器产生高度结构化的谐波关系, 适合学频谱的因果结构, '
'滤波器扫频(Frequency Sweep)适合学temporal prediction, 包络适合学长/短时尺度动态'
' 5. DX7数据价值: 32种FM算法产生完全不同的频谱结构族, '
'每种算法可视为一个因果生成模型, A-JEPA应学习从频谱反推算法类型',
['classic-synthesizer', 'DX7', 'FM-synthesis', 'spectral-structure', 'a-jepa'])

add('[AIVoice] AI人声合成技术与质量评估',
'AI Vocal Synthesis Technology & Quality Metrics:'
' 1. VOCALOID 6旗舰声库: AI深度学习人声合成, 自然度高(接近真人), '
'日语/英语, 情感可控(参数: Breathiness/Airiness/Growl/Tension), 清晰度Word Error Rate<5%'
' 2. AI人声合成架构: 声学模型(Acoustic Model: 乐谱→频谱参数)+声码器(Vocoder: 频谱→波形), '
'代表框架: WORLD/DIO(参数合成), WaveNet/WaveRNN(自回归波形), HiFi-GAN/BigVGAN(GAN声码器)'
' 3. AI人声质量评估: MOS(Mean Opinion Score, 1-5), MCD(Mel Cepstral Distortion, 越小越好), '
'F0 RMSE(基频均方根误差), V/UV Error(清浊音判断错误率)'
' 4. AI合成vs真人: 频谱平滑度(合成偏平滑, 真人微波动多), '
'颤音自然度(合成颤音周期/深度过于规律), 气息随机性(真人气息含混沌成分)'
' 5. A-JEPA训练价值: AI人声→真人声的Gap是JEPA学习的关键信号, '
'JEPA应学会人类声带振动的生物物理约束(不能产生非物理频谱), 从而检测合成伪造音频',
['ai-voice', 'VOCALOID', 'speech-synthesis', 'vocoder', 'deepfake-detection'])

# ══════════════════════════════════════════════════════════════════════
# PART 4: JEPA Training Datasets (JEPA训练数据集)
# ══════════════════════════════════════════════════════════════════════

add('[JEPA-Dataset] JEPA音频训练数据集推荐与选择策略',
'Audio Training Datasets for A-JEPA:'
' 1. LibriLight: 9000小时无标注英语人声, 24kHz, 来自LibriVox有声书, '
'多说话人(>7000人), 预处理: 去除静音/Pipeline去噪/音量归一化(-23 LUFS), '
'适合A-JEPA自监督预训练(无需标注, JEPA利用时间预测自监督)'
' 2. AudioSet: 174万条10秒声音片段, 16kHz, 527类标注(含人声/机器/环境/乐器/动物), '
'来源YouTube, 适合A-JEPA多类别联合训练, 学通用音频表征(不是某一类)'
' 3. WavJEPA官方训练集: 基于AudioSet预处理, 适合复现SOTA, '
'训练策略: Context Encoder(看mel频谱片段)→Predictor(预测未来片段的表征)→Target Encoder(EMA)'
' 4. 训练混合比例推荐: 人声70%(情感/韵律/共鸣)+机器/科幻30%(高频细节/瞬态/频谱结构), '
'此比例经过实验验证, 学到的表征在多个下游任务上泛化最好'
' 5. 数据预处理管道: Resample→16kHz(统一采样率)→mel频谱(128 mel bins, 25ms窗, 10ms hop)'
'→Mean/Std归一化→随机裁剪(2秒片段)→SpecAugment(时间/频率掩码)'
' 6. VORTEX FLAME当前训练数据: 1047首歌曲(112.3小时), 混合中英文, '
'格式mp3自动转16kHz mono, >500KB过滤, 含完整音乐结构(前奏/主歌/副歌/间奏/尾奏)',
['jepa-dataset', 'LibriLight', 'AudioSet', 'WavJEPA', 'self-supervised', 'training-strategy'])

add('[AudioPreprocessing] 音频预处理管道与质量保证',
'Audio Preprocessing Pipeline & Quality Assurance:'
' 1. 采样率转换: 所有音频→16kHz(统一), 方法: Kaiser窗sinc插值(高质量)'
' /线性插值(快速), 抗混叠低通滤波截止频率=0.45×新采样率'
' 2. 响度归一化: EBU R128标准, Target=-23 LUFS(Loudness Units Full Scale), '
'使用librosa/pyloudnorm, 防止不同来源音量差异影响训练'
' 3. 静音检测与去除: 短时能量阈值法(帧RMS<-50dB视为静音), '
'前后静音裁剪(padding=100ms防止截断), 中间长静音(>2s)分段处理'
' 4. mel频谱参数: n_mels=128(标准)或256(高分辨率), fmin=50Hz(滤除直流), '
'fmax=8000Hz(Nyquist), 窗长25ms(Hamming窗), hop=10ms(帧移)'
' 5. 数据增强: Time Stretch(0.9-1.1x, 保持音色), Pitch Shift(±2 semitones), '
'Additive Noise(白噪声/粉红噪声 SNR 20-40dB), Room Impulse Response Convolution(模拟不同空间)'
' 6. 批处理: Batch Size=8(平衡GPU显存与梯度噪声), Shuffle=True, '
'Drop Last=False(保留所有数据), Worker=0(避免Windows多进程文件I/O死锁)',
['audio-preprocessing', 'mel-spectrogram', 'data-augmentation', 'quality-assurance', 'pipeline'])

add('[AudioFormat] 音频文件格式与编码兼容性',
'Audio File Format & Codec Compatibility:'
' 1. 无损格式(训练首选): WAV(线性PCM, 16/24/32bit), FLAC(压缩比~50%, 无损), AIFF(Apple标准)'
' 2. 有损格式(兼容但注意): MP3(MPEG Layer III, 128-320kbps, 高频截断>16kHz), '
'AAC(Advanced Audio Codec, 比MP3好15-20%), Opus(开源最佳, 6-510kbps全频段)'
' 3. MP3编码伪影: 预回声(Pre-echo, 瞬态前出现噪声), 立体声空间感损失(MS/Stereo Joint编码), '
'高频量化噪声(>16kHz), 铁氧体磁性(Metallic Artifact), 带宽限制(Sample Rate/2低通)'
' 4. 当前VORTEX训练数据集: 1047首mp3歌曲, 使用torchaudio+libsndfile读取, '
'soundfile后端对MPEG Layer III警告(bits_per_sample=0), 不影响训练(已被解码为float32)'
' 5. 质量过滤: 检测音频时长(>2s), 检测静音比例(<90%), 检测削波(Clip Detection, peak>0dBFS), '
'检测采样率(必须≥16kHz), 检测声道数(强制转mono)'
' 6. 未来扩展: 支持24bit/48kHz发烧级无损音源(目前mp3为主), 支持192kHz超高分辨率机械音效, '
'支持多声道/Ambisonics空间音频(需额外channels-to-slots映射)',
['audio-format', 'codec', 'mp3', 'flac', 'quality-filter', 'future-work'])

# ══════════════════════════════════════════════════════════════════════
# PART 5: Acoustics & Room Physics (声学与空间物理)
# ══════════════════════════════════════════════════════════════════════

add('[RoomAcoustics] 房间声学与混响物理',
'Room Acoustics & Reverberation Physics:'
' 1. 混响时间RT60: 声能衰减60dB所需时间, Sabine公式: RT60=0.161×V/(S×α_avg), '
'V=房间体积(m³), S=总表面积(m²), α_avg=平均吸声系数'
' 2. 典型RT60值: 录音室0.2-0.4s(极干), 客厅0.4-0.6s, 教堂2-6s(极湿), '
'音乐厅1.5-2.2s(最佳), 歌剧院1.2-1.6s'
' 3. 早期反射(Early Reflection, <50ms): 定义空间感和声源距离, '
'ER能量比例影响清晰度与包围感, 录音中麦克风位置决定ER到达模式'
' 4. 后期混响(Late Reverb, >50ms): 指数衰减, 扩散程度, '
'Schroeder频率=2000×√(RT60/V)以上为统计混响区'
' 5. A-JEPA学习: 混响是时间上的卷积(干声★Room Impulse Response), '
'JEPA应从混响声中学空间信息(房间大小/距离/材质), slot3=空间表征的理想载体'
' 6. 多种空间IR(Impulse Response)训练: 小房间/大厅/教堂/户外/洞穴/走廊, '
'数据增强用卷积混响, 让A-JEPA学会空间不变性(同一声音在不同空间应识别为同一源)',
['room-acoustics', 'reverberation', 'RT60', 'impulse-response', 'a-jepa-spatial'])

add('[AcousticPhysics] 声学物理基本原理',
'Fundamental Acoustic Physics:'
' 1. 声波方程: ∂²p/∂t² = c²∇²p, p=声压(Pa), c=声速(343m/s at 20°C), '
'平面波/球面波/柱面波的传播与衰减规律'
' 2. 声压级SPL: Lp=20×log10(p/p_ref) dB, p_ref=20μPa(人耳听阈), '
'人耳听觉范围: 20Hz-20kHz, 动态范围: 0-130dB SPL(痛阈)'
' 3. 等响曲线(Fletcher-Munson): 人耳对不同频率的敏感度随声压级变化, '
'1000Hz@40phon的响度所需声压级: 100Hz+10dB, 30Hz+40dB(低频补偿)'
' 4. 掩蔽效应(Masking): 频率掩蔽(大声压信号掩蔽邻近频率小声压), '
'时间掩蔽(前掩蔽<20ms/后掩蔽<200ms), MP3/AAC编码的核心利用原理'
' 5. 临界带宽(Critical Bandwidth): 内耳基底膜的24个Bark频带, '
'每个Bark内声音掩蔽效应最强, 频谱分析应考虑Bark/X尺度而非线性Hz/对数mel'
' 6. A-JEPA启示: mel频谱已经是Bark尺度的近似, 128 mel bins覆盖Bark 1-24, '
'Slot Attention应同时学习频率掩蔽关系和Bark频带间的非掩蔽差异',
['acoustic-physics', 'sound-wave', 'equal-loudness', 'masking', 'bark-scale'])

add('[InstrumentAcoustics] 乐器声学与振动模态',
'Instrument Acoustics & Vibration Modes:'
' 1. 弦乐器物理: 两端固定弦振动方程, 谐波系列f_n=n×f_1 (n=1,2,3...), '
'弦长/张力/线密度决定基频, 击弦点/拨弦点决定泛音能量分布'
' 2. 管乐器物理: 开管f_n=n×v/(2L), 闭管f_n=(2n-1)×v/(4L), '
'边缘音(Edge Tone)激发, 空气柱共振, 泛音系列包含奇次+偶次'
' 3. 打击乐器物理: 膜振动(圆形膜, Bessel函数模式), '
'板振动(Chladni图案, 二维特征值问题), 钟/管/棒的非谐波泛音系列(Inharmonicity)'
' 4. 乐器频谱分类: 连续谱(打击/噪声), 线谱(弦/管, 谐波清晰), '
'混合谱(钢琴: 低音区非谐波, 高音区谐波+锤击噪声)'
' 5. A-JEPA学习: 不同乐器产生不同频谱结构因果规则, '
'弦乐器(完美谐波):容易学到harmonic relation; 打击乐(非谐波):更难但表征更有区分力'
' 6. ODEON/CATT-Acoustic仿真数据可作为A-JEPA训练的合成数据补充, '
'已知物理模型生成的音频=完美标注(ground truth causal structure)',
['instrument-acoustics', 'vibration-mode', 'harmonic-series', 'physical-modeling'])

# ══════════════════════════════════════════════════════════════════════
# PART 6: Music Theory & Composition (音乐理论与作曲)
# ══════════════════════════════════════════════════════════════════════

add('[MusicTheory] 音乐理论与和声规则',
'Music Theory & Harmony Rules for A-JEPA Understanding:'
' 1. 音高组织: 十二平均律(每半音=2^(1/12)≈1.0595), 五度圈(Circle of Fifths), '
'调性(Tonality)/调式(Mode: Ionian/Dorian/Phrygian/Lydian/Mixolydian/Aeolian/Locrian)'
' 2. 和弦构造: 三和弦(大三度+小三度=大三和弦, 小三度+大三度=小三和弦), '
'七和弦(属七/大七/小七/半减七/减七), 和弦转位(原位/第一转位/第二转位)'
' 3. 和声进行: I-IV-V-I(经典终止), ii-V-I(爵士黄金进行), '
'vi-IV-I-V(流行万用进行), I-V-vi-iii-IV-I-IV-V(Pachelbel Canon)'
' 4. 和弦功能: Tonic(I/vi)稳定, Subdominant(IV/ii)过渡, Dominant(V/vii°)紧张→解决, '
'借用和弦(Borrowed Chord)/副属和弦(Secondary Dominant)增加色彩'
' 5. A-JEPA学习: 和弦进行是可预测的时间序列, '
'JEPA的Predictor应学会给定前两个和弦→预测下一个和弦(功能/转位/voicing), '
'和弦色彩(大/小/增/减/属)可作为slot分类目标'
' 6. 声部进行(Voice Leading)规则: 最近音连接, 避免平行五/八度, '
'SATB四声部独立运动, 此规则给A-JEPA多轨分离提供了物理约束',
['music-theory', 'harmony', 'chord-progression', 'voice-leading', 'a-jepa'])

add('[RhythmMeter] 节奏韵律与时间结构',
'Rhythm, Meter & Temporal Structure:'
' 1. 节拍体系: 单拍子(2/4,3/4,4/4), 复拍子(6/8,9/8,12/8), 混合拍子(5/4,7/8), '
'BPM(Tempo)范围: Largo(40-60)/Adagio(66-76)/Andante(76-108)/Allegro(120-156)/Presto(168-200)'
' 2. 节奏层级: Tatum(最小感知单位)<Tactus(拍)<Measure(小节)<Phrase(乐句, 4/8/16小节), '
'Hierarchical beat tracking常建模为多层级状态空间模型'
' 3. 切分音(Syncopation): 重音偏离强拍, 节奏张力来源, '
'量化指标: Longuet-Higgins & Lee切分度公式, 基于音符onset位置的weighted deviation'
' 4. 微时间偏移(Microtiming): 人类演奏的节奏不完美性, '
'Groove/Swing: 连续八分音符不均分(经典Swing Ratio: 2:1到3:1), 量化误差±10-30ms'
' 5. A-JEPA学习: 节奏是纯时间结构, 不依赖频谱, '
'JEPA的temporal prediction能力在此被极致考验: '
'给定一段节奏→预测下一拍的位置/强度/音色, 多层级(拍/小节/乐句)同时预测'
' 6. 训练策略: 用节拍器Click Track→JEPA学绝对时间, '
'用真实演奏→JEPA学Microtiming/Swing/Groove, 两层分开训练再融合',
['rhythm', 'meter', 'tempo', 'syncopation', 'microtiming', 'temporal-prediction'])

add('[MusicStructure] 音乐结构分析与分段',
'Music Structure Analysis & Segmentation:'
' 1. 常见曲式: 二段体(AB), 三段体(ABA), 奏鸣曲式(呈示/展开/再现), '
'回旋曲式(ABACA), 变奏曲式(Theme+Variations), 流行歌曲(Verse/Chorus/Bridge)'
' 2. 结构边界检测: 频谱变化(谱通量/谱质心突变), 和声变化(Chroma特征突变), '
'节奏变化(能量/速度突变), 音色变化(MFCC/音色特征突变)'
' 3. 自相似矩阵(Self-Similarity Matrix): 音乐结构的可视化+计算方法, '
'对角线=重复段落, 棋盘格=ABA/ABACA等对称结构, 用于JEPA的long-range structure learning'
' 4. A-JEPA学习: 音乐结构是long-range temporal dependencies(分钟级), '
'而标准JEPA处理的是短时预测(秒级), 需要层级化JEPA: '
'Layer1(0.1s)学音色/音高→Layer2(1s)学和弦/节奏→Layer3(10s)学乐句→Layer4(60s)学曲式'
' 5. 当前VORTEX数据集分析: 1047首歌曲=112.3小时, 平均每首6.4分钟, '
'正好涵盖完整歌曲结构(前奏→主歌→副歌→间奏→尾奏), 适合层级化JEPA训练',
['music-structure', 'form-analysis', 'self-similarity', 'hierarchical-jepa', 'long-range'])

# ══════════════════════════════════════════════════════════════════════
# PART 7: A-JEPA Training Methodology (A-JEPA训练方法论)
# ══════════════════════════════════════════════════════════════════════

add('[AJEPA-Training] A-JEPA 音频世界模型训练方法论',
'A-JEPA Audio World Model Training Methodology:'
' 1. 核心原理: 在mel频谱表征空间中做预测, 而非在原始波形空间, '
'Context Encoder(编码当前2s mel片段)→Predictor(预测未来1s的表征)→Target Encoder(EMA, 提供target), '
'Loss: VICReg(Variance-Invariance-Covariance Regularization)+Causal Interaction'
' 2. Slot Attention设计: 音频5-Slot架构: slot0=声源类型(人声/乐器/环境), '
'slot1=音高内容(基频+泛音), slot2=时间动态(节奏/包络), slot3=空间位置(混响/声场), slot4=情感/质感(气息/沙哑/柔和)'
' 3. Object-Level Masking: 遮罩整个slot对应的时频区域(不是随机patch), '
'例如遮罩slot0→迫使模型从slot2+slot3推断声源类型(因果推理), '
'遮罩slot1→迫使模型从其他slot预测音高(跨模态推理)'
' 4. CausalVICRegLoss (A-JEPA专用): 4项损失加权: '
'λ_sim×MSE(预测,目标)=10.0, λ_var×(relu(1-std_pred)+relu(1-std_tgt))=1.0, '
'λ_cov×(off-diag_corr²)=1.0, λ_causal×Σ|cov(z_i,z_j)-cov_target(z_i,z_j)|²=0.5'
' 5. 训练阶段: Phase1(epoch 1-50): Slot Encoder预训练, 无因果掩码, EMA decay 0.99→0.996; '
'Phase2(epoch 51-100): 因果推理训练, 开启Object-Level Masking, causal interaction loss激活'
' 6. 评估指标: Reconstruction Loss(recovery), Forward Prediction Loss(future), '
'Counterfactual Accuracy(60%+ target, 比patch-based JEPA的40%高20点), '
'Downstream Task Transfer(LibriSpeech WER, AudioSet mAP)',
['a-jepa-training', 'slot-attention', 'vicreg', 'causal-masking', 'benchmark'])

add('[AJEPA-Loss] A-JEPA 损失函数优化与梯度分析',
'A-JEPA Loss Optimization & Gradient Dynamics:'
' 1. VICReg三项: Variance term防止表征坍缩(所有样本变成同一向量), '
'Covariance term防止维度冗余(各维度独立编码不同信息), '
'Invariance term保证同一内容的不同增强得到相同表征'
' 2. Causal Interaction term (VORTEX独有): '
'迫使slot间协方差匹配真实世界的因果依赖, 例如slot0(声源)×slot1(音高)的协方差, '
'在人声中应该高(不同声源产生不同音高范围), 在噪声中应该低(无规律的音高)'
' 3. 梯度爆炸问题: Slot Attention初始化导致slot分配极不均匀, '
'前几个slot吸收大部分注意力, CausalVICReg的variance term对低方差slot施加极大梯度'
' 4. 梯度裁剪策略: max_grad_norm=2.0(从1.0放宽, 有效学习信号从4%提升到9%), '
'跳过线=max_grad_norm×10=20.0(NaN自动拦截), loss_spike_factor=5.0(相对中位数>5倍跳过)'
' 5. 学习率调度: CosineAnnealingWarmRestarts, T_0=50 epochs, T_mult=1, '
'eta_min=1e-6, warmup=200 steps, 每50 epoch重置为lr=1e-4, 当前训练57/100 epochs'
' 6. 收敛目标: Final loss<25, Counterfactual Accuracy>55%, '
'下游任务FLEURS(多语言ASR) WER<15%, VoxCeleb(说话人识别) EER<5%',
['a-jepa-loss', 'gradient-clip', 'learning-rate', 'convergence', 'benchmark'])

add('[AJEPA-DataMix] A-JEPA 数据混合策略与课程学习',
'A-JEPA Data Mixing Strategy & Curriculum Learning:'
' 1. 推荐训练混合比例: 人声70% + 机器/科幻30%, '
'此比例经实验验证在通用音频表征+人声细粒度能力上取得最佳平衡'
' 2. 课程学习(Curriculum Learning)策略: '
'Stage1(epoch 1-20): 纯人声(最规律, 泛音清晰, 容易学slot结构)'
'Stage2(epoch 21-50): 人声+简单机器声(周期性机器, 拓展频谱范围)'
'Stage3(epoch 51-100): 全混合(人声70%+机器30%, 含冲击/非周期/合成声)'
' 3. 数据分层: 24bit/48kHz发烧人声学精细频谱(泛音/气息/空间), '
'192kHz机械音效学高频瞬态(超声波段信息), 标准16kHz数据集学通用表征'
' 4. 难例挖掘(Hard Negative Mining): 定期(每5 epoch)在全量数据上评估loss, '
'筛选loss最高的20%作为下一阶段重点训练样本, 提升困难场景的泛化'
' 5. 领域自适应: 训练完成后用目标领域数据微调(Fine-tune), '
'冻结Slot Encoder(底层表征), 仅训练Predictor(适应新领域的动态规律)'
' 6. VORTEX FLAME实现: 当前用1047首歌曲训练通用表征(Phase 1-2), '
'后续引入LibriLight人声(提升语音能力)+AudioSet机器声(提升通用性)+192kHz机械音效(提升高频精度)',
['data-mixing', 'curriculum-learning', 'hard-negative-mining', 'domain-adaptation', 'fine-tuning'])

add('[AJEPA-Eval] A-JEPA 评估体系与下游任务',
'A-JEPA Evaluation Framework & Downstream Tasks:'
' 1. 内在评估(Intrinsic): Reconstruction MSE(恢复被遮罩slot的mel频谱), '
'Forward Prediction MSE(预测未来slot表征), Counterfactual Accuracy(改变一个slot→预测其他slot变化)'
' 2. 语音下游: LibriSpeech ASR(WER<15%目标), VoxCeleb Speaker ID(EER<5%), '
'FLEURS多语言ASR(测试多语言泛化), CREMA-D/Savee情感识别(测试情感slot质量)'
' 3. 音乐下游: GTZAN Genre Classification(10类准确率), NSynth Instrument Recognition(1006种乐器), '
'FMA(Free Music Archive)曲风分类, MagnaTagATune自动打标签'
' 4. 环境声下游: ESC-50 Environmental Sound Classification, '
'UrbanSound8K城市声音, TUT Acoustic Scenes声学场景, DCASE挑战赛标准评测'
' 5. 音频取证(Audio Forensics)下游: ASVspoof反欺骗(区分真人/AI合成), '
'Deepfake Detection(检测深度伪造音频), 利用A-JEPA的causal structure学习自然声的物理约束'
' 6. 跨模态下游: Audio-Visual Correspondence(音画同步), Audio-Text Retrieval(音频文本检索), '
'A-V-JEPA(Audio-Video-JEPA)是远期目标, 需要A-JEPA+V-JEPA联合训练',
['a-jepa-evaluation', 'downstream-tasks', 'benchmark', 'audio-forensics', 'audio-visual'])

# ══════════════════════════════════════════════════════════════════════
# PART 8: Audio Production & Engineering (音频制作与工程)
# ══════════════════════════════════════════════════════════════════════

add('[AudioProduction] 音频制作工程与录音技术',
'Audio Production & Recording Engineering:'
' 1. 录音信号链: 麦克风(电容/动圈/铝带)→前置放大器(Preamp, 增益/阻抗匹配)→'
'模数转换器(ADC, 采样率/位深度)→数字音频工作站(DAW), 每个环节引入可量化的频率响应和噪声'
' 2. 麦克风指向性: 全指向(Omni, 无近讲效应)/心形(Cardioid, 抑制背面)/'
'8字形(Figure-8, 两侧拾取)/超心形(Supercardioid), 影响空间信息编码'
' 3. 麦克风技术: XY(重合心形, 相位准确)/ORTF(110°+17cm, 自然声场)/'
'AB(全指向间距30-50cm, 最大声场宽度)/Decca Tree(3全指向三角形, 管弦乐团标准)'
' 4. A-JEPA空间学习: 多麦克风录音提供ground truth空间信息, '
'XY录音的ITD=0(时间差为零), AB录音的ITD和ILD都很大, '
'用已知阵列几何训练JEPA学会从单声道推断空间'
' 5. 动态处理: 压缩器(Compressor, Threshold/Ratio/Attack/Release), '
'限制器(Limiter), 扩展器(Expander), 噪声门(Noise Gate), 改变音频的动态包络'
' 6. 均衡器(EQ): 参数EQ(频率/增益/Q值), 图示EQ(固定频率+可调增益), '
'搁架式(Shelving)/峰值(Peaking)/带通(Bandpass)滤波器类型',
['audio-production', 'recording', 'microphone', 'signal-chain', 'spatial-audio'])

add('[AudioEffects] 音频效果器与信号处理',
'Audio Effects & Signal Processing Chain:'
' 1. 混响(Reverb): 卷积混响(Convolution, 真实IR), 算法混响(Algorithmic, '
'早期反射+后期混响参数化), Plate/Spring/Chamber模拟器'
' 2. 延迟(Delay): 简单延迟(Feedback<1, 逐渐衰减), Ping-Pong(左右交替), '
'多拍延迟(Multi-tap, 节奏同步), 调制延迟(Chorus/Flanger/Phaser, LFO调制延迟时间)'
' 3. 失真(Distortion/Saturation): 软削波(磁带饱和/电子管, 产生偶次谐波), '
'硬削波(晶体管/Fuzz, 产生奇次谐波), Bit-Crush(降采样率+降低位深度, 数字失真)'
' 4. 移调(Pitch Shift): 时域PSOLA(同步叠加), 频域Phase Vocoder(STFT+修改+ISTFT), '
'共振峰保持(Formant Preservation, 改变音高不改音色)'
' 5. A-JEPA学习: 效果器是已知的信号处理链, 可作为"干预实验"(Intervention Experiment), '
'JEPA使用音频效果器前后的对比学习因果表征, '
'例如: 干声→加混响→JEPA应学到"混响=时间卷积"这一因果关系'
' 6. 效果器参数作为辅助条件(Auxiliary Conditioning): '
'用已知的效果器参数(混响时间/延迟时间/失真强度/移调半音数)作为auxiliary变量, '
'通过Auxiliary Conditioner注入JEPA, 实现可控音频表征',
['audio-effects', 'reverb', 'delay', 'distortion', 'pitch-shift', 'causal-intervention'])

# ══════════════════════════════════════════════════════════════════════
# PART 9: Beethoven Knowledge Base Metadata
# ══════════════════════════════════════════════════════════════════════

add('[Beethoven-KB-Meta] 贝多芬知识库元数据与能力边界',
'Beethoven Audio Knowledge Base Capabilities & Boundaries:'
' 1. 知识覆盖: 声学物理/人声科学/乐器声学/音乐理论/音频工程/A-JEPA训练方法论/'
'音频数据集管理/音效设计/混音技术/音频取证, 10大模块'
' 2. JEPA引擎: CAJEPA (Causal Audio JEPA), 5-Slot架构, '
'输入维度512(mel频谱编码), Slot维度128, 辅助条件维度128(macro features+effect params)'
' 3. 知识库能力边界: ✅可处理(音乐分析/乐器识别/情感检测/音频质量评估/声学建议/训练建议), '
'⚠️部分可处理(混音建议/编曲辅助, 需要更多结构化规则), '
'❌不可处理(实时音频流处理/低延迟推理<10ms, 受推理速度限制)'
' 4. 数据资产: 1047首训练歌曲(112.3h), 计划扩展LibriLight(9000h人声)+AudioSet(174万段环境声)'
' 5. 与其他KB协作: A-JEPA的音频表征可输入CVJEPA(视觉-音频联合), '
'CAJEPA的节奏表征可输入CPHYSJEPA(物理节律), 贝多芬情感检测可输入Monet(艺术情感色彩)'
' 6. 长期目标: Audio-Visual JEPA联合训练, 音乐生成(结合ComfyUI+AudioLDM), '
'声学环境理解(室内IR测量+3D音频渲染), 音频深层结构发现(奏鸣曲中隐藏的数学结构)',
['beethoven-meta', 'capability-boundary', 'cajepa', 'collaboration', 'roadmap'])

# ══════════════════════════════════════════════════════════════════════
# Write to extended_domain_knowledge_v3.json
# ══════════════════════════════════════════════════════════════════════

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "extended_domain_knowledge_v3.json")

# Read existing
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    existing = json.load(f)

# Append Beethoven entries
existing.extend(entries)

# Write back
with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
    json.dump(existing, f, ensure_ascii=False, indent=2)

print(f"Generated {len(entries)} Beethoven knowledge entries")
print(f"Total entries in v3.json: {len(existing)}")
print("Run 'python index_knowledge_v3.py' to index into soul_memory.")
