# VORTEX FLAME 设计规范 v2.0

**生成工具**: Open Design MCP Adapter (`open_design_adapter.py`)
**评估灵魂**: 莫奈 (美学评分) / 塞尚 (结构逻辑) / 达芬奇 (工程集成) / 爱因斯坦 (数据纵深) / 梵高 (情感节奏)
**支持设计系统**: Vercel (黑白极简) + Linear (极简科技) + Kami (温暖纸张) + Figma (设计工具)
**生成日期**: 2026-05-31
**Daemon状态**: 离线概念模式 (启动命令: `pnpm tools-dev`)

---

## 一、设计理念

### 1.1 核心哲学: "剥离色彩，直抵形式"

五灵魂设计共识:
- **塞尚**: 所有视觉元素必须可解构为基本几何体——矩形、六边形、圆。没有一个像素是装饰性的。
- **莫奈**: 灰度梯度是光本身。高光→暗部的过渡节奏决定呼吸感。
- **爱因斯坦**: 视觉纵深 = 数据纵深。外层大圈是宏观规律，内层小圈是微观核心。
- **达芬奇**: 工具集成 = 模块化面板。每个面板独立可折叠，不使用时退让，需要时展开。
- **梵高**: 密度即情感。密集的线条布局 = 能量，稀疏的留白 = 宁静。

### 1.2 设计系统选择矩阵

| 灵魂 | 推荐系统 | 风格 | 核心色 |
|------|---------|------|--------|
| 塞尚 | Vercel | 黑白极简 | #000000 #FFFFFF |
| 爱因斯坦 | Stripe | 金融专业 | #635BFF #0A2540 #F6F9FC |
| 达芬奇 | Linear | 极简科技 | #5E6AD2 #1C1F26 #F4F4F6 |
| 莫奈 | Kami | 温暖纸张 | #1A365D #F5E6D3 #8B7355 |
| 梵高 | Figma | 设计工具 | #A259FF #0D0D0D #FFFFFF |

**选定主导系统**: **Vercel + VORTEX FLAME 定制** — 融合黑白极简的克制 + 自研灰度梯度体系

---

## 二、色彩系统

### 2.1 基础灰度体系 (来自莫奈的光影理论)

```
--c0: #080808   (最深底色 — 暗部)
--c1: #121212   (面板背景 — 深灰)
--c2: #1a1a1a   (卡片背景 — 中深)
--c3: #242424   (边框/分隔 — 中灰)
--c4: #2e2e2e   (hover/高亮边框 — 浅中)
```

### 2.2 文字层级 (爱因斯坦的数据纵深理论)

```
--t0: #f0f0f0   (主标题/强调 — 最高对比度)
--t1: #c0c0c0   (正文 — 标准阅读)
--t2: #808080   (辅助文字 — 降低优先级)
--t3: #505050   (占位符/失效 — 最低可见度)
```

### 2.3 功能色 (单点缀策略)

```
--acc: #c8a951   (唯一点缀色 — 暖金，仅用于: 激活态/链接hover/选中边框)
--ok:  #6b8      (成功状态 — 仅status indicator)
--warn:#da8      (警告状态)
--err: #c66      (错误状态)
```

### 2.4 五灵魂灵感色 (仅用于灵魂标签icon微差异)

```
cezanne:  #8a9a8a   (结构绿灰)
einstein: #9a8a7a   (数据暖灰)
davinci:  #7a8a9a   (工程蓝灰)
monet:    #8a8a9a   (美学中性灰)
vangogh:  #9a8a8a   (情感暖灰)
```

---

## 三、字体排版 (Typography)

### 3.1 字体栈

| 用途 | 字体 | 权重 |
|------|------|------|
| UI正文 | Inter | 300/400/500 |
| 标题/中文 | Noto Serif SC | 300/400 |
| 代码/meta | JetBrains Mono | 400/500 |

### 3.2 字号层级

| 层级 | 大小 | 用途 | 行高 |
|------|------|------|------|
| H1 | 2rem (32px) | 页面标题 | 1.3 |
| H2 | 1.25rem (20px) | Section标题 | 1.4 |
| H3 | 0.875rem (14px) | 卡片标题 | 1.5 |
| Body | 0.8125rem (13px) | 正文 | 1.7 |
| Caption | 0.6875rem (11px) | 辅助说明 | 1.6 |
| Code | 0.75rem (12px) | 代码块 | 19px固定 |
| Meta | 0.625rem (10px) | 标签/徽章 | 1.4 |

### 3.3 字间距

- 页面标题: `letter-spacing: -0.01em`
- Section标签: `letter-spacing: 0.06em` (大写)
- 代码: 无额外间距 (等宽字体自带)

---

## 四、间距系统

### 4.1 基础单位: 4px grid

| Token | 值 | 用途 |
|-------|-----|------|
| xs | 4px | 紧密元素间距 |
| sm | 8px | 卡片内间距/标签间距 |
| md | 12px | 面板内边距 |
| lg | 16px | 面板间距 |
| xl | 24px | Section间距 |
| 2xl | 40px | 页面区块间距 |
| 3xl | 56px | 大区块分隔 |

### 4.2 圆角

- 按钮/输入框: `4px` (--r)
- 卡片/面板: `8px` (--rl)
- 标签/徽章: `12px` (pill shape)
- 圆形元素: `50%`

### 4.3 边框

- 分隔线: `1px solid var(--c3)`
- 输入框: `1px solid var(--c4)`
- Hover激活: `1px solid var(--acc)`
- 所有边框不透明度: 100% (非半透明 — 塞尚几何原则)

---

## 五、布局系统

### 5.1 主控台 (index.html) — 四面板网格

```
┌────┬──────────┬──────────────┬──────────┐
│NAV │  TREE    │    EDITOR    │   CHAT   │
│44px│  220px   │    1fr       │  340px   │
│    │          │              │          │
│    │          │  ┌──────────┐│          │
│    │          │  │ tab bar  ││ souls    │
│    │          │  ├──────────┤│ msgs     │
│    │          │  │ editor   ││          │
│    │          │  │          ││ input    │
│    │          │  ├──────────┤│          │
│    │          │  │ status   ││          │
│    │          │  └──────────┘│          │
└────┴──────────┴──────────────┴──────────┘
```

- **NAV**: 固定44px，左侧垂直排列icon+tooltip
- **TREE**: 固定220px，上下文相关的文件浏览器
- **EDITOR**: flex:1 自适应，tabbed多文件编辑
- **CHAT**: 固定340px，14灵魂多选+消息+输入

### 5.2 面板显示规则

| 模式 | NAV | TREE | EDITOR | CHAT |
|------|-----|------|--------|------|
| files | ✓ | ✓ | ✓ | ✓ |
| files (no chat) | ✓ | ✓ | 1fr | ✗ |
| chatfull | ✓ | ✗ | ✗ | ✗ |
| dashboard | ✓ | ✗ | 全宽 | ✓ |
| skills | ✓ | ✗ | 全宽 | ✓ |

### 5.3 文档页 (docs.html) — 两栏

```
┌──────┬──────────────────────────────────┐
│SIDEBAR│        ARTICLE                  │
│240px  │        max-width:780px          │
│       │                                 │
│ 入门   │  crumbs > h1 > lede             │
│ 灵魂   │  h2 + content + pre/code        │
│ 工作流 │  h2 + table / callout           │
│ API   │  h2 + api-endpoint cards        │
└──────┴──────────────────────────────────┘
```

### 5.4 设计系统页 (design.html) — 单列流

```
max-width:1100px, 居中
├── Logo展示 (flex row)
├── 几何解构 (3 card grid)
├── 调色盘 (auto-fill grid)
├── 技能市场 (table + grid)
```

### 5.5 画廊页 (gallery.html) — 叙事流

```
max-width:900px, 居中
├── Header: 标题 + 设计哲学引言
├── 笔触区: 灰度条 + 情感按钮 + 画布
├── 光影区: 调色盘 + 理论卡片(3列) + 涟漪生成
├── 图形区: 三幅SVG作品卡片
```

---

## 六、交互规范

### 6.1 过渡动画

- 所有hover/active切换: `transition: all 0.12s ease`
- 消息入场: `@keyframes msgIn { from{opacity:0;translateY(4px)} }` 200ms
- 面板切换: 即时显示/隐藏 (无动画 — 保持即时响应)
- Loading dots: 1.4s bounce循环

### 6.2 Hover模式

- 卡片/行hover: `border-color` 从 `var(--c3)` → `var(--c4)`
- 按钮hover: `color` 从 `var(--t3)` → `var(--acc)`, `border-color` → `var(--acc)`
- 灵魂标签激活: `border-color → var(--acc)`, `color → var(--acc)`, 背景微金
- **不做**: box-shadow, transform scale, backdrop-filter (保持平面几何)

### 6.3 知识库多选 (新功能)

- **单点击**: 切换选中/取消 (toggle)
- **选中态**: 金色边框 + 金色文字 + 微金背景
- **多选态**: 多个标签可同时激活
- **发送逻辑**: 选中 = 指定这些KB同时响应; 未选 = 自动路由

### 6.4 键盘快捷键

| 快捷键 | 功能 |
|--------|------|
| Ctrl+P | 打开文件搜索 |
| Ctrl+S | 保存当前文件 |
| Ctrl+B | 切换右侧聊天面板 |
| Enter | 发送消息 (聊天输入框) |
| Shift+Enter | 换行 (聊天输入框) |
| Tab | 缩进 (编辑器) |

---

## 七、组件规范

### 7.1 基础组件树

```
App
├── NavBar (left toolbar)
│   ├── Logo
│   ├── NavItem × 4 (files/chat/dashboard/skills)
│   └── NavItem × 2 (toggle/search)
├── TreePanel (file browser)
│   ├── TreeHeader
│   ├── TreePath (breadcrumb)
│   └── TreeBody
│       └── TreeRow × N (file/directory)
├── MainPanel (editor area)
│   ├── TabBar
│   │   └── Tab × N
│   ├── EditorWrap
│   │   ├── EditorEmpty (welcome state)
│   │   └── EditorArea (active editing)
│   │       ├── EditorLines (line numbers)
│   │       └── EditorText (textarea)
│   └── StatusBar
│       ├── StatusDot + StatusText
│       ├── StatusMem
│       ├── StatusCursor
│       └── StatusLang
├── ChatPanel (right sidebar)
│   ├── ChatHeader
│   ├── ChatSouls (multi-select KB tags)
│   ├── ChatMessages
│   │   ├── ChatWelcome (empty state)
│   │   └── ChatMsg × N (user/soul bubbles)
│   └── ChatInputWrap
│       ├── ChatInput (textarea)
│       └── ChatSend (button)
├── Dashboard (full-page view)
│   ├── StatCard × 4
│   └── BarRow × 14 (soul memory bars)
├── SkillsPage (full-page view)
│   └── SkillCard × N
├── HermesPage (full-page view)
│   ├── HermesCard (models)
│   └── BarRow × 14 (model mapping)
└── SearchOverlay (modal)
    ├── SearchInput
    └── SearchResults
        └── SearchItem × N
```

### 7.2 文档页组件

```
DocsPage
├── Topbar
│   ├── Brand (logo + "VORTEX FLAME · 文档")
│   ├── SearchInput
│   └── TopbarActions (page links)
├── Sidebar
│   ├── Group × 4
│   │   ├── GroupLabel
│   │   └── SidebarLink × N
└── Article
    ├── Crumbs (breadcrumb)
    ├── H1 + Lede
    ├── H2 + content (per section)
    │   ├── Callout (info/success/warning)
    │   ├── Pre > Code
    │   ├── Table
    │   ├── SoulGrid > SoulCard × N
    │   └── ApiEndpoint
    └── Pager (prev/next nav)
```

### 7.3 画廊页组件

```
GalleryPage
├── Topbar
├── Header (H1 + subtitle + intro)
├── BrushSection
│   ├── Palette (grayscale swatches)
│   ├── EmotionSelector (buttons × 5)
│   └── ArtCanvas (dynamic brushstroke render)
├── LightSection
│   ├── Palette (grayscale swatches)
│   ├── TheoryGrid > TheoryCard × 3
│   ├── GenerateButton
│   └── ArtCanvas (dynamic ripple render)
└── FormsSection
    └── GalleryGrid > GalleryCard × 3
        ├── Preview (SVG)
        ├── Caption (h4 + p + tag)
```

---

## 八、响应式策略

### 8.1 断点

| 断点 | 宽度 | 策略 |
|------|------|------|
| Desktop | >1024px | 完整四面板/两栏布局 |
| Tablet | 768-1024px | 折叠TREE面板, CHAT保持 |
| Mobile | <768px | 单面板堆叠, 侧边栏隐藏 |

### 8.2 各页面响应式规则

**index.html**:
- `<768px`: TREE隐藏, CHAT变为底部面板
- `<500px`: 全屏单面板, 通过NAV切换视图

**docs.html**:
- `<768px`: 侧边栏隐藏, 文章区占满宽度

**design.html / gallery.html**:
- card grid: `repeat(auto-fill, minmax(260px, 1fr))` 自动适配

---

## 九、资产清单

### 9.1 SVG Icons (14个抽象几何)

| Icon ID | 几何形态 | 物理隐喻 | 对应灵魂 |
|---------|---------|---------|---------|
| hexagon | 嵌套六边形 | 结构、分层 | 塞尚 |
| atom | 三椭圆+核心 | 原子轨道 | 爱因斯坦 |
| orbit | 多层轨道+星点 | 天体运行 | 伽利略 |
| tree | 树状分形 | 生命进化 | 达尔文 |
| diamond | 菱形套叠 | 决策矩阵 | 策略 |
| balance | 天平 | 法律均衡 | 孟德斯鸠 |
| spiral | 螺旋曲线 | 黄金比例 | 达芬奇 |
| mountain | 山脉折线 | 地质层次 | 洪堡 |
| seed | 水滴+弧线 | 种子生长 | 袁隆平 |
| enso | 不完整圆 | 禅、圆满 | 硅酌 |
| scroll | 书卷+横线 | 历史书写 | 希罗多德 |
| ripple | 三圈扩散 | 水波光影 | 莫奈 |
| swirl | 旋涡曲线 | 情感旋流 | 梵高 |
| wave | 波形+基线 | 声波振动 | 贝多芬 |

### 9.2 页面文件清单

| 文件 | 大小 | 状态 |
|------|------|------|
| index.html | ~12KB | 已重设计 |
| docs.html | ~6KB | 已重设计 |
| design.html | ~7KB | 已重设计 |
| gallery.html | ~9KB | 已重设计 (含文字优化) |

### 9.3 设计Token文件

- `vf_web/design_tokens.css` — CSS变量全集
- `DESIGN_SPEC.md` — 本文档
- `COMPONENT_LIBRARY.md` — 组件库文档

---

## 十、前端实施检查清单

- [x] 全部CSS变量统一为 `--c0-c4 / --t0-t3 / --acc`
- [x] 移除所有 `linearGradient` / `radialGradient` / `backdrop-filter`
- [x] 移除 `noise-overlay` 伪元素
- [x] 移除所有 `box-shadow` / `transform: translateY()` hover效果
- [x] Hover仅通过 `border-color` 变化实现
- [x] 所有过渡 `0.12s ease`
- [x] 字体栈: Inter + Noto Serif SC + JetBrains Mono
- [x] 仅一个点缀色 `#c8a951` (暖金)
- [x] 14个灵魂icon统一灰度stroke (opacity分层)
- [x] 知识库多选功能 (toggleSoul/activeSouls数组)
- [ ] favicon已添加到index + gallery (docs/design缺失)
- [ ] Open Design daemon启动后重跑在线评分

---

## 附录A: Open Design 适配器诊断结果

```
Daemon: 离线 (http://127.0.0.1:3456)
可用技能: 12个 (generate_ppt/aesthetic_score/design_system/layout_suggest/
          color_palette/typography/export_html/export_pdf/export_pptx/
          visual_reasoning/prompt_choreography/guizang_ppt)
设计系统: 8个 (linear/vercel/stripe/apple/cursor/figma/kami/guizang_ppt)
美学评分: 离线启发式模式 (daemon在线可得精确分)
```

## 附录B: 后续优化路线

1. **启动 Open Design daemon** → 获得精确5维美学评分 + OKLch调色板
2. **生成鬼藏PPT** → VORTEX FLAME 产品介绍演示
3. **导出PDF设计规范** → 可打印版本
4. **添加动画微交互** → UI Movement 参考, 灰度体系下动效是关键
5. **接入视觉推理API** → 自动检测设计一致性问题
