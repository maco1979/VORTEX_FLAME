# VORTEX FLAME 组件库 v2.0

**设计工具**: Open Design MCP Adapter (`open_design_adapter.py`)
**设计系统**: Vercel (黑白极简) + VORTEX FLAME 定制灰度体系
**五灵魂指导**: 塞尚 (结构) / 莫奈 (光影) / 爱因斯坦 (纵深) / 达芬奇 (集成) / 梵高 (节奏)

---

## 一、布局容器

### 1.1 App Shell (index.html)

```html
<div class="app" id="app">
  <nav class="nav">...</nav>
  <aside class="tree-panel">...</aside>
  <main class="main-panel">...</main>
  <aside class="chat-panel">...</aside>
</div>
```

```css
.app {
  display: grid;
  grid-template-columns: var(--nav-w) var(--tree-w) 1fr var(--chat-w);
  height: 100vh;
}
/* variants: no-tree (tree=0), no-chat (chat=0) */
```

**Props**: 无 (纯CSS状态)
**Slots**: nav, tree-panel, main-panel, chat-panel
**Variants**: `no-tree`, `no-chat`, `no-tree.no-chat`

---

### 1.2 Topbar (docs/design/gallery)

```html
<header class="topbar">
  <div class="brand">
    <svg viewBox="0 0 32 32" fill="none">
      <path d="M16 4L28 10v12L16 28 4 22V10z"
            stroke="var(--acc)" stroke-width="1.2" fill="none"/>
    </svg>
    VORTEX FLAME · [页面名]
  </div>
  <div class="topbar-actions">
    <a href="index.html" class="topbar-btn">控制台</a>
    <a href="docs.html" class="topbar-btn">文档</a>
    <a href="gallery.html" class="topbar-btn">画廊</a>
    <a href="design.html" class="topbar-btn">设计</a>
  </div>
</header>
```

```css
.topbar {
  background: var(--c1);
  border-bottom: 1px solid var(--c3);
  padding: 12px 24px;
  display: flex;
  align-items: center;
  position: sticky; top: 0; z-index: 100;
}
```

**Props**:
- `brand`: 左侧品牌区域
- `actions`: 右侧链接按钮组

---

### 1.3 Gallery Header

```html
<div class="header">
  <h1>标题</h1>
  <div class="sub">副标题</div>
  <div class="intro">设计哲学引言</div>
</div>
```

---

## 二、导航组件

### 2.1 NavBar (侧边工具栏)

```html
<nav class="nav">
  <div class="nav-logo" onclick="location.reload()">
    <svg viewBox="0 0 32 32" fill="none">[logo]</svg>
  </div>
  <div class="nav-item active" data-page="files">
    <svg>[icon]</svg>
    <span class="tip">文件</span>
  </div>
  <!-- ... more items ... -->
  <div class="nav-spacer"></div>
  <div class="nav-item">[toggle]</div>
  <div class="nav-item">[search]</div>
</nav>
```

```css
.nav {
  background: var(--c1);
  display: flex; flex-direction: column;
  align-items: center; padding: 8px 0; gap: 4px;
  border-right: 1px solid var(--c3);
}
.nav-item {
  width: 32px; height: 32px; border-radius: var(--r);
  display: flex; align-items: center; justify-content: center;
  cursor: pointer; color: var(--t3);
  transition: all .15s; position: relative;
}
.nav-item:hover { background: var(--c3); color: var(--t1); }
.nav-item.active { background: var(--c3); color: var(--acc); }
.nav-item .tip {
  position: absolute; left: 38px; top: 50%;
  transform: translateY(-50%);
  background: var(--c2); padding: 3px 10px;
  border-radius: 3px; font-size: 10px; color: var(--t1);
  white-space: nowrap; pointer-events: none; opacity: 0;
  border: 1px solid var(--c4); z-index: 100;
  transition: opacity .12s;
}
.nav-item:hover .tip { opacity: 1; }
```

**Props**:
- `data-page`: 页面标识 (files/chatfull/dashboard/skills)
- `active`: 当前激活状态
- `tip`: 悬停提示文字

---

### 2.2 TreePanel (文件浏览器)

```html
<aside class="tree-panel" id="treePanel">
  <div class="tree-header">
    <span>文件浏览器</span>
    <div class="tree-actions">
      <button onclick="refreshTree()">↻</button>
      <button onclick="newFile()">+</button>
    </div>
  </div>
  <div class="tree-path" id="treePath">路径</div>
  <div class="tree-body" id="treeBody">
    <div class="tree-row dir" data-path="...">
      <span class="tree-arrow">▸</span>
      <span class="tree-icon">▵</span>
      <span class="tree-name">文件夹名</span>
    </div>
    <div class="tree-row file" data-path="...">
      <span class="tree-icon">file icon</span>
      <span class="tree-name">文件名</span>
    </div>
  </div>
</aside>
```

**Row variants**: `dir` (可展开), `file` (点击打开)
**Arrow states**: `▸` (折叠), `▾` (展开)

---

### 2.3 Sidebar (文档页)

```html
<nav class="sidebar">
  <div class="group">
    <div class="group-label">分组标签</div>
    <a href="#" class="active" onclick="show('section')">链接</a>
    <a href="#" onclick="show('section')">链接</a>
  </div>
</nav>
```

---

## 三、选项卡 / 标签

### 3.1 TabBar (编辑器标签)

```html
<div class="tab-bar" id="tabBar">
  <div class="tab active">
    文件名
    <span class="tab-dirty">●</span>
    <span class="tab-close">×</span>
  </div>
</div>
```

```css
.tab-bar {
  display: flex; align-items: stretch;
  height: 30px; background: var(--c1);
  border-bottom: 1px solid var(--c3);
  overflow-x: auto; flex-shrink: 0;
}
.tab {
  display: flex; align-items: center; gap: 5px;
  padding: 0 12px; font-size: 11px; cursor: pointer;
  border-right: 1px solid var(--c3); color: var(--t3);
  white-space: nowrap; min-width: 90px;
}
.tab.active {
  background: var(--c2); color: var(--t0);
  border-bottom: 2px solid var(--acc);
}
```

**State**: active / inactive / dirty (●金色圆点)

### 3.2 ChatSouls (知识库多选标签)

```html
<div class="chat-souls" id="chatSouls">
  <div class="chat-soul-tag active">
    <svg>[soul icon]</svg>
    塞尚
  </div>
  <div class="chat-soul-tag">
    <svg>[soul icon]</svg>
    莫奈
  </div>
</div>
```

```css
.chat-souls {
  display: flex; gap: 3px; padding: 6px 10px;
  flex-wrap: wrap; border-bottom: 1px solid var(--c4);
}
.chat-soul-tag {
  padding: 3px 9px; border-radius: 12px; font-size: 10px;
  cursor: pointer; white-space: nowrap;
  border: 1px solid var(--c4); color: var(--t3);
  transition: all .12s;
  display: flex; align-items: center; gap: 3px;
}
.chat-soul-tag:hover { border-color: var(--t2); color: var(--t1); }
.chat-soul-tag.active {
  border-color: var(--acc); color: var(--acc);
  background: rgba(200,169,81,.06);
}
```

**Behavior**: 点击toggle选中/取消, 支持多选 (塞尚几何: 选中=激活, 未选=休眠)

---

## 四、编辑器

### 4.1 EditorArea

```html
<div class="editor-area" id="editorArea">
  <div class="editor-lines" id="editorLines">
    <div>1</div><div>2</div>...
  </div>
  <div class="editor-text">
    <textarea id="editorText" spellcheck="false"></textarea>
  </div>
</div>
```

```css
.editor-lines {
  width: 44px; flex-shrink: 0;
  background: var(--c1); padding: 10px 0;
  border-right: 1px solid var(--c3);
  text-align: right; user-select: none;
}
.editor-text textarea {
  width: 100%; height: 100%;
  background: transparent; border: none;
  outline: none; color: var(--t0);
  font-family: var(--ffm); font-size: 12px;
  line-height: 19px; padding: 10px 12px;
  resize: none; tab-size: 4;
}
```

### 4.2 EditorEmpty (欢迎状态)

```html
<div class="editor-empty">
  <svg>[VF logo]</svg>
  <div class="editor-empty-title">VORTEX FLAME</div>
  <div class="editor-empty-sub">选择文件开始编辑</div>
  <div>
    <kbd>Ctrl+P</kbd> 搜索 <kbd>Ctrl+S</kbd> 保存
  </div>
</div>
```

---

## 五、聊天组件

### 5.1 ChatMessage

```html
<!-- 用户消息 -->
<div class="chat-msg user">
  <div class="bubble">消息内容</div>
</div>

<!-- 灵魂回复 -->
<div class="chat-msg soul">
  <div class="chat-msg-avatar">
    <svg>[soul icon]</svg>
  </div>
  <div class="chat-msg-body">
    <div class="chat-msg-name">塞尚 <span>94%</span></div>
    <div class="chat-msg-text">回复内容</div>
  </div>
</div>
```

**User bubble**: 右对齐, 深灰背景 `var(--c3)`, 圆角 `8px 8px 3px 8px`
**Soul bubble**: 左对齐, 带头像, 中灰背景 `var(--c2)`, 圆角 `3px 4px 4px 4px`

### 5.2 Loading Indicator

```html
<div class="loading-dots">
  <span></span><span></span><span></span>
</div>
```

```css
@keyframes dotBounce {
  0%,80%,100% { transform: scale(.5); opacity: .2; }
  40% { transform: scale(1); opacity: 1; }
}
```

### 5.3 ChatWelcome

```html
<div class="chat-welcome">
  <svg>[VF logo]</svg>
  <h3>VORTEX FLAME</h3>
  <p>14 个专业知识库，按需路由</p>
  <div class="chat-hints">
    <div class="chat-hint" onclick="quickAsk('审查')">审查</div>
    <div class="chat-hint" onclick="quickAsk('解释')">解释</div>
  </div>
</div>
```

---

## 六、状态指示器

### 6.1 StatusBar

```html
<div class="status-bar">
  <span>
    <span class="status-dot ok"></span>
    <span id="statusText">在线</span>
  </span>
  <span id="statusMem">33k 条目</span>
  <span class="spacer"></span>
  <span id="statusCursor">Ln 1, Col 1</span>
  <span id="statusLang">python</span>
</div>
```

**Dot variants**: `ok` (green), `warn` (orange), `err` (red)

---

## 七、数据展示

### 7.1 StatCard (仪表盘)

```html
<div class="dash-stats">
  <div class="stat-card">
    <div class="stat-label">知识库</div>
    <div class="stat-value">14</div>
    <div class="stat-sub">/ 14 活跃</div>
  </div>
</div>
```

### 7.2 BarRow (进度条)

```html
<div class="bar-row">
  <span class="bar-label">
    <svg>[soul icon]</svg> 塞尚
  </span>
  <div class="bar-track">
    <div class="bar-fill" style="width:45%"></div>
  </div>
  <span class="bar-val">11.2k</span>
</div>
```

### 7.3 Table

```html
<table class="table">
  <thead><tr><th>列1</th><th>列2</th></tr></thead>
  <tbody><tr><td>值</td><td>值</td></tr></tbody>
</table>
```

### 7.4 Callout

```html
<div class="callout">
  <div class="label">提示</div>
  内容文字...
</div>
```

**Variants**: `.callout.success` (默认), `.callout.info`

---

## 八、文档专用

### 8.1 Code Block

```html
<pre><code>代码内容</code></pre>
```

### 8.2 API Endpoint

```html
<div class="api-endpoint">
  <span class="api-method">POST</span>
  <span class="api-path">/api/ask</span>
  <div class="api-desc">多专家提问</div>
</div>
```

### 8.3 SoulCard (文档灵魂展示)

```html
<div class="soul-card">
  <div class="icon"><svg>[soul icon]</svg></div>
  <div class="name">塞尚 Cezanne</div>
  <div class="domain">代码 · 逻辑</div>
  <div class="tier">A</div>
</div>
```

### 8.4 Pager (上下页导航)

```html
<div class="pager">
  <a href="#"><small>上一页</small>标题</a>
  <a href="#"><small>下一页</small>标题</a>
</div>
```

---

## 九、画廊专用

### 9.1 Palette (灰度调色盘)

```html
<div class="palette">
  <div class="pcolor" style="background:#f0f0f0" title="#f0f0f0"></div>
</div>
```

### 9.2 Emotion Button

```html
<button class="ebtn" data-e="hope">希望 — 疏朗亮灰</button>
```

**State**: active (金色边框+微金背景)

### 9.3 TheoryCard

```html
<div class="tcard">
  <h4>同时对比 Simultaneous Contrast</h4>
  <p>描述文字...</p>
</div>
```

### 9.4 GalleryCard

```html
<div class="gcard">
  <div class="preview"><svg>[art]</svg></div>
  <div class="caption">
    <h4>作品名</h4>
    <p>描述</p>
    <span class="tag">标签</span>
  </div>
</div>
```

---

## 十、搜索/模态

### 10.1 SearchOverlay

```html
<div class="overlay" onclick="close if click outside">
  <div class="search-dlg">
    <input id="searchInput" placeholder="搜索文件..." autofocus>
    <div class="search-results" id="searchResults">
      <div class="search-item">
        <div class="sf">文件名 <span class="sm">:line</span></div>
        <div class="sl">匹配行内容</div>
      </div>
    </div>
  </div>
</div>
```

---

## 十一、按钮 / 输入

### 11.1 Primary Button

```html
<button class="gen-btn">生成涟漪 · Ripples</button>
```

### 11.2 Text Input

```html
<input class="chat-input" placeholder="输入问题...">
```

### 11.3 Topbar Button

```html
<a href="..." class="topbar-btn">控制台</a>
```

---

## 十二、文件图标映射

```
.py → ¶    .js → {     .ts → t
.tsx → r   .html → ⊞   .css → #
.json → [  .yaml → ·   .yml → ·
.md → ❡    .sql → ⊡    .sh → $
.svg → ◇
```

---

## 十三、动画定义

| 动画 | 时长 | 缓动 | 用途 |
|------|------|------|------|
| msgIn | 0.2s | ease | 消息入场 |
| dotBounce | 1.4s | ease-in-out | 加载指示器 |
| tab/focus | 0.12s | ease | hover/active过渡 |
| 面板切换 | 即时 | - | 保持响应速度 |

---

## 附录: 组件JS接口

### ChatPanel API

```js
renderChatSouls()       // 渲染14灵魂多选标签
toggleSoul(id)          // 切换单个灵魂选中态
sendChat()              // 发送消息 (读取activeSouls数组)
quickAsk(prompt)        // 快捷提问
clearChat()             // 清空对话
```

### Editor API

```js
openFile(path)          // 打开文件到新tab
closeTab(idx)           // 关闭tab
saveFile()              // Ctrl+S 保存
renderTabs()            // 重新渲染tab栏
renderEditor()          // 重新渲染编辑器内容
```

### Tree API

```js
loadTree(path, el)      // 加载目录树
toggleDir(row, path)    // 展开/折叠目录
refreshTree()           // 刷新整个树
newFile()               // 新建文件
```
