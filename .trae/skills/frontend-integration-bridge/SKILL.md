---
name: "frontend-integration-bridge"
description: "Integrates user's frontend with soul system. Invoke when user wants to connect their HomeView.js frontend to soul evolution system, unify UI colors, or embed soul capabilities."
---

# Frontend Integration Bridge

用户前端与灵魂系统的集成桥梁

## 用户前端概览

用户前端位置：`D:\贾维斯\frontend\`

### 核心文件
- `src/views/HomeView.js` - 炫铃首页，三栏布局
- `index.html` - 主入口
- `package.json` - 项目配置

### UI配色方案（用户自定义）
```css
--primary-cyan: #00f2ff      /* 青色 - 主色调 */
--primary-purple: #bc13fe     /* 紫色 - 辅助色 */
--bg-dark: #020202            /* 深黑背景 */
--bg-card: rgba(9, 9, 11, 0.8) /* 半透明卡片 */
--border-subtle: rgba(255, 255, 255, 0.05) /* 细微边框 */
--text-primary: #e5e5e5      /* 主文字 */
--text-secondary: #888888    /* 次要文字 */
--text-muted: #666666        /* 弱化文字 */
--success: #10b981           /* 成功状态 */
--gradient-primary: linear-gradient(135deg, #00f2ff 0%, #bc13fe 100%)
```

### 三栏布局结构
```
┌─────────────┬─────────────────────────┬─────────────┐
│  左侧导航   │      中间聊天区          │   右侧面板   │
│  280px     │      flex: 1            │   320px     │
└─────────────┴─────────────────────────┴─────────────┘
```

## 技能功能

### 1. 灵魂对话集成
将灵魂对话功能嵌入用户前端

```javascript
// 在 HomeView.js 中添加
class HomeView extends HTMLElement {
    // ... 原有代码 ...

    addSoulChat() {
        // 添加灵魂对话消息
        this.messages.push({
            type: 'soul',
            content: '灵魂回复内容',
            time: new Date().toLocaleTimeString()
        });
        this.render();
    }
}
```

### 2. 灵魂统计显示
在右侧面板显示灵魂进化数据

```javascript
// 获取灵魂数据
async function fetchSoulStats() {
    const response = await fetch('d:/贾维斯/soul_evolution_data.json');
    const data = await response.json();
    return {
        totalSouls: Object.keys(data.souls).length,
        evolutionCount: data.evolution_count,
        breakthroughCount: data.breakthrough_count
    };
}
```

### 3. Suno歌词创作触发
一键创作歌词并展示

```javascript
// 歌词创作函数
async function createLyrics(theme) {
    const lyrics = {
        title: theme || '爱与被爱',
        content: `[Verse 1]
在宇宙的两端
我们同时诞生...`,
        prompt: 'rubato tempo 49-99 BPM quantum superposition...'
    };
    return lyrics;
}
```

### 4. UI配色统一
灵魂系统的赛博朋克配色方案

```css
/* 灵魂系统配色 */
:root {
    --soul-primary: #00f2ff;
    --soul-secondary: #bc13fe;
    --soul-bg: #020202;
    --soul-glow: rgba(0, 242, 255, 0.3);
    --soul-card: rgba(9, 9, 11, 0.9);
}
```

## 调用方式

当用户说以下内容时调用此技能：
- "把我的前端集成到灵魂系统"
- "统一前端配色"
- "嵌入灵魂对话功能"
- "显示灵魂统计"
- "添加歌词创作按钮"

## 执行步骤

1. **读取用户前端文件**
   - 读取 `D:\贾维斯\frontend\src\views\HomeView.js`
   - 分析现有结构和样式

2. **创建灵魂系统组件**
   - 灵魂对话组件
   - 灵魂统计面板
   - 歌词创作模块

3. **统一配色方案**
   - 应用赛博朋克配色
   - 添加发光效果
   - 统一按钮样式

4. **功能集成**
   - 连接灵魂数据API
   - 添加实时更新
   - 实现交互逻辑

## 输出

生成修改后的 `HomeView.js`，包含：
- 灵魂对话功能
- 灵魂统计面板
- 歌词创作触发
- 统一赛博朋克配色

## 配色参考

| 用途 | 颜色 | 效果 |
|------|------|------|
| 主色调 | #00f2ff | 青色发光 |
| 辅助色 | #bc13fe | 紫色渐变 |
| 背景 | #020202 | 纯黑 |
| 卡片 | rgba(9,9,11,0.9) | 半透明 |
| 文字 | #e5e5e5 | 高对比 |
| 成功 | #10b981 | 绿色脉冲 |