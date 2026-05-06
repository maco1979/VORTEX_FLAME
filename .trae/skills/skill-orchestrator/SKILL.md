---
name: "skill-orchestrator"
description: "技能编排器 - 总自动化中枢。Invoke when user wants to understand the complete skill automation system, or needs to run cross-drive skill scanning with automatic execution."
---

# 技能编排器 (Skill Orchestrator)

## 定位

**技能编排器**是整个技能系统的"总指挥"，负责：
1. 协调所有技能的自动扫描和注册
2. 管理技能间的调用依赖
3. 实现完全自动化的工作流

## 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                     SKILL ORCHESTRATOR                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Auto        │  │ Cross-Drive │  │ Workflow            │ │
│  │ Discovery   │→ │ Scanner     │→ │ Engine              │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│         ↓                ↓                   ↓            │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              SKILL REGISTRY (skills_registry.json)   │  │
│  └─────────────────────────────────────────────────────┘  │
│                            ↓                               │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              DISPATCHER (Skill Dispatcher)           │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 核心功能

### 1. 全自动技能发现

```python
class SkillOrchestrator:
    def __init__(self):
        self.scanner = CrossDriveSkillScanner()
        self.registry = SkillRegistry()
        self.dispatcher = SkillDispatcher()
        self.workflow_engine = WorkflowEngine()

    def full_auto_init(self):
        """完全自动化初始化"""

        # Step 1: 扫描所有驱动器
        print("🔍 正在扫描所有硬盘...")
        all_skills = self.scanner.scan_all_drives()

        # Step 2: 更新注册表
        print("📝 更新技能注册表...")
        self.registry.update(all_skills)

        # Step 3: 初始化调度器
        print("⚙️ 初始化调度器...")
        self.dispatcher.load_all_skills()

        # Step 4: 验证所有技能
        print("✅ 验证技能完整性...")
        validation = self.registry.validate_all()

        print(f"\n✨ 初始化完成!")
        print(f"   发现技能: {len(all_skills)} 个")
        print(f"   可用技能: {validation['valid_count']} 个")

        return validation
```

### 2. 跨驱动器扫描

```python
class CrossDriveSkillScanner:
    """跨驱动器技能扫描器"""

    def __init__(self):
        self.target_pattern = ".trae/skills/**/SKILL.md"
        self.found_skills = []

    def scan_all_drives(self) -> List[Dict]:
        """扫描所有可用驱动器"""

        skills = []
        drives = self.get_available_drives()

        print(f"📂 检测到驱动器: {', '.join(drives)}")

        for drive in drives:
            print(f"   扫描 {drive}...")
            drive_skills = self.scan_drive(drive)
            skills.extend(drive_skills)
            print(f"      发现 {len(drive_skills)} 个技能")

        # 去重
        unique_skills = self.deduplicate_skills(skills)
        return unique_skills

    def get_available_drives(self) -> List[str]:
        """获取所有可用驱动器"""

        drives = []

        # Windows 驱动器检测
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            drive = f"{letter}:\\"
            if os.path.exists(drive):
                drives.append(drive)

        return drives

    def scan_drive(self, drive: str) -> List[Dict]:
        """扫描单个驱动器"""

        found = []

        try:
            for path in Path(drive).glob(self.target_pattern):
                skill_info = {
                    "name": path.parent.name,
                    "path": str(path),
                    "drive": drive,
                    "size": path.stat().st_size,
                    "modified": datetime.fromtimestamp(path.stat().st_mtime),
                    "description": self.extract_description(path)
                }
                found.append(skill_info)
        except PermissionError:
            print(f"      ⚠️ 无权限访问 {drive}")

        return found

    def extract_description(self, skill_path: Path) -> str:
        """从 SKILL.md 提取描述"""

        try:
            content = skill_path.read_text(encoding="utf-8")
            # 解析 YAML frontmatter
            match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
            if match:
                yaml_content = match.group(1)
                desc_match = re.search(r'description:\s*["\'](.*?)["\']', yaml_content)
                if desc_match:
                    return desc_match.group(1)
        except Exception:
            pass

        return ""
```

### 3. 自动化工作流引擎

```python
class WorkflowEngine:
    """工作流引擎 - 连接多个技能形成自动化流水线"""

    # 预定义工作流
    WORKFLOWS = {
        "music_creation": [
            "ai-quantum-music-mind",
            "fkj-music-creation",
            "music-creation-executor",
            "multi-soul-evaluation"
        ],
        "skill_management": [
            "skill-discovery-scanner",
            "skill-dispatcher",
            "auto-skill-factory"
        ],
        "full_auto": [
            "skill-discovery-scanner",
            "system-auto-init-fix",
            "auto-skill-factory",
            "vortex-flame-soul-system"
        ]
    }

    def run_workflow(self, workflow_name: str, context: Dict = None) -> Dict:
        """运行工作流"""

        if workflow_name not in self.WORKFLOWS:
            return {"status": "error", "message": f"未知工作流: {workflow_name}"}

        workflow_steps = self.WORKFLOWS[workflow_name]
        results = []

        print(f"\n🚀 开始执行工作流: {workflow_name}")
        print(f"   步骤数: {len(workflow_steps)}\n")

        for i, skill_name in enumerate(workflow_steps, 1):
            print(f"[{i}/{len(workflow_steps)}] 执行: {skill_name}")

            result = self.execute_skill(skill_name, context)
            results.append({
                "skill": skill_name,
                "result": result
            })

            if result.get("status") == "error":
                print(f"   ❌ 失败: {result.get('message')}")
                break
            else:
                print(f"   ✅ 完成")

        return {
            "workflow": workflow_name,
            "steps_completed": len(results),
            "results": results
        }

    def execute_skill(self, skill_name: str, context: Dict = None) -> Dict:
        """执行单个技能"""
        # 调用 SkillDispatcher
        dispatcher = SkillDispatcher()
        return dispatcher.dispatch_by_name(skill_name, context)
```

## 总自动化命令

| 命令 | 功能 |
|------|------|
| `全自动化` | 扫描+初始化+执行主工作流 |
| `扫描所有技能` | 跨驱动器扫描技能文件 |
| `执行音乐创作流程` | 运行 music_creation 工作流 |
| `执行技能管理工作流` | 运行 skill_management 工作流 |
| `一键启动` | 完整初始化 + 所有服务启动 |

## 自动化流程图

```
用户: "全自动化"
           ↓
┌──────────────────────────────────────┐
│  1. skill-discovery-scanner          │
│     扫描所有硬盘 .trae/skills/        │
└──────────────────────────────────────┘
           ↓
┌──────────────────────────────────────┐
│  2. skill-dispatcher                  │
│     加载所有技能到注册表               │
└──────────────────────────────────────┘
           ↓
┌──────────────────────────────────────┐
│  3. auto-skill-factory               │
│     验证并修复技能                     │
└──────────────────────────────────────┘
           ↓
┌──────────────────────────────────────┐
│  4. vortex-flame-soul-system         │
│     执行主工作流                       │
└──────────────────────────────────────┘
           ↓
           ✅ 完成
```

## 技能注册表格式

```json
{
  "version": "1.0",
  "last_scan": "2026-04-06T10:30:00",
  "total_skills": 30,
  "skills": {
    "ai-quantum-music-mind": {
      "path": "D:\\贾维斯\\.trae\\skills\\ai-quantum-music-mind",
      "drive": "D:",
      "status": "active",
      "dependencies": [],
      "version": "1.0"
    },
    "music-creation-executor": {
      "path": "D:\\贾维斯\\.trae\\skills\\music-creation-executor",
      "drive": "D:",
      "status": "active",
      "dependencies": ["ai-quantum-music-mind"],
      "version": "1.0"
    }
  }
}
```

## 使用示例

```python
from skill_orchestrator import SkillOrchestrator

# 创建编排器
orchestrator = SkillOrchestrator()

# 全自动化初始化
orchestrator.full_auto_init()

# 运行音乐创作工作流
result = orchestrator.workflow_engine.run_workflow("music_creation", {
    "input": "奋斗主题歌曲"
})

# 或者直接说"全自动化"
orchestrator.full_auto()
```

## 与灵魂系统的集成

VORTEX FLAME 灵魂系统通过 `vortex-flame-soul-system` 技能与编排器交互：

```
灵魂对话请求
       ↓
vortex-flame-soul-system (execute.py)
       ↓
skill-orchestrator.run_workflow()
       ↓
调度相关技能执行
       ↓
返回结果给灵魂
```

---

**技能编排器 = 扫描器 + 注册表 + 调度器 + 工作流引擎**

**一句话理解：用户输入 → 编排器协调 → 多技能协作 → 自动完成任务**
