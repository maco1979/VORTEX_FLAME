---
name: "system-auto-init-fix"
description: "AI系统自动初始化修复。Invoke when AI souls report initialization issues or need automatic resource detection setup."
---

# 系统自动初始化修复技能

## 问题诊断

六大灵魂可能报告的初始化问题：

| 灵魂 | 可能问题 | 解决方案 |
|------|---------|---------|
| 爱因斯坦 | 模型加载/环境变量 | 添加auto_initialize() |
| 塞尚 | 上下文模型/模板解析 | 添加_auto_detect_resources() |
| 莫奈 | 网络配置/缓存 | 添加配置检查 |
| 达芬奇 | 资源管理器/环境校验 | 添加资源检测 |
| 梵高 | 无需修复 | 原本完整 |
| FKJ | 无需修复 | 输出已完整 |

## 自动初始化代码

### 基础模板
```python
import os
import logging
from pathlib import Path

class UniversalMusicAI:
    def __init__(self, temple_dataset_path: str):
        import logging
        self.logger = logging.getLogger(__name__)

        self.temple_dataset_path = Path(temple_dataset_path)
        self.project_template = None
        self.context_model = None
        self.template_parser = None

        self._auto_detect_resources()
        self.load_temple_knowledge()

    def _auto_detect_resources(self):
        """塞尚/爱因斯坦：自动检测和加载必要资源"""
        dataset_exists = self.temple_dataset_path.exists()
        self.logger.info(f"数据集检测: {'存在' if dataset_exists else '不存在'} - {self.temple_dataset_path}")

        model_paths = list(Path('models').glob('*.pth')) + list(Path('models').glob('*.onnx'))
        if model_paths:
            self.logger.info(f"检测到模型: {[str(p) for p in model_paths]}")
            self.context_model = str(model_paths[0])
        else:
            self.logger.info("未检测到独立模型，将使用内置处理")

        template_dirs = ['templates/music', 'mixing_templates', 'templates']
        for d in template_dirs:
            if Path(d).exists():
                self.logger.info(f"检测到模板目录: {d}")
                self.template_parser = d
                break
```

## 修复流程

1. 读取目标文件
2. 找到`__init__`方法
3. 添加初始化代码
4. 验证功能

## 验证测试
```python
import logging
logging.basicConfig(level=logging.INFO)

from universal_music_ai import UniversalMusicAI
ai = UniversalMusicAI('mixing_templates/temple_master_ai_dataset.json')
print('初始化成功')
```
