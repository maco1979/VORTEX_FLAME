---
name: "slice-origin-tracker"
description: "Hash-chain traceability system for model inference results. Invoke when tracing which model produced a result, verifying chain integrity, or querying inference origin."
---

# SliceOriginTracker - 切片溯源系统

## 功能
每个推理结果携带唯一origin_id，可溯源到：
- 哪个模型（model_name）
- 哪个深度切片（slice_name + depth_range）
- 冻结状态（deterministic_zeno vs probabilistic）
- 时间戳（timestamp）

## 哈希链不可篡改
block_hash = SHA256(prev_hash + record_data)
每条记录包含block_hash和prev_hash，形成链式结构

## Origin ID格式
Ori-{model前4字符}-{depth//1000}K-{SHA256前16位}
示例：Ori-eins-20K-7fa699a336716740

## 核心API
```python
sot = SliceOriginTracker()

# 注册溯源记录
record = sot.register(model_slice, depth, result_signal, frozen, task_type)

# 通过origin_id溯源
record = sot.trace(origin_id)

# 按模型溯源
records = sot.trace_by_model('einstein-soul-7b')

# 按深度范围溯源
records = sot.trace_by_depth_range(18000, 27000)

# 按冻结状态溯源
records = sot.trace_by_frozen(True)

# 验证哈希链完整性
result = sot.verify_chain()
```

## 溯源记录字段
| 字段 | 说明 |
|------|------|
| origin_id | 唯一溯源标识 |
| model_name | 模型名称 |
| tier | 模型层级(7B/1.5B/HUGE) |
| soul_binding | 绑定灵魂 |
| depth | 推理深度 |
| slice_name | 切片名称 |
| slice_range | 切片深度范围 |
| frozen | 是否冻结 |
| inference_mode | 推理模式 |
| result_signal | 结果信号值 |
| task_type | 任务类型 |
| block_hash | 当前块哈希 |
| prev_hash | 前一块哈希 |

## 源码位置
D:\贾维斯\VORTEX_FLAME\vortex_flame_v62.py - class SliceOriginTracker

## 测试验证
76/76压力测试全部通过，哈希链完整性验证通过
