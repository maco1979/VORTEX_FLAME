---
name: "project-delivery-system"
description: "项目交付系统 - 从写代码到交付项目的完整方法论。Invoke when AI needs to handle long-term project development, code iteration, or delivery tasks."
---

# 项目交付系统

## 核心理念

从"写代码"转向"交付项目" - 目标不是死磕某一行代码，而是把整个项目的"燃尽图"拉平到终点。

## 一、战略定调

### 1. 设定"里程碑闸门"
- 把大任务拆解成多个可运行的Mini版本
- **规则**：不达到下个里程碑，绝不重构上一个模块
- **示例**：先完成基础列表展示 → 优化搜索速度 → 打磨UI细节

### 2. 拥抱"持续集成" (CI)
- 每写一个功能就跑一次测试
- 不要把所有代码写在一起才测试
- 避免面对一堵错误墙，降低修复成本

## 二、战术执行：解决"卡壳"的三类场景

### 场景 A：代码写不出来 / 逻辑卡住（未完成）

**战术：先写"伪代码"，后填"真代码"**
```python
# 示例
# 步骤1：伪代码框架
def calculate_user_score(user):
    # TODO: 这里需要计算用户积分
    # 1. 获取用户基础信息
    # 2. 计算活跃度
    # 3. 加成会员等级
    pass

# 步骤2：填充真代码
def calculate_user_score(user):
    base_score = user.login_count * 10
    activity_bonus = calculate_activity(user)
    member_multiplier = get_member_multiplier(user.level)
    return base_score * activity_bonus * member_multiplier
```

### 场景 B：单元测试报错 / 功能异常（未通过）

**战术：二分法定位（Binary Search Debug）**
```python
# 步骤1：注释掉一半代码
def test_user_flow():
    # test_login()  # 注释掉
    # test_dashboard()  # 注释掉
    test_payment()  # 保留
    test_logout()  # 保留

# 步骤2：观察报错是否消失
# - 消失：问题在注释部分
# - 还在：问题在保留部分

# 步骤3：继续二分，快速定位元凶
```

### 场景 C：依赖库冲突 / 环境搭建失败（未通过）

**战术：环境隔离（Sandbox）**
```bash
# 创建干净环境
python -m venv fresh_env
source fresh_env/bin/activate  # Linux/Mac
# 或
fresh_env\Scripts\activate  # Windows

# 从零开始安装依赖
pip install -r requirements.txt

# 不要在旧环境里东删西改
```

## 三、防守反击：建立"永不掉线"的系统

### 1. 建立"熔断机制"
- **规则**：一个任务连续卡了2小时还没进展，必须立刻停下
- **动作**：
  - 查文档
  - 换思路
  - 休息一下
  - 不要在死胡同里死磕

### 2. 日志驱动开发 (Log-Driven Development)
```python
import logging

# 在写代码时，就把关键日志埋好
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_payment(order):
    logger.info(f"开始处理订单: {order.id}")
    
    try:
        result = payment_gateway.charge(order.amount)
        logger.info(f"支付成功: {result.transaction_id}")
        return result
    except NetworkError as e:
        logger.error(f"网络超时: {e}")
        raise
    except InsufficientFunds as e:
        logger.error(f"余额不足: {e}")
        raise
```

## 四、终极心法：版本管理

### Git分支策略
```bash
# dev分支：试验田
git checkout -b dev

# 完成功能后合并
git add .
git commit -m "feat: 完成用户登录功能"
git checkout main
git merge dev

# 保证始终有一个可运行的版本
```

### 版本回退安全网
```bash
# 如果今天失败了
git log --oneline  # 查看历史版本
git checkout <commit-hash>  # 回退到安全版本
```

## 五、工程师口诀

> **小步快跑，频繁提交；先跑通，再跑快；日志开路，不做死磕。**

## 六、实战应用流程

### 遇到卡壳时的决策树

```
问题出现
    ├─ 代码写不出来？
    │   └─ 写伪代码 → 拆解小填空 → 逐步填充
    │
    ├─ 测试报错？
    │   └─ 二分法注释 → 快速定位 → 精准修复
    │
    └─ 环境问题？
        └─ 新建干净环境 → 从零安装 → 避免配置债务
```

### 每日工作流

1. **早晨**：拉取最新代码，确认环境正常
2. **上午**：专注核心功能开发（伪代码→真代码）
3. **下午**：持续集成测试，小步提交
4. **傍晚**：代码审查，日志检查，准备次日任务
5. **遇到卡壳**：触发熔断机制，不超过2小时

## 七、质量检查清单

- [ ] 伪代码框架是否清晰？
- [ ] 关键日志是否埋好？
- [ ] 单元测试是否通过？
- [ ] 是否有可回退的安全版本？
- [ ] 是否在2小时内解决不了就触发熔断？

## 八、常见陷阱

1. **完美主义陷阱**：不要追求一步到位，先跑通再优化
2. **死磕陷阱**：超过2小时必须换思路
3. **大爆炸集成陷阱**：不要攒一堆代码才测试
4. **配置债务陷阱**：环境有问题就重建，不要修修补补

## 九、成功案例模板

```markdown
## 项目：[项目名称]
- 里程碑1：[功能] - [完成时间]
- 里程碑2：[功能] - [完成时间]
- 里程碑3：[功能] - [完成时间]

### 遇到的问题
- 问题：[描述]
- 战术：[使用的战术]
- 结果：[解决情况]

### 日志关键信息
- [关键日志片段]
```

## 十、团队协作建议

1. **每日站会**：同步进度，暴露卡点
2. **代码审查**：互相检查，避免盲点
3. **知识共享**：记录解决方案，建立知识库
4. **结对编程**：复杂问题两人一起攻克

---

**记住**：项目交付是一场马拉松，不是百米冲刺。保持节奏，持续交付，最终必达终点！
