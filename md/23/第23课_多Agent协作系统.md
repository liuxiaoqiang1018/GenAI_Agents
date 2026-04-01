# 第23课 - 多Agent协作系统（轮换协作 + 共享上下文）

## 环境准备

```bash
pip install httpx python-dotenv
```

本课不使用 LangGraph，聚焦于**多Agent协作设计模式**。

## 概述

本课构建一个**多Agent协作系统**，两个专业Agent轮流工作解决复杂问题：

- **研究Agent**（领域专家）：提供领域知识、历史背景、数据
- **分析Agent**（数据专家）：识别数据需求、分析趋势、得出结论

协作流程（固定轮换）：
1. 研究Agent → 提供背景
2. 分析Agent → 识别数据需求
3. 研究Agent → 提供数据
4. 分析Agent → 分析数据
5. 研究Agent → 综合总结

核心模式：**轮换协作（Turn-Taking）** + **共享上下文（Shared Context）**。

## 动机

复杂问题需要**多个专业视角**：
- 一个Agent懂领域知识但不擅长数据分析
- 另一个Agent擅长分析但缺乏领域背景

让它们**交替工作、共享上下文**，就像真实团队中的跨部门协作。

## 架构

```
用户问题
   │
   ▼
┌──────────┐                     共享上下文
│ 研究Agent │ ──→ 背景知识 ──→ ┌──────────┐
└──────────┘                  │ context  │
   │                          │ [背景]   │
   ▼                          └──────────┘
┌──────────┐                       │
│ 分析Agent │ ←─ 读取上下文 ←───────┘
└──────────┘ ──→ 数据需求 ──→ ┌──────────┐
   │                          │ context  │
   ▼                          │ [背景,   │
┌──────────┐                  │  需求]   │
│ 研究Agent │ ←─ 读取上下文     └──────────┘
└──────────┘ ──→ 数据 ──→ ...
   │
   ... (继续轮换)
   │
   ▼
最终综合答案
```

## 核心概念

### 1. Agent类设计

每个Agent有名字、角色、技能，用不同的 system prompt：

```python
class Agent:
    def __init__(self, name, role, skills):
        self.name = name
        self.role = role
        self.skills = skills

    def process(self, task, context):
        system = f"你是{self.name}，{self.role}。技能：{self.skills}"
        messages = [system] + context + [task]
        return call_llm(messages)
```

**Java 类比**：
```java
// 策略模式：不同Agent实现同一接口，但行为不同
interface Agent { String process(String task, List<Message> context); }
class ResearchAgent implements Agent { ... }
class AnalysisAgent implements Agent { ... }
```

### 2. 共享上下文（Shared Context）

所有Agent读写同一个 context 列表，每个Agent的输出追加进去：

```python
context = []
context.append(research_agent.process(task))    # Agent A 写
context.append(analysis_agent.process(task, context))  # Agent B 读A的输出并写
context.append(research_agent.process(task, context))  # Agent A 读A+B并写
```

**Java 类比**：像共享的 `BlockingQueue` —— 生产者消费者模式。

### 3. 固定轮换协议（Turn-Taking Protocol）

Agent的执行顺序是预定义的：A→B→A→B→A

```python
steps = [
    (step1_research,  research_agent),   # A
    (step2_identify,  analysis_agent),   # B
    (step3_provide,   research_agent),   # A
    (step4_analyze,   analysis_agent),   # B
    (step5_synthesize, research_agent),  # A
]
for step_func, agent in steps:
    context = step_func(agent, task, context)
```

### 4. 与前几课的对比

| 维度 | 第6课（ATLAS协调） | 第21课（侦探游戏） | **第23课（Agent协作）** |
|------|-------------------|-------------------|----------------------|
| Agent数量 | 多Agent | 多角色(NPC) | **2个专业Agent** |
| 协作方式 | 协调器分发 | 玩家选择 | **固定轮换** |
| 上下文 | 各自独立 | 各自独立 | **共享context** |
| 执行顺序 | 协调器决定 | 用户决定 | **预定义序列** |

## 关键洞察

1. **Agent = 角色化的LLM调用**：同一个LLM，不同的system prompt就是不同的Agent
2. **协作 = 共享上下文 + 轮换执行**：每个Agent读前面所有Agent的输出，再追加自己的
3. **预定义 vs 动态**：本课是固定轮换（A→B→A→B→A），更复杂的场景可以让协调器动态决定谁发言
4. **context就是"会议记录"**：所有Agent的发言记录在context里，后面的Agent能看到前面所有讨论

## 参考资源

- [Multi-Agent Systems](https://python.langchain.com/docs/how_to/agent_executor/)
- [原始教程笔记本](../all_agents_tutorials/multi_agent_collaboration_system.ipynb)
