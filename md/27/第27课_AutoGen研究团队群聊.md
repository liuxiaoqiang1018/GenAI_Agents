# 第27课 - AutoGen研究团队群聊（群聊 + 发言权转移 + 多角色协作）

## 环境准备

```bash
pip install httpx python-dotenv
```

## 概述

本课构建一个**多Agent群聊系统**，5个Agent共享一个对话线程，由Manager根据转移规则动态决定谁发言：

- **管理员**（Admin）：人类用户，审批计划
- **规划师**（Planner）：制定研究计划
- **开发者**（Developer）：编写代码
- **执行者**（Executor）：执行代码
- **质检员**（QA）：审查结果质量

核心模式：**群聊（Group Chat）** —— 所有Agent共享同一对话线程，Manager根据**发言权转移图**动态选择下一个发言者。

## 动机

第23课的协作是**固定轮换**（A→B→A→B→A），但真实团队不是这样工作的：
- 规划师说完，可能是开发者接话，也可能是管理员插话
- 开发者写完代码，应该交给执行者，而不是规划师
- 质检发现问题，可以打回给任何人

需要**动态的发言权控制**。

## 架构

```
发言权转移图（有向图）：

  管理员 ──→ 规划师
  管理员 ──→ 质检员
  规划师 ──→ 管理员 / 开发者 / 质检员
  开发者 ──→ 执行者 / 质检员 / 管理员
  执行者 ──→ 开发者
  质检员 ──→ 规划师 / 开发者 / 执行者 / 管理员

群聊线程（所有人共享）：
  [管理员] 请研究AI在法律领域的应用
  [规划师] 计划：1.搜索 2.分析 3.总结
  [管理员] 批准
  [开发者] 编写搜索代码...
  [执行者] 执行结果：找到5个仓库
  [质检员] 结果完整，通过
```

## 核心概念

### 1. 群聊 vs 轮换协作

| 维度 | 第23课（轮换协作） | **第27课（群聊）** |
|------|-------------------|-------------------|
| 发言顺序 | 固定（A→B→A→B→A） | **动态（Manager决定）** |
| 上下文 | 共享 context 列表 | **共享对话线程** |
| Agent数量 | 2个 | **5个** |
| 控制方式 | 硬编码顺序 | **转移规则图** |

### 2. 发言权转移图（Speaker Transition Graph）

用有向图定义"谁之后可以是谁"：

```python
transitions = {
    "管理员": ["规划师", "质检员"],
    "规划师": ["管理员", "开发者", "质检员"],
    "开发者": ["执行者", "质检员", "管理员"],
    "执行者": ["开发者"],
    "质检员": ["规划师", "开发者", "执行者", "管理员"],
}
```

**Java 类比**：
```java
// 像状态机的转移表
Map<State, Set<State>> transitions = Map.of(
    ADMIN, Set.of(PLANNER, QA),
    PLANNER, Set.of(ADMIN, DEVELOPER, QA),
    DEVELOPER, Set.of(EXECUTOR, QA, ADMIN),
    ...
);
```

### 3. Manager 选择下一个发言者

Manager（由LLM驱动）根据当前对话内容和转移规则，决定谁来发言：

```python
def select_next_speaker(current, context, transitions):
    candidates = transitions[current]  # 当前发言者之后可选的候选人
    # 让LLM根据对话内容选择最合适的候选人
    next_speaker = llm_select(candidates, context)
    return next_speaker
```

## 关键洞察

1. **群聊 = 共享对话线程 + 动态发言权**：和微信群聊类似，所有人看同一个聊天记录
2. **转移图 = 业务规则**：执行者只能把话语权给开发者（执行完要开发者看结果），这是工作流规则
3. **质检员是枢纽**：QA可以转给任何人，因为发现问题后需要灵活打回
4. **Manager = 主持人**：根据对话进展判断该轮到谁说话

## 参考资源

- [AutoGen Group Chat](https://microsoft.github.io/autogen/docs/tutorial/conversation-patterns/)
- [原始教程笔记本](../all_agents_tutorials/research_team_autogen.ipynb)
