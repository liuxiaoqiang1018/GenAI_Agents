# 第4课 - LangGraph 图工作流

## 概述

LangGraph 是 LangChain 团队推出的图工作流框架，用**有向图**来编排多步 LLM 任务。每个节点是一个处理函数，边定义了节点之间的执行顺序。

本课构建一个**文本分析流水线**，依次经过三个节点：
1. **文本分类** — 把输入文本归类为：新闻 / 博客 / 研究 / 其他
2. **实体提取** — 从文本中识别人物、组织、地点
3. **文本摘要** — 生成一句话摘要

## 环境准备

```bash
# 创建并激活虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# 安装依赖
pip install langgraph langchain-openai langchain-core python-dotenv
```

- `langgraph` — LangGraph 图工作流框架
- `langchain-openai` / `langchain-core` — LangChain 核心组件
- `python-dotenv` — 加载 .env 环境变量

`main_debug.py` 不需要 LangGraph，只需要 `openai` 和 `python-dotenv`。

## 动机

很多 AI 应用不是"问一个问题就完事"，而是需要**多步骤、有顺序**地处理。例如：
- 先分类，再根据类别做不同处理
- 先提取关键信息，再总结

用普通代码写也能实现，但 LangGraph 的优势是：
- **可视化**：图结构一目了然
- **状态管理**：自动在节点间传递数据
- **可扩展**：加节点、改路由只需要几行代码
- **条件路由**：可以根据前一步的结果决定下一步走哪

## 核心概念

### 1. State（状态）
一个 TypedDict，定义了整个工作流中流动的数据结构。每个节点都能读取和修改 State。

```python
class State(TypedDict):
    text: str              # 输入文本
    classification: str     # 分类结果
    entities: List[str]     # 实体列表
    summary: str            # 摘要
```

**Java 类比**：就像一个 `Map<String, Object>`，在 Pipeline 的每个 Handler 之间传递。

### 2. Node（节点）
一个普通的 Python 函数，接收 State，返回要更新的字段。

```python
def classification_node(state: State) -> dict:
    # 调用 LLM 做分类
    return {"classification": "新闻"}
```

**Java 类比**：就像 `Function<State, Map<String, Object>>`，或者 Servlet Filter 里的每一个 Filter。

### 3. Edge（边）
定义节点之间的执行顺序。

```python
workflow.add_edge("分类节点", "实体提取节点")  # 分类完了做实体提取
workflow.add_edge("实体提取节点", "摘要节点")   # 实体提取完了做摘要
```

**Java 类比**：就像 `CompletableFuture.thenApply()` 的链式调用。

### 4. StateGraph + compile（图构建与编译）
把节点和边组装成图，编译后就可以执行了。

```python
workflow = StateGraph(State)
workflow.add_node("classify", classification_node)
workflow.set_entry_point("classify")
app = workflow.compile()
result = app.invoke({"text": "..."})
```

**Java 类比**：类似 Spring 的 `BeanDefinition` 阶段（add_node）和 `refresh()` 阶段（compile）。

## 架构图

```
输入文本
   │
   ▼
┌──────────┐
│ 文本分类  │  → 调用 LLM，返回分类标签
└──────────┘
   │
   ▼
┌──────────┐
│ 实体提取  │  → 调用 LLM，返回人物/组织/地点列表
└──────────┘
   │
   ▼
┌──────────┐
│ 文本摘要  │  → 调用 LLM，返回一句话摘要
└──────────┘
   │
   ▼
  结束
```

这是一个**线性管道**（无条件路由）。三个节点依次执行，每个节点都会调用一次 LLM，共调用 3 次。

## 关键代码标注

| 概念 | 在 main.py 中的位置 | 作用 |
|------|---------------------|------|
| State | `class State(TypedDict)` | 定义工作流数据结构 |
| Node | `classification_node` 等3个函数 | 每个节点调用 LLM 处理一个子任务 |
| Edge | `workflow.add_edge(...)` | 定义执行顺序 |
| Graph | `StateGraph(State)` | 图容器，管理节点和边 |
| compile | `workflow.compile()` | 编译成可执行的应用 |
| invoke | `app.invoke(state)` | 执行整个工作流 |

## 与前几课的对比

| 维度 | 第2课（ReAct Agent） | 第3课（数据分析） | 本课（LangGraph） |
|------|---------------------|-------------------|------------------|
| 模式 | LLM 自主决策调不调工具 | LLM 生成查询代码 + 重试 | 人工定义执行顺序 |
| 流程控制 | LLM 驱动（自主循环） | LLM 驱动 + 错误重试 | **开发者定义图结构** |
| LLM 调用次数 | 不确定（取决于决策） | 不确定（取决于重试） | **固定3次** |
| 适用场景 | 需要灵活决策 | 需要试错 | **步骤固定、可预测** |

## 额外思考

1. **条件路由**：LangGraph 支持 `add_conditional_edges()`，可以根据分类结果走不同分支（如"新闻"走一条路，"研究"走另一条路）
2. **并行执行**：实体提取和摘要其实互不依赖，理论上可以并行
3. **持久化**：LangGraph 支持 checkpointing，可以中断后恢复
4. **人机协作**：可以在节点之间插入人工审核步骤

## 参考资源

- [LangGraph 官方文档](https://python.langchain.com/docs/langgraph)
- [原始教程笔记本](../all_agents_tutorials/langgraph-tutorial.ipynb)
