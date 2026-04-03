# Java 端 AI Agent 框架对照

根据本项目使用的 Python 框架，以下是 Java 生态的对应方案。

## 框架对照表

| Python 框架 | Java 对应框架 | 说明 |
|---|---|---|
| **LangChain** | **LangChain4j** | 最直接的对应，API 设计非常相似，支持 RAG、Tool、Memory、Chain |
| **LangGraph** | **LangChain4j + 自建状态机** | LangGraph 暂无官方 Java 版，可用 LangChain4j 的 AI Service + 状态管理实现 |
| **PydanticAI** | **LangChain4j 的结构化输出** | LangChain4j 支持直接映射到 Java POJO，类似 Pydantic 的结构化输出 |
| **CrewAI / AutoGen** | **LangChain4j Multi-Agent** | 多 Agent 协作可用 LangChain4j 手动编排，或看 **JAgent** 等社区项目 |
| **OpenAI Swarm** | **Spring AI** | Spring 生态的 AI 框架，对 Java 开发者最友好 |
| **MCP (Model Context Protocol)** | **MCP Java SDK** | Anthropic 官方提供了 Java/Kotlin 的 MCP SDK |

## 核心推荐（两条路线）

### 路线一：LangChain4j（与本项目最接近）

```xml
<dependency>
    <groupId>dev.langchain4j</groupId>
    <artifactId>langchain4j</artifactId>
</dependency>
<dependency>
    <groupId>dev.langchain4j</groupId>
    <artifactId>langchain4j-open-ai</artifactId>
</dependency>
```

- 和 Python LangChain 概念一一对应（Chain、Agent、Tool、Memory、RAG）
- 支持 OpenAI / 各种 LLM
- **你学的 Python 知识可以直接迁移**

### 路线二：Spring AI（Spring 生态）

```xml
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-openai-spring-boot-starter</artifactId>
</dependency>
```

- Spring Boot 风格，注解驱动
- 适合做企业级项目、微服务集成
- 社区活跃，迭代快

## 概念映射（Python → Java）

| Python 概念 | LangChain4j 对应 |
|---|---|
| `ChatOpenAI` | `OpenAiChatModel` |
| `Tool / Function Calling` | `@Tool` 注解 |
| `Memory (对话记忆)` | `ChatMemory` 接口 |
| `RAG (检索增强)` | `EmbeddingStore` + `ContentRetriever` |
| `Structured Output` | `AiServices` 接口直接返回 Java 对象 |
| `Agent (ReAct)` | `AiServices` + `@Tool` 自动实现 ReAct 循环 |
| `Multi-Agent` | 多个 `AiServices` 实例互相调用 |
| `MCP` | `langchain4j-mcp` 模块 或 官方 MCP Java SDK |

## LangGraph 深入解析

### LangGraph 是什么

**一句话：LangGraph 是用来构建"有状态、多步骤"AI Agent 工作流的框架。**

### LangGraph 和 LangChain 的区别

| | LangChain | LangGraph |
|---|---|---|
| 核心思路 | 链式调用（A → B → C 顺序执行） | **图（Graph）**：节点 + 边，支持循环、分支、条件跳转 |
| 适合场景 | 简单的 RAG、单轮问答 | 复杂 Agent、多步推理、多 Agent 协作 |
| 状态管理 | 弱 | **强**，内置 State 在节点间传递和持久化 |
| 流程控制 | 线性 | 支持循环、条件分支、人工介入（Human-in-the-loop） |

### 用 Java 类比

- **LangChain** ≈ 一个 `Pipeline`，像 Java 的 `Stream` 链式操作
- **LangGraph** ≈ 一个**状态机 / 工作流引擎**，像 Java 的 `Spring State Machine` 或 `Activiti/Flowable` 工作流引擎

### 它解决什么问题

```
用户提问 → Agent思考 → 调用工具 → 拿到结果 → 再思考 → 再调工具 → ... → 最终回答
                    ↑___________________________|
                         这个"循环"就是 LangGraph 的核心
```

没有 LangGraph 时，你要自己写 `while` 循环来实现 ReAct（思考-行动-观察）。LangGraph 把这个**循环、分支、状态传递**标准化了。

### 在本项目中的地位

本项目 46+ 课中，大量中高级课程都用了 LangGraph，比如：
- 自适应 RAG
- 多 Agent 协作
- 规划与执行
- 人工介入审批
- 纠错重试循环

**可以说，LangGraph 是从"简单 Agent"走向"生产级 Agent"的关键框架。**

### 重要性

**非常重要。** 理由：

1. **真实场景的 Agent 都不是线性的**——需要循环、判断、回退、等待人类审批
2. **它是 LangChain 官方的"下一代"Agent 方案**——LangChain 团队把复杂 Agent 的未来押在了 LangGraph 上
3. **本项目课程中大量使用**——不理解 LangGraph，后面很多课看不懂

### Java 端替代方案

LangGraph 的核心能力就三个：**状态管理 + 图流转 + 循环/分支控制**。Java 生态有现成的方案覆盖这些。

#### 方案一：LangGraph4j（社区版，最直接）

GitHub 上已有社区实现的 **LangGraph4j**，API 设计模仿 Python 版：

```java
var workflow = new StateGraph<>(AgentState::new)
    .addNode("agent", this::callAgent)
    .addNode("tools", this::callTools)
    .addEdge(START, "agent")
    .addConditionalEdges("agent", this::shouldContinue, 
        Map.of("continue", "tools", "end", END))
    .addEdge("tools", "agent");  // 循环回去

var app = workflow.compile();
var result = app.invoke(initialState);
```

和 Python 版几乎一模一样，学习成本最低。

#### 方案二：Spring State Machine

Spring 生态，Java 开发者熟悉：

```java
@Configuration
@EnableStateMachine
public class AgentStateMachine extends StateMachineConfigurerAdapter<AgentState, AgentEvent> {
    
    @Override
    public void configure(StateMachineTransitionConfigurer<AgentState, AgentEvent> transitions) {
        transitions
            .withExternal().source(THINKING).target(TOOL_CALLING).event(NEED_TOOL)
            .and()
            .withExternal().source(TOOL_CALLING).target(THINKING).event(GOT_RESULT)
            .and()
            .withExternal().source(THINKING).target(DONE).event(FINAL_ANSWER);
    }
}
```

优点是企业级成熟，缺点是配置偏重。

#### 方案三：Flowable / Camunda（工作流引擎）

如果 Agent 流程复杂到需要可视化编排、人工审批、超时重试：

```java
// BPMN 流程定义，节点是 LLM 调用
runtimeService.startProcessInstanceByKey("agent-workflow", variables);
```

适合**生产级、需要运维监控的 Agent 系统**。

#### 方案四：手写（最轻量）

LangGraph 本质就是一个 `while` 循环 + `Map` 状态：

```java
public class SimpleAgentGraph {
    Map<String, Object> state = new HashMap<>();
    
    public String run(String input) {
        state.put("input", input);
        
        while (true) {
            // 节点1：LLM 思考
            String llmResponse = callLLM(state);
            
            // 条件分支：需要调工具还是直接回答？
            if (needsTool(llmResponse)) {
                // 节点2：调用工具
                String toolResult = executeTool(llmResponse);
                state.put("tool_result", toolResult);
                // 循环回去继续思考
            } else {
                return llmResponse;  // 结束
            }
        }
    }
}
```

简单场景完全够用，不需要引入重框架。

#### 方案推荐选择

| 场景 | 推荐方案 |
|---|---|
| 学习阶段 / 简单 Agent | **手写** 或 **LangGraph4j** |
| 中等复杂度，Spring 项目 | **Spring State Machine** + LangChain4j |
| 生产级，需要监控和审批 | **Flowable / Camunda** |
| 想和 Python 版代码一一对应 | **LangGraph4j** |

## 建议

作为 Java 开发者，**LangChain4j 是首选**——在 Python 项目中学的所有概念（Tool Calling、RAG、Memory、Multi-Agent、结构化输出）都能直接对应过去。如果项目是 Spring Boot 体系，可以再加上 **Spring AI** 做集成层。

两个框架不冲突，可以一起用。
