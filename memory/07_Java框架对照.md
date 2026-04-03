# Java 端 AI Agent 框架对照

> 完整版含代码示例见 `java/md/Java_AI_Agent框架对照.md`

## 框架映射表

| Python 框架 | Java 对应 | 说明 |
|---|---|---|
| **LangChain** | **LangChain4j** | 最直接的对应，API 设计非常相似 |
| **LangGraph** | **LangGraph4j / Spring State Machine / Flowable** | 社区版或工作流引擎替代 |
| **PydanticAI** | **LangChain4j 结构化输出** | AiServices 接口直接返回 Java POJO |
| **CrewAI / AutoGen** | **LangChain4j Multi-Agent** | 手动编排多个 AiServices 实例 |
| **OpenAI Swarm** | **Spring AI** | Spring Boot 风格，注解驱动 |
| **MCP** | **MCP Java SDK** | Anthropic 官方 Java/Kotlin SDK |

## 概念映射

| Python 概念 | LangChain4j 对应 |
|---|---|
| `ChatOpenAI` | `OpenAiChatModel` |
| `Tool / Function Calling` | `@Tool` 注解 |
| `Memory` | `ChatMemory` 接口 |
| `RAG` | `EmbeddingStore` + `ContentRetriever` |
| `Structured Output` | `AiServices` 接口返回 Java 对象 |
| `Agent (ReAct)` | `AiServices` + `@Tool` 自动 ReAct |
| `Multi-Agent` | 多个 `AiServices` 实例互相调用 |

## 核心依赖

```xml
<!-- 路线一：LangChain4j -->
<dependency>
    <groupId>dev.langchain4j</groupId>
    <artifactId>langchain4j</artifactId>
</dependency>
<dependency>
    <groupId>dev.langchain4j</groupId>
    <artifactId>langchain4j-open-ai</artifactId>
</dependency>

<!-- 路线二：Spring AI -->
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-openai-spring-boot-starter</artifactId>
</dependency>
```

## LangGraph Java 替代方案选择

| 场景 | 推荐方案 |
|---|---|
| 学习阶段 / 简单 Agent | **手写 while 循环** 或 **LangGraph4j** |
| 中等复杂度，Spring 项目 | **Spring State Machine** + LangChain4j |
| 生产级，需要监控和审批 | **Flowable / Camunda** |
| 想和 Python 版一一对应 | **LangGraph4j**（社区版） |

## 建议

**LangChain4j 是首选**——Python 项目中学的所有概念（Tool Calling、RAG、Memory、Multi-Agent、结构化输出）都能直接对应过去。Spring Boot 项目可以再加 **Spring AI** 做集成层，两者不冲突。
