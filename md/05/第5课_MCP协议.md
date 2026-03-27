# 第5课 - MCP 协议（Model Context Protocol）

## 概述

MCP（模型上下文协议）是 Anthropic 推出的开放协议，**标准化了 AI 应用如何接入外部工具和数据源**。

类比：MCP 就像 AI 世界的 **USB-C 接口**——不管什么设备（工具），只要实现了 MCP 协议，任何 AI 应用都能即插即用。

## 环境准备

```bash
# 创建并激活虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# 安装依赖
pip install mcp openai python-dotenv
```

- `mcp` — MCP 协议的 Python SDK（包含 Server 和 Client）
- `openai` — 调用 LLM（通过 OpenAI 兼容接口连 MiniMax）
- `python-dotenv` — 加载 .env 环境变量

`main_debug.py` 不需要 `mcp` 包，只需要 `openai` 和 `python-dotenv`。

## 动机

在前几课中，工具定义是硬编码在代码里的（第2课 ReAct 的搜索工具、第3课的 DataFrame 查询工具）。问题是：

- 每个 Agent 都要重新定义一遍工具
- 工具升级了，所有用到的 Agent 都要改
- 不同框架的工具格式不一样（LangChain 的 Tool vs PydanticAI 的 @agent.tool）

MCP 的解决方案：**把工具做成独立的 Server，Agent 通过标准协议来发现和调用**。

## 架构

```
┌─────────────────────────────────────────────────┐
│                   MCP Host                       │
│         （你的 AI 应用 / Agent）                  │
│                                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │ MCP      │    │ MCP      │    │ MCP      │   │
│  │ Client A │    │ Client B │    │ Client C │   │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘   │
└───────┼──────────────┼──────────────┼───────────┘
        │              │              │
   stdio/SSE      stdio/SSE      stdio/SSE
        │              │              │
   ┌────┴─────┐  ┌────┴─────┐  ┌────┴─────┐
   │ MCP      │  │ MCP      │  │ MCP      │
   │ Server A │  │ Server B │  │ Server C │
   │ (天气)   │  │ (数据库) │  │ (文件)   │
   └──────────┘  └──────────┘  └──────────┘
```

三个角色：
| 角色 | 职责 | Java 类比 |
|------|------|-----------|
| **Host** | AI 应用，包含 LLM，决定何时调用工具 | Spring Boot 应用（Controller 层） |
| **Client** | 维护与 Server 的连接，转发请求 | HTTP Client / RestTemplate |
| **Server** | 暴露工具，接收调用，返回结果 | 微服务（暴露 REST API） |

## 通信方式

MCP 支持两种传输方式：
1. **stdio**：通过标准输入/输出通信（本课使用）。Host 启动 Server 子进程，通过管道通信
2. **SSE**（Server-Sent Events）：通过 HTTP 通信，适合远程 Server

**Java 类比**：
- stdio 类似于 `ProcessBuilder` 启动子进程 + `InputStream/OutputStream` 通信
- SSE 类似于 HTTP 长连接（WebSocket 的简化版）

## 核心流程

```
1. Host 启动 Server 子进程
   └→ Java 类比：Runtime.exec("python mcp_server.py")

2. Client 发送 initialize 请求
   └→ Java 类比：httpClient.post("/handshake")

3. Client 调用 list_tools() 发现工具
   └→ Java 类比：httpClient.get("/tools") → List<ToolDefinition>

4. Host 把工具描述告诉 LLM（放进 System Prompt 或 tools 参数）
   └→ Java 类比：把 API 文档拼进 Prompt

5. 用户提问 → LLM 决定调用某个工具
   └→ Java 类比：Controller 根据请求路由到对应 Service

6. Client 调用 call_tool(name, args) 执行工具
   └→ Java 类比：httpClient.post("/tools/execute", body)

7. 结果返回给 LLM → LLM 生成最终回答
```

## 本课文件说明

| 文件 | 作用 |
|------|------|
| `mcp_server.py` | MCP Server —— 暴露"城市信息查询"和"单位换算"两个工具 |
| `main.py` | MCP Host+Client（框架版）—— 用 mcp 库连接 Server，用 OpenAI API 做 LLM 推理 |
| `main_debug.py` | 纯手写版 —— 不用 mcp 库，手动通过 subprocess + JSON 模拟 MCP 协议 |

## 与前几课的对比

| 维度 | 第2课（ReAct） | 第4课（LangGraph） | 本课（MCP） |
|------|---------------|-------------------|------------|
| 工具定义 | 硬编码在 Agent 里 | 无工具（纯 LLM 节点） | **独立 Server 暴露** |
| 工具发现 | 写死的工具列表 | 不涉及 | **动态发现（list_tools）** |
| 工具调用 | 直接函数调用 | 不涉及 | **跨进程 RPC 调用** |
| 可复用性 | 低（和 Agent 绑定） | 不涉及 | **高（任何 Host 都能用）** |

## 关键洞察

1. **MCP Server 就是一个暴露工具的微服务**：用 `@mcp.tool()` 装饰器注册工具，和 Spring 的 `@GetMapping` 异曲同工
2. **MCP Client 就是 HTTP Client 的变体**：连接、发现、调用，底层是 JSON-RPC 协议
3. **LLM 不知道 MCP 的存在**：LLM 只看到工具描述（在 prompt 或 tools 参数里），MCP 是 Host 和 Server 之间的协议
4. **核心价值是解耦**：工具开发者和 Agent 开发者可以独立工作，只要遵守 MCP 协议

## 参考资源

- [MCP 官方文档](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [原始教程笔记本](../all_agents_tutorials/mcp-tutorial.ipynb)
