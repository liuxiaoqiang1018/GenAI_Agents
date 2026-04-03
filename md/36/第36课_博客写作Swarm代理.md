# 第36课：博客写作 Swarm 代理（Blog Writer Swarm）

## 课程概述

本课使用 **OpenAI Swarm** 框架实现 5 个 Agent 接力协作写博客：管理员定主题 → 规划者拟大纲 → 研究者调研内容 → 写手撰写文章 → 编辑润色定稿 → 保存为 Markdown 文件。

**核心模式**：Swarm Agent 接力传递（Handoff）+ 5 角色分工流水线

## 架构流程

```
管理员 → 规划者 → 研究者 → 写手 → 编辑 → 保存文件
(定主题) (拟大纲) (调研)  (撰写) (润色)  (完成)
```

每个 Agent 完成任务后，调用 `transfer_to_xxx()` 函数把控制权交给下一个 Agent。这就是 Swarm 的**接力模式（Handoff）**。

## 五个 Agent 角色

| Agent | 职责 | 输出 |
|---|---|---|
| **Admin（管理员）** | 接收用户主题，启动项目 | 传递主题给规划者 |
| **Planner（规划者）** | 根据主题拟定博客大纲和章节结构 | 章节标题列表 |
| **Researcher（研究者）** | 对每个章节进行详细调研 | 每节的调研笔记 |
| **Writer（写手）** | 根据大纲和调研写完整博客文章 | 博客正文 |
| **Editor（编辑）** | 审阅润色，修正语法和表达 | 定稿并保存文件 |

## 关键概念标注

| 概念 | 在代码中的位置 | Java 类比 |
|---|---|---|
| **Swarm Agent** | `Agent(name, instructions, functions)` | 带指令的 Service |
| **Handoff（接力）** | `transfer_to_xxx()` 返回下一个 Agent 对象 | 责任链模式 |
| **context_variables** | 跨 Agent 共享的上下文变量 | ThreadLocal / Context |
| **instructions 函数** | 返回 system prompt 的函数，可访问 context | 动态配置 |
| **functions 列表** | Agent 可调用的工具/转接函数 | 接口方法列表 |

## Swarm vs LangGraph 对比

| 维度 | Swarm | LangGraph |
|---|---|---|
| 流程控制 | Agent 自主决定何时转交 | 图结构预定义边 |
| 状态管理 | context_variables 简单 dict | TypedDict + reducer |
| 复杂度 | 极简，几行代码一个 Agent | 完整的图定义 |
| 适合场景 | 线性接力、对话式 | 复杂分支、循环、并行 |
| 可视化 | 无 | 内置 Mermaid 图 |

## 环境准备

```bash
python -m venv venv
venv\Scripts\activate

pip install httpx python-dotenv
```

注意：原教程使用 OpenAI Swarm 包（实验性），本课用 httpx 手写 Swarm 的接力机制，无需安装 swarm。

## 运行方式

```bash
cd md/36
python main.py        # 模拟 Swarm 接力的框架版
python main_debug.py  # 透明版（纯手写接力流程）
```
