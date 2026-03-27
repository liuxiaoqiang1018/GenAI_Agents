# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 语言

本项目使用中文进行交流。

## 项目概述

GenAI_Agents 是一个生成式 AI 代理教程和实现的综合资源库，包含 46+ 个 Jupyter Notebook，涵盖从初级到高级的 AI 代理实现。由 Nir Diamant 维护。

## 技术栈

- **语言**: Python (Jupyter Notebooks)
- **主要框架**: LangChain, LangGraph, PydanticAI
- **其他框架**: CrewAI, AutoGen, OpenAI Swarm, MCP (Model Context Protocol)
- **LLM**: OpenAI (GPT-4, GPT-4o-mini)
- **环境变量**: 通过 `python-dotenv` 加载，需要 `OPENAI_API_KEY`

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 Jupyter
jupyter notebook

# 运行单个笔记本
jupyter nbconvert --to notebook --execute all_agents_tutorials/<notebook>.ipynb
```

## 项目结构

- `all_agents_tutorials/` — 所有代理实现笔记本（核心内容）
  - `scripts/` — 辅助脚本（如 `mcp_server.py`）
  - `data/` — 笔记本专用数据（`chinook.db`, `ATLAS_data/`）
- `data/` — 项目级数据文件（EU法规文本等）
- `images/` — 架构图（SVG 格式，由 Mermaid 生成）
- `audio/` — 音频示例文件

## 笔记本标准结构

新建或修改笔记本必须遵循以下结构：

1. 标题和概述
2. 详细说明（动机、关键组件、架构、优势）
3. 架构图（Mermaid → SVG，存放在 `/images`，引用格式：`![Name](../images/name.svg)`）
4. 包安装（`!pip install` 在实现部分开头）
5. 实现代码（带注释的分步实现）
6. 使用示例
7. 与简单代理的对比
8. 额外考虑和改进建议
9. 参考资源

每个代码单元格前必须有描述其目的的 markdown 单元格。

## 贡献流程

1. 笔记本放在 `all_agents_tutorials/` 目录
2. 更新 README.md：在列表和表格中添加新代理条目，需正确分类、编号，并递增后续编号
3. 提交前清除笔记本中不必要的输出
4. PR 目标分支为 `main`

## LangChain API 注意事项

LangChain 的导入路径和 API 会变化。最近一次更新（commit `cbeab13`）修复了过时的导入和调用方式。修改笔记本时注意检查 API 兼容性。
