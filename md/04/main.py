"""
第4课 - LangGraph 图工作流（框架版）

核心概念：StateGraph + Node + Edge = 图工作流
三个节点依次处理：文本分类 → 实体提取 → 文本摘要
每个节点调用一次 LLM，共 3 次调用。
"""

import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# ========== 定义 State：工作流中流动的数据 ==========

class State(TypedDict):
    text: str                # 输入文本
    classification: str      # 分类结果
    entities: List[str]      # 提取的实体
    summary: str             # 摘要


# ========== 初始化 LLM ==========

llm = ChatOpenAI(
    model=os.getenv('MODEL_NAME', 'gpt-4o-mini'),
    temperature=0,
    base_url=os.getenv('API_BASE_URL'),
    api_key=os.getenv('API_KEY'),
    use_responses_api=False,
    default_headers={"User-Agent": "Mozilla/5.0"},  # 第三方API可能屏蔽SDK默认UA
)


# ========== 节点1：文本分类 ==========

def classification_node(state: State):
    """把文本分类为：新闻、博客、研究、其他"""
    prompt = PromptTemplate(
        input_variables=["text"],
        template="请将以下文本分类为：新闻、博客、研究、其他（只返回分类名称）。\n\n文本：{text}\n\n分类：",
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))

    print('\n' + '=' * 50)
    print('【节点1 - 文本分类】')
    print('=' * 50)
    print(f'>>> 发送给 LLM 的 Prompt:')
    print(f'    {message.content[:200]}...')

    response = llm.invoke([message])
    classification = response.content.strip()

    print(f'>>> LLM 返回: {classification}')
    print(f'>>> Token 用量: {response.response_metadata.get("token_usage", "N/A")}')

    return {"classification": classification}


# ========== 节点2：实体提取 ==========

def entity_extraction_node(state: State):
    """从文本中提取人物、组织、地点"""
    prompt = PromptTemplate(
        input_variables=["text"],
        template="请从以下文本中提取所有实体（人物、组织、地点），用逗号分隔返回。\n\n文本：{text}\n\n实体：",
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))

    print('\n' + '=' * 50)
    print('【节点2 - 实体提取】')
    print('=' * 50)
    print(f'>>> 发送给 LLM 的 Prompt:')
    print(f'    {message.content[:200]}...')

    response = llm.invoke([message])
    entities_text = response.content.strip()
    entities = [e.strip() for e in entities_text.split(",")]

    print(f'>>> LLM 返回: {entities_text}')
    print(f'>>> 解析后的实体列表: {entities}')

    return {"entities": entities}


# ========== 节点3：文本摘要 ==========

def summarization_node(state: State):
    """生成一句话摘要"""
    prompt = PromptTemplate(
        input_variables=["text"],
        template="请用一句话概括以下文本的核心内容。\n\n文本：{text}\n\n摘要：",
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))

    print('\n' + '=' * 50)
    print('【节点3 - 文本摘要】')
    print('=' * 50)
    print(f'>>> 发送给 LLM 的 Prompt:')
    print(f'    {message.content[:200]}...')

    response = llm.invoke([message])
    summary = response.content.strip()

    print(f'>>> LLM 返回: {summary}')

    return {"summary": summary}


# ========== 构建图工作流 ==========

def build_workflow():
    """构建 LangGraph 工作流：分类 → 实体提取 → 摘要"""
    workflow = StateGraph(State)

    # 添加节点
    workflow.add_node("classification", classification_node)
    workflow.add_node("entity_extraction", entity_extraction_node)
    workflow.add_node("summarization", summarization_node)

    # 添加边（定义执行顺序）
    workflow.set_entry_point("classification")
    workflow.add_edge("classification", "entity_extraction")
    workflow.add_edge("entity_extraction", "summarization")
    workflow.add_edge("summarization", END)

    # 编译
    app = workflow.compile()
    return app


# ========== 分析函数 ==========

def analyze_text(app, text: str) -> dict:
    """用工作流分析一段文本"""
    print('\n' + '#' * 60)
    print('#  文本分析开始')
    print('#' * 60)
    print(f'\n输入文本: {text[:100]}{"..." if len(text) > 100 else ""}')

    state_input = {"text": text}
    result = app.invoke(state_input)

    print('\n' + '=' * 50)
    print('【最终结果】')
    print('=' * 50)
    print(f'  分类: {result["classification"]}')
    print(f'  实体: {result["entities"]}')
    print(f'  摘要: {result["summary"]}')
    print()

    return result


# ========== 运行 ==========

def print_graph(app):
    """打印工作流的图结构"""
    print('=' * 50)
    print('【工作流图结构】')
    print('=' * 50)
    try:
        mermaid = app.get_graph().draw_mermaid()
        print(mermaid)
        print()
        try:
            png_data = app.get_graph().draw_mermaid_png()
            png_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workflow_graph.png')
            with open(png_path, 'wb') as f:
                f.write(png_data)
            print(f'>>> 图片已保存到: {png_path}')
        except Exception as e:
            print(f'>>> PNG 导出失败: {e}')
            print('>>> 把上面的 Mermaid 代码粘贴到 https://mermaid.live 查看')
    except Exception as e:
        print(f'>>> 图可视化失败: {e}')
    print()


if __name__ == '__main__':
    print('第4课 - LangGraph 图工作流')
    print(f'模型: {os.getenv("MODEL_NAME")}')
    print(f'API: {os.getenv("API_BASE_URL")}')
    print()

    # 构建工作流
    app = build_workflow()

    # 打印图结构
    print_graph(app)

    # 示例文本
    sample_texts = [
        """
华为近日发布了全新的鸿蒙操作系统4.0版本，这是华为自主研发的分布式操作系统。
该系统将在深圳举行的开发者大会上正式亮相，华为消费者业务CEO余承东表示，
鸿蒙4.0在性能和安全性方面都有显著提升，预计将覆盖手机、平板、智能手表等多种终端设备。
        """.strip(),
        """
最近我在学习 LangGraph 框架，发现它用图结构来编排 AI 工作流真的很优雅。
每个节点就是一个处理步骤，边定义了执行顺序。跟传统的 Pipeline 模式相比，
它更灵活，支持条件路由和循环，非常适合构建复杂的 AI 应用。
        """.strip(),
    ]

    print('--- 示例分析 ---')
    for text in sample_texts:
        analyze_text(app, text)

    # 交互模式
    print('\n输入文本进行分析，输入 /quit 退出\n')
    while True:
        user_input = input('请输入文本: ').strip()

        if not user_input:
            continue

        if user_input == '/quit':
            print('再见！')
            break

        analyze_text(app, user_input)
