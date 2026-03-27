"""
第4课 - LangGraph 内部机制演示（不使用 LangGraph 框架）

目的：让你看清 LangGraph 的 StateGraph 内部做了什么：
  1. State 就是一个 dict，在节点之间传递
  2. Node 就是一个函数，接收 state，返回要更新的字段
  3. Edge 就是一个执行顺序列表
  4. compile() 就是把节点和边组装好
  5. invoke() 就是按照边的顺序依次调用节点函数

对比 main.py（LangGraph 框架版），理解框架帮你省了什么。
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 初始化 API 客户端
client = OpenAI(
    api_key=os.getenv('API_KEY'),
    base_url=os.getenv('API_BASE_URL'),
    default_headers={"User-Agent": "Mozilla/5.0"},
)
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini')


# ================================================================
#  【LLM 层】—— 调用大模型 API
# ================================================================

def call_llm(prompt: str) -> dict:
    """
    调用 LLM，返回响应内容和元信息。

    Java 类比：
        public LLMResponse callLLM(String prompt) {
            HttpRequest req = buildRequest(prompt);
            HttpResponse resp = httpClient.send(req);
            return parseResponse(resp);
        }
    """
    messages = [
        {'role': 'user', 'content': prompt},
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0,
    )

    content = response.choices[0].message.content.strip()
    usage = {
        'prompt_tokens': response.usage.prompt_tokens,
        'completion_tokens': response.usage.completion_tokens,
        'total_tokens': response.usage.total_tokens,
    }

    return {'content': content, 'usage': usage}


# ================================================================
#  【节点层】—— 每个节点就是一个函数
# ================================================================

def classification_node(state: dict) -> dict:
    """
    节点1：文本分类

    Java 类比：
        public class ClassificationHandler implements Function<State, Map<String, Object>> {
            public Map<String, Object> apply(State state) {
                String prompt = buildPrompt(state.getText());
                String result = llm.call(prompt);
                return Map.of("classification", result);
            }
        }
    """
    prompt = f"请将以下文本分类为：新闻、博客、研究、其他（只返回分类名称）。\n\n文本：{state['text']}\n\n分类："

    print()
    print('=' * 60)
    print('【节点1 - 文本分类】')
    print('=' * 60)
    print()
    print('>>> 发送给 LLM 的完整 Prompt:')
    print('-' * 40)
    print(prompt)
    print('-' * 40)
    print()

    result = call_llm(prompt)

    print(f'>>> LLM 返回内容: {result["content"]}')
    print(f'>>> Token 用量: {result["usage"]}')
    print()

    # 节点返回要更新的字段（不是整个 state，只返回变化的部分）
    return {"classification": result["content"]}


def entity_extraction_node(state: dict) -> dict:
    """
    节点2：实体提取

    Java 类比：同上，换了 prompt 模板。
    """
    prompt = f"请从以下文本中提取所有实体（人物、组织、地点），用逗号分隔返回。\n\n文本：{state['text']}\n\n实体："

    print()
    print('=' * 60)
    print('【节点2 - 实体提取】')
    print('=' * 60)
    print()
    print('>>> 发送给 LLM 的完整 Prompt:')
    print('-' * 40)
    print(prompt)
    print('-' * 40)
    print()

    result = call_llm(prompt)
    entities = [e.strip() for e in result["content"].split(",")]

    print(f'>>> LLM 返回内容: {result["content"]}')
    print(f'>>> 解析后的实体列表: {entities}')
    print(f'>>> Token 用量: {result["usage"]}')
    print()

    return {"entities": entities}


def summarization_node(state: dict) -> dict:
    """
    节点3：文本摘要

    Java 类比：同上，换了 prompt 模板。
    """
    prompt = f"请用一句话概括以下文本的核心内容。\n\n文本：{state['text']}\n\n摘要："

    print()
    print('=' * 60)
    print('【节点3 - 文本摘要】')
    print('=' * 60)
    print()
    print('>>> 发送给 LLM 的完整 Prompt:')
    print('-' * 40)
    print(prompt)
    print('-' * 40)
    print()

    result = call_llm(prompt)

    print(f'>>> LLM 返回内容: {result["content"]}')
    print(f'>>> Token 用量: {result["usage"]}')
    print()

    return {"summary": result["content"]}


# ================================================================
#  【图引擎层】—— 这就是 LangGraph 的 StateGraph 内部逻辑
# ================================================================

class SimpleGraph:
    """
    手写的图引擎，模拟 LangGraph 的 StateGraph。

    Java 类比：
        public class SimpleGraph {
            private Map<String, Function<State, Map>> nodes = new LinkedHashMap<>();
            private List<String[]> edges = new ArrayList<>();
            private String entryPoint;

            public void addNode(String name, Function<State, Map> fn) { ... }
            public void addEdge(String from, String to) { ... }
            public void setEntryPoint(String name) { ... }
            public SimpleGraph compile() { return this; }
            public State invoke(State input) { ... }
        }
    """

    def __init__(self):
        self.nodes = {}       # 节点名 → 函数
        self.edges = []       # [(from, to), ...]
        self.entry_point = None

    def add_node(self, name: str, func):
        """添加节点。Java 类比：nodes.put(name, func)"""
        self.nodes[name] = func

    def add_edge(self, from_node: str, to_node: str):
        """添加边。Java 类比：edges.add(new String[]{from, to})"""
        self.edges.append((from_node, to_node))

    def set_entry_point(self, name: str):
        """设置入口节点。Java 类比：this.entryPoint = name"""
        self.entry_point = name

    def compile(self):
        """
        编译图：计算执行顺序。

        LangGraph 的 compile() 做的事情：
        1. 验证图结构（节点都存在？入口设了？）
        2. 根据边计算拓扑排序（执行顺序）
        3. 返回可执行的 CompiledGraph 对象

        我们这里简化为：从入口节点开始，沿着边找出顺序。
        """
        print()
        print('=' * 60)
        print('【图编译阶段 - compile()】')
        print('=' * 60)
        print()
        print(f'>>> 注册的节点: {list(self.nodes.keys())}')
        print(f'>>> 注册的边: {self.edges}')
        print(f'>>> 入口节点: {self.entry_point}')

        # 计算执行顺序
        self.execution_order = []
        current = self.entry_point

        while current:
            self.execution_order.append(current)
            # 找下一个节点
            next_node = None
            for from_n, to_n in self.edges:
                if from_n == current:
                    next_node = to_n if to_n != '__end__' else None
                    break
            current = next_node

        print(f'>>> 计算出的执行顺序: {self.execution_order}')
        print()
        return self

    def invoke(self, initial_state: dict) -> dict:
        """
        执行工作流。这就是 LangGraph 的 app.invoke() 内部逻辑：

        1. 初始化 state
        2. 按执行顺序依次调用每个节点
        3. 每个节点返回的 dict 合并到 state 中
        4. 返回最终 state
        """
        state = dict(initial_state)  # 复制一份，避免修改原始数据

        print()
        print('#' * 60)
        print('#  开始执行工作流 invoke()')
        print('#' * 60)
        print()
        print(f'>>> 初始 State: { {k: (v[:50] + "..." if isinstance(v, str) and len(v) > 50 else v) for k, v in state.items()} }')

        for i, node_name in enumerate(self.execution_order, 1):
            print()
            print(f'>>> ====== 第 {i}/{len(self.execution_order)} 步：执行节点 [{node_name}] ======')

            # 调用节点函数
            node_func = self.nodes[node_name]
            updates = node_func(state)

            # 合并返回值到 state（这就是 LangGraph 的状态更新机制）
            print(f'>>> 节点 [{node_name}] 返回的更新: {updates}')
            state.update(updates)
            print(f'>>> 更新后的 State 包含字段: {list(state.keys())}')

        return state


# ================================================================
#  【运行】
# ================================================================

if __name__ == '__main__':
    print('第4课 - LangGraph 内部机制演示（不使用框架）')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {os.getenv("API_BASE_URL")}')
    print()
    print('这个程序展示 LangGraph 的 StateGraph 内部做了什么。')
    print('核心发现：图引擎本质上就是"按顺序调函数 + 合并返回值到共享 state"')
    print()

    # ==========================================
    # 第1步：构建图（等价于 LangGraph 的代码）
    # ==========================================
    print('=' * 60)
    print('【构建图 - 等价于 LangGraph 的 StateGraph 操作】')
    print('=' * 60)
    print()
    print('LangGraph 写法:')
    print('  workflow = StateGraph(State)')
    print('  workflow.add_node("classification", classification_node)')
    print('  workflow.add_node("entity_extraction", entity_extraction_node)')
    print('  workflow.add_node("summarization", summarization_node)')
    print('  workflow.set_entry_point("classification")')
    print('  workflow.add_edge("classification", "entity_extraction")')
    print('  workflow.add_edge("entity_extraction", "summarization")')
    print('  workflow.add_edge("summarization", END)')
    print('  app = workflow.compile()')
    print()
    print('我们的等价手写版:')

    graph = SimpleGraph()
    graph.add_node("classification", classification_node)
    graph.add_node("entity_extraction", entity_extraction_node)
    graph.add_node("summarization", summarization_node)
    graph.set_entry_point("classification")
    graph.add_edge("classification", "entity_extraction")
    graph.add_edge("entity_extraction", "summarization")
    graph.add_edge("summarization", "__end__")

    app = graph.compile()

    # ==========================================
    # 第2步：执行示例
    # ==========================================
    sample_texts = [
        """华为近日发布了全新的鸿蒙操作系统4.0版本，这是华为自主研发的分布式操作系统。
该系统将在深圳举行的开发者大会上正式亮相，华为消费者业务CEO余承东表示，
鸿蒙4.0在性能和安全性方面都有显著提升，预计将覆盖手机、平板、智能手表等多种终端设备。""",
        """最近我在学习 LangGraph 框架，发现它用图结构来编排 AI 工作流真的很优雅。
每个节点就是一个处理步骤，边定义了执行顺序。跟传统的 Pipeline 模式相比，
它更灵活，支持条件路由和循环，非常适合构建复杂的 AI 应用。""",
    ]

    print('\n--- 示例分析 ---')
    for text in sample_texts:
        result = app.invoke({"text": text.strip()})

        print()
        print('=' * 60)
        print('【最终结果 - invoke() 返回的完整 State】')
        print('=' * 60)
        print(f'  分类: {result["classification"]}')
        print(f'  实体: {result["entities"]}')
        print(f'  摘要: {result["summary"]}')
        print()

    # 交互模式
    print('\n输入文本进行分析，输入 /quit 退出\n')
    while True:
        user_input = input('请输入文本: ').strip()

        if not user_input:
            continue

        if user_input == '/quit':
            print('再见！')
            break

        result = app.invoke({"text": user_input})

        print()
        print('=' * 60)
        print('【最终结果】')
        print('=' * 60)
        print(f'  分类: {result["classification"]}')
        print(f'  实体: {result["entities"]}')
        print(f'  摘要: {result["summary"]}')
        print()
