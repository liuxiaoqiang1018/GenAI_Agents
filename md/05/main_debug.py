"""
第5课 - MCP 内部机制演示（不使用 mcp 库）

目的：让你看清 MCP 协议的本质——就是通过 subprocess + JSON-RPC 通信：
  1. Host 启动 Server 子进程
  2. 通过 stdin/stdout 发送 JSON-RPC 消息
  3. Server 返回工具列表（list_tools）
  4. Host 调用工具（call_tool）
  5. LLM 根据工具结果生成回答

对比 main.py（使用 mcp 库），理解 mcp 库帮你封装了什么。

注意：本文件不依赖真实的 MCP Server 进程，而是用模拟数据来展示完整流程。
这样可以专注于理解协议本身，不被进程管理的复杂性干扰。
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
#  【Server 层】模拟 MCP Server 的工具注册和执行
# ================================================================

# 模拟城市数据（和 mcp_server.py 一致）
CITY_DATA = {
    "北京": {
        "人口": "2189万", "面积": "16410平方公里", "GDP": "43760亿元",
        "著名景点": ["故宫", "长城", "天坛", "颐和园"],
        "气候": "温带季风气候", "平均气温": "13°C",
    },
    "上海": {
        "人口": "2487万", "面积": "6341平方公里", "GDP": "47218亿元",
        "著名景点": ["外滩", "东方明珠", "豫园", "南京路"],
        "气候": "亚热带季风气候", "平均气温": "17°C",
    },
    "深圳": {
        "人口": "1768万", "面积": "1997平方公里", "GDP": "34606亿元",
        "著名景点": ["世界之窗", "欢乐谷", "大梅沙", "莲花山"],
        "气候": "亚热带海洋性气候", "平均气温": "23°C",
    },
    "成都": {
        "人口": "2119万", "面积": "14335平方公里", "GDP": "22075亿元",
        "著名景点": ["武侯祠", "锦里", "大熊猫基地", "宽窄巷子"],
        "气候": "亚热带湿润气候", "平均气温": "16°C",
    },
    "杭州": {
        "人口": "1237万", "面积": "16850平方公里", "GDP": "20059亿元",
        "著名景点": ["西湖", "灵隐寺", "千岛湖", "宋城"],
        "气候": "亚热带季风气候", "平均气温": "17°C",
    },
}

CONVERSION_RATES = {
    "人民币_美元": 0.14, "美元_人民币": 7.15,
    "公里_英里": 0.621, "英里_公里": 1.609,
    "千克_磅": 2.205, "磅_千克": 0.454,
}


class MockMCPServer:
    """
    模拟 MCP Server，展示 Server 内部的工具注册和执行机制。

    Java 类比：
        public class MCPServer {
            private Map<String, ToolHandler> tools = new HashMap<>();

            public void registerTool(String name, ToolHandler handler) {
                tools.put(name, handler);
            }

            public List<ToolDefinition> listTools() {
                return tools.values().stream()
                    .map(ToolHandler::getDefinition)
                    .collect(toList());
            }

            public String callTool(String name, Map<String, Object> args) {
                return tools.get(name).execute(args);
            }
        }
    """

    def __init__(self):
        # 工具注册表（name → {definition, handler}）
        self.tools = {}

    def register_tool(self, name: str, description: str, parameters: dict, handler):
        """注册工具。Java 类比：tools.put(name, new ToolHandler(def, fn))"""
        self.tools[name] = {
            "definition": {
                "name": name,
                "description": description,
                "inputSchema": parameters,
            },
            "handler": handler,
        }

    def list_tools(self) -> list:
        """
        返回所有工具的定义。
        这就是 MCP 协议的 tools/list 请求的响应内容。
        """
        return [t["definition"] for t in self.tools.values()]

    def call_tool(self, name: str, arguments: dict) -> str:
        """
        执行工具。
        这就是 MCP 协议的 tools/call 请求的处理逻辑。
        """
        if name not in self.tools:
            return f"错误：工具 '{name}' 不存在"
        return self.tools[name]["handler"](arguments)


# 工具实现函数
def query_city_handler(args: dict) -> str:
    city_name = args.get("city_name", "")
    if city_name not in CITY_DATA:
        available = "、".join(CITY_DATA.keys())
        return f"未找到城市'{city_name}'的信息。支持的城市：{available}"
    info = CITY_DATA[city_name]
    lines = [f"【{city_name}】城市信息："]
    for key, value in info.items():
        if isinstance(value, list):
            lines.append(f"  {key}：{'、'.join(value)}")
        else:
            lines.append(f"  {key}：{value}")
    return "\n".join(lines)


def convert_unit_handler(args: dict) -> str:
    value = args.get("value", 0)
    from_unit = args.get("from_unit", "")
    to_unit = args.get("to_unit", "")
    key = f"{from_unit}_{to_unit}"
    if key not in CONVERSION_RATES:
        return f"不支持 {from_unit} → {to_unit} 的换算"
    result = value * CONVERSION_RATES[key]
    return f"{value} {from_unit} = {result:.2f} {to_unit}"


# ================================================================
#  【协议层】模拟 MCP 的 JSON-RPC 通信
# ================================================================

def simulate_mcp_request(method: str, params: dict = None) -> dict:
    """
    模拟发送 MCP JSON-RPC 请求。

    真实的 MCP 通信是通过 stdin/stdout 发送 JSON：
    → {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    ← {"jsonrpc": "2.0", "id": 1, "result": {"tools": [...]}}

    Java 类比：
        public class MCPClient {
            private ProcessBuilder pb;
            private OutputStream stdin;
            private InputStream stdout;

            public JSONObject sendRequest(String method, JSONObject params) {
                JSONObject request = new JSONObject();
                request.put("jsonrpc", "2.0");
                request.put("method", method);
                request.put("params", params);
                stdin.write(request.toString().getBytes());
                return JSONObject.parse(stdout.readLine());
            }
        }
    """
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params or {},
    }
    return request


def simulate_mcp_response(result: any) -> dict:
    """模拟 MCP JSON-RPC 响应"""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": result,
    }


# ================================================================
#  【LLM 层】调用大模型
# ================================================================

def call_llm(messages: list, tools: list = None) -> dict:
    kwargs = {
        'model': MODEL_NAME,
        'messages': messages,
        'temperature': 0,
    }
    if tools:
        kwargs['tools'] = tools
    return client.chat.completions.create(**kwargs)


# ================================================================
#  【Host 层】完整的 Agent 流程
# ================================================================

def agent_run(question: str, server: MockMCPServer) -> str:
    """
    完整的 MCP Host 运行流程：

    1. 通过 MCP 协议发现工具（list_tools）
    2. 把工具描述转成 LLM 能理解的格式
    3. 发送用户问题 + 工具描述给 LLM
    4. 如果 LLM 要调用工具 → 通过 MCP 协议执行（call_tool）
    5. 把结果返回给 LLM 生成最终回答
    """

    # ==========================================
    # 第1步：MCP 工具发现
    # ==========================================
    print()
    print('=' * 60)
    print('【1 - MCP 工具发现（list_tools）】')
    print('=' * 60)
    print()

    # 模拟发送 JSON-RPC 请求
    request = simulate_mcp_request("tools/list")
    print('>>> 发送给 MCP Server 的 JSON-RPC 请求:')
    print(json.dumps(request, indent=2, ensure_ascii=False))
    print()

    # Server 处理请求
    tool_definitions = server.list_tools()

    # 模拟 JSON-RPC 响应
    response = simulate_mcp_response({"tools": tool_definitions})
    print('>>> MCP Server 返回的 JSON-RPC 响应:')
    print(json.dumps(response, indent=2, ensure_ascii=False))
    print()

    print(f'>>> 发现 {len(tool_definitions)} 个工具:')
    for td in tool_definitions:
        print(f'    - {td["name"]}: {td["description"][:60]}...')
    print()

    # ==========================================
    # 第2步：把 MCP 工具转成 OpenAI function calling 格式
    # ==========================================
    print('=' * 60)
    print('【2 - 工具格式转换（MCP → OpenAI function calling）】')
    print('=' * 60)
    print()
    print('>>> MCP 协议返回的工具定义和 OpenAI 的 tools 参数格式不同')
    print('>>> Host 需要做一次转换（这就是 mcp 库帮你做的事之一）')
    print()

    openai_tools = []
    for td in tool_definitions:
        tool = {
            "type": "function",
            "function": {
                "name": td["name"],
                "description": td["description"],
                "parameters": td["inputSchema"],
            }
        }
        openai_tools.append(tool)
        print(f'>>> 转换后的工具: {json.dumps(tool, indent=2, ensure_ascii=False)[:300]}')
        print()

    # ==========================================
    # 第3步：发送给 LLM
    # ==========================================
    print('=' * 60)
    print('【3 - 发送给 LLM（带工具描述）】')
    print('=' * 60)
    print()

    system_msg = "你是一个AI助手，可以使用外部工具查询城市信息和做单位换算。回答用中文。"
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": question},
    ]

    print(f'>>> System Prompt: {system_msg}')
    print(f'>>> 用户问题: {question}')
    print(f'>>> 附带 {len(openai_tools)} 个工具定义')
    print()

    print('>>> 调用 LLM...')
    llm_response = call_llm(messages, tools=openai_tools)
    choice = llm_response.choices[0]

    print(f'>>> finish_reason: {choice.finish_reason}')
    print(f'>>> Token: prompt={llm_response.usage.prompt_tokens}, completion={llm_response.usage.completion_tokens}')
    print()

    # 情况1：LLM 直接回答
    if choice.finish_reason == 'stop':
        answer = choice.message.content
        print(f'>>> LLM 直接回答（不需要工具）: {answer}')
        return answer

    # 情况2：LLM 要调用工具
    if choice.message.tool_calls:
        tool_call = choice.message.tool_calls[0]
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)

        print(f'>>> LLM 决定调用工具: {func_name}')
        print(f'>>> LLM 生成的参数: {json.dumps(func_args, ensure_ascii=False)}')
        print()

        # ==========================================
        # 第4步：通过 MCP 协议执行工具
        # ==========================================
        print('=' * 60)
        print('【4 - MCP 工具执行（call_tool）】')
        print('=' * 60)
        print()

        # 模拟发送 JSON-RPC 请求
        call_request = simulate_mcp_request("tools/call", {
            "name": func_name,
            "arguments": func_args,
        })
        print('>>> 发送给 MCP Server 的 JSON-RPC 请求:')
        print(json.dumps(call_request, indent=2, ensure_ascii=False))
        print()

        # Server 执行工具
        tool_result = server.call_tool(func_name, func_args)

        call_response = simulate_mcp_response({"content": [{"type": "text", "text": tool_result}]})
        print('>>> MCP Server 返回的 JSON-RPC 响应:')
        print(json.dumps(call_response, indent=2, ensure_ascii=False))
        print()

        # ==========================================
        # 第5步：把工具结果返回给 LLM
        # ==========================================
        print('=' * 60)
        print('【5 - 把工具结果返回给 LLM 生成最终回答】')
        print('=' * 60)
        print()

        messages.append(choice.message.model_dump())
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result,
        })

        print(f'>>> 工具结果: {tool_result}')
        print('>>> 再次调用 LLM...')
        print()

        final_response = call_llm(messages, tools=openai_tools)
        final_answer = final_response.choices[0].message.content

        print(f'>>> LLM 最终回答: {final_answer}')
        return final_answer

    return choice.message.content or "（无响应）"


# ================================================================
#  【运行】
# ================================================================

if __name__ == '__main__':
    print('第5课 - MCP 内部机制演示（不使用 mcp 库）')
    print(f'模型: {MODEL_NAME}')
    print(f'API: {os.getenv("API_BASE_URL")}')
    print()
    print('这个程序展示 MCP 协议内部做了什么：')
    print('  - MCP Server 就是一个工具注册表')
    print('  - MCP Client 通过 JSON-RPC 与 Server 通信')
    print('  - Host 负责把工具描述喂给 LLM，并协调工具调用')
    print()

    # 创建模拟的 MCP Server
    server = MockMCPServer()

    # 注册工具（等价于 mcp_server.py 中的 @mcp.tool()）
    server.register_tool(
        name="query_city",
        description="查询中国城市的基本信息，包括人口、面积、GDP、著名景点、气候等。",
        parameters={
            "type": "object",
            "properties": {
                "city_name": {
                    "type": "string",
                    "description": "城市名称（如北京、上海、深圳、成都、杭州）",
                }
            },
            "required": ["city_name"],
        },
        handler=query_city_handler,
    )

    server.register_tool(
        name="convert_unit",
        description="单位换算工具，支持货币、距离、重量的换算。",
        parameters={
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "要换算的数值"},
                "from_unit": {"type": "string", "description": "原始单位（如人民币、公里、千克）"},
                "to_unit": {"type": "string", "description": "目标单位（如美元、英里、磅）"},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
        handler=convert_unit_handler,
    )

    print(f'已注册 {len(server.tools)} 个工具')
    print()

    # 示例问题
    example_questions = [
        "成都有哪些著名景点？",
        "500 人民币等于多少美元？",
    ]

    print('--- 示例问题 ---')
    for q in example_questions:
        print()
        print('#' * 60)
        print(f'#  用户提问: {q}')
        print('#' * 60)

        result = agent_run(q, server)

        print()
        print('=' * 60)
        print('【最终返回给用户的回复】')
        print('=' * 60)
        print(f'AI: {result}')
        print()

    # 交互模式
    print('\n输入问题，输入 /quit 退出\n')
    while True:
        user_input = input('你: ').strip()
        if not user_input:
            continue
        if user_input == '/quit':
            print('再见！')
            break

        print()
        print('#' * 60)
        print(f'#  用户提问: {user_input}')
        print('#' * 60)

        result = agent_run(user_input, server)

        print()
        print('=' * 60)
        print('【最终返回给用户的回复】')
        print('=' * 60)
        print(f'AI: {result}\n')
