"""
第29课：气象灾害管理 AI 代理（LangGraph 框架版）

架构：获取天气 → 社交媒体监控 → 分析灾害类型 → 评估严重程度 → 数据日志
      → [条件路由] → 紧急响应/市政工程/民防响应 → [人工审批] → 发送告警/拒绝

核心模式：条件路由 + Human-in-the-Loop + 多部门分发
"""

import os
import json
import time
import random
import httpx
from typing import Dict, TypedDict, List, Literal
from datetime import datetime
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

load_dotenv()

# ===== 配置 =====
API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_RETRIES = 3


# ===== LLM 调用（httpx 标准模板）=====
def call_llm(messages: list, temperature: float = 0.7) -> str:
    """标准 LLM 调用模板：httpx + 300s超时 + 重试 + 空响应检查"""
    url = f"{API_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature
    }

    for attempt in range(MAX_RETRIES):
        try:
            print(f"  [LLM调用] 第{attempt+1}次请求...")
            resp = httpx.post(url, json=payload, headers=headers, timeout=300)
            data = resp.json()
            choices = data.get("choices")
            if not choices:
                raise ValueError(f"空响应: {data}")
            content = choices[0]["message"]["content"]
            # 清理模型思考过程标签
            import re
            content = re.sub(r'<think>[\s\S]*?</think>\s*', '', content).strip()
            print(f"  [LLM响应] {content[:100]}...")
            return content
        except Exception as e:
            print(f"  [LLM错误] 第{attempt+1}次失败: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep((attempt + 1) * 3)
            else:
                raise


# ===== 状态定义 =====
class WeatherState(TypedDict):
    city: str                    # 城市名称
    weather_data: Dict           # 天气数据
    disaster_type: str           # 灾害类型
    severity: str                # 严重程度
    response: str                # 响应计划
    messages: List[str]          # 日志消息
    alerts: List[str]            # 告警记录
    social_media_reports: List[str]  # 社交媒体报告
    human_approved: bool         # 人工审批结果


# ===== 模拟天气数据（不依赖外部API）=====
def get_simulated_weather_data(scenario: str = "high") -> Dict:
    """生成模拟天气数据用于测试"""
    scenarios = {
        "high": {
            "weather": "强雷暴伴随暴雨和大风",
            "wind_speed": 32.5,
            "cloud_cover": 95,
            "sea_level": 1015,
            "temperature": 35.5,
            "humidity": 90,
            "pressure": 960
        },
        "medium": {
            "weather": "中雨伴随阵风",
            "wind_speed": 15.2,
            "cloud_cover": 75,
            "sea_level": 1012,
            "temperature": 22.3,
            "humidity": 70,
            "pressure": 1005
        },
        "low": {
            "weather": "小雨",
            "wind_speed": 8.5,
            "cloud_cover": 45,
            "sea_level": 1013,
            "temperature": 20.1,
            "humidity": 60,
            "pressure": 1015
        }
    }
    return scenarios.get(scenario.lower(), scenarios["medium"])


# ===== 节点函数 =====

def get_weather_data(state: WeatherState) -> dict:
    """节点1：获取天气数据（使用模拟数据）"""
    print("\n" + "=" * 60)
    print("【节点1】获取天气数据")
    print("=" * 60)

    # 如果已有天气数据（测试模式），直接使用
    if state["weather_data"]:
        print(f"  使用预设的模拟天气数据: {state['city']}")
        for k, v in state["weather_data"].items():
            print(f"    {k}: {v}")
        return {
            **state,
            "messages": state["messages"] + [f"使用模拟天气数据: {state['city']}"]
        }

    # 否则生成默认模拟数据
    weather_data = get_simulated_weather_data("high")
    print(f"  为 {state['city']} 生成模拟天气数据:")
    for k, v in weather_data.items():
        print(f"    {k}: {v}")

    return {
        **state,
        "weather_data": weather_data,
        "messages": state["messages"] + [f"天气数据获取成功: {state['city']}"]
    }


def social_media_monitoring(state: WeatherState) -> dict:
    """节点2：社交媒体监控（模拟）"""
    print("\n" + "=" * 60)
    print("【节点2】社交媒体监控")
    print("=" * 60)

    simulated_reports = [
        "当地报告水位上涨，出现轻微洪水。",
        "大风导致部分城区停电。",
        "市民反映高温导致身体不适加剧。",
        "社交媒体报告严重风暴破坏了当地基础设施。",
        "暴雨导致交通严重中断。",
        "目前未发现与天气相关的异常社交媒体报告。"
    ]

    report = random.choice(simulated_reports)
    print(f"  社交媒体报告: {report}")

    return {
        **state,
        "social_media_reports": state["social_media_reports"] + [report],
        "messages": state["messages"] + [f"社交媒体监控完成: {report}"]
    }


def analyze_disaster_type(state: WeatherState) -> dict:
    """节点3：分析灾害类型（LLM调用）"""
    print("\n" + "=" * 60)
    print("【节点3】分析灾害类型")
    print("=" * 60)

    weather_data = state["weather_data"]
    social_info = ""
    if state["social_media_reports"]:
        social_info = f"\n社交媒体报告: {'; '.join(state['social_media_reports'])}"

    prompt = (
        f"根据以下天气条件，判断是否存在潜在的气象灾害。\n"
        f"天气条件:\n"
        f"- 天气描述: {weather_data.get('weather', 'N/A')}\n"
        f"- 风速: {weather_data.get('wind_speed', 'N/A')} m/s\n"
        f"- 温度: {weather_data.get('temperature', 'N/A')}°C\n"
        f"- 湿度: {weather_data.get('humidity', 'N/A')}%\n"
        f"- 气压: {weather_data.get('pressure', 'N/A')} hPa"
        f"{social_info}\n"
        f"请将灾害分类为以下类型之一: 台风、洪水、热浪、强风暴、寒潮、无直接威胁\n"
        f"只输出灾害类型，不要输出其他内容。"
    )

    print(f"  发送给LLM的提示: {prompt[:200]}...")

    messages = [{"role": "user", "content": prompt}]
    disaster_type = call_llm(messages, temperature=0.3)

    print(f"  灾害类型判定: {disaster_type}")

    return {
        **state,
        "disaster_type": disaster_type.strip(),
        "messages": state["messages"] + [f"灾害类型: {disaster_type.strip()}"]
    }


def assess_severity(state: WeatherState) -> dict:
    """节点4：评估严重程度（LLM调用）"""
    print("\n" + "=" * 60)
    print("【节点4】评估严重程度")
    print("=" * 60)

    weather_data = state["weather_data"]
    prompt = (
        f"已识别的灾害类型: '{state['disaster_type']}'\n"
        f"天气条件:\n"
        f"- 天气: {weather_data.get('weather', 'N/A')}\n"
        f"- 风速: {weather_data.get('wind_speed', 'N/A')} m/s\n"
        f"- 温度: {weather_data.get('temperature', 'N/A')}°C\n"
        f"- 气压: {weather_data.get('pressure', 'N/A')} hPa\n"
        f"请评估严重程度，只回复以下四个级别之一: Critical、High、Medium、Low"
    )

    print(f"  发送给LLM的提示: {prompt[:200]}...")

    messages = [{"role": "user", "content": prompt}]
    severity = call_llm(messages, temperature=0.3)

    print(f"  严重程度评估: {severity}")

    return {
        **state,
        "severity": severity.strip(),
        "messages": state["messages"] + [f"严重程度: {severity.strip()}"]
    }


def data_logging(state: WeatherState) -> dict:
    """节点5：数据日志记录"""
    print("\n" + "=" * 60)
    print("【节点5】数据日志记录")
    print("=" * 60)

    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "city": state["city"],
        "weather_data": state["weather_data"],
        "disaster_type": state["disaster_type"],
        "severity": state["severity"],
    }

    log_file = os.path.join(os.path.dirname(__file__), "disaster_log.txt")
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
        print(f"  日志已写入: {log_file}")
    except Exception as e:
        print(f"  日志写入失败: {e}")

    return {
        **state,
        "messages": state["messages"] + ["数据日志记录完成"]
    }


def emergency_response(state: WeatherState) -> dict:
    """节点6a：紧急响应计划（高/危急级别）"""
    print("\n" + "=" * 60)
    print("【节点6a】紧急响应部门 - 生成应急方案")
    print("=" * 60)

    prompt = (
        f"为{state['city']}的{state['disaster_type']}灾害（严重程度: {state['severity']}）"
        f"制定一份紧急响应计划。包括立即需要采取的行动。请用中文回答，控制在200字以内。"
    )

    messages = [{"role": "user", "content": prompt}]
    response = call_llm(messages)

    print(f"  紧急响应计划: {response[:200]}...")

    return {
        **state,
        "response": response,
        "messages": state["messages"] + ["紧急响应计划已生成"]
    }


def civil_defense_response(state: WeatherState) -> dict:
    """节点6b：民防响应计划"""
    print("\n" + "=" * 60)
    print("【节点6b】民防部门 - 生成公共安全方案")
    print("=" * 60)

    prompt = (
        f"为{state['city']}的{state['disaster_type']}灾害（严重程度: {state['severity']}）"
        f"制定一份民防响应计划。重点关注公共安全措施。请用中文回答，控制在200字以内。"
    )

    messages = [{"role": "user", "content": prompt}]
    response = call_llm(messages)

    print(f"  民防响应计划: {response[:200]}...")

    return {
        **state,
        "response": response,
        "messages": state["messages"] + ["民防响应计划已生成"]
    }


def public_works_response(state: WeatherState) -> dict:
    """节点6c：市政工程响应计划"""
    print("\n" + "=" * 60)
    print("【节点6c】市政工程部门 - 生成基础设施保护方案")
    print("=" * 60)

    prompt = (
        f"为{state['city']}的{state['disaster_type']}灾害（严重程度: {state['severity']}）"
        f"制定一份市政工程响应计划。重点关注基础设施保护。请用中文回答，控制在200字以内。"
    )

    messages = [{"role": "user", "content": prompt}]
    response = call_llm(messages)

    print(f"  市政工程响应计划: {response[:200]}...")

    return {
        **state,
        "response": response,
        "messages": state["messages"] + ["市政工程响应计划已生成"]
    }


def get_human_verification(state: WeatherState) -> dict:
    """节点7：人工审批（低/中级别需要人工确认）"""
    print("\n" + "=" * 60)
    print("【节点7】人工审批")
    print("=" * 60)

    severity = state["severity"].strip().lower()

    if severity in ["low", "medium"]:
        print(f"  低/中级别告警需要人工审批:")
        print(f"  城市: {state['city']}")
        print(f"  灾害类型: {state['disaster_type']}")
        print(f"  天气: {state['weather_data'].get('weather', 'N/A')}")
        print(f"  温度: {state['weather_data'].get('temperature', 'N/A')}°C")
        print(f"  风速: {state['weather_data'].get('wind_speed', 'N/A')} m/s")
        print(f"  严重程度: {state['severity']}")
        print(f"  响应计划: {state['response'][:100]}...")
        print()

        while True:
            user_input = input("  输入 'y' 批准发送告警，'n' 拒绝: ").lower().strip()
            if user_input in ['y', 'n']:
                approved = user_input == 'y'
                print(f"  人工审批结果: {'已批准' if approved else '已拒绝'}")
                break
            print("  请输入 'y' 或 'n'")

        return {
            **state,
            "human_approved": approved,
            "messages": state["messages"] + [f"人工审批: {'已批准' if approved else '已拒绝'}"]
        }
    else:
        print(f"  高/危急级别({severity})自动批准，无需人工审批")
        return {
            **state,
            "human_approved": True,
            "messages": state["messages"] + [f"自动批准: {severity}级别"]
        }


def send_email_alert(state: WeatherState) -> dict:
    """节点8a：发送告警（模拟，不实际发邮件）"""
    print("\n" + "=" * 60)
    print("【节点8a】发送告警通知")
    print("=" * 60)

    weather_data = state["weather_data"]
    alert_content = (
        f"气象告警 - {state['city']}\n"
        f"{'=' * 40}\n"
        f"灾害类型: {state['disaster_type']}\n"
        f"严重程度: {state['severity']}\n"
        f"\n当前天气:\n"
        f"  天气描述: {weather_data.get('weather', 'N/A')}\n"
        f"  温度: {weather_data.get('temperature', 'N/A')}°C\n"
        f"  风速: {weather_data.get('wind_speed', 'N/A')} m/s\n"
        f"  湿度: {weather_data.get('humidity', 'N/A')}%\n"
        f"  气压: {weather_data.get('pressure', 'N/A')} hPa\n"
        f"\n响应计划:\n{state['response']}\n"
        f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    print(f"  [模拟邮件发送]\n{alert_content}")

    return {
        **state,
        "messages": state["messages"] + [f"告警通知已发送: {state['city']}"],
        "alerts": state["alerts"] + [f"告警发送: {datetime.now()}"]
    }


def handle_no_approval(state: WeatherState) -> dict:
    """节点8b：处理审批拒绝"""
    print("\n" + "=" * 60)
    print("【节点8b】审批被拒绝")
    print("=" * 60)

    message = (
        f"告警未发送 - {state['city']} - "
        f"严重程度'{state['severity']}'经人工判断为非紧急，审批被拒绝。"
    )
    print(f"  {message}")

    return {
        **state,
        "messages": state["messages"] + [message]
    }


# ===== 路由函数 =====

def route_response(state: WeatherState) -> Literal["emergency_response", "civil_defense_response", "public_works_response"]:
    """条件路由：根据灾害类型和严重程度分发到不同部门"""
    disaster = state["disaster_type"].strip().lower()
    severity = state["severity"].strip().lower()

    print(f"\n  [路由决策] 灾害类型: {disaster}, 严重程度: {severity}")

    if severity in ["critical", "high"]:
        print(f"  → 路由到: 紧急响应部门（高/危急级别）")
        return "emergency_response"
    elif "洪水" in disaster or "风暴" in disaster or "flood" in disaster or "storm" in disaster:
        print(f"  → 路由到: 市政工程部门（洪水/风暴相关）")
        return "public_works_response"
    else:
        print(f"  → 路由到: 民防部门（其他情况）")
        return "civil_defense_response"


def verify_approval_router(state: WeatherState) -> Literal["send_email_alert", "handle_no_approval"]:
    """审批路由：根据人工审批结果决定是否发送告警"""
    if state["human_approved"]:
        print(f"  → 审批通过，路由到: 发送告警")
        return "send_email_alert"
    else:
        print(f"  → 审批拒绝，路由到: 拒绝处理")
        return "handle_no_approval"


# ===== 构建 LangGraph 工作流 =====

def build_workflow():
    """构建并编译工作流图"""
    workflow = StateGraph(WeatherState)

    # 添加节点
    workflow.add_node("get_weather", get_weather_data)
    workflow.add_node("social_media_monitoring", social_media_monitoring)
    workflow.add_node("analyze_disaster", analyze_disaster_type)
    workflow.add_node("assess_severity", assess_severity)
    workflow.add_node("data_logging", data_logging)
    workflow.add_node("emergency_response", emergency_response)
    workflow.add_node("civil_defense_response", civil_defense_response)
    workflow.add_node("public_works_response", public_works_response)
    workflow.add_node("get_human_verification", get_human_verification)
    workflow.add_node("send_email_alert", send_email_alert)
    workflow.add_node("handle_no_approval", handle_no_approval)

    # 添加边 — 线性部分
    workflow.add_edge("get_weather", "social_media_monitoring")
    workflow.add_edge("social_media_monitoring", "analyze_disaster")
    workflow.add_edge("analyze_disaster", "assess_severity")
    workflow.add_edge("assess_severity", "data_logging")

    # 条件路由1：根据灾害类型和严重程度分发
    workflow.add_conditional_edges("data_logging", route_response)

    # 紧急响应直接发邮件（不需人工审批）
    workflow.add_edge("emergency_response", "send_email_alert")

    # 民防和市政工程需要人工审批
    workflow.add_edge("civil_defense_response", "get_human_verification")
    workflow.add_edge("public_works_response", "get_human_verification")

    # 条件路由2：根据审批结果决定
    workflow.add_conditional_edges("get_human_verification", verify_approval_router)

    # 结束节点
    workflow.add_edge("send_email_alert", END)
    workflow.add_edge("handle_no_approval", END)

    # 设置入口
    workflow.set_entry_point("get_weather")

    return workflow.compile()


# ===== 主函数 =====

def main():
    print("=" * 60)
    print("  气象灾害应急管理系统（LangGraph 框架版）")
    print("=" * 60)

    app = build_workflow()

    print("\n选择运行模式:")
    print("  1. 高严重度测试（自动走紧急响应，不需人工审批）")
    print("  2. 中严重度测试（需要人工审批）")
    print("  3. 低严重度测试（需要人工审批）")

    choice = input("\n请选择 (1/2/3，默认1): ").strip() or "1"
    scenario_map = {"1": "high", "2": "medium", "3": "low"}
    scenario = scenario_map.get(choice, "high")

    city = input("输入城市名称（默认: 上海）: ").strip() or "上海"

    initial_state: WeatherState = {
        "city": city,
        "weather_data": get_simulated_weather_data(scenario),
        "disaster_type": "",
        "severity": "",
        "response": "",
        "messages": [],
        "alerts": [],
        "social_media_reports": [],
        "human_approved": False
    }

    print(f"\n开始执行气象灾害分析: {city} ({scenario}级别场景)")
    print("=" * 60)

    result = app.invoke(initial_state)

    # 打印最终结果
    print("\n" + "=" * 60)
    print("【执行完毕】最终状态汇总")
    print("=" * 60)
    print(f"  城市: {result['city']}")
    print(f"  灾害类型: {result['disaster_type']}")
    print(f"  严重程度: {result['severity']}")
    print(f"  人工审批: {'已批准' if result['human_approved'] else '已拒绝'}")
    print(f"  告警记录: {result['alerts']}")
    print(f"\n  执行日志:")
    for msg in result["messages"]:
        print(f"    - {msg}")


if __name__ == "__main__":
    main()
