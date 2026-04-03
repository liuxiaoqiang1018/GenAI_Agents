"""
第29课：气象灾害管理 AI 代理（透明调试版）

不使用任何框架，纯手写 Agent 流程，打印每个阶段的完整 Prompt 和调用日志。
让你看清 LangGraph 条件路由 + Human-in-the-Loop 的内部机制。
"""

import os
import json
import time
import random
import httpx
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ===== 配置 =====
API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_RETRIES = 3


# ===== LLM 调用 =====
def call_llm(messages: list, temperature: float = 0.7) -> str:
    """标准 LLM 调用模板"""
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
            resp = httpx.post(url, json=payload, headers=headers, timeout=300)
            data = resp.json()
            choices = data.get("choices")
            if not choices:
                raise ValueError(f"空响应: {data}")
            content = choices[0]["message"]["content"]
            # 清理模型思考过程标签
            import re
            content = re.sub(r'<think>[\s\S]*?</think>\s*', '', content).strip()
            return content
        except Exception as e:
            print(f"    !! LLM调用失败(第{attempt+1}次): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep((attempt + 1) * 3)
            else:
                raise


# ===== 模拟天气数据 =====
def get_simulated_weather_data(scenario: str = "high") -> dict:
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


def main():
    print("=" * 70)
    print("  气象灾害应急管理系统（透明调试版 - 无框架）")
    print("=" * 70)

    # 选择场景
    print("\n选择测试场景:")
    print("  1. 高严重度（自动走紧急响应，不需人工审批）")
    print("  2. 中严重度（需要人工审批）")
    print("  3. 低严重度（需要人工审批）")
    choice = input("\n请选择 (1/2/3，默认1): ").strip() or "1"
    scenario_map = {"1": "high", "2": "medium", "3": "low"}
    scenario = scenario_map.get(choice, "high")
    city = input("输入城市名称（默认: 上海）: ").strip() or "上海"

    # 初始化状态（对应 LangGraph 的 WeatherState）
    state = {
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

    # ================================================================
    # 阶段1：获取天气数据
    # ================================================================
    print("\n" + "=" * 70)
    print("【阶段1】获取天气数据")
    print("=" * 70)
    print(f"  城市: {state['city']}")
    print(f"  天气数据（模拟）:")
    for k, v in state["weather_data"].items():
        print(f"    {k}: {v}")
    state["messages"].append(f"天气数据获取成功: {city}")

    # ================================================================
    # 阶段2：社交媒体监控
    # ================================================================
    print("\n" + "=" * 70)
    print("【阶段2】社交媒体监控（模拟）")
    print("=" * 70)

    simulated_reports = [
        "当地报告水位上涨，出现轻微洪水。",
        "大风导致部分城区停电。",
        "市民反映高温导致身体不适加剧。",
        "社交媒体报告严重风暴破坏了当地基础设施。",
        "暴雨导致交通严重中断。",
        "目前未发现与天气相关的异常社交媒体报告。"
    ]
    report = random.choice(simulated_reports)
    state["social_media_reports"].append(report)
    print(f"  随机选取的社交媒体报告: {report}")
    state["messages"].append(f"社交媒体监控: {report}")

    # ================================================================
    # 阶段3：分析灾害类型（LLM调用）
    # ================================================================
    print("\n" + "=" * 70)
    print("【阶段3】分析灾害类型 — LLM 调用")
    print("=" * 70)

    weather_data = state["weather_data"]
    social_info = f"\n社交媒体报告: {'; '.join(state['social_media_reports'])}"

    disaster_prompt = (
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

    disaster_messages = [{"role": "user", "content": disaster_prompt}]

    print(f"\n  >>> 发送给 LLM 的完整 Prompt <<<")
    print(f"  {'-' * 60}")
    print(f"  [user]: {disaster_prompt}")
    print(f"  {'-' * 60}")

    print(f"\n  >>> 调用 LLM: POST {API_BASE_URL}/chat/completions <<<")
    print(f"  模型: {MODEL_NAME}")
    print(f"  temperature: 0.3")

    disaster_type = call_llm(disaster_messages, temperature=0.3)
    state["disaster_type"] = disaster_type.strip()

    print(f"\n  >>> LLM 响应 <<<")
    print(f"  {'-' * 60}")
    print(f"  {disaster_type}")
    print(f"  {'-' * 60}")

    state["messages"].append(f"灾害类型: {state['disaster_type']}")

    # ================================================================
    # 阶段4：评估严重程度（LLM调用）
    # ================================================================
    print("\n" + "=" * 70)
    print("【阶段4】评估严重程度 — LLM 调用")
    print("=" * 70)

    severity_prompt = (
        f"已识别的灾害类型: '{state['disaster_type']}'\n"
        f"天气条件:\n"
        f"- 天气: {weather_data.get('weather', 'N/A')}\n"
        f"- 风速: {weather_data.get('wind_speed', 'N/A')} m/s\n"
        f"- 温度: {weather_data.get('temperature', 'N/A')}°C\n"
        f"- 气压: {weather_data.get('pressure', 'N/A')} hPa\n"
        f"请评估严重程度，只回复以下四个级别之一: Critical、High、Medium、Low"
    )

    severity_messages = [{"role": "user", "content": severity_prompt}]

    print(f"\n  >>> 发送给 LLM 的完整 Prompt <<<")
    print(f"  {'-' * 60}")
    print(f"  [user]: {severity_prompt}")
    print(f"  {'-' * 60}")

    print(f"\n  >>> 调用 LLM: POST {API_BASE_URL}/chat/completions <<<")

    severity = call_llm(severity_messages, temperature=0.3)
    state["severity"] = severity.strip()

    print(f"\n  >>> LLM 响应 <<<")
    print(f"  {'-' * 60}")
    print(f"  {severity}")
    print(f"  {'-' * 60}")

    state["messages"].append(f"严重程度: {state['severity']}")

    # ================================================================
    # 阶段5：数据日志记录
    # ================================================================
    print("\n" + "=" * 70)
    print("【阶段5】数据日志记录")
    print("=" * 70)

    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "city": state["city"],
        "weather_data": state["weather_data"],
        "disaster_type": state["disaster_type"],
        "severity": state["severity"],
    }
    print(f"  日志内容: {json.dumps(log_data, ensure_ascii=False, indent=2)}")

    log_file = os.path.join(os.path.dirname(__file__), "disaster_log_debug.txt")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
    print(f"  已写入: {log_file}")

    state["messages"].append("数据日志记录完成")

    # ================================================================
    # 阶段6：条件路由决策（这就是 LangGraph 的 add_conditional_edges）
    # ================================================================
    print("\n" + "=" * 70)
    print("【阶段6】条件路由决策")
    print("=" * 70)

    disaster_lower = state["disaster_type"].strip().lower()
    severity_lower = state["severity"].strip().lower()

    print(f"  灾害类型: {state['disaster_type']}")
    print(f"  严重程度: {state['severity']}")
    print()
    print(f"  路由规则:")
    print(f"    if severity in [critical, high]  → 紧急响应（自动发告警）")
    print(f"    elif '洪水/flood/storm/风暴' in disaster → 市政工程（需人工审批）")
    print(f"    else → 民防响应（需人工审批）")
    print()

    if severity_lower in ["critical", "high"]:
        route = "emergency_response"
        department = "紧急响应部门"
        needs_human = False
    elif any(kw in disaster_lower for kw in ["洪水", "风暴", "flood", "storm"]):
        route = "public_works_response"
        department = "市政工程部门"
        needs_human = True
    else:
        route = "civil_defense_response"
        department = "民防部门"
        needs_human = True

    print(f"  >>> 路由结果: {route} ({department})")
    print(f"  >>> 需要人工审批: {'是' if needs_human else '否'}")

    # ================================================================
    # 阶段7：生成响应计划（LLM调用）
    # ================================================================
    print("\n" + "=" * 70)
    print(f"【阶段7】{department} - 生成响应计划 — LLM 调用")
    print("=" * 70)

    if route == "emergency_response":
        response_prompt = (
            f"为{state['city']}的{state['disaster_type']}灾害（严重程度: {state['severity']}）"
            f"制定一份紧急响应计划。包括立即需要采取的行动。请用中文回答，控制在200字以内。"
        )
    elif route == "public_works_response":
        response_prompt = (
            f"为{state['city']}的{state['disaster_type']}灾害（严重程度: {state['severity']}）"
            f"制定一份市政工程响应计划。重点关注基础设施保护。请用中文回答，控制在200字以内。"
        )
    else:
        response_prompt = (
            f"为{state['city']}的{state['disaster_type']}灾害（严重程度: {state['severity']}）"
            f"制定一份民防响应计划。重点关注公共安全措施。请用中文回答，控制在200字以内。"
        )

    response_messages = [{"role": "user", "content": response_prompt}]

    print(f"\n  >>> 发送给 LLM 的完整 Prompt <<<")
    print(f"  {'-' * 60}")
    print(f"  [user]: {response_prompt}")
    print(f"  {'-' * 60}")

    print(f"\n  >>> 调用 LLM <<<")

    response_plan = call_llm(response_messages)
    state["response"] = response_plan

    print(f"\n  >>> LLM 响应 <<<")
    print(f"  {'-' * 60}")
    print(f"  {response_plan}")
    print(f"  {'-' * 60}")

    state["messages"].append(f"{department}响应计划已生成")

    # ================================================================
    # 阶段8：人工审批（Human-in-the-Loop）
    # ================================================================
    if needs_human:
        print("\n" + "=" * 70)
        print("【阶段8】人工审批（Human-in-the-Loop）")
        print("=" * 70)
        print(f"  这就是 LangGraph 的 Human-in-the-Loop 机制:")
        print(f"  框架在此暂停图的执行，等待人类输入，再决定走哪条边。")
        print()
        print(f"  城市: {state['city']}")
        print(f"  灾害类型: {state['disaster_type']}")
        print(f"  严重程度: {state['severity']}")
        print(f"  响应计划: {state['response'][:150]}...")
        print()

        while True:
            user_input = input("  输入 'y' 批准发送告警，'n' 拒绝: ").lower().strip()
            if user_input in ['y', 'n']:
                state["human_approved"] = (user_input == 'y')
                break
            print("  请输入 'y' 或 'n'")

        print(f"  人工审批结果: {'已批准' if state['human_approved'] else '已拒绝'}")
        state["messages"].append(f"人工审批: {'已批准' if state['human_approved'] else '已拒绝'}")

        # 审批路由（对应 verify_approval_router）
        print(f"\n  [审批路由] human_approved={state['human_approved']}")
        if state["human_approved"]:
            print(f"  → 路由到: 发送告警")
        else:
            print(f"  → 路由到: 拒绝处理")

    else:
        state["human_approved"] = True
        print(f"\n  高/危急级别，自动跳过人工审批")
        state["messages"].append(f"自动批准: {severity_lower}级别")

    # ================================================================
    # 阶段9：发送告警 或 拒绝处理
    # ================================================================
    if state["human_approved"]:
        print("\n" + "=" * 70)
        print("【阶段9】发送告警通知")
        print("=" * 70)

        alert_content = (
            f"气象告警 - {state['city']}\n"
            f"{'=' * 40}\n"
            f"灾害类型: {state['disaster_type']}\n"
            f"严重程度: {state['severity']}\n"
            f"\n当前天气:\n"
            f"  天气描述: {state['weather_data'].get('weather', 'N/A')}\n"
            f"  温度: {state['weather_data'].get('temperature', 'N/A')}°C\n"
            f"  风速: {state['weather_data'].get('wind_speed', 'N/A')} m/s\n"
            f"  湿度: {state['weather_data'].get('humidity', 'N/A')}%\n"
            f"  气压: {state['weather_data'].get('pressure', 'N/A')} hPa\n"
            f"\n响应计划:\n{state['response']}\n"
            f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(f"  [模拟邮件内容]\n{alert_content}")
        state["alerts"].append(f"告警发送: {datetime.now()}")
        state["messages"].append(f"告警通知已发送: {state['city']}")

    else:
        print("\n" + "=" * 70)
        print("【阶段9】审批被拒绝 - 告警未发送")
        print("=" * 70)
        message = f"告警未发送 - {state['city']} - 严重程度'{state['severity']}'经人工判断为非紧急"
        print(f"  {message}")
        state["messages"].append(message)

    # ================================================================
    # 最终汇总
    # ================================================================
    print("\n" + "=" * 70)
    print("【执行完毕】最终状态汇总")
    print("=" * 70)
    print(f"  城市: {state['city']}")
    print(f"  灾害类型: {state['disaster_type']}")
    print(f"  严重程度: {state['severity']}")
    print(f"  路由部门: {department}")
    print(f"  人工审批: {'已批准' if state['human_approved'] else '已拒绝'}")
    print(f"  告警记录: {state['alerts']}")
    print(f"\n  完整执行日志:")
    for i, msg in enumerate(state["messages"], 1):
        print(f"    {i}. {msg}")

    print("\n" + "=" * 70)
    print("  调试总结：LangGraph 在本课做了什么")
    print("=" * 70)
    print(f"  1. StateGraph 定义了 WeatherState，在节点间传递共享状态")
    print(f"  2. add_edge() 定义了线性流程：天气→监控→分析→评估→日志")
    print(f"  3. add_conditional_edges() 实现条件路由：")
    print(f"     - route_response: 根据灾害类型和严重程度分发到3个部门")
    print(f"     - verify_approval_router: 根据审批结果决定发告警或拒绝")
    print(f"  4. Human-in-the-Loop: get_human_verification 暂停等待人工输入")
    print(f"  5. 本质上就是一个 if-elif-else 的分支 + input() 的暂停")
    print(f"     框架把这些组织成了一个可视化的图结构")


if __name__ == "__main__":
    main()
