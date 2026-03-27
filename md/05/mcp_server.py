"""
第5课 - MCP Server（工具服务端）

这个 Server 暴露两个工具：
1. 查询城市信息（模拟数据，不依赖外部 API）
2. 单位换算（人民币↔美元，公里↔英里等）

启动方式：python mcp_server.py
Host 通过 stdio 连接这个进程来发现和调用工具。
"""

from mcp.server.fastmcp import FastMCP

# 创建 MCP Server
mcp = FastMCP("city_tools")


# ========== 模拟数据 ==========

CITY_DATA = {
    "北京": {
        "人口": "2189万",
        "面积": "16410平方公里",
        "GDP": "43760亿元",
        "著名景点": ["故宫", "长城", "天坛", "颐和园"],
        "气候": "温带季风气候",
        "平均气温": "13°C",
    },
    "上海": {
        "人口": "2487万",
        "面积": "6341平方公里",
        "GDP": "47218亿元",
        "著名景点": ["外滩", "东方明珠", "豫园", "南京路"],
        "气候": "亚热带季风气候",
        "平均气温": "17°C",
    },
    "深圳": {
        "人口": "1768万",
        "面积": "1997平方公里",
        "GDP": "34606亿元",
        "著名景点": ["世界之窗", "欢乐谷", "大梅沙", "莲花山"],
        "气候": "亚热带海洋性气候",
        "平均气温": "23°C",
    },
    "成都": {
        "人口": "2119万",
        "面积": "14335平方公里",
        "GDP": "22075亿元",
        "著名景点": ["武侯祠", "锦里", "大熊猫基地", "宽窄巷子"],
        "气候": "亚热带湿润气候",
        "平均气温": "16°C",
    },
    "杭州": {
        "人口": "1237万",
        "面积": "16850平方公里",
        "GDP": "20059亿元",
        "著名景点": ["西湖", "灵隐寺", "千岛湖", "宋城"],
        "气候": "亚热带季风气候",
        "平均气温": "17°C",
    },
}


# ========== 工具1：查询城市信息 ==========

@mcp.tool()
async def query_city(city_name: str) -> str:
    """
    查询中国城市的基本信息，包括人口、面积、GDP、著名景点、气候等。

    Parameters:
    - city_name: 城市名称（如"北京"、"上海"、"深圳"、"成都"、"杭州"）

    Returns:
    - 城市信息的格式化文本
    """
    if city_name not in CITY_DATA:
        available = "、".join(CITY_DATA.keys())
        return f"未找到城市'{city_name}'的信息。目前支持的城市：{available}"

    info = CITY_DATA[city_name]
    lines = [f"【{city_name}】城市信息："]
    for key, value in info.items():
        if isinstance(value, list):
            lines.append(f"  {key}：{'、'.join(value)}")
        else:
            lines.append(f"  {key}：{value}")

    return "\n".join(lines)


# ========== 工具2：单位换算 ==========

CONVERSION_RATES = {
    "人民币_美元": 0.14,
    "美元_人民币": 7.15,
    "公里_英里": 0.621,
    "英里_公里": 1.609,
    "千克_磅": 2.205,
    "磅_千克": 0.454,
    "摄氏度_华氏度": lambda x: x * 9 / 5 + 32,
    "华氏度_摄氏度": lambda x: (x - 32) * 5 / 9,
}


@mcp.tool()
async def convert_unit(value: float, from_unit: str, to_unit: str) -> str:
    """
    单位换算工具，支持货币、距离、重量、温度的换算。

    Parameters:
    - value: 要换算的数值
    - from_unit: 原始单位（如"人民币"、"公里"、"千克"、"摄氏度"）
    - to_unit: 目标单位（如"美元"、"英里"、"磅"、"华氏度"）

    Returns:
    - 换算结果的格式化文本
    """
    key = f"{from_unit}_{to_unit}"

    if key not in CONVERSION_RATES:
        supported = [k.replace("_", " → ") for k in CONVERSION_RATES.keys()]
        return f"不支持 {from_unit} → {to_unit} 的换算。支持的换算：\n" + "\n".join(supported)

    rate = CONVERSION_RATES[key]

    if callable(rate):
        result = rate(value)
    else:
        result = value * rate

    return f"{value} {from_unit} = {result:.2f} {to_unit}"


# ========== 启动 Server ==========

if __name__ == "__main__":
    mcp.run()
