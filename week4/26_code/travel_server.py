from fastmcp import FastMCP

# 初始化 MCP 服务
mcp = FastMCP("TravelPlanner")


@mcp.tool()
async def search_flights(origin: str, destination: str, date: str, cabin_class: str = "Economy"):
    """
    查询航班信息。
    :param origin: 出发城市
    :param destination: 目的地城市
    :param date: 出发日期 (YYYY-MM-DD)
    :param cabin_class: 舱位 (Economy, Business, First)
    """
    # 模拟 API 调用逻辑
    return {
        "flights": [
            {"flight_no": "CA1234", "price": "¥1500", "departure": "10:00"},
            {"flight_no": "MU5678", "price": "¥1850", "departure": "14:30"}
        ],
        "status": "success"
    }

@mcp.tool()
async def find_hotels(city: str, checkin_date: str, budget_range: str):
    """
    查询酒店及价格。
    :param city: 目标城市
    :param checkin_date: 入住日期
    :param budget_range: 预算范围 (如 "500-1000")
    """
    return [
        {"name": "城市中心酒店", "rating": 4.5, "price": "¥680"},
        {"name": "精品民宿", "rating": 4.8, "price": "¥920"}
    ]

@mcp.tool()
async def get_weather(city: str, date: str):
    """
    获取目的地天气。
    :param city: 城市名
    :param date: 日期
    """
    return {"city": city, "date": date, "forecast": "晴朗", "temp": "15°C - 22°C"}

@mcp.tool()
async def plan_route(start: str, end: str, mode: str = "transit"):
    """
    规划景点间的交通路线。
    :param start: 起点名称
    :param end: 终点名称
    :param mode: 出行方式 (driving, walking, transit)
    """
    return f"从 {start} 到 {end} 的 {mode} 方案：预计耗时 30 分钟，距离 5 公里。"

# --- RESOURCES (资源：提供结构化数据参考) ---
@mcp.resource("attractions://{city}")
def get_attractions(city: str) -> str:
    """获取城市热门景点列表"""
    data = {
        "西安": "1. 秦始皇陵兵马俑\n2. 大雁塔\n3. 西安城墙",
        "巴黎": "1. 埃菲尔铁塔\n2. 卢浮宫\n3. 凯旋门"
    }
    return data.get(city, "暂无景点数据。")

@mcp.resource("guides://{city}")
def get_city_guide(city: str) -> str:
    """
    获取城市的背景文化介绍。
    包含：文化底蕴、当地礼仪、必吃美食。
    """
    # 模拟从数据库或 Markdown 文件加载内容
    guides = {
        "西安": """
# 西安旅游深度指南

## 🏛 文化底蕴
西安（古称长安）是十三朝古都，丝绸之路的起点。这里的每一寸土地都充满了历史感。

## 🥢 必吃美食
* **肉夹馍**：被称为“中国式汉堡”。
* **羊肉泡馍**：推荐去回民街尝试，记得自己动手掰馍。
* **凉皮**：酸辣爽口，夏季必备。

## ⚠️ 旅游礼仪与建议
* **尊重习俗**：在回民街游览时，请尊重穆斯林的宗教信仰和生活习惯。
* **避开高峰**：城墙骑行建议选在傍晚，可以看落日，且避开高温。
        """,
        "巴黎": """
# 巴黎旅游深度指南

## 🎨 文化底蕴
艺术之都与时尚之都。不仅有海明威笔下“流动的盛宴”，还有世界顶级的艺术馆。

## 🥐 必吃美食
* **法式长棍面包 (Baguette)**：随处可见的灵魂美食。
* **马卡龙**：Ladurée 或 Pierre Hermé 是经典之选。
* **油封鸭 (Confit de Canard)**：经典的法式主菜。

## ⚠️ 旅游礼仪与建议
* **问候礼仪**：进入商店时，先说一句 "Bonjour" 是非常基本的礼貌。
* **用餐节奏**：法国人用餐时间较长，不要催促服务员结账。
        """
    }
    
    return guides.get(city, f"抱歉，目前还没有关于 {city} 的详细文化指南，建议查阅维基百科。")


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8080)