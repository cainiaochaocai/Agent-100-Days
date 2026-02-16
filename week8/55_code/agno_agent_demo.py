"""
Week 8 基于 Agno 开发 Agent：天气 + 当前时间工具，端到端运行

- 使用 Agno Agent(model=..., tools=[...], instructions=...)
- 工具为普通函数，由框架自动转 schema 并执行“模型 → 工具 → 模型”循环
- 需先安装：pip install agno；并配置 OPENAI_API_KEY

建议在项目根目录执行：python week8/55_code/agno_agent_demo.py
"""

from dotenv import load_dotenv
load_dotenv()

import os
assert os.getenv("DASHSCOPE_API_KEY"), "请先在项目根目录 .env 中配置 DASHSCOPE_API_KEY"

from agno.agent import Agent
from agno.models.dashscope import DashScope


def get_weather(city: str) -> str:
    """当用户询问某地天气时，使用此工具查询该城市的天气。
    Args:
        city (str): 要查询天气的城市名称。
    """
    fake_db = {"北京": "晴，25°C", "上海": "多云，22°C", "深圳": "阴，28°C"}
    return fake_db.get(city, f"{city}：暂无数据")


def get_current_time() -> str:
    """当用户问当前时间或日期时，使用此工具。"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


agent = Agent(
    model=DashScope(
        id="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    tools=[get_weather, get_current_time],
    instructions="你是助手。根据用户问题选择合适的工具获取信息，然后给出简洁回答。",
    markdown=True,
)


def main() -> None:
    print("Agno Agent 示例（天气 + 当前时间）")
    print("输入问题，Agent 将自动选择工具并多轮调用后回答。输入 quit 退出。\n")
    while True:
        user_input = input("你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        agent.print_response(user_input, stream=True)
        print()


if __name__ == "__main__":
    main()
