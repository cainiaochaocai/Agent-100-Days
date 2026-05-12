"""
Week 8 基于 LangChain 开发 Agent：多轮推理与工具调用

- 定义两个工具：get_weather、get_current_time
- 使用 llm.bind_tools(tools) 绑定，手写 Agent 循环直到无 tool_calls 或达 max_steps
- 接收用户输入 → 多轮推理与工具调用 → 返回最终结果

运行前在项目根目录配置 .env 中的 OPENAI_API_KEY。
建议在项目根目录执行：python week8/54_code/langchain_agent_demo.py
"""

from dotenv import load_dotenv
load_dotenv()

import os
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

assert os.getenv("OPENAI_API_KEY"), "请先在项目根目录 .env 中配置 OPENAI_API_KEY"


@tool
def get_weather(city: str) -> str:
    """当用户询问某地天气时，使用此工具查询该城市的天气。"""
    fake_db = {"北京": "晴，25°C", "上海": "多云，22°C", "深圳": "阴，28°C"}
    return fake_db.get(city, f"{city}：暂无数据")


@tool
def get_current_time() -> str:
    """当用户问当前时间或日期时，使用此工具。"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


llm = ChatOpenAI(model="qwen-plus", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
tools = [get_weather, get_current_time]
llm_with_tools = llm.bind_tools(tools)
tool_map = {t.name: t for t in tools}

SYSTEM = "你是助手。根据用户问题选择合适的工具获取信息，然后给出简洁回答。若无需工具可直接回答。"


def run_agent(user_input: str, system: str | None = SYSTEM, max_steps: int = 10) -> str:
    """手写 Agent 循环：多轮推理与工具调用，直到无 tool_calls 或达到 max_steps。"""
    messages = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=user_input))

    for step in range(max_steps):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            return response.content or "（无回复）"

        for call in response.tool_calls:
            name = call.get("name")
            args = call.get("args") or {}
            tid = call.get("id", "")
            if name not in tool_map:
                messages.append(ToolMessage(content=f"未知工具: {name}", tool_call_id=tid))
                continue
            try:
                result = tool_map[name].invoke(args)
                messages.append(ToolMessage(content=str(result), tool_call_id=tid))
            except Exception as e:
                messages.append(ToolMessage(content=f"执行失败: {e}", tool_call_id=tid))

    return "达到最大步数，未得到最终回复。"


def main() -> None:
    print("LangChain Agent 示例（天气 + 当前时间）")
    print("输入问题，Agent 将自动选择工具并多轮调用后回答。输入 quit 退出。\n")
    while True:
        user_input = input("你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        reply = run_agent(user_input)
        print("助手:", reply, "\n")


if __name__ == "__main__":
    main()
