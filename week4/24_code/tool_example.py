from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv

load_dotenv()

assert os.getenv("OPENAI_API_KEY"), "请先配置 OPENAI_API_KEY"

@tool
def get_weather(city: str) -> str:
    """当用户询问实时天气情况时，使用该工具获取指定城市的天气信息。"""
    fake_weather_db = {
        "上海": "小雨，湿度 90%",
        "北京": "晴，温度 10°C"
    }
    return fake_weather_db.get(city, "未查询到该城市天气")


llm = ChatOpenAI(
    model="qwen-max", 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

tools = [get_weather]
tool_map = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)


response = llm_with_tools.invoke([
    HumanMessage(content="帮我查一下今天上海的天气，顺便判断适不适合洗车")
])

print(response)

tool_messages = []

for call in response.tool_calls:
    tool_name = call["name"]
    tool_args = call["args"]

    tool = tool_map[tool_name]
    result = tool.invoke(tool_args)

    tool_messages.append(
        ToolMessage(
            content=str(result),
            tool_call_id=call["id"]
        )
    )

print(tool_messages)

final_response = llm_with_tools.invoke([
    HumanMessage(content="帮我查一下今天上海的天气，顺便判断适不适合洗车"),
    response,
    *tool_messages
])

print(final_response.content)