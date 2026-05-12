"""
Week 11 为 Agent 加入 HITL（审批型）：执行前需人批准

- 在 Week 8/9 风格 Agent 基础上，对「需审批」工具在执行前暂停，呈现意图/动作/参数，等人 y/n 后再执行或拒绝
- 工具：get_weather、get_current_time、search_order（将 search_order 设为需审批，模拟敏感操作）
- 流程：Agent 决定调用需审批工具 → 暂停 → 打印待执行信息 → 等待用户输入 → 通过则执行并继续，拒绝则注入原因并继续

运行前在项目根目录配置 .env 中的 OPENAI_API_KEY。
在项目根目录执行：python week11/76_code/agent_with_hitl_demo.py
"""

from dotenv import load_dotenv
load_dotenv()

import os
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

assert os.getenv("OPENAI_API_KEY"), "请先在项目根目录 .env 中配置 OPENAI_API_KEY"
api_key = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# ---------- 工具定义 ----------

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


@tool
def search_order(order_id: str) -> str:
    """根据订单号查询订单状态。仅支持已知订单号（示例中仅 ORD001 有数据）。涉及订单信息，需人工确认后执行。"""
    if order_id.strip().upper() == "ORD001":
        return "订单 ORD001：已支付，配送中"
    return f"错误：订单不存在（{order_id}）。请确认订单号是否正确。"


# 标记哪些工具需要执行前审批（HITL Approval）
TOOLS_REQUIRING_APPROVAL = {"search_order"}

llm = ChatOpenAI(model="qwen-plus", base_url=BASE_URL, api_key=api_key)
tools = [get_weather, get_current_time, search_order]
llm_with_tools = llm.bind_tools(tools)
tool_map = {t.name: t for t in tools}

SYSTEM = (
    "你是助手。根据用户问题选择合适的工具获取信息，然后给出简洁回答。"
    "若用户拒绝某次工具执行，请根据拒绝原因调整或结束，不要重复相同请求。"
)


def request_human_approval(goal: str, tool_name: str, args: dict) -> str:
    """
    审批型 HITL：呈现当前意图、待执行动作与参数、风险提示，等待人输入。
    返回 "approve" | "reject" 或 "reject:原因"
    """
    print("\n  ---------- [HITL 审批] ----------")
    print("  当前意图：", goal[:100] + ("..." if len(goal) > 100 else ""))
    print("  待执行动作：", tool_name)
    print("  参数：", args)
    print("  风险提示：涉及订单查询，请确认是否允许执行。")
    print("  输入 y 通过 / n 或 n:原因 拒绝 ----------")
    raw = input("  你的决策 (y/n): ").strip().lower()
    if raw.startswith("y") or raw == "yes":
        return "approve"
    if raw.startswith("n") or raw == "no":
        if ":" in raw:
            return "reject:" + raw.split(":", 1)[1].strip()
        return "reject:用户拒绝执行"
    return "reject:无效输入，视为拒绝"


def run_agent_with_hitl(user_input: str, system: str | None = SYSTEM, max_steps: int = 10, verbose: bool = True) -> str:
    """带审批型 HITL 的 Agent：对需审批工具先 request_human_approval，再根据决策执行或注入拒绝原因。"""
    messages = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=user_input))

    for step in range(max_steps):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            return (response.content or "").strip()

        tool_messages = []
        for call in response.tool_calls:
            name = call.get("name")
            args = call.get("args") or {}
            tid = call.get("id", "")

            if name not in tool_map:
                tool_messages.append(ToolMessage(content=f"未知工具: {name}", tool_call_id=tid))
                continue

            # ---------- HITL：需审批工具先暂停，等人决策 ----------
            if name in TOOLS_REQUIRING_APPROVAL:
                decision = request_human_approval(user_input, name, args)
                if decision == "approve":
                    try:
                        result = tool_map[name].invoke(args)
                        tool_messages.append(ToolMessage(content=str(result), tool_call_id=tid))
                    except Exception as e:
                        tool_messages.append(ToolMessage(content=f"执行失败: {e}", tool_call_id=tid))
                else:
                    reason = decision.replace("reject:", "").strip() if "reject:" in decision else "用户拒绝执行"
                    tool_messages.append(
                        ToolMessage(
                            content=f"[人工拒绝] {reason}。请根据此反馈调整回复或结束，不要再次请求执行该操作。",
                            tool_call_id=tid,
                        )
                    )
                continue

            # 不需审批的工具直接执行
            try:
                result = tool_map[name].invoke(args)
                tool_messages.append(ToolMessage(content=str(result), tool_call_id=tid))
            except Exception as e:
                tool_messages.append(ToolMessage(content=f"执行失败: {e}", tool_call_id=tid))

        messages.extend(tool_messages)

    return "达到最大步数，未得到最终回复。"


def main() -> None:
    print("Week 11 带 HITL（审批型）的 Agent 示例")
    print("search_order 为需审批工具，执行前会暂停并等待你输入 y/n。")
    print("输入 quit 退出。\n")
    while True:
        user_input = input("你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        reply = run_agent_with_hitl(user_input, verbose=True)
        print("助手:", reply, "\n")


if __name__ == "__main__":
    main()
