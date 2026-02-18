"""
Week 9 为 Agent 加入反思节点：可配置触发条件 + 继续/调整/停止分支

- 在 Week 8 LangChain Agent 基础上增加「反思」独立模块与触发逻辑
- 工具：get_weather、get_current_time、search_order（仅 ORD001 成功，便于触发失败反思）
- 配置：失败时反思、每 N 步反思（可选）
- 反思输出：decision (continue/adjust/stop) + reason + suggestion + stop_type，驱动主循环分支

运行前在项目根目录配置 .env（如 OPENAI_API_KEY 与 Week 8 一致）。
在项目根目录执行：python week9/62_code/agent_with_reflection_demo.py
"""

from dotenv import load_dotenv
load_dotenv()

import os
import json
import re
from typing import Any

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

api_key = os.getenv("OPENAI_API_KEY")
assert api_key, "请在项目根目录 .env 中配置 OPENAI_API_KEY"

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
    """根据订单号查询订单状态。仅支持已知订单号（示例中仅 ORD001 有数据）。"""
    if order_id.strip().upper() == "ORD001":
        return "订单 ORD001：已支付，配送中"
    return f"错误：订单不存在（{order_id}）。请确认订单号是否正确。"


# ---------- 模型与绑定 ----------

llm = ChatOpenAI(model="qwen-plus", base_url=BASE_URL, api_key=api_key)
tools = [get_weather, get_current_time, search_order]
llm_with_tools = llm.bind_tools(tools)
tool_map = {t.name: t for t in tools}

# 反思用同一模型，但不绑定工具，用于生成结构化反思结论
llm_reflection = ChatOpenAI(model="qwen-plus", base_url=BASE_URL, api_key=api_key)

SYSTEM = (
    "你是助手。根据用户问题选择合适的工具获取信息，然后给出简洁回答。"
    "若工具返回错误信息，请根据错误调整（如确认参数或换策略），不要重复相同错误。"
)


# ---------- 反思配置（贴近实际开发：可改配置观察行为） ----------

REFLECTION_CONFIG = {
    "reflection_enabled": True,
    "reflect_on_failure": True,
    "max_failures_before_reflect": 1,
    "reflect_every_n_steps": None,  # 例如 3 表示每 3 步反思一次；None 表示不启用
}


def _tool_message_is_failure(content: str) -> bool:
    """判断该步工具执行是否视为「失败」（用于触发反思）。"""
    if not content:
        return False
    return "错误：" in content or "执行失败:" in content or "失败" in content


def should_trigger_reflection(
    step: int,
    last_step_failed: bool,
    failure_count: int,
    timed_out: bool = False,
) -> bool:
    """是否在本步后触发反思。"""
    cfg = REFLECTION_CONFIG
    if not cfg.get("reflection_enabled", True):
        return False
    if cfg.get("reflect_on_failure") and last_step_failed:
        if failure_count >= cfg.get("max_failures_before_reflect", 1):
            return True
    if cfg.get("reflect_on_timeout") and timed_out:
        return True
    n = cfg.get("reflect_every_n_steps")
    if n is not None and step > 0 and step % n == 0:
        return True
    return False


REFLECTION_SYSTEM = """你是一个「反思」模块，不执行工具，只根据当前状态和最近执行结果做元级判断。
请严格输出一个 JSON 对象，不要其他文字。字段如下：
- decision: 只能是 "continue" | "adjust" | "stop" 之一
- reason: 简短原因（一句话）
- suggestion: 仅当 decision 为 "adjust" 时给出具体建议（如换参数、换工具、先确认再执行）；否则为 null
- stop_type: 仅当 decision 为 "stop" 时为 "done" | "failed" | "human_required" 之一；否则为 null

判断准则：
- 若上一步只是小问题、可继续按当前思路执行，选 continue
- 若上一步失败或结果异常、需要改变做法（如改参数、换工具），选 adjust 并写清 suggestion
- 若认为不应再继续自动执行（如多次失败、或需人工确认），选 stop，并设 stop_type（failed 或 human_required）
"""


def run_reflection(
    goal: str,
    step: int,
    max_steps: int,
    recent_summary: str,
    error_info: str | None,
) -> dict[str, Any]:
    """调用反思 LLM，返回结构化结论。"""
    user_text = (
        f"当前目标：{goal}\n"
        f"当前步数：{step} / 最大步数：{max_steps}\n"
        f"最近步骤摘要：\n{recent_summary}\n"
    )
    if error_info:
        user_text += f"本步错误信息：{error_info}\n"
    user_text += "请输出上述 JSON。"

    msg = llm_reflection.invoke(
        [SystemMessage(content=REFLECTION_SYSTEM), HumanMessage(content=user_text)]
    )
    content = (msg.content or "").strip()
    # 允许模型返回 ```json ... ``` 包裹
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if m:
        content = m.group(1).strip()
    try:
        out = json.loads(content)
    except json.JSONDecodeError:
        out = {
            "decision": "continue",
            "reason": "反思输出解析失败，默认继续",
            "suggestion": None,
            "stop_type": None,
        }
    return out


def run_agent_with_reflection(
    user_input: str,
    system: str | None = SYSTEM,
    max_steps: int = 10,
    verbose: bool = True,
) -> str:
    """带反思节点的 Agent：工具执行后按配置触发反思，按 decision 分支。"""
    messages = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=user_input))

    failure_count = 0
    for step in range(max_steps):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            return response.content or "（无回复）"

        # 执行工具并收集结果
        tool_messages: list[ToolMessage] = []
        last_error: str | None = None
        last_step_failed = False

        for call in response.tool_calls:
            name = call.get("name")
            args = call.get("args") or {}
            tid = call.get("id", "")
            if name not in tool_map:
                tool_messages.append(
                    ToolMessage(content=f"未知工具: {name}", tool_call_id=tid)
                )
                last_step_failed = True
                last_error = f"未知工具: {name}"
                continue
            try:
                result = tool_map[name].invoke(args)
                content = str(result)
                tool_messages.append(ToolMessage(content=content, tool_call_id=tid))
                if _tool_message_is_failure(content):
                    last_step_failed = True
                    last_error = content
            except Exception as e:
                tool_messages.append(
                    ToolMessage(content=f"执行失败: {e}", tool_call_id=tid)
                )
                last_step_failed = True
                last_error = str(e)

        if last_step_failed:
            failure_count += 1

        # ---------- 反思触发与分支（贴近实际：先判断再调用） ----------
        # 注意：带 tool_calls 的 assistant 消息必须紧接对应的 tool 消息，不能中间插入其他 role，
        # 故先追加 tool_messages，再根据反思结果决定是否追加「反思建议」。
        messages.extend(tool_messages)

        if should_trigger_reflection(step, last_step_failed, failure_count, False):
            recent = _summarize_recent(messages, max_chars=500)
            ref = run_reflection(
                goal=user_input[:200],
                step=step,
                max_steps=max_steps,
                recent_summary=recent,
                error_info=last_error,
            )
            if verbose:
                print(
                    f"  [反思] decision={ref.get('decision')} reason={ref.get('reason')}"
                )

            if ref.get("decision") == "stop":
                stop_type = ref.get("stop_type") or "failed"
                return (
                    f"（反思决定停止）{ref.get('reason', '')} "
                    f"[stop_type={stop_type}]"
                )
            if ref.get("decision") == "adjust" and ref.get("suggestion"):
                messages.append(
                    HumanMessage(
                        content=f"【反思建议】{ref['suggestion']} 请根据此建议决定下一步，不要重复相同错误。"
                    )
                )

    return "达到最大步数，未得到最终回复。"


def _summarize_recent(messages: list, max_chars: int = 500) -> str:
    """最近几步的简短摘要，供反思输入。"""
    parts = []
    for m in messages[-10:]:
        if isinstance(m, AIMessage):
            tc = getattr(m, "tool_calls", None) or []
            if tc:
                parts.append("模型调用: " + ", ".join(c.get("name", "") for c in tc))
        elif isinstance(m, ToolMessage):
            parts.append("工具结果: " + (m.content[:80] + "..." if len(m.content) > 80 else m.content))
    s = "\n".join(parts)
    return s[-max_chars:] if len(s) > max_chars else s


def main() -> None:
    print("Week 9 带反思节点的 Agent 示例")
    print("工具：天气、当前时间、订单查询（仅 ORD001 成功，其它会触发失败与反思）")
    print("配置：失败时触发反思（见 REFLECTION_CONFIG）")
    print("输入 quit 退出。\n")
    while True:
        user_input = input("你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        reply = run_agent_with_reflection(user_input, verbose=True)
        print("助手:", reply, "\n")


if __name__ == "__main__":
    main()
