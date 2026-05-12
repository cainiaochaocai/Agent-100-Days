"""
Week 10 任务分解模式 Agent：分解 → 执行子任务 → 汇总

- 用户目标先经「分解」得到子任务列表（线性顺序），再按序执行每个子任务，最后汇总为最终回复
- 工具与 Week 8/9 一致：get_weather、get_current_time、search_order
- 流程：decompose_goal(goal) -> [step1, step2, ...] -> 每步 run_subtask(step, context) -> summarize(goal, results)

运行前在项目根目录配置 .env 中的 OPENAI_API_KEY。
在项目根目录执行：python week10/69_code/task_decomposition_agent_demo.py
"""

from dotenv import load_dotenv
load_dotenv()

import os
import json
import re

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

assert os.getenv("OPENAI_API_KEY"), "请先在项目根目录 .env 中配置 OPENAI_API_KEY"
api_key = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# ---------- 工具定义（与 Week 8/9 一致） ----------

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
llm_no_tools = ChatOpenAI(model="qwen-plus", base_url=BASE_URL, api_key=api_key)
tools = [get_weather, get_current_time, search_order]
llm_with_tools = llm.bind_tools(tools)
tool_map = {t.name: t for t in tools}

STEP_SYSTEM = "你是助手。根据当前子任务和已有信息，选择合适的工具完成这一子任务，给出简洁结果。若无需工具可直接回答。"
MAX_STEP_TURNS = 5


def decompose_goal(goal: str) -> list[str]:
    """将用户目标分解为有序子任务列表。"""
    prompt = f"""请将用户的以下目标分解为 2～5 个可单独执行的子任务，按执行顺序输出。
每个子任务应能对应一次或少量工具调用即可完成，不要漏掉用户目标中的关键动作。
只输出一个 JSON 数组，每项是一个子任务描述字符串，不要其他文字。例如：["查北京天气", "查订单 ORD001 状态", "用一句话总结"]

用户目标：{goal}
"""
    msg = llm_no_tools.invoke([HumanMessage(content=prompt)])
    content = (msg.content or "").strip()
    m = re.search(r"\[[\s\S]*\]", content)
    if m:
        try:
            raw = json.loads(m.group(0))
            return [str(x) if not isinstance(x, str) else x for x in raw][:10]
        except (json.JSONDecodeError, TypeError):
            pass
    # 兜底：按行或句号拆分
    fallback = [s.strip() for s in re.split(r"[。\n]+", goal) if s.strip()][:5]
    return fallback if fallback else [goal]


def run_subtask(subtask: str, goal: str, prior_results: list[str], max_turns: int = MAX_STEP_TURNS) -> str:
    """执行单个子任务：带工具的 Agent 循环，上下文包含目标与前序结果。"""
    context = goal
    if prior_results:
        context += "\n\n已完成步骤的结果：\n" + "\n".join(f"- {r}" for r in prior_results)
    query = f"当前子任务：{subtask}\n\n整体目标与已有信息：\n{context}"

    messages = [
        SystemMessage(content=STEP_SYSTEM),
        HumanMessage(content=query),
    ]
    for _ in range(max_turns):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        if not getattr(response, "tool_calls", None):
            return (response.content or "").strip()
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
    return "（该子任务达到最大步数未结束）"


def summarize(goal: str, results: list[str]) -> str:
    """根据目标与各步结果生成最终回复。"""
    parts = "\n".join(f"步骤{i+1} 结果：{r}" for i, r in enumerate(results))
    prompt = f"""用户目标：{goal}

各步骤执行结果：
{parts}

请用一段简洁的话汇总以上信息，直接回答用户，不要重复罗列步骤。"""
    msg = llm_no_tools.invoke([HumanMessage(content=prompt)])
    return (msg.content or "").strip()


def run_task_decomposition_agent(goal: str, verbose: bool = True) -> str:
    """任务分解 Agent：分解 → 顺序执行子任务 → 汇总。"""
    subtasks = decompose_goal(goal)
    if verbose:
        print("  [分解] 子任务:", subtasks)
    results = []
    for i, sub in enumerate(subtasks):
        if verbose:
            print(f"  [执行] {i+1}/{len(subtasks)}: {sub[:50]}...")
        out = run_subtask(sub, goal, results)
        results.append(out)
        if verbose:
            print(f"         -> {out[:80]}{'...' if len(out) > 80 else ''}")
    final = summarize(goal, results)
    return final


def main() -> None:
    print("Week 10 任务分解模式 Agent")
    print("流程：分解目标 → 按序执行子任务 → 汇总为最终回复")
    print("示例目标：查北京天气、查订单 ORD001 状态、用一句话总结。输入 quit 退出。\n")
    while True:
        user_input = input("目标: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        reply = run_task_decomposition_agent(user_input, verbose=True)
        print("回复:", reply, "\n")


if __name__ == "__main__":
    main()
