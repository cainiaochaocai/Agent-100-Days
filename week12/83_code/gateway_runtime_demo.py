"""
Week 12 Gateway + Agent Runtime 最小可用版本

- Gateway：接收请求、路由决策、会话管理、调用 Runtime、返回响应
- Router：规则路由（关键词 → agent_id），不匹配时回退到 general_agent
- SessionManager：内存存储会话与历史，支持创建/查询/更新
- AgentRuntime：上下文组装、LLM 调用、工具执行循环，返回最终回复
- 工具：get_weather、get_current_time、search_order（与 Week 8/9 一致）

运行前在项目根目录配置 .env（OPENAI_API_KEY）。
在项目根目录执行：python week12/83_code/gateway_runtime_demo.py
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

# ---------- 环境与模型 ----------

api_key = os.getenv("OPENAI_API_KEY") 
assert api_key, "请在项目根目录 .env 中配置 OPENAI_API_KEY"
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

llm = ChatOpenAI(model="qwen-plus", base_url=BASE_URL, api_key=api_key)


# ---------- 工具定义（与 Week 8/9 一致） ----------

@tool
def get_weather(city: str) -> str:
    """当用户询问某地天气时，使用此工具查询该城市的天气。输入为城市名，如北京、上海。"""
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


TOOLS = [get_weather, get_current_time, search_order]
TOOL_MAP = {t.name: t for t in TOOLS}


# ---------- Router：规则路由，回退到 general_agent ----------

ROUTE_RULES = [
    (["研究", "分析", "调查", "深度"], "research_agent"),
    (["查询", "数据", "报表", "销售额", "订单"], "general_agent"),  # 本 demo 只有 general
]


def route(message: str, session: dict[str, Any]) -> str:
    """根据用户消息和会话做路由决策。返回 agent_id。"""
    msg_lower = (message or "").strip().lower()
    for keywords, agent_id in ROUTE_RULES:
        if any(kw in msg_lower for kw in keywords):
            return agent_id
    return "general_agent"


# ---------- SessionManager：内存会话与历史 ----------

class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, dict] = {}
        self._history: dict[str, list[dict]] = {}

    def create_session(self, user_id: str, channel: str) -> dict:
        session_id = f"{channel}_{user_id}_{uuid.uuid4().hex[:8]}"
        session = {
            "id": session_id,
            "user_id": user_id,
            "channel": channel,
            "created_at": None,
            "last_agent": None,
            "last_message": None,
        }
        self._sessions[session_id] = session
        self._history[session_id] = []
        return session

    def get_session(self, session_id: str) -> dict | None:
        return self._sessions.get(session_id)

    def update_session(self, session_id: str, updates: dict) -> None:
        if session_id in self._sessions:
            self._sessions[session_id].update(updates)

    def append_history(self, session_id: str, role: str, content: str) -> None:
        if session_id in self._history:
            self._history[session_id].append({"role": role, "content": content})

    def get_history(self, session_id: str, limit: int = 20) -> list[dict]:
        hist = self._history.get(session_id, [])
        return hist[-limit:] if limit else hist


# ---------- ContextBuilder：组装 messages ----------

SYSTEM_PROMPT = (
    "你是助手。根据用户问题选择合适的工具获取信息，然后给出简洁回答。"
    "若工具返回错误信息，请根据错误调整（如确认参数），不要重复相同错误。"
)


def build_context(
    user_input: str,
    history: list[dict],
    session_id: str,
    system: str = SYSTEM_PROMPT,
) -> list:
    """组装 LLM 可用的 messages：SystemMessage + 历史 + 当前 UserMessage。"""
    messages: list = []
    if system:
        messages.append(SystemMessage(content=system))
    for h in history:
        role, content = h.get("role", ""), h.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=user_input))
    return messages


# ---------- ToolRegistry + ToolExecutor ----------

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    def register(self, tool_fn: Any) -> None:
        self._tools[tool_fn.name] = tool_fn

    def get(self, name: str) -> Any | None:
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        return list(self._tools.keys())


def execute_tool(name: str, args: dict) -> str:
    """执行工具，统一返回字符串；异常时返回错误说明。"""
    tool_fn = TOOL_MAP.get(name)
    if not tool_fn:
        return f"错误：未知工具 {name}"
    try:
        result = tool_fn.invoke(args or {})
        return str(result) if result is not None else ""
    except Exception as e:
        return f"执行失败: {e}"


# ---------- AgentRuntime：核心循环 ----------

def run_runtime(
    user_input: str,
    context: list[dict],
    session_id: str,
    session_manager: SessionManager,
    max_steps: int = 10,
) -> str:
    """
    单 Agent Runtime：组装上下文 → LLM 调用 → 工具执行循环 → 返回最终回复。
    执行过程中将 assistant 回复与工具结果追加到 session 历史（仅保留最后一轮用于演示简化）。
    """
    messages = build_context(
        user_input,
        context,
        session_id,
    )

    for step in range(max_steps):
        llm_with_tools = llm.bind_tools(TOOLS)
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            # 最终回复
            reply = (response.content or "").strip() or "（无回复）"
            session_manager.append_history(session_id, "user", user_input)
            session_manager.append_history(session_id, "assistant", reply)
            return reply

        # 执行工具并追加 ToolMessage（必须紧接 AIMessage，不能插入其他 role）
        for call in response.tool_calls:
            name = call.get("name", "")
            args = call.get("args") or {}
            tid = call.get("id", "")
            content = execute_tool(name, args)
            messages.append(ToolMessage(content=content, tool_call_id=tid))

    session_manager.append_history(session_id, "user", user_input)
    session_manager.append_history(session_id, "assistant", "达到最大步数，未得到最终回复。")
    return "达到最大步数，未得到最终回复。"


# ---------- Gateway ----------

class Gateway:
    def __init__(
        self,
        session_manager: SessionManager,
    ) -> None:
        self.session_manager = session_manager

    def handle_request(
        self,
        user_id: str,
        channel: str,
        message: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        接收请求 → 获取/创建会话 → 路由决策 → 调用 Runtime → 更新会话 → 返回响应。
        """
        if not session_id:
            session = self.session_manager.create_session(user_id, channel)
            session_id = session["id"]
        else:
            session = self.session_manager.get_session(session_id)
            if not session:
                return {
                    "response": "错误：会话不存在或已过期。",
                    "session_id": session_id or "",
                    "error": "session_not_found",
                }

        agent_id = route(message, session)
        history = self.session_manager.get_history(session_id)

        response = run_runtime(
            user_input=message,
            context=history,
            session_id=session_id,
            session_manager=self.session_manager,
        )

        self.session_manager.update_session(session_id, {
            "last_agent": agent_id,
            "last_message": message,
        })

        return {
            "response": response,
            "session_id": session_id,
            "agent_id": agent_id,
        }


# ---------- CLI 入口 ----------

def main() -> None:
    session_manager = SessionManager()
    gateway = Gateway(session_manager)

    print("Week 12 Gateway + Runtime 最小可用版本")
    print("流程：输入 → Gateway 路由 → Runtime（工具调用）→ 返回结果")
    print("工具：天气、当前时间、订单查询（仅 ORD001 有数据）")
    print("输入 quit 退出。\n")

    user_id = "cli_user"
    channel = "cli"
    session_id: str | None = None

    while True:
        try:
            user_input = input("你: ").strip()
        except EOFError:
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        result = gateway.handle_request(user_id, channel, user_input, session_id)
        session_id = result.get("session_id") or session_id
        print("助手:", result.get("response", ""))
        if result.get("agent_id"):
            print("  [路由:", result["agent_id"], "]")
        print()


if __name__ == "__main__":
    main()
