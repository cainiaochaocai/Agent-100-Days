"""
Week 14 两个场景 Agent 在平台上运行

在 Week 13 插件化平台基础上，增加多 Agent 路由与两个垂直场景 Runtime：
- deep_research_agent：研究类问题 → 规划（LLM 拆子问题）→ 检索（模拟搜索）→ 合成（LLM 报告）
- chatbi_agent：BI 类问题 → SQL 生成（LLM）→ Guardrail（仅 SELECT）→ 执行（内存表）→ 解释（LLM）
- general_agent：其余问题 → 沿用 Week 13 的 run_runtime（工具调用）

运行前在项目根目录配置 .env（OPENAI_API_KEY 或 DASHSCOPE_API_KEY）。
CLI：python week14/97_code/platform_with_scenario_agents_demo.py --channel cli
Web：python week14/97_code/platform_with_scenario_agents_demo.py --channel web
"""

from __future__ import annotations

import argparse
import os
import re
import uuid
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

# ---------- 环境 ----------

api_key = os.getenv("OPENAI_API_KEY") 
assert api_key, "请在项目根目录 .env 中配置 OPENAI_API_KEY "
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

llm = ChatOpenAI(model="qwen-plus", base_url=BASE_URL, api_key=api_key)


# ========== 路由：多 Agent ==========

ROUTE_RULES = [
    (["研究", "分析", "调查", "深度"], "deep_research_agent"),
    (["查询", "数据", "报表", "销售额", "月活", "订单统计"], "chatbi_agent"),
    (["天气", "时间", "订单"], "general_agent"),
]


def route(message: str, session: dict) -> str:
    msg_lower = (message or "").strip().lower()
    for keywords, agent_id in ROUTE_RULES:
        if any(kw in msg_lower for kw in keywords):
            return agent_id
    return "general_agent"


# ========== SessionManager（与 Week 13 一致）==========

class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, dict] = {}
        self._history: dict[str, list[dict]] = {}

    def create_session(self, user_id: str, channel: str) -> dict:
        session_id = f"{channel}_{user_id}_{uuid.uuid4().hex[:8]}"
        session = {"id": session_id, "user_id": user_id, "channel": channel, "last_agent": None, "last_message": None}
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


# ========== General Agent：工具调用（与 Week 13 一致）==========

@tool
def get_weather(city: str) -> str:
    """当用户询问某地天气时使用。输入为城市名。"""
    fake_db = {"北京": "晴，25°C", "上海": "多云，22°C", "深圳": "阴，28°C"}
    return fake_db.get(city, f"{city}：暂无数据")


@tool
def get_current_time() -> str:
    """当用户问当前时间或日期时使用。"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def search_order(order_id: str) -> str:
    """根据订单号查询订单状态。示例中仅 ORD001 有数据。"""
    if order_id.strip().upper() == "ORD001":
        return "订单 ORD001：已支付，配送中"
    return f"错误：订单不存在（{order_id}）。"


TOOLS_GENERAL = [get_weather, get_current_time, search_order]
TOOL_MAP = {t.name: t for t in TOOLS_GENERAL}
SYSTEM_GENERAL = "你是助手。根据用户问题选择合适的工具获取信息，然后给出简洁回答。若工具返回错误，请根据错误调整。"


def run_general(
    user_input: str,
    context: list[dict],
    session_id: str,
    session_manager: SessionManager,
    max_steps: int = 10,
) -> str:
    messages = [SystemMessage(content=SYSTEM_GENERAL)]
    for h in context:
        role, content = h.get("role", ""), h.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=user_input))

    llm_with_tools = llm.bind_tools(TOOLS_GENERAL)
    for step in range(max_steps):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        if not getattr(response, "tool_calls", None):
            reply = (response.content or "").strip() or "（无回复）"
            session_manager.append_history(session_id, "user", user_input)
            session_manager.append_history(session_id, "assistant", reply)
            return reply
        for call in response.tool_calls:
            name = call.get("name", "")
            args = call.get("args") or {}
            tid = call.get("id", "")
            fn = TOOL_MAP.get(name)
            content = str(fn.invoke(args)) if fn else f"错误：未知工具 {name}"
            messages.append(ToolMessage(content=content, tool_call_id=tid))
    session_manager.append_history(session_id, "user", user_input)
    session_manager.append_history(session_id, "assistant", "达到最大步数。")
    return "达到最大步数，未得到最终回复。"


# ========== DeepResearch Agent：规划 → 检索（模拟）→ 合成 ==========

SEARCH_SIMULATE_DB = {
    "AI 医疗": "AI 在医疗中的应用包括：影像诊断（如肺结节检测）、辅助问诊、药物研发中的分子筛选、医院管理（排班、病历摘要）。数据来源：行业报告与学术综述。",
    "医疗诊断": "医疗诊断场景中，AI 可用于影像识别（CT/MRI）、病理切片分析、辅助诊断建议。需注意合规与医生最终决策权。",
    "药物研发": "AI 在药物研发中用于靶点发现、分子设计、临床试验患者筛选等，可缩短研发周期。",
}


def run_deep_research(
    user_input: str,
    context: list[dict],
    session_id: str,
    session_manager: SessionManager,
) -> str:
    """简化版 DeepResearch：规划（LLM 拆子问题）→ 模拟检索 → 合成报告。"""
    # 1. 规划：LLM 将问题拆成 2～3 个子问题
    plan_prompt = f"""用户的研究问题是：{user_input}
请将上述问题拆成 2～3 个可独立检索的子问题，每行一个，不要编号外的其他内容。例如：
子问题1
子问题2"""
    plan_msg = llm.invoke([HumanMessage(content=plan_prompt)])
    plan_text = (plan_msg.content or "").strip()
    lines = [s.strip() for s in plan_text.split("\n") if s.strip()][:3]
    sub_questions = [re.sub(r"^[\d\.\、\s]+", "", s).strip() or s for s in lines]

    # 2. 检索（模拟）：按关键词匹配到预设内容，组成证据池
    evidence_pool = []
    for sq in sub_questions:
        added = False
        for kw, content in SEARCH_SIMULATE_DB.items():
            if kw in sq or kw in user_input:
                evidence_pool.append({"sub_question": sq, "source": kw, "content": content})
                added = True
        if not added:
            evidence_pool.append({"sub_question": sq, "source": "综合", "content": f"关于「{sq}」的综述：涉及多类应用与案例，需结合具体场景进一步检索。"})

    evidence_text = "\n\n".join(
        f"[{e['source']}] {e['sub_question']}\n{e['content']}" for e in evidence_pool[:6]
    )

    # 3. 合成：LLM 根据证据写简短报告
    synth_prompt = f"""用户的研究问题：{user_input}

以下是从多来源收集到的证据（已按子问题组织）：

{evidence_text}

请根据上述证据，写一段简短的研究报告（200～400 字），包含：1）摘要；2）主要发现（分点）；3）结论。不要编造证据中未出现的内容。"""
    report_msg = llm.invoke([HumanMessage(content=synth_prompt)])
    report = (report_msg.content or "").strip() or "（未能生成报告）"
    report = f"【DeepResearch 报告】\n\n{report}\n\n（本示例使用模拟检索，真实场景可接入搜索/RAG 工具。）"

    session_manager.append_history(session_id, "user", user_input)
    session_manager.append_history(session_id, "assistant", report)
    return report


# ========== ChatBI Agent：SQL 生成 → Guardrail → 执行（内存表）→ 解释 ==========

ORDERS_SCHEMA = """
可用表：
- orders: order_id (str), amount (float), created_at (str 日期 YYYY-MM-DD)

仅允许 SELECT，禁止 INSERT/UPDATE/DELETE/DROP。只查询 orders 表。
"""

# 内存演示数据
DEMO_ORDERS = [
    {"order_id": "ORD001", "amount": 1000.0, "created_at": "2025-01-15"},
    {"order_id": "ORD002", "amount": 2500.0, "created_at": "2025-01-20"},
    {"order_id": "ORD003", "amount": 800.0, "created_at": "2025-02-01"},
    {"order_id": "ORD004", "amount": 3200.0, "created_at": "2025-02-10"},
]


def _guardrail_sql(sql: str) -> tuple[bool, str]:
    """只允许 SELECT，且不得包含危险关键字。"""
    sql_upper = sql.upper().strip()
    for kw in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE"]:
        if kw in sql_upper:
            return False, f"Guardrail：不允许 {kw} 操作，仅支持 SELECT 查询。"
    if "SELECT" not in sql_upper:
        return False, "Guardrail：仅支持 SELECT 查询。"
    return True, sql


def _execute_memory_sql(sql: str) -> list[tuple]:
    """简化：仅支持对 DEMO_ORDERS 的简单 SELECT 查询；否则返回示例汇总。"""
    sql_upper = sql.upper()
    if "SUM" in sql_upper and "AMOUNT" in sql_upper:
        total = sum(o["amount"] for o in DEMO_ORDERS)
        return [(total,)]
    if "COUNT" in sql_upper:
        return [(len(DEMO_ORDERS),)]
    if "AVG" in sql_upper and "AMOUNT" in sql_upper:
        avg = sum(o["amount"] for o in DEMO_ORDERS) / len(DEMO_ORDERS)
        return [(avg,)]
    return [(r["order_id"], r["amount"], r["created_at"]) for r in DEMO_ORDERS]


def run_chatbi(
    user_input: str,
    context: list[dict],
    session_id: str,
    session_manager: SessionManager,
) -> str:
    """简化版 ChatBI：LLM 生成 SQL → Guardrail → 内存执行 → LLM 解释。"""
    # 1. 生成 SQL
    sql_prompt = f"""{ORDERS_SCHEMA}

用户问题：{user_input}

请只输出一条 SQL（仅 SELECT），不要其他解释。若无法从表中得出答案，输出：-- 无法从 orders 表得出"""
    sql_msg = llm.invoke([HumanMessage(content=sql_prompt)])
    raw_sql = (sql_msg.content or "").strip()
    raw_sql = re.sub(r"^```\w*\n?", "", raw_sql)
    raw_sql = re.sub(r"\n?```\s*$", "", raw_sql).strip()
    if raw_sql.startswith("--"):
        session_manager.append_history(session_id, "user", user_input)
        session_manager.append_history(session_id, "assistant", "当前仅支持对 orders 表的查询（如销售额、订单数等）。")
        return "当前仅支持对 orders 表的查询（如销售额、订单数等）。"

    ok, sql = _guardrail_sql(raw_sql)
    if not ok:
        session_manager.append_history(session_id, "user", user_input)
        session_manager.append_history(session_id, "assistant", sql)
        return sql

    # 2. 执行（内存）
    try:
        rows = _execute_memory_sql(sql)
    except Exception as e:
        rows = []
        result_text = f"执行错误: {e}"
    else:
        result_text = str(rows[:10])

    # 3. 自然语言解释
    explain_prompt = f"""用户问题：{user_input}
生成的 SQL：{sql}
查询结果（前几条）：{result_text}

请用一两句话解释该结果，直接回答用户问题。"""
    explain_msg = llm.invoke([HumanMessage(content=explain_prompt)])
    explanation = (explain_msg.content or "").strip() or result_text
    reply = f"【ChatBI】\n{explanation}\n\n（SQL：{sql}）"

    session_manager.append_history(session_id, "user", user_input)
    session_manager.append_history(session_id, "assistant", reply)
    return reply


# ========== Gateway：按 agent_id 分发 ==========

class Gateway:
    def __init__(self, session_manager: SessionManager) -> None:
        self.session_manager = session_manager

    def handle_request(
        self,
        user_id: str,
        channel: str,
        message: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        if not session_id:
            session = self.session_manager.create_session(user_id, channel)
            session_id = session["id"]
        else:
            session = self.session_manager.get_session(session_id)
            if not session:
                return {"response": "错误：会话不存在或已过期。", "session_id": session_id or "", "error": "session_not_found"}

        agent_id = route(message, session)
        history = self.session_manager.get_history(session_id)

        if agent_id == "deep_research_agent":
            response = run_deep_research(message, history, session_id, self.session_manager)
        elif agent_id == "chatbi_agent":
            response = run_chatbi(message, history, session_id, self.session_manager)
        else:
            response = run_general(message, history, session_id, self.session_manager)

        self.session_manager.update_session(session_id, {"last_agent": agent_id, "last_message": message})
        return {"response": response, "session_id": session_id, "agent_id": agent_id}


# ========== CLI / Web 入口 ==========

def main_cli() -> None:
    sm = SessionManager()
    gateway = Gateway(sm)
    print("Week 14 两个场景 Agent 在平台上运行")
    print("路由：含「研究/分析/调查」→ DeepResearch；含「查询/数据/报表/销售额」→ ChatBI；其余 → General（天气/时间/订单）")
    print("输入 quit 退出。\n")
    session_id = None
    while True:
        try:
            line = input("你: ").strip()
        except EOFError:
            break
        if not line or line.lower() in ("quit", "exit", "q"):
            break
        result = gateway.handle_request("cli_user", "cli", line, session_id)
        session_id = result.get("session_id")
        print("助手:", result.get("response", ""))
        print("  [路由:", result.get("agent_id"), "]\n")


def main_web() -> None:
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("Web 模式需要安装 flask: pip install flask")
        return
    app = Flask(__name__)
    sm = SessionManager()
    gateway = Gateway(sm)

    @app.route("/api/chat", methods=["POST"])
    def chat():
        data = request.get_json(force=True, silent=True) or {}
        user_id = data.get("user_id", "web_user")
        message = data.get("message", "")
        session_id = data.get("session_id")
        result = gateway.handle_request(user_id, "web", message, session_id)
        return jsonify(result)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "agents": "general,deep_research,chatbi"})

    print("Week 14 两个场景 Agent — Web 通道")
    print("POST http://localhost:5000/api/chat  Body: {\"message\":\"研究一下 AI 在医疗的应用\"} 或 {\"message\":\"查询销售额\"}")
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", choices=["cli", "web"], default="cli", help="通道：cli 或 web")
    args = parser.parse_args()
    if args.channel == "web":
        main_web()
    else:
        main_cli()
