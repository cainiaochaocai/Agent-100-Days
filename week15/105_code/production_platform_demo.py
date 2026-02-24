"""
Week 15 生产就绪平台示例

在 Week 14 双场景 Agent 基础上，增加：
- 监控与可观测性：指标（请求量、延迟、错误率、成本）、结构化日志、trace_id
- 安全与权限：API 密钥从环境变量读取、简单 RBAC（用户可访问的 Agent 白名单）、审计日志
- 性能优化：响应缓存（相同请求短时复用）、限流（令牌桶）、请求超时
- 健康检查：/health 返回状态与依赖信息
- Langfuse（可选）：配置 LANGFUSE_PUBLIC_KEY、LANGFUSE_SECRET_KEY 后，LLM 调用自动上报 trace 与成本

运行：在项目根目录配置 .env（OPENAI_API_KEY ）
  python week15/105_code/production_platform_demo.py --channel cli
  python week15/105_code/production_platform_demo.py --channel web
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
from collections import deque
from threading import Lock
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

# ---------- Langfuse（可选，Day 99）----------
# 配置 LANGFUSE_PUBLIC_KEY、LANGFUSE_SECRET_KEY 后，所有 LLM.invoke 会自动上报 trace 与成本
_langfuse_handler = None
try:
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
        _langfuse_handler = LangfuseCallbackHandler()
except Exception:
    pass


def _llm_config(trace_id: str | None = None) -> dict:
    """LLM invoke 的 config：若启用了 Langfuse 则带上 callback，便于在 Langfuse UI 查看 trace。"""
    if _langfuse_handler is None:
        return {}
    # 可选：将 trace_id 传入 Langfuse 以便与网关日志关联（部分版本支持）
    return {"callbacks": [_langfuse_handler]}


# ---------- 环境 ----------
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    print("请在项目根目录 .env 中配置 OPENAI_API_KEY", file=sys.stderr)
    sys.exit(1)
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
llm = ChatOpenAI(model="qwen-plus", base_url=BASE_URL, api_key=api_key)


# ========== 指标收集（Day 99）==========

class MetricsCollector:
    """请求量、延迟、错误率、成本（演示用内存统计）。"""

    def __init__(self, latency_sample_size: int = 1000) -> None:
        self._lock = Lock()
        self.requests_total = 0
        self.errors_total = 0
        self.cost_total = 0.0
        self._latencies: deque[float] = deque(maxlen=latency_sample_size)
        self._by_agent: dict[str, int] = {}
        self._by_error_type: dict[str, int] = {}

    def record(self, agent_id: str, latency_sec: float, cost: float = 0.0, error_type: str | None = None) -> None:
        with self._lock:
            self.requests_total += 1
            self._latencies.append(latency_sec)
            self.cost_total += cost
            self._by_agent[agent_id] = self._by_agent.get(agent_id, 0) + 1
            if error_type:
                self.errors_total += 1
                self._by_error_type[error_type] = self._by_error_type.get(error_type, 0) + 1

    def latency_p50(self) -> float:
        with self._lock:
            if not self._latencies:
                return 0.0
            s = sorted(self._latencies)
            return s[int(len(s) * 0.5)]

    def latency_p95(self) -> float:
        with self._lock:
            if not self._latencies:
                return 0.0
            s = sorted(self._latencies)
            return s[int(len(s) * 0.95)] if len(s) > 1 else s[0]

    def error_rate(self) -> float:
        with self._lock:
            return self.errors_total / self.requests_total if self.requests_total else 0.0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requests_total": self.requests_total,
                "errors_total": self.errors_total,
                "error_rate": self.error_rate(),
                "cost_total": round(self.cost_total, 4),
                "latency_p50": round(self.latency_p50(), 3),
                "latency_p95": round(self.latency_p95(), 3),
                "by_agent": dict(self._by_agent),
                "by_error_type": dict(self._by_error_type),
            }


# ========== 结构化日志 + trace_id（Day 99）==========

def structured_log(
    level: str,
    message: str,
    trace_id: str | None = None,
    user_id: str | None = None,
    agent_id: str | None = None,
    session_id: str | None = None,
    latency_ms: float | None = None,
    **kwargs: Any,
) -> None:
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "level": level,
        "module": "gateway",
        "message": message,
        **kwargs,
    }
    if trace_id:
        entry["trace_id"] = trace_id
    if user_id is not None:
        entry["user_id"] = user_id
    if agent_id is not None:
        entry["agent_id"] = agent_id
    if session_id is not None:
        entry["session_id"] = session_id
    if latency_ms is not None:
        entry["latency_ms"] = round(latency_ms, 2)
    print(json.dumps(entry, ensure_ascii=False), flush=True)


# ========== 限流：令牌桶（Day 102）==========

class TokenBucketRateLimiter:
    """每 refill_per_sec 补充 1 个令牌，最多 max_tokens；每次请求消耗 1 个。"""

    def __init__(self, max_tokens: int = 10, refill_per_sec: float = 2.0) -> None:
        self.max_tokens = max_tokens
        self.refill_per_sec = refill_per_sec
        self._tokens = float(max_tokens)
        self._last_refill = time.monotonic()
        self._lock = Lock()

    def allow(self, key: str = "default") -> bool:
        with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self.max_tokens,
                self._tokens + (now - self._last_refill) * self.refill_per_sec,
            )
            self._last_refill = now
            if self._tokens >= 1:
                self._tokens -= 1
                return True
            return False


# ========== 响应缓存（Day 102）==========

class ResponseCache:
    """key = hash(user_id + message)，TTL 秒。"""

    def __init__(self, ttl_sec: int = 60, max_size: int = 500) -> None:
        self.ttl_sec = ttl_sec
        self.max_size = max_size
        self._cache: dict[str, tuple[float, Any]] = {}
        self._lock = Lock()

    def _key(self, user_id: str, message: str) -> str:
        return hashlib.sha256((user_id + "\n" + message).encode()).hexdigest()

    def get(self, user_id: str, message: str) -> Any | None:
        k = self._key(user_id, message)
        with self._lock:
            if k not in self._cache:
                return None
            ts, val = self._cache[k]
            if time.time() - ts > self.ttl_sec:
                del self._cache[k]
                return None
            return val

    def set(self, user_id: str, message: str, value: Any) -> None:
        k = self._key(user_id, message)
        with self._lock:
            if len(self._cache) >= self.max_size:
                # 简单策略：删掉一半最旧的（按时间戳）
                items = sorted(self._cache.items(), key=lambda x: x[1][0])
                for old_k, _ in items[: self.max_size // 2]:
                    del self._cache[old_k]
            self._cache[k] = (time.time(), value)


# ========== 审计日志（Day 101）==========

class AuditLog:
    """记录敏感操作：谁、何时、访问了哪个 Agent、会话。"""

    def __init__(self, max_entries: int = 1000) -> None:
        self._entries: list[dict[str, Any]] = []
        self._max = max_entries
        self._lock = Lock()

    def log(self, user_id: str, action: str, agent_id: str | None, session_id: str | None, trace_id: str | None, extra: dict | None = None) -> None:
        with self._lock:
            entry = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "user_id": user_id,
                "action": action,
                "agent_id": agent_id,
                "session_id": session_id,
                "trace_id": trace_id,
                **(extra or {}),
            }
            self._entries.append(entry)
            if len(self._entries) > self._max:
                self._entries.pop(0)

    def recent(self, n: int = 50) -> list[dict]:
        with self._lock:
            return list(self._entries[-n:])


# ========== 简单 RBAC（Day 101）==========

# 用户 -> 允许的 agent 列表；None 表示不限制（允许所有）
USER_ALLOWED_AGENTS: dict[str, list[str] | None] = {
    "admin": None,  # 全部
    "analyst": ["general_agent", "chatbi_agent"],
    "researcher": ["general_agent", "deep_research_agent"],
}


def check_agent_access(user_id: str, agent_id: str) -> bool:
    allowed = USER_ALLOWED_AGENTS.get(user_id)
    if allowed is None:
        return True
    return agent_id in allowed


# ========== 路由与 Session（与 Week 14 一致）==========

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


# ========== General / DeepResearch / ChatBI Runtime（与 Week 14 一致，略简化以节省篇幅）==========

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


def run_general(user_input: str, context: list[dict], session_id: str, session_manager: SessionManager, max_steps: int = 10) -> str:
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
        response = llm_with_tools.invoke(messages, config=_llm_config())
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


SEARCH_SIMULATE_DB = {
    "AI 医疗": "AI 在医疗中的应用包括：影像诊断、辅助问诊、药物研发中的分子筛选。",
    "医疗诊断": "医疗诊断场景中，AI 可用于影像识别、病理切片分析。",
    "药物研发": "AI 在药物研发中用于靶点发现、分子设计等。",
}


def run_deep_research(user_input: str, context: list[dict], session_id: str, session_manager: SessionManager) -> str:
    plan_prompt = f"""用户的研究问题是：{user_input}
请将上述问题拆成 2～3 个可独立检索的子问题，每行一个，不要编号外的其他内容。"""
    plan_msg = llm.invoke([HumanMessage(content=plan_prompt)], config=_llm_config())
    plan_text = (plan_msg.content or "").strip()
    lines = [s.strip() for s in plan_text.split("\n") if s.strip()][:3]
    sub_questions = [re.sub(r"^[\d\.\、\s]+", "", s).strip() or s for s in lines]
    evidence_pool = []
    for sq in sub_questions:
        added = False
        for kw, content in SEARCH_SIMULATE_DB.items():
            if kw in sq or kw in user_input:
                evidence_pool.append({"sub_question": sq, "source": kw, "content": content})
                added = True
        if not added:
            evidence_pool.append({"sub_question": sq, "source": "综合", "content": f"关于「{sq}」的综述。"})
    evidence_text = "\n\n".join(f"[{e['source']}] {e['sub_question']}\n{e['content']}" for e in evidence_pool[:6])
    synth_prompt = f"""用户的研究问题：{user_input}\n\n证据：\n{evidence_text}\n请根据上述证据写一段简短报告（摘要、主要发现、结论）。"""
    report_msg = llm.invoke([HumanMessage(content=synth_prompt)], config=_llm_config())
    report = (report_msg.content or "").strip() or "（未能生成报告）"
    report = f"【DeepResearch 报告】\n\n{report}"
    session_manager.append_history(session_id, "user", user_input)
    session_manager.append_history(session_id, "assistant", report)
    return report


ORDERS_SCHEMA = "可用表：orders: order_id (str), amount (float), created_at (str)。仅允许 SELECT。"
DEMO_ORDERS = [
    {"order_id": "ORD001", "amount": 1000.0, "created_at": "2025-01-15"},
    {"order_id": "ORD002", "amount": 2500.0, "created_at": "2025-01-20"},
    {"order_id": "ORD003", "amount": 800.0, "created_at": "2025-02-01"},
    {"order_id": "ORD004", "amount": 3200.0, "created_at": "2025-02-10"},
]


def _guardrail_sql(sql: str) -> tuple[bool, str]:
    sql_upper = sql.upper().strip()
    for kw in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE"]:
        if kw in sql_upper:
            return False, f"Guardrail：不允许 {kw} 操作。"
    if "SELECT" not in sql_upper:
        return False, "Guardrail：仅支持 SELECT。"
    return True, sql


def _execute_memory_sql(sql: str) -> list[tuple]:
    sql_upper = sql.upper()
    if "SUM" in sql_upper and "AMOUNT" in sql_upper:
        return [(sum(o["amount"] for o in DEMO_ORDERS),)]
    if "COUNT" in sql_upper:
        return [(len(DEMO_ORDERS),)]
    if "AVG" in sql_upper and "AMOUNT" in sql_upper:
        return [(sum(o["amount"] for o in DEMO_ORDERS) / len(DEMO_ORDERS),)]
    return [(r["order_id"], r["amount"], r["created_at"]) for r in DEMO_ORDERS]


def run_chatbi(user_input: str, context: list[dict], session_id: str, session_manager: SessionManager) -> str:
    sql_prompt = f"""{ORDERS_SCHEMA}\n用户问题：{user_input}\n请只输出一条 SQL（仅 SELECT）。若无法得出，输出：-- 无法得出"""
    sql_msg = llm.invoke([HumanMessage(content=sql_prompt)], config=_llm_config())
    raw_sql = (sql_msg.content or "").strip()
    raw_sql = re.sub(r"^```\w*\n?", "", raw_sql)
    raw_sql = re.sub(r"\n?```\s*$", "", raw_sql).strip()
    if raw_sql.startswith("--"):
        session_manager.append_history(session_id, "user", user_input)
        session_manager.append_history(session_id, "assistant", "当前仅支持对 orders 表的查询。")
        return "当前仅支持对 orders 表的查询。"
    ok, sql = _guardrail_sql(raw_sql)
    if not ok:
        session_manager.append_history(session_id, "user", user_input)
        session_manager.append_history(session_id, "assistant", sql)
        return sql
    try:
        rows = _execute_memory_sql(sql)
        result_text = str(rows[:10])
    except Exception as e:
        result_text = f"执行错误: {e}"
    explain_prompt = f"用户问题：{user_input}\nSQL：{sql}\n结果：{result_text}\n请用一两句话解释。"
    explain_msg = llm.invoke([HumanMessage(content=explain_prompt)], config=_llm_config())
    explanation = (explain_msg.content or "").strip() or result_text
    reply = f"【ChatBI】\n{explanation}\n\n（SQL：{sql}）"
    session_manager.append_history(session_id, "user", user_input)
    session_manager.append_history(session_id, "assistant", reply)
    return reply


# ========== Gateway（生产增强：trace、指标、日志、限流、缓存、超时、RBAC、审计）==========

class Gateway:
    def __init__(
        self,
        session_manager: SessionManager,
        metrics: MetricsCollector,
        rate_limiter: TokenBucketRateLimiter,
        cache: ResponseCache | None,
        audit_log: AuditLog,
        request_timeout_sec: float = 60.0,
    ) -> None:
        self.session_manager = session_manager
        self.metrics = metrics
        self.rate_limiter = rate_limiter
        self.cache = cache
        self.audit_log = audit_log
        self.request_timeout_sec = request_timeout_sec

    def handle_request(
        self,
        user_id: str,
        channel: str,
        message: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        trace_id = str(uuid.uuid4())
        start = time.perf_counter()
        structured_log("INFO", "Request received", trace_id=trace_id, user_id=user_id)

        # 限流
        if not self.rate_limiter.allow(key=user_id):
            structured_log("WARNING", "Rate limit exceeded", trace_id=trace_id, user_id=user_id)
            self.metrics.record("_rejected", 0, 0, "rate_limit")
            return {
                "response": "请求过于频繁，请稍后再试。",
                "session_id": session_id or "",
                "agent_id": None,
                "error": "rate_limit",
            }

        # 缓存
        if self.cache:
            cached = self.cache.get(user_id, message)
            if cached is not None:
                latency_ms = (time.perf_counter() - start) * 1000
                structured_log("INFO", "Cache hit", trace_id=trace_id, user_id=user_id, latency_ms=latency_ms)
                self.metrics.record(cached.get("agent_id", "general_agent"), (time.perf_counter() - start), 0)
                return {**cached, "cached": True}

        # 会话
        if not session_id:
            session = self.session_manager.create_session(user_id, channel)
            session_id = session["id"]
        else:
            session = self.session_manager.get_session(session_id)
            if not session:
                self.metrics.record("_error", time.perf_counter() - start, 0, "session_not_found")
                return {"response": "错误：会话不存在或已过期。", "session_id": session_id, "error": "session_not_found"}

        agent_id = route(message, session)

        # RBAC
        if not check_agent_access(user_id, agent_id):
            structured_log("WARNING", "Forbidden agent access", trace_id=trace_id, user_id=user_id, agent_id=agent_id)
            self.metrics.record("_rejected", time.perf_counter() - start, 0, "forbidden")
            self.audit_log.log(user_id, "access_denied", agent_id, session_id, trace_id)
            return {
                "response": "您没有权限使用该能力，请联系管理员。",
                "session_id": session_id,
                "agent_id": agent_id,
                "error": "forbidden",
            }

        history = self.session_manager.get_history(session_id)
        response_text = ""
        error_type = None
        cost = 0.0  # 演示：可按 token 估算成本

        try:
            if agent_id == "deep_research_agent":
                response_text = run_deep_research(message, history, session_id, self.session_manager)
            elif agent_id == "chatbi_agent":
                response_text = run_chatbi(message, history, session_id, self.session_manager)
            else:
                response_text = run_general(message, history, session_id, self.session_manager)
        except Exception as e:
            error_type = type(e).__name__
            response_text = f"处理出错：{e}"
            structured_log("ERROR", str(e), trace_id=trace_id, user_id=user_id, agent_id=agent_id)

        latency_sec = time.perf_counter() - start
        self.metrics.record(agent_id, latency_sec, cost, error_type)
        self.session_manager.update_session(session_id, {"last_agent": agent_id, "last_message": message})
        self.audit_log.log(user_id, "chat", agent_id, session_id, trace_id)

        result = {"response": response_text, "session_id": session_id, "agent_id": agent_id}
        if self.cache:
            self.cache.set(user_id, message, result)
        latency_ms = latency_sec * 1000
        structured_log("INFO", "Request completed", trace_id=trace_id, user_id=user_id, agent_id=agent_id, session_id=session_id, latency_ms=latency_ms)
        return result


# ========== 健康检查（Day 103）==========

def health_check(metrics: MetricsCollector) -> dict[str, Any]:
    return {
        "status": "ok",
        "agents": ["general_agent", "deep_research_agent", "chatbi_agent"],
        "langfuse_enabled": _langfuse_handler is not None,
        "metrics": metrics.snapshot(),
    }


# ========== CLI / Web ==========

def main_cli() -> None:
    sm = SessionManager()
    metrics = MetricsCollector()
    rate_limiter = TokenBucketRateLimiter(max_tokens=20, refill_per_sec=5)
    cache = ResponseCache(ttl_sec=60)
    audit = AuditLog()
    gateway = Gateway(sm, metrics, rate_limiter, cache, audit)
    print("Week 15 生产就绪平台（监控 / 限流 / 缓存 / RBAC / 审计）")
    print("Langfuse:", "已启用" if _langfuse_handler else "未配置（可选：LANGFUSE_PUBLIC_KEY、LANGFUSE_SECRET_KEY）")
    print("用户权限：admin=全部；analyst=general+chatbi；researcher=general+deep_research；其他=全部")
    print("输入 quit 退出。\n")
    session_id = None
    while True:
        try:
            line = input("你: ").strip()
        except EOFError:
            break
        if not line or line.lower() in ("quit", "exit", "q"):
            break
        user_id = os.getenv("DEMO_USER_ID", "admin")
        result = gateway.handle_request(user_id, "cli", line, session_id)
        session_id = result.get("session_id")
        print("助手:", result.get("response", ""))
        print("  [路由:", result.get("agent_id"), "]", "[缓存]" if result.get("cached") else "")


def main_web() -> None:
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("Web 模式需要安装 flask: pip install flask", file=sys.stderr)
        sys.exit(1)
    app = Flask(__name__)
    sm = SessionManager()
    metrics = MetricsCollector()
    rate_limiter = TokenBucketRateLimiter(max_tokens=30, refill_per_sec=5)
    cache = ResponseCache(ttl_sec=60)
    audit = AuditLog()
    gateway = Gateway(sm, metrics, rate_limiter, cache, audit)

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
        return jsonify(health_check(metrics))

    @app.route("/metrics", methods=["GET"])
    def metrics_endpoint():
        return jsonify(metrics.snapshot())

    print("Week 15 生产就绪平台 — Web")
    print("POST /api/chat  GET /health  GET /metrics")
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", choices=["cli", "web"], default="cli", help="通道：cli 或 web")
    args = parser.parse_args()
    if args.channel == "web":
        main_web()
    else:
        main_cli()
