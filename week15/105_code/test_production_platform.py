"""
Week 15 生产就绪平台 — 单元测试与集成测试示例

测试内容（Day 100）：
- 单元测试：路由、Guardrail、RBAC、指标、限流、缓存、审计
- 集成测试：Gateway 限流与权限（不依赖 LLM）

运行：在项目根目录
  pytest week15/105_code/test_production_platform.py -v
  pytest week15/105_code/test_production_platform.py -v -k "test_route or test_guardrail"
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import pytest

# 从同目录加载 production_platform_demo（105_code 非合法包名，无法直接 import）
# 测试时设置占位 API Key，避免 demo 在 import 时 sys.exit
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-unit-tests")
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "production_platform_demo",
    os.path.join(_here, "production_platform_demo.py"),
)
demo = importlib.util.module_from_spec(_spec)
sys.modules["production_platform_demo"] = demo
_spec.loader.exec_module(demo)

route = demo.route
_guardrail_sql = demo._guardrail_sql
check_agent_access = demo.check_agent_access
MetricsCollector = demo.MetricsCollector
TokenBucketRateLimiter = demo.TokenBucketRateLimiter
ResponseCache = demo.ResponseCache
AuditLog = demo.AuditLog
SessionManager = demo.SessionManager
Gateway = demo.Gateway


# ---------- 单元测试 ----------

class TestRoute:
    def test_research_keywords(self):
        assert route("研究一下 AI 在医疗的应用", {}) == "deep_research_agent"
        assert route("分析市场趋势", {}) == "deep_research_agent"
        assert route("调查用户反馈", {}) == "deep_research_agent"

    def test_chatbi_keywords(self):
        assert route("查询销售额", {}) == "chatbi_agent"
        assert route("数据报表", {}) == "chatbi_agent"
        assert route("订单统计", {}) == "chatbi_agent"

    def test_general_keywords(self):
        assert route("北京天气", {}) == "general_agent"
        assert route("现在几点", {}) == "general_agent"
        assert route("查订单 ORD001", {}) == "general_agent"

    def test_default_fallback(self):
        assert route("你好", {}) == "general_agent"
        assert route("", {}) == "general_agent"


class TestGuardrailSql:
    def test_allow_select(self):
        ok, sql = _guardrail_sql("SELECT * FROM orders")
        assert ok is True
        assert "SELECT" in sql

    def test_reject_delete(self):
        ok, msg = _guardrail_sql("DELETE FROM orders WHERE 1=1")
        assert ok is False
        assert "DELETE" in msg

    def test_reject_update(self):
        ok, msg = _guardrail_sql("UPDATE orders SET amount=0")
        assert ok is False

    def test_reject_drop(self):
        ok, msg = _guardrail_sql("DROP TABLE orders")
        assert ok is False

    def test_reject_non_select(self):
        ok, msg = _guardrail_sql("INSERT INTO orders VALUES (1,2,3)")
        assert ok is False


class TestRBAC:
    def test_admin_all(self):
        assert check_agent_access("admin", "general_agent") is True
        assert check_agent_access("admin", "deep_research_agent") is True
        assert check_agent_access("admin", "chatbi_agent") is True

    def test_analyst_allowed(self):
        assert check_agent_access("analyst", "general_agent") is True
        assert check_agent_access("analyst", "chatbi_agent") is True
        assert check_agent_access("analyst", "deep_research_agent") is False

    def test_researcher_allowed(self):
        assert check_agent_access("researcher", "general_agent") is True
        assert check_agent_access("researcher", "deep_research_agent") is True
        assert check_agent_access("researcher", "chatbi_agent") is False

    def test_unknown_user_allowed(self):
        # 未配置用户视为允许（或按业务改为拒绝）
        assert check_agent_access("unknown_user", "general_agent") is True


class TestMetricsCollector:
    def test_record_and_snapshot(self):
        m = MetricsCollector()
        m.record("general_agent", 0.5, 0.01, None)
        m.record("general_agent", 0.6, 0.02, None)
        m.record("chatbi_agent", 1.0, 0.05, "TimeoutError")
        s = m.snapshot()
        assert s["requests_total"] == 3
        assert s["errors_total"] == 1
        assert s["by_agent"]["general_agent"] == 2
        assert s["by_agent"]["chatbi_agent"] == 1
        assert s["by_error_type"]["TimeoutError"] == 1
        assert s["cost_total"] == pytest.approx(0.08)


class TestTokenBucketRateLimiter:
    def test_allow_within_limit(self):
        r = TokenBucketRateLimiter(max_tokens=2, refill_per_sec=0.1)
        assert r.allow() is True
        assert r.allow() is True
        assert r.allow() is False

    def test_refill(self):
        r = TokenBucketRateLimiter(max_tokens=2, refill_per_sec=10.0)
        assert r.allow() is True
        assert r.allow() is True
        assert r.allow() is False
        time.sleep(0.2)
        assert r.allow() is True


class TestResponseCache:
    def test_set_get(self):
        c = ResponseCache(ttl_sec=10, max_size=10)
        assert c.get("u1", "msg1") is None
        c.set("u1", "msg1", {"response": "ok", "agent_id": "general_agent"})
        assert c.get("u1", "msg1") == {"response": "ok", "agent_id": "general_agent"}
        assert c.get("u1", "msg2") is None
        assert c.get("u2", "msg1") is None


class TestAuditLog:
    def test_log_and_recent(self):
        a = AuditLog(max_entries=5)
        a.log("user1", "chat", "general_agent", "s1", "trace1")
        a.log("user2", "access_denied", "deep_research_agent", None, "trace2")
        recent = a.recent(10)
        assert len(recent) == 2
        assert recent[0]["user_id"] == "user1"
        assert recent[1]["action"] == "access_denied"


# ---------- 集成测试（不调 LLM，仅测 Gateway 限流/权限/会话）----------

class TestGatewayIntegration:
    def test_rate_limit_returns_error(self):
        sm = SessionManager()
        metrics = MetricsCollector()
        # 只给 0 个令牌，全部被限流
        limiter = TokenBucketRateLimiter(max_tokens=0, refill_per_sec=0)
        gateway = Gateway(sm, metrics, limiter, None, AuditLog())
        out = gateway.handle_request("admin", "cli", "你好", None)
        assert out.get("error") == "rate_limit"
        assert "请稍后再试" in out.get("response", "")

    def test_forbidden_agent_returns_error(self):
        sm = SessionManager()
        metrics = MetricsCollector()
        limiter = TokenBucketRateLimiter(max_tokens=10, refill_per_sec=1)
        gateway = Gateway(sm, metrics, limiter, None, AuditLog())
        # analyst 不允许 deep_research_agent
        out = gateway.handle_request("analyst", "cli", "研究一下 AI", None)
        assert out.get("error") == "forbidden"
        assert "没有权限" in out.get("response", "")
