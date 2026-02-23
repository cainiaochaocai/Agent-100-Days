"""
Week 13 完整平台基础功能：Gateway + Runtime + 插件系统 + 多通道

- 插件接口：ProviderPlugin、ToolPlugin、MemoryPlugin、ChannelPlugin
- ProviderPlugin：DashScopeProviderPlugin（统一 invoke / bind_tools）
- ToolPlugin：BuiltinToolPlugin（天气/时间/订单）、可选 FileToolPlugin
- MemoryPlugin：InMemoryMemoryPlugin、可选 SQLiteMemoryPlugin
- ChannelPlugin：CLIChannelPlugin、WebChannelPlugin（Flask）
- PluginManager：注册与获取各类型插件
- Gateway / SessionManager / Router / Runtime 与 Week 12 一致，Runtime 通过插件调用 LLM 与工具

运行前在项目根目录配置 .env（OPENAI_API_KEY ）。
CLI：python week13/90_code/platform_with_plugins_demo.py --channel cli
Web：python week13/90_code/platform_with_plugins_demo.py --channel web
     然后 POST http://localhost:5000/api/chat {"user_id":"web_user","message":"查北京天气"}
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import uuid
from abc import ABC, abstractmethod
from typing import Any

from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

# ---------- 环境 ----------

api_key = os.getenv("OPENAI_API_KEY") 
assert api_key, "请在项目根目录 .env 中配置 OPENAI_API_KEY"
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")


# ========== 插件接口（Day 85）==========

class BaseProviderPlugin(ABC):
    @abstractmethod
    def invoke(self, messages: list, **kwargs) -> Any:
        """调用 LLM，返回 AIMessage（含 tool_calls）。"""
        pass

    def bind_tools(self, tools: list) -> Any:
        """返回绑定工具后的可调用对象（用于 tool_calls）。"""
        raise NotImplementedError

    def get_model_info(self) -> dict:
        return {"name": "unknown", "version": "0"}


class BaseToolPlugin(ABC):
    @abstractmethod
    def get_tools(self) -> list:
        pass

    @abstractmethod
    def execute(self, tool_name: str, args: dict) -> str:
        pass

    def get_plugin_info(self) -> dict:
        return {"name": self.__class__.__name__, "tools": [t.name for t in self.get_tools()]}


class BaseMemoryPlugin(ABC):
    @abstractmethod
    def write(self, user_id: str, content: str, metadata: dict | None = None) -> str:
        """返回 memory_id。"""
        pass

    @abstractmethod
    def retrieve(self, user_id: str, query: str, limit: int = 10) -> list[dict]:
        pass

    def update(self, memory_id: str, updates: dict) -> None:
        pass

    def delete(self, memory_id: str) -> None:
        pass


class BaseChannelPlugin(ABC):
    @abstractmethod
    def receive(self) -> tuple[str, str, str | None]:
        """返回 (user_id, message, session_id?)。"""
        pass

    @abstractmethod
    def send(self, user_id: str, message: str) -> None:
        pass

    def get_channel_info(self) -> dict:
        return {"type": "unknown"}


# ========== ProviderPlugin 实现（Day 86）==========

class DashScopeProviderPlugin(BaseProviderPlugin):
    def __init__(self, model: str = "qwen-plus", base_url: str = BASE_URL, api_key: str = ""):
        self._llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key)
        self._model = model

    def invoke(self, messages: list, **kwargs) -> Any:
        return self._llm.invoke(messages, **kwargs)

    def bind_tools(self, tools: list) -> Any:
        return self._llm.bind_tools(tools)

    def get_model_info(self) -> dict:
        return {"name": "dashscope", "model": self._model}


# ========== ToolPlugin 实现（Day 86）==========

@tool
def get_weather(city: str) -> str:
    """当用户询问某地天气时，使用此工具查询该城市的天气。输入为城市名。"""
    fake_db = {"北京": "晴，25°C", "上海": "多云，22°C", "深圳": "阴，28°C"}
    return fake_db.get(city, f"{city}：暂无数据")


@tool
def get_current_time() -> str:
    """当用户问当前时间或日期时，使用此工具。"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def search_order(order_id: str) -> str:
    """根据订单号查询订单状态。示例中仅 ORD001 有数据。"""
    if order_id.strip().upper() == "ORD001":
        return "订单 ORD001：已支付，配送中"
    return f"错误：订单不存在（{order_id}）。请确认订单号是否正确。"


class BuiltinToolPlugin(BaseToolPlugin):
    """内置工具：天气、时间、订单。"""

    def get_tools(self) -> list:
        return [get_weather, get_current_time, search_order]

    def execute(self, tool_name: str, args: dict) -> str:
        tool_map = {t.name: t for t in self.get_tools()}
        fn = tool_map.get(tool_name)
        if not fn:
            return f"错误：未知工具 {tool_name}"
        try:
            result = fn.invoke(args or {})
            return str(result) if result is not None else ""
        except Exception as e:
            return f"执行失败: {e}"


class FileToolPlugin(BaseToolPlugin):
    """自定义 ToolPlugin：读文件、列目录（写文件省略，避免误删）。"""

    def get_tools(self) -> list:
        from langchain_core.tools import tool
        @tool
        def read_file(path: str) -> str:
            """读取文本文件内容。输入为文件路径。"""
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()[:4000] or "(空文件)"
            except FileNotFoundError:
                return f"错误：文件不存在 {path}"
            except Exception as e:
                return f"读取失败: {e}"

        @tool
        def list_dir(path: str) -> str:
            """列出目录下的文件和子目录。输入为目录路径。"""
            try:
                names = os.listdir(path or ".")
                return "\n".join(names[:100]) or "(空目录)"
            except FileNotFoundError:
                return f"错误：目录不存在 {path}"
            except Exception as e:
                return f"列出失败: {e}"

        return [read_file, list_dir]

    def execute(self, tool_name: str, args: dict) -> str:
        tool_map = {t.name: t for t in self.get_tools()}
        fn = tool_map.get(tool_name)
        if not fn:
            return f"错误：未知工具 {tool_name}"
        try:
            result = fn.invoke(args or {})
            return str(result) if result is not None else ""
        except Exception as e:
            return f"执行失败: {e}"


# ========== MemoryPlugin 实现（Day 87）==========

class InMemoryMemoryPlugin(BaseMemoryPlugin):
    def __init__(self) -> None:
        self._memories: dict[str, dict] = {}
        self._id_counter = 0

    def write(self, user_id: str, content: str, metadata: dict | None = None) -> str:
        self._id_counter += 1
        mid = f"mem_{self._id_counter}"
        self._memories[mid] = {"user_id": user_id, "content": content, "metadata": metadata or {}}
        return mid

    def retrieve(self, user_id: str, query: str, limit: int = 10) -> list[dict]:
        result = []
        for mid, m in self._memories.items():
            if m["user_id"] == user_id and (not query or query in m["content"]):
                result.append({"memory_id": mid, **m})
        return result[:limit]


class SQLiteMemoryPlugin(BaseMemoryPlugin):
    def __init__(self, db_path: str = ":memory:") -> None:
        self.db = sqlite3.connect(db_path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS memories (memory_id TEXT PRIMARY KEY, user_id TEXT, content TEXT, metadata TEXT)"
        )
        self.db.commit()

    def write(self, user_id: str, content: str, metadata: dict | None = None) -> str:
        mid = f"sql_{uuid.uuid4().hex[:12]}"
        self.db.execute(
            "INSERT INTO memories (memory_id, user_id, content, metadata) VALUES (?, ?, ?, ?)",
            (mid, user_id, content, json.dumps(metadata or {})),
        )
        self.db.commit()
        return mid

    def retrieve(self, user_id: str, query: str, limit: int = 10) -> list[dict]:
        if query:
            cur = self.db.execute(
                "SELECT memory_id, user_id, content, metadata FROM memories WHERE user_id = ? AND content LIKE ? LIMIT ?",
                (user_id, f"%{query}%", limit),
            )
        else:
            cur = self.db.execute(
                "SELECT memory_id, user_id, content, metadata FROM memories WHERE user_id = ? LIMIT ?",
                (user_id, limit),
            )
        rows = cur.fetchall()
        return [
            {"memory_id": r[0], "user_id": r[1], "content": r[2], "metadata": json.loads(r[3] or "{}") if r[3] else {}}
            for r in rows
        ]


# ========== ChannelPlugin 实现（Day 87）==========

class CLIChannelPlugin(BaseChannelPlugin):
    def __init__(self) -> None:
        self.user_id = "cli_user"

    def receive(self) -> tuple[str, str, str | None]:
        try:
            line = input("你: ").strip()
        except EOFError:
            return self.user_id, "quit", None
        return self.user_id, line, None

    def send(self, user_id: str, message: str) -> None:
        print("助手:", message)

    def get_channel_info(self) -> dict:
        return {"type": "cli", "user_id": self.user_id}


class WebChannelPlugin(BaseChannelPlugin):
    """通过 Flask 暴露 HTTP，receive 由 HTTP 请求驱动，不在此处阻塞。"""

    def __init__(self) -> None:
        self.user_id = "web_user"

    def receive(self) -> tuple[str, str, str | None]:
        raise NotImplementedError("Web 通道由 HTTP 请求驱动，不在此处 receive")

    def send(self, user_id: str, message: str) -> None:
        pass  # 由 Flask 在请求上下文中返回 response

    def get_channel_info(self) -> dict:
        return {"type": "web"}


# ========== PluginManager（Day 85）==========

class PluginManager:
    def __init__(self) -> None:
        self._providers: dict[str, BaseProviderPlugin] = {}
        self._tool_plugins: list[BaseToolPlugin] = []
        self._memory: BaseMemoryPlugin | None = None
        self._channels: dict[str, BaseChannelPlugin] = {}

    def register_provider(self, name: str, plugin: BaseProviderPlugin) -> None:
        self._providers[name] = plugin

    def get_provider(self, name: str | None = None) -> BaseProviderPlugin | None:
        if name:
            return self._providers.get(name)
        return next(iter(self._providers.values()), None)

    def register_tool_plugin(self, plugin: BaseToolPlugin) -> None:
        self._tool_plugins.append(plugin)

    def get_all_tools(self) -> list:
        tools = []
        for p in self._tool_plugins:
            tools.extend(p.get_tools())
        return tools

    def execute_tool(self, tool_name: str, args: dict) -> str:
        for p in self._tool_plugins:
            for t in p.get_tools():
                if t.name == tool_name:
                    return p.execute(tool_name, args)
        return f"错误：未知工具 {tool_name}"

    def register_memory(self, plugin: BaseMemoryPlugin) -> None:
        self._memory = plugin

    def get_memory(self) -> BaseMemoryPlugin | None:
        return self._memory

    def register_channel(self, name: str, plugin: BaseChannelPlugin) -> None:
        self._channels[name] = plugin

    def get_channel(self, name: str) -> BaseChannelPlugin | None:
        return self._channels.get(name)


# ========== SessionManager / Router / ContextBuilder（与 Week 12 一致）==========

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


ROUTE_RULES = [
    (["研究", "分析"], "research_agent"),
    (["查询", "数据", "订单", "天气", "时间"], "general_agent"),
]


def route(message: str, session: dict) -> str:
    msg_lower = (message or "").strip().lower()
    for keywords, agent_id in ROUTE_RULES:
        if any(kw in msg_lower for kw in keywords):
            return agent_id
    return "general_agent"


SYSTEM_PROMPT = (
    "你是助手。根据用户问题选择合适的工具获取信息，然后给出简洁回答。"
    "若工具返回错误信息，请根据错误调整，不要重复相同错误。"
)


def build_context(
    user_input: str,
    history: list[dict],
    memory_plugin: BaseMemoryPlugin | None,
    user_id: str,
    system: str = SYSTEM_PROMPT,
) -> list:
    messages = []
    if system:
        messages.append(SystemMessage(content=system))
    if memory_plugin and user_id:
        mems = memory_plugin.retrieve(user_id, user_input[:50], limit=3)
        if mems:
            mem_text = "\n".join(m["content"][:200] for m in mems)
            messages.append(SystemMessage(content=f"用户相关记忆（仅供参考）：\n{mem_text}"))
    for h in history:
        role, content = h.get("role", ""), h.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=user_input))
    return messages


# ========== Runtime（通过 PluginManager 调用 Provider + ToolPlugin）==========

def run_runtime(
    user_input: str,
    context: list[dict],
    session_id: str,
    user_id: str,
    session_manager: SessionManager,
    plugin_manager: PluginManager,
    max_steps: int = 10,
) -> str:
    provider = plugin_manager.get_provider()
    if not provider:
        return "错误：未配置 LLM Provider。"
    tools = plugin_manager.get_all_tools()
    if not tools:
        return "错误：未注册任何工具插件。"
    llm_with_tools = provider.bind_tools(tools)
    memory = plugin_manager.get_memory()

    messages = build_context(user_input, context, memory, user_id)

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
            content = plugin_manager.execute_tool(name, args)
            messages.append(ToolMessage(content=content, tool_call_id=tid))

    session_manager.append_history(session_id, "user", user_input)
    session_manager.append_history(session_id, "assistant", "达到最大步数。")
    return "达到最大步数，未得到最终回复。"


# ========== Gateway ==========

class Gateway:
    def __init__(self, session_manager: SessionManager, plugin_manager: PluginManager) -> None:
        self.session_manager = session_manager
        self.plugin_manager = plugin_manager

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

        response = run_runtime(
            user_input=message,
            context=history,
            session_id=session_id,
            user_id=user_id,
            session_manager=self.session_manager,
            plugin_manager=self.plugin_manager,
        )

        self.session_manager.update_session(session_id, {"last_agent": agent_id, "last_message": message})
        return {"response": response, "session_id": session_id, "agent_id": agent_id}


# ========== 入口：CLI / Web ==========

def create_platform() -> tuple[Gateway, PluginManager]:
    """创建并注册所有插件的平台实例。"""
    pm = PluginManager()
    pm.register_provider("dashscope", DashScopeProviderPlugin(api_key=api_key))
    pm.register_tool_plugin(BuiltinToolPlugin())
    pm.register_tool_plugin(FileToolPlugin())
    pm.register_memory(InMemoryMemoryPlugin())
    pm.register_channel("cli", CLIChannelPlugin())
    pm.register_channel("web", WebChannelPlugin())

    sm = SessionManager()
    gateway = Gateway(sm, pm)
    return gateway, pm


def main_cli() -> None:
    gateway, pm = create_platform()
    channel_plugin = pm.get_channel("cli")
    print("Week 13 完整平台（插件化）— CLI 通道")
    print("插件：DashScope Provider、BuiltinTool、FileTool、InMemoryMemory、CLI Channel")
    print("输入 quit 退出。\n")
    session_id = None
    while True:
        user_id, message, _ = channel_plugin.receive()
        if not message or message.lower() in ("quit", "exit", "q"):
            break
        result = gateway.handle_request(user_id, "cli", message, session_id)
        session_id = result.get("session_id")
        channel_plugin.send(user_id, result.get("response", ""))
        if result.get("agent_id"):
            print("  [路由:", result["agent_id"], "]")


def main_web() -> None:
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("Web 模式需要安装 flask: pip install flask")
        return
    app = Flask(__name__)
    gateway, pm = create_platform()

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
        return jsonify({"status": "ok", "plugins": "provider,tool,memory,channel"})

    print("Week 13 完整平台（插件化）— Web 通道")
    print("POST http://localhost:5000/api/chat  Body: {\"user_id\":\"web_user\",\"message\":\"查北京天气\"}")
    print("GET  http://localhost:5000/health")
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", choices=["cli", "web"], default="cli", help="通道：cli 或 web")
    args = parser.parse_args()
    if args.channel == "web":
        main_web()
    else:
        main_cli()
