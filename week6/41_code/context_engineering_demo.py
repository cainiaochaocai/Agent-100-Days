"""
Week 6 上下文工程实战：多轮对话 + 时间窗口 + 参考内容注入

- 用 LangChain 组装每拍的 Context：System + 历史（最近 K 轮）+ 当前问题
- 可选：在本轮 user 消息中注入「参考内容」（模拟 RAG），放在用户问题前
- 使用 tiktoken 估算当前 messages 的 token 数

运行前在项目根目录配置 .env 中的 OPENAI_API_KEY。
建议在项目根目录执行：python week6/41_code/context_engineering_demo.py
"""

from dotenv import load_dotenv
load_dotenv() 

import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import tiktoken

assert os.getenv("OPENAI_API_KEY"), "请先在项目根目录 .env 中配置 OPENAI_API_KEY"

llm = ChatOpenAI(model="qwen-plus", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

SYSTEM = (
    "你是公司知识库助手。若存在【参考内容】，请仅根据参考内容回答，并尽量简洁；"
    "无法从资料得出时请说明。"
)
REF_SAMPLE = """[1] 差旅报销每人每年上限 5000 元，超出需特批。
[2] 报销需在返回后 30 日内提交 OA。"""


def keep_last_k_turns(history: list, k: int) -> list:
    """只保留最近 k 轮（每轮 2 条：user + assistant）。"""
    if k <= 0:
        return []
    return history[-(2 * k) :] if len(history) >= 2 * k else history


def count_tokens(messages: list, model: str = "gpt-4o-mini") -> int:
    """估算 messages 的总 token 数（按 OpenAI 编码）。"""
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    for m in messages:
        content = getattr(m, "content", None) or str(m)
        if isinstance(content, list):
            # 多模态等
            for part in content:
                if hasattr(part, "get") and "text" in part:
                    total += len(enc.encode(part["text"]))
        else:
            total += len(enc.encode(content))
    return total


def build_messages(
    history: list,
    user_input: str,
    ref: str | None,
    k: int = 3,
) -> list:
    """组装本拍要发给 LLM 的 messages：System + 最近 k 轮历史 + 当前 user（可选带参考内容）。"""
    window = keep_last_k_turns(history, k)
    if ref:
        user_content = f"【参考内容】\n{ref}\n\n【用户问题】\n{user_input}"
    else:
        user_content = user_input
    return (
        [SystemMessage(content=SYSTEM)]
        + window
        + [HumanMessage(content=user_content)]
    )


def main() -> None:
    history: list = []
    print("上下文工程实战（时间窗口 K=3）")
    print("输入 'ref' 开启本轮参考内容注入，'quit' 退出。\n")
    while True:
        user_input = input("你: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        use_ref = user_input.lower() == "ref"
        if use_ref:
            user_input = input("（请再输入问题）你: ").strip()
            if not user_input:
                continue
        ref = REF_SAMPLE if use_ref else None
        messages = build_messages(history, user_input, ref, k=3)
        n_tokens = count_tokens(messages)
        print(f"[Context 约 {n_tokens} token，历史仅保留最近 3 轮]\n")
        resp = llm.invoke(messages)
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=resp.content))
        print("助手:", resp.content, "\n")


if __name__ == "__main__":
    main()
