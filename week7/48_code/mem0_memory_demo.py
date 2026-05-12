"""
Week 7 mem0 记忆功能实战：对话中检索记忆注入 Context，回复后写入新记忆

- 每轮：用当前用户输入做 memory.search()，将相关记忆注入 System
- 调用 LLM 得到回复后，用 memory.add() 把本轮 [user, assistant] 写入
- 需先安装：pip install mem0ai；并配置 OPENAI_API_KEY

建议在项目根目录执行：python week7/48_code/mem0_memory_demo.py
"""
from dotenv import load_dotenv
load_dotenv()

import os
from mem0 import Memory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

assert os.getenv("OPENAI_API_KEY"), "请先在项目根目录 .env 中配置 OPENAI_API_KEY"

# mem0 默认用 OpenAI 做推断与嵌入，会读 OPENAI_API_KEY
memory = Memory()
llm = ChatOpenAI(model="qwen-plus", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

BASE_SYSTEM = "你是贴心助手。若下面有【用户记忆】，请结合记忆回答，并保持友好一致。"


def chat_with_memories(user_input: str, user_id: str = "default_user") -> str:
    # 1. 检索相关记忆
    related = memory.search(query=user_input, user_id=user_id, limit=3)
    results = related.get("results") or []
    memories_str = "\n".join(f"- {r.get('memory', r)}" for r in results)

    # 2. 拼 System Prompt + 当前用户输入
    if memories_str:
        system_content = f"{BASE_SYSTEM}\n\n【用户记忆】\n{memories_str}"
    else:
        system_content = BASE_SYSTEM
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_input),
    ]
    resp = llm.invoke(messages)
    assistant_content = resp.content

    # 3. 本轮到记忆（让 mem0 从对话中推断并写入）
    memory.add(
        [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_content},
        ],
        user_id=user_id,
    )
    return assistant_content


def main() -> None:
    print("mem0 记忆实战：每轮检索相关记忆注入回复，回复后自动写入新记忆")
    print("输入 quit 退出。可先聊几句偏好（如喜欢科幻片），再问「我喜欢什么类型的电影？」观察是否用到记忆。\n")
    user_id = "demo_user"
    while True:
        user_input = input("你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        reply = chat_with_memories(user_input, user_id=user_id)
        print("助手:", reply, "\n")


if __name__ == "__main__":
    main()
