import gradio as gr
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv

load_dotenv()

assert os.getenv("OPENAI_API_KEY"), "请先配置 OPENAI_API_KEY"



# 存储不同用户的记忆
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_ai_response(user_message: str, chat_history: list) -> str:
    """
    调用大模型，生成回复内容
    - user_message: 当前用户输入
    - chat_history: 过往对话历史（可用于上下文）
    """

    english_tutor_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a friendly English tutor.

            Help the learner improve step by step.
            Keep responses short, clear, and conversational.

            When correcting:
            - First show the corrected sentence.
            - Then give a very brief reason (1–2 short sentences).
            - Use simple, everyday English.
            - Encourage the learner to try again.

            Do not give long explanations or grammar lectures.
        """),
        MessagesPlaceholder(variable_name="chat_history"), # 记忆存放处
        ("user", "{user_message}"),
    ])

    model = ChatOpenAI(model="qwen-flash", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    output_parser = StrOutputParser()

    # 使用 LCEL 表达式将 prompt、model、parser 串联起来
    chain = english_tutor_prompt | model | output_parser

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="user_message",
        history_messages_key="chat_history",
    )
    response = chain_with_history.invoke(
        {"user_message": user_message},
        config={"configurable": {"session_id": "user_001"}}
    )
    return response


def chat_handler(message: str, history: list) -> str:
    """
    Gradio ChatInterface 的回调函数
    负责：
    1. 接收用户输入
    2. 调用 LLM 生成回复
    3. 返回给前端展示
    """
    return get_ai_response(message, history)


# 使用 Gradio 专门为聊天机器人设计的高层接口
chat_ui = gr.ChatInterface(
    fn=chat_handler,
    title="英语学习助手",
    description="一个基于 LLM 的对话式英语学习助手示例"
)


if __name__ == "__main__":
    chat_ui.launch(share=True)  # share=True 会生成公网访问链接