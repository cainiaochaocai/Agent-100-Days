
from typing import Mapping, Any
from langchain_core.messages import AIMessageChunk, BaseMessageChunk
from langchain_openai.chat_models import base
from typing import cast

# 保存原始方法
_original_create_chunk = base._convert_delta_to_message_chunk


def _patched_convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    """
    Monkey patch:
    - 保留 OpenAI / Qwen 扩展字段（reasoning_content 等）
    """
    message_chunk = _original_create_chunk(_dict, default_class)

    try:
        # print(f"patch reasoning_content: {_dict}, {default_class}")
        role = cast(str, _dict.get("role"))
        additional_kwargs: dict = {}
        if _dict.get("reasoning_content"):
            additional_kwargs["reasoning_content"] = _dict["reasoning_content"]
        if role == "assistant" or default_class == AIMessageChunk:
            message_chunk.additional_kwargs = additional_kwargs
    except Exception:
        # 兜底，绝不能影响主流程
        pass

    return message_chunk


# 打猴子补丁
base._convert_delta_to_message_chunk = _patched_convert_delta_to_message_chunk

########################  以上代码必须放在开头  #############################
###############################################################

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

def get_model(is_reasoning):
    if is_reasoning:
        print("Using deep thinking...")
        return ChatOpenAI(model="qwen-flash", 
                          base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                          extra_body={"enable_thinking": True})
    else:
        return ChatOpenAI(model="qwen-max", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


def build_chain(deep_thinking: bool):
    model = get_model(deep_thinking)
    chain = english_tutor_prompt | model 

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="user_message",
        history_messages_key="chat_history",
    )
    return chain_with_history

def stream_ai_response(user_message: str, session_id: str, deep_thinking: bool):
    """
    调用大模型，生成回复内容
    - user_message: 当前用户输入
    - session_id: 会话ID（用于区分不同用户）
    """
    chain_with_history = build_chain(deep_thinking)

    answer_buffer = ""
    thinking_buffer = ""

    for chunk in chain_with_history.stream(
        {"user_message": user_message},
        config={"configurable": {"session_id": session_id}}
    ):
        if not isinstance(chunk, AIMessageChunk):
            continue

        # ===== 1. 深度思考（不进历史）=====
        if "reasoning_content" in chunk.additional_kwargs:
            thinking_buffer += chunk.additional_kwargs["reasoning_content"]

            # 只用于前端展示
            yield (
                f"<thinking>{thinking_buffer}</thinking>\n\n"
                f"{answer_buffer}"
            )

        # ===== 2. 最终回答（进历史）=====
        if chunk.content:
            answer_buffer += chunk.content

            yield (
                f"<thinking>{thinking_buffer}</thinking>\n\n"
                f"{answer_buffer}"
            )

def chat_handler(message: str, history: list, deep_thinking: bool):
    """
    Gradio ChatInterface 的回调函数
    负责：
    1. 接收用户输入
    2. 调用 LLM 生成回复
    3. 返回给前端展示
    """
    session_id = "user_001"  # 在实际应用中，应根据用户身份动态生成
    for partial in stream_ai_response(message, session_id, deep_thinking):
        yield partial


# 使用 Gradio 专门为聊天机器人设计的高层接口
chat_ui = gr.ChatInterface(
    fn=chat_handler,
    additional_inputs=[
        gr.Checkbox(label="深度思考", value=False)
    ],
    title="英语学习助手",
    description="支持普通模式 / 深度思考模式的英语学习助手"
)


if __name__ == "__main__":
    chat_ui.launch(share=True)  # share=True 会生成公网访问链接