import gradio as gr
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

assert os.getenv("OPENAI_API_KEY"), "请先配置 OPENAI_API_KEY"



def get_ai_response(user_message: str, chat_history: list) -> str:
    """
    调用大模型，生成回复内容
    - user_message: 当前用户输入
    - chat_history: 过往对话历史（可用于上下文）
    """

    english_tutor_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an English Learning Assistant.

            Your primary role is to help learners improve their English through:
            - clear explanations
            - gentle corrections
            - practical examples
            - guided practice

            Follow these principles at all times:
            1. Be encouraging and patient. Never criticize the learner.
            2. Correct mistakes politely and explain why they are wrong.
            3. Use simple, clear language unless the learner asks for advanced explanations.
            4. Always prefer examples over abstract rules.
            5. Adapt your response to the learner’s English level when possible.

            When responding:
            - If the learner makes a mistake, first show the corrected version.
            - Then explain the correction briefly.
            - Then provide 1–2 example sentences.
            - If appropriate, ask a short follow-up question to encourage practice.
            """.strip()
        ),
        ("human", "{user_message}")
    ])

    model = ChatOpenAI(model="qwen-flash", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    output_parser = StrOutputParser()

    # 使用 LCEL 表达式将 prompt、model、parser 串联起来
    chain = english_tutor_prompt | model | output_parser

    result = chain.invoke({"user_message": user_message})
    return result


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