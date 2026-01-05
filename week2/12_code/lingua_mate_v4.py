import gradio as gr
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
import time
import whisper
import edge_tts
from dotenv import load_dotenv

load_dotenv()

assert os.getenv("OPENAI_API_KEY"), "请先配置 OPENAI_API_KEY"

asr_model = whisper.load_model("turbo")


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


def speech_to_text(audio_path: str) -> str:
    """
    把用户语音转成文本
    """
    transcribed = asr_model.transcribe(audio_path)
    return transcribed["text"]

def text_to_speech(text: str) -> str:
    """
    输入文本 → 输出音频文件路径
    """
    print(f"[TTS] Processing: {text}")
    audio_path = f"./output_{int(time.time())}.mp3"
    communicate = edge_tts.Communicate(text, "en-GB-SoniaNeural")
    with open(audio_path, "wb") as file:
        for chunk in communicate.stream_sync():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
    return audio_path


def stream_ai_response(user_message: str, session_id: str):
    """
    调用大模型，生成回复内容
    - user_message: 当前用户输入
    - session_id: 会话ID（用于区分不同用户）
    """
   
    partial_answer = ""

    for chunk in chain_with_history.stream(
        {"user_message": user_message},
        config={"configurable": {"session_id": session_id}}
    ):
        if chunk:
            partial_answer += chunk
            yield partial_answer

def process_voice_and_stream(audio_path: str, history: list):
    """
    Gradio ChatInterface 的回调函数
    负责：
    1. 接收用户输入
    2. 调用 LLM 生成回复
    3. 返回给前端展示
    """
    user_text = speech_to_text(audio_path)
    if not user_text:
        yield history, None
        return
    
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": ""})
    yield history, None

    session_id = "user_001"  # 在实际应用中，应根据用户身份动态生成

    full_response = ""
    for partial in stream_ai_response(user_text, session_id):
        full_response = partial
        history[-1]["content"] = full_response
        # 实时逐块推送文本到 Chatbot
        yield history, None

    audio_reply = text_to_speech(full_response)
    yield history, audio_reply

with gr.Blocks(theme=gr.themes.Soft()) as chat_ui:
    gr.Markdown("# 🎙️ 流式多模态英语助手")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="请开口说英语 (Speak English)"
            )
            audio_output = gr.Audio(label="AI 语音回复", autoplay=True)
            
        with gr.Column(scale=2):
            # 使用 type="messages" 适配最新的 LangChain/Gradio 格式
            chatbot = gr.Chatbot(label="对话记录")
            clear_btn = gr.Button("清空对话")

    # 交互绑定
    # 当音频改变时触发
    audio_input.stop_recording(
        fn=process_voice_and_stream,
        inputs=[audio_input, chatbot],
        outputs=[chatbot, audio_output]
    )
    
    clear_btn.click(lambda: [], None, chatbot)


if __name__ == "__main__":
    chat_ui.launch(share=True)  # share=True 会生成公网访问链接