from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

assert os.getenv("OPENAI_API_KEY"), "请先配置 OPENAI_API_KEY"

llm = ChatOpenAI(model="qwen-flash", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

for chunk in llm.stream("解释什么是 Agent"):
    print(chunk.content, end="", flush=True)
