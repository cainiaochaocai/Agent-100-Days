from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

assert os.getenv("OPENAI_API_KEY"), "请先配置 OPENAI_API_KEY"

prompt = PromptTemplate.from_template(
    "Write an English paragraph about {topic} and list 3 vocabulary words."
)
model = ChatOpenAI(model="qwen-flash", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
output_parser = StrOutputParser()

# 使用 LCEL 表达式将 prompt、model、parser 串联起来
chain = prompt | model | output_parser

result = chain.invoke({"topic": "climate change"})
print(result)