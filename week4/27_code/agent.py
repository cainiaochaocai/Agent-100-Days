

# agent.py

from langchain.agents import create_agent
from skill_engine import SkillEngine
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

assert os.getenv("OPENAI_API_KEY"), "请先配置 OPENAI_API_KEY"

engine = SkillEngine("skills")

tools = engine.load_scripts_as_tools()

skill_prompts = engine.load_all_skill_prompts()

templates_registry = engine.load_templates()

system_prompt = f"""
你是一个具备多项专业技能的 Agent。

## 技能列表
{list(skill_prompts.keys())}

## 技能指令集
{skill_prompts}

## 可用输出模板 (Resources)
当技能指令要求使用特定模板时，请参考以下内容进行填充。
{templates_registry}

## 工作要求
1. 识别用户意图是否匹配上述技能。
2. 如果匹配，严格执行指令中的步骤。
3. 最终输出必须严格遵循对应技能 resources/templates 下的格式。
"""

print(system_prompt)

llm = init_chat_model(
                model="qwen-plus",
                model_provider="openai",
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )


agent = create_agent(
    tools=tools,
    model=llm,
    debug=True,
    system_prompt=system_prompt,
)

if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "帮我总结这篇文章：https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview"}]}
    )
    print(result['messages'][-1].content)
