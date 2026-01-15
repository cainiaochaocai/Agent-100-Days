import importlib.util
from pathlib import Path
from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field


class TemplateInput(BaseModel):
    skill_name: str = Field(description="技能文件夹的名称，例如 'web-researcher'")
    template_name: str = Field(description="模板文件的全名，例如 'report_template.md'")
    
class SkillEngine:
    def __init__(self, skills_root="skills"):
        self.skills_root = Path(skills_root)

    def load_skill_prompt(self, skill_path: Path) -> str:
        content = skill_path.read_text(encoding="utf-8")
        if "---" in content:
            return content.split("---", 2)[2].strip()
        return content

    def load_templates(self):
        """扫描所有技能下的 templates 文件夹，返回 {skill_name: {template_name: content}}"""
        all_templates = {}
        for skill in self.skills_root.iterdir():
            template_dir = skill / "resources" / "templates"
            if not template_dir.exists():
                continue
            
            skill_templates = {}
            for temp_file in template_dir.glob("*.*"): # 支持 .md, .json, .txt 等
                skill_templates[temp_file.name] = temp_file.read_text(encoding="utf-8")
            
            if skill_templates:
                all_templates[skill.name] = skill_templates
        return all_templates

    def load_scripts_as_tools(self):
        tools = []
        # 基础脚本工具加载
        for skill in self.skills_root.iterdir():
            scripts_dir = skill / "resources" / "scripts"
            if not scripts_dir.exists():
                continue

            for script_file in scripts_dir.glob("*.py"):
                tool = self._load_script_as_tool(skill.name, script_file)
                if tool:
                    tools.append(tool)
        
        tools.append(self._create_template_reader_tool())
        return tools

    def _create_template_reader_tool(self):
        def read_template(skill_name: str, template_name: str) -> str:
            path = self.skills_root / skill_name / "resources" / "templates" / template_name
            if path.exists():
                return path.read_text(encoding="utf-8")
            return f"错误：在技能 '{skill_name}' 中找不到模板 '{template_name}'。"

        # 返回 StructuredTool 而不是 Tool
        return StructuredTool.from_function(
            func=read_template,
            name="read_skill_template",
            description="读取特定技能的输出模板资源。当你需要按照特定格式排版输出结果时使用。",
            args_schema=TemplateInput # 绑定参数定义
        )

    def _load_script_as_tool(self, skill_name: str, script_path: Path):
        module_name = f"{skill_name}_{script_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "run"):
            return None

        return Tool(
            name=f"{skill_name}_{script_path.stem}", # 注意：LangChain工具名建议用下划线
            func=module.run,
            description=f"执行技能 {skill_name} 的脚本: {script_path.stem}"
        )

    def load_all_skill_prompts(self):
        prompts = {}
        for skill in self.skills_root.iterdir():
            skill_md = skill / "SKILL.md"
            if skill_md.exists():
                prompts[skill.name] = self.load_skill_prompt(skill_md)
        return prompts

if __name__ == "__main__":
    engine = SkillEngine("skills")
    # 1. 动态加载工具
    tools = engine.load_scripts_as_tools()
    print(tools)

    # 2. 加载所有 Skill 中枢
    skill_prompts = engine.load_all_skill_prompts()
    print(skill_prompts)