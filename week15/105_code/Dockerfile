# Week 15 生产就绪平台示例 — 容器化（Day 103）
# 在项目根目录构建：docker build -t agent-platform-week15 -f week15/105_code/Dockerfile .
# 运行：docker run -p 5000:5000 -e OPENAI_API_KEY=xxx agent-platform-week15

FROM python:3.12-slim

WORKDIR /app

COPY week15/105_code/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY week15/105_code/production_platform_demo.py ./

ENV PYTHONUNBUFFERED=1
EXPOSE 5000

CMD ["python", "production_platform_demo.py", "--channel", "web"]
