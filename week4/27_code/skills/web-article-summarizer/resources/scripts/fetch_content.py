import requests
from bs4 import BeautifulSoup

def run(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 AgentSkillBot"
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # 移除无关元素
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = "\n".join(p.get_text().strip() for p in soup.find_all("p"))
    return text



if __name__ == "__main__":
    import sys
    url = sys.argv[1]
    content = run(url)
    print(content[:5000])  # 防止极端长度
