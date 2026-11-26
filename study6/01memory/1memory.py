'''
用户画像保存
'''

import os
import re

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

assert os.getenv("OPENAI_API_KEY"), "请先在项目根目录 .env 中配置 OPENAI_API_KEY"

llm = ChatOpenAI(model="qwen-plus", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
history: list = []  # 对话历史（HumanMessage / AIMessage）
user_preferences: list[str] = []
K = 3

BASE_SYSTEM = "你是贴心助手。若存在【用户偏好】，请严格遵守其中的要求。"

def keep_last_k_turns(history: list, k: int) -> list:
    """只保留最近 k 轮（每轮 2 条：user + assistant）。"""
    if k <= 0:
        return []
    return history[-(2 * k) :] if len(history) >= 2 * k else history

def extract_preference(user_input: str) -> str | None:
    """简单规则：'记住：XXX' 或 '我偏好 XXX' 则返回 XXX。"""
    m = re.search(r"记住[：:]\s*(.+)", user_input)
    if m:
        return m.group(1).strip()
    m = re.search(r"我偏好\s*(.+)", user_input)
    if m:
        return m.group(1).strip()
    return None

def build_messages(user_input: str) -> list:
    pref_block = "用户偏好：" + ";".join(user_preferences) if user_preferences else ""
    system_content = (
        BASE_SYSTEM + ("\n\n【用户偏好】\n" + pref_block if pref_block else "")
    )
    window = keep_last_k_turns(history, K)
    return (
        [SystemMessage(content=system_content)]
        + window
        + [HumanMessage(content=user_input)]
    )

def main() -> None:
    print("LangChain 记忆实战（1.0+ 兼容）：对话历史（最近 3 轮）+ 用户偏好记忆")
    print("输入「记住：XXX」或「我偏好 XXX」可添加偏好，后续回复将遵守。输入 quit 退出。\n")
    while True:
        user_input = input("你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        pref = extract_preference(user_input)
        if pref:
            user_preferences.append(pref)
            print(f"[已记住偏好：{pref}]\n")
        messages = build_messages(user_input)
        resp = llm.invoke(messages)
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=resp.content))
        print("助手:", resp.content, "\n")

if __name__ == "__main__":
    main()



