import os
import re

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

# 关掉 mem0 的匿名遥测：既不会再有 PostHog 日志，
# 也不会额外打开 ~/.mem0/migrations_qdrant 这个内部集合（多进程时会抢不到锁）
os.environ.setdefault("MEM0_TELEMETRY", "False")

from mem0 import Memory

assert os.getenv("OPENAI_API_KEY"), "请先在项目根目录 .env 中配置 OPENAI_API_KEY"

llm = ChatOpenAI(model="qwen-plus", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
BASE_SYSTEM = "你是贴心助手。若下面有【用户记忆】，请结合记忆回答，并保持友好一致。"

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# mem0 默认用 OpenAI 官方接口，这里改用 DashScope 的兼容模式（LLM + Embedding 都走它）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")

memory = Memory.from_config(
    {
        "llm": {
            "provider": "openai",
            "config": {
                "model": "qwen-plus",
                "api_key": DASHSCOPE_API_KEY,
                "openai_base_url": DASHSCOPE_BASE_URL,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-v3",  # DashScope 向量模型
                "api_key": DASHSCOPE_API_KEY,
                "openai_base_url": DASHSCOPE_BASE_URL,
                "embedding_dims": 1024,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                # 旧集合 mem0 是 1536 维、且没有 bm25 槽位，换个新集合更省事
                "collection_name": "mem0_dashscope",
                "embedding_model_dims": 1024,
                "path": os.path.join(os.path.expanduser("~"), ".mem0", "qdrant"),
            },
        },
    }
)

def chat_with_memories(user_input: str, user_id: str = "default_user") -> str:
    # 1. 检索相关记忆
    related = memory.search(query=user_input, filters={"user_id": user_id}, top_k=3)
    results = related.get("results") or []
    memories_str = "\n".join(f"- {r.get('memory', r)}" for r in results)

    # 2. 拼 System Prompt + 当前用户输入
    if memories_str:
        system_content = f"{BASE_SYSTEM}\n\n【用户记忆】\n{memories_str}"
    else:
        system_content = BASE_SYSTEM
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_input),
    ]
    resp = llm.invoke(messages)
    assistant_content = resp.content

    # 3. 本轮到记忆（让 mem0 从对话中推断并写入）
    memory.add(
        [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_content},
        ],
        user_id=user_id,
    )
    return assistant_content


def main() -> None:
    print("mem0 记忆实战：每轮检索相关记忆注入回复，回复后自动写入新记忆")
    print("输入 quit 退出。可先聊几句偏好（如喜欢科幻片），再问「我喜欢什么类型的电影？」观察是否用到记忆。\n")
    user_id = "demo_user"
    while True:
        user_input = input("你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        reply = chat_with_memories(user_input, user_id=user_id)
        print("助手:", reply, "\n")


if __name__ == "__main__":
    main()
