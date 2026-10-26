import gradio as gr
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv


load_dotenv()

assert os.getenv("OPENAI_API_KEY"), "请先配置 OPENAI_API_KEY"

store = {}

def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{user_message}")
])
model = ChatOpenAI(model="qwen-flash", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
output_parser = StrOutputParser()

# 使用 LCEL 表达式将 prompt、model、parser 串联起来
chain = prompt | model | output_parser

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="user_message",
    history_messages_key="chat_history",
)

def invoke_ai_response_stream(user_message: str, session_id: str) -> str:

    partial_answer = ""

    for chunk in chain_with_history.stream(
        {"user_message": user_message},
        config={"configurable": {"session_id": session_id}}
    ):
        if chunk:
            partial_answer += chunk
            yield partial_answer

def chat_handler(user_message: str, history: list) -> str:

    # 实际使用可以动态改变
    session_id = "user_001"
    for partial in invoke_ai_response_stream(user_message, session_id):
        yield partial


def dump_store() -> str:
    """把 store 里的会话记忆渲染成 Markdown，看看模型每轮实际看到了什么。"""
    if not store:
        return "（store 还是空的，先聊一句再看）"
    role_names = {"human": "用户", "ai": "AI", "system": "系统"}
    lines = []
    for session_id, session_history in store.items():
        lines.append(f"### session_id: `{session_id}`（{len(session_history.messages)} 条）")
        for message in session_history.messages:
            role = role_names.get(message.type, message.type)
            lines.append(f"- **{role}**：{message.content}")
        lines.append("")
    return "\n".join(lines)


with gr.Blocks(title="主题解答") as demo:
    chat_interface = gr.ChatInterface(
        fn=chat_handler,
        title="主题解答",
        description="输入一个问题，AI 会结合上下文回答（边生成边显示）。",
    )
    show_memory_btn = gr.Button("查看 store 里的记忆")
    memory_view = gr.Markdown()
    show_memory_btn.click(fn=dump_store, outputs=memory_view)

if __name__ == "__main__":
    demo.launch(share=True) # share=True 会生成公网访问链接