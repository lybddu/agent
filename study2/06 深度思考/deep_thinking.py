
from typing import Mapping, Any
from langchain_core.messages import AIMessageChunk, BaseMessageChunk
from langchain_openai.chat_models import base
from typing import cast

# 保存原始方法
_original_create_chunk = base._convert_delta_to_message_chunk


def _patched_convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    """
    Monkey patch:
    - 保留 OpenAI / Qwen 扩展字段（reasoning_content 等）
    """
    message_chunk = _original_create_chunk(_dict, default_class)

    try:
        # print(f"patch reasoning_content: {_dict}, {default_class}")
        role = cast(str, _dict.get("role"))
        additional_kwargs: dict = {}
        if _dict.get("reasoning_content"):
            additional_kwargs["reasoning_content"] = _dict["reasoning_content"]
        if role == "assistant" or default_class == AIMessageChunk:
            message_chunk.additional_kwargs = additional_kwargs
    except Exception:
        # 兜底，绝不能影响主流程
        pass

    return message_chunk

## 打开猴子补丁
base._convert_delta_to_message_chunk = _patched_convert_delta_to_message_chunk

###########################  上面必须保持不变  ###########################

import gradio as gr
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
# model = ChatOpenAI(model="qwen-flash", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

# 是否开启深度思考模式
def get_model(is_reasoning):
    if is_reasoning:
        print("开启深度思考模式")
        # 开关是 enable_thinking（不是 enable_telemetry）；
        # 且 qwen-flash 不支持思考模式，这里换成 qwen-plus
        return ChatOpenAI(model="qwen-plus",
                          base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                          extra_body={"enable_thinking": True})
    else:
        return ChatOpenAI(model="qwen-flash",
                          base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


output_parser = StrOutputParser()


# 构建带记忆的 chain，可以选择是否开启深度思考模式
def build_chain(deep_thinking: bool):
    model = get_model(is_reasoning=deep_thinking)
    chain = prompt | model

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_history,
        input_messages_key="user_message",
        history_messages_key="chat_history",
    )
    return chain_with_history

def stream_ai_response(user_message: str, session_id: str, deep_thinking: bool):

    chain_with_history = build_chain(deep_thinking)

    answer_buffer = ""
    thinking_buffer = ""

    for chunk in chain_with_history.stream(
        {"user_message": user_message},
        config={"configurable": {"session_id": session_id}}
    ):
        if not isinstance(chunk, AIMessageChunk):
            continue

        # 思考过程
        if chunk.additional_kwargs.get("reasoning_content"):
            thinking_buffer += chunk.additional_kwargs["reasoning_content"]

        # 最终回答
        if chunk.content:
            answer_buffer += chunk.content

        # 没有思考内容时（普通模式，或模型不支持），不要显示一对空的 <thinking></thinking>
        prefix = f"<thinking>{thinking_buffer}</thinking>\n\n" if thinking_buffer else ""
        yield f"{prefix}{answer_buffer}"


def chat_handler(message: str, history: list, deep_thinking: bool):
    """
    Gradio ChatInterface 的回调函数
    负责：
    1. 接收用户输入
    2. 调用 LLM 生成回复
    3. 返回给前端展示
    """
    session_id = "user_001"  # 在实际应用中，应根据用户身份动态生成
    for partial in stream_ai_response(message, session_id, deep_thinking):
        yield partial


# 使用 Gradio 专门为聊天机器人设计的高层接口
chat_ui = gr.ChatInterface(
    fn=chat_handler,
    additional_inputs=[
        gr.Checkbox(label="深度思考", value=False)
    ],
    title="主题介绍",
    description="支持普通模式 / 深度思考模式的主题介绍"
)

if __name__ == "__main__":
    chat_ui.launch(share=True)