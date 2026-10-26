from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
import gradio as gr

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "请先配置 OPENAI_API_KEY"


def invoke_ai_response(topic: str, history: list) -> str:
    """
    调用大模型，生成关于指定主题的英文段落和词汇列表
    - topic: 用户当前输入的主题
    - history: 历史对话，ChatInterface 会自动传入，这里暂不使用
    """
    prompt = PromptTemplate.from_template(
        "写一篇关于{topic}的段落，并列出3个词汇。"
    )
    model = ChatOpenAI(model="qwen-flash", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    output_parser = StrOutputParser()

    # 使用 LCEL 表达式将 prompt、model、parser 串联起来
    chain = prompt | model | output_parser

    result = chain.invoke({"topic": topic})
    return result

# gr 会生成一个聊天界面，用户输入主题后，AI 会生成一段内容和相关词汇
chat_interface = gr.ChatInterface(
    fn=invoke_ai_response,
    title="AI Response",
    description="输入一个主题，AI 将生成一段内容和相关词汇。",
)

if __name__ == "__main__":
    chat_interface.launch(share=True) # share=True 会生成公网访问链接