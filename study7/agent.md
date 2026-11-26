# 一、 什么是agent

Agent = 在运行中依赖 LLM（或类似模型）进行“感知 → 决策 → 行动”，并根据行动结果再次感知与决策，形成闭环的多步系统。

# 二、 Agent 的核心特征：感知、决策、行动、循环与自主性

Agent 的价值在于：在“目标明确、边界清晰”的前提下，把“如何拆解、如何选工具、如何组合结果”交给模型在运行中决策，
从而支持主动执行、智能决策与持续优化。对开发者而言，需要从关注“流程怎么一步步写死”，转换到关注“目标、方法与策略怎么设定”，
让 Agent 在边界内自主完成闭环。


# 三、 Agent 基本组成

## 3.1 大脑（LLM）

## 3.2 工具 （Tools）   

## 3.3 技能（Skills）

## 3.4 记忆与上下文：状态与历史的承载

## 3.5 规划与执行循环, 各组成部分如何协同

# 四、 Agent 的失败模式

## 4.1 失败类型与缓解方向

| 失败类型       | 表现             | 常见成因               | 缓解方向                         |
| ---------- | -------------- | ------------------ | ---------------------------- |
| **幻觉**     | 编造信息、记错观察      | 上下文噪音、结果呈现不清       | 强调“仅依据工具结果”、结构化呈现            |
| **错误工具选择** | 选错工具、多余/遗漏调用   | 工具描述与场景边界不清        | 清晰描述、只暴露必要工具、System Prompt约束 |
| **错误参数**   | 类型/格式/取值错误、漏填  | 模型“猜”参数、schema 不清晰 | 明确 schema、示例、工具内校验           |
| **错误时机**   | 前置条件未满足就调用、未确认 | 缺少“先查再动”的规则说明      | 在 System 中规定顺序与确认策略          |

## 4.2 循环死锁


| 现象       | 表现                   | 缓解手段                            |
| -------- | -------------------- | ------------------------------- |
| **重复调用** | 同/等价 tool_call 反复执行  | 重复检测 + 注入“已执行过，请换策略或结束”         |
| **无法结束** | 始终有 tool_calls、无最终回复 | 最大步数、超时、在 System Prompt 中写清结束条件 |


# 五、 基于 Agno 开发 Agent

**Agno** 是一个面向“可学习”的多智能体系统**的 Python 框架：强调 Agent 在多次交互中**记住用户、积累知识、从决策中学习，而不是“每次对话都从零开始”。

环境要求：`agno`、`openai`（或其它 Agno 支持的模型依赖）、`python-dotenv`。本节示例需 `pip install agno`；若项目未包含，可单独安装后运行示例。

## 5.1 agent 定义

```python
from agno.agent import Agent
from agno.models.dashscope import DashScope

agent = Agent(
    model=DashScope(
        id="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    tools=[get_weather, get_current_time],  # 见下
    instructions="你是助手。根据用户问题选择合适的工具获取信息，然后给出简洁回答。",
    markdown=True,
)
agent.print_response("我叫小明，喜欢简洁回答。", user_id="user_1", session_id="s1", stream=True)
agent.print_response("你还记得我吗？", user_id="user_1", session_id="s2", stream=True) 
```


