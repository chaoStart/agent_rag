from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # 对话消息列表（含工具调用和结果），使用 add_messages reducer 自动追加
    messages: Annotated[list, add_messages]
    # 用户原始问题
    question: str
    # 请求中传入的知识库 ID（作为默认检索范围）
    kb_ids: list[str]
    # 累积的检索来源（chunks + doc_aggs）
    references: dict
    # 当前迭代次数
    iteration: int
    # 模型配置（temperature, top_p 等）
    model_config: dict
    # 已消耗的 token 数，用于防止上下文溢出
    total_tokens: int
