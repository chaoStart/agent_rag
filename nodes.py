import copy
import json
import logging
from typing import Generator
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, BaseMessage
from typing import Dict, Any, List, Union, Tuple
from agent.state import AgentState
from agent.tools import ALL_TOOLS
from agent.prompts import AGENT_SYSTEM_PROMPT
from agent.utils import calculate_messages_tokens
# 最大迭代次数
MAX_ITERATIONS = 8


def create_llm(llm_model, model_config: dict) -> ChatOpenAI:
    """根据数据库中的模型信息创建 ChatOpenAI 实例。

    Args:
        llm_model: 数据库中的模型对象 (Models ORM 实例)
        model_config: 模型配置字典 (temperature, top_p 等)
    """
    base_url = llm_model.docurl

    # 适配逻辑：如果 docurl 以 /chat/completions 结尾，则去除该后缀
    # 使用 rstrip 或者正则替换，防止多次出现或大小写问题
    if base_url.endswith("/chat/completions"):
        base_url = base_url[:-len("/chat/completions")]
    elif base_url.endswith("/chat/completions/"):
        base_url = base_url[:-len("/chat/completions/")]

    # 确保去除后末尾没有多余的斜杠（可选，视具体库的实现而定，通常保留或去除均可，但保持一致较好）
    # base_url = base_url.rstrip("/")

    llm = ChatOpenAI(
        base_url=base_url,
        api_key=llm_model.api_key,
        model=llm_model.name,
        temperature=model_config.get("temperature", 0.7),
        max_tokens=model_config.get("max_tokens", 4096),
        streaming=True,
    )
    return llm


def build_messages_for_llm(state: AgentState) -> list:
    """从 state 构建发送给 LLM 的消息列表。"""
    messages: List[BaseMessage] = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]

    if "messages" in state and state["messages"]:
        messages.extend(state["messages"])

    if len(messages) < 2:
        return messages

    deduped: List[BaseMessage] = []
    seen: Dict[Tuple[str, str], bool] = {}  # key: (消息类型, 内容签名)

    for msg in messages:
        msg_type = type(msg).__name__
        content = str(getattr(msg, 'content', ''))

        # 为ToolMessage生成签名（忽略tool_call_id）
        if isinstance(msg, ToolMessage):
            signature = (msg_type, content, str(getattr(msg, 'name', '')))
        else:
            # 其他类型直接用 content 作为签名
            signature = (msg_type, content)

        # 如果该签名已出现过，则跳过（去重）
        if signature in seen:
            continue

        seen[signature] = True
        deduped.append(msg)

    # 校验上下文信息长度是否超过模型最大上下文
    token_info = calculate_messages_tokens(deduped)
    # 如果超过上下文限制，不回复用户
    if token_info['total_tokens'] > 120000:   # 禁止回答
        warning_msg = AIMessage(
            content="查询文档长度已超出模型最大上下文，无法回复用户问题。"
        )
        deduped.append(warning_msg)
    return deduped


def agent_node_streaming(state: AgentState, llm_with_tools: ChatOpenAI) -> Generator:
    """Agent 节点的流式执行。

    此函数是一个生成器，yield 两种类型的事件:
    - {"type": "token", "data": "..."} — 流式文本 token
    - {"type": "tool_calls", "data": [...]} — 检测到工具调用
    - {"type": "thinking", "data": "..."} — 思考过程
    - {"type": "agent_plan", "data": "..."} — 第一轮执行计划
    - {"type": "done", "message": AIMessage} — 完成，返回完整的 AIMessage

    最终返回更新后的 state。
    """
    messages = build_messages_for_llm(state)
    iteration = state.get("iteration", 0)

    # 流式调用 LLM
    full_content = ""
    tool_calls = []
    thinking_content = ""

    try:
        for chunk in llm_with_tools.stream(messages):
            # 处理 tool_calls
            if chunk.tool_call_chunks:
                for tc_chunk in chunk.tool_call_chunks:
                    # 收集工具调用片段
                    # LangChain 会自动拼装 tool_call_chunks 到最终的 AIMessage
                    pass

            # 处理文本内容
            if chunk.content:
                content = chunk.content
                full_content += content
                yield {"type": "token", "data": content}

            # 处理 additional_kwargs 中可能的 thinking（部分模型支持）
            if hasattr(chunk, 'additional_kwargs') and chunk.additional_kwargs:
                reasoning = chunk.additional_kwargs.get("reasoning_content", "")
                if reasoning:
                    thinking_content += reasoning
                    yield {"type": "thinking", "data": reasoning}

    except Exception as e:
        logging.exception("LLM streaming error")
        yield {"type": "error", "data": str(e)}
        return

    # 获取完整的 AIMessage（通过非流式收集）
    # 使用 invoke 获取完整响应以正确获取 tool_calls
    try:
        full_response = llm_with_tools.invoke(messages)
    except Exception as e:
        logging.exception("LLM invoke error for tool_calls collection")
        # 如果 invoke 也失败，构造一个基本的 AIMessage
        full_response = AIMessage(content=full_content)

    yield {"type": "done", "message": full_response}


def execute_tools(tool_calls: list, state: AgentState) -> list[ToolMessage]:
    """执行工具调用并返回 ToolMessage 列表。

    Args:
        tool_calls: AIMessage 中的 tool_calls 列表
        state: 当前状态（用于注入默认 kb_ids）
    """
    tool_map = {t.name: t for t in ALL_TOOLS}
    tool_messages = []

    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = copy.deepcopy(tc["args"]) # 深拷贝解决参数传递问题

        # --- 修正 JSON 字符串参数 ---
        for key, val in list(tool_args.items()):
            if isinstance(val, str):
                val_stripped = val.strip()
                if (val_stripped.startswith('[') and val_stripped.endswith(']')) or \
                        (val_stripped.startswith('{') and val_stripped.endswith('}')):
                    try:
                        tool_args[key] = json.loads(val_stripped)
                    except (json.JSONDecodeError, ValueError):
                        pass

        # --- 注入权限参数 (用于工具内部逻辑) ---
        request_kb_ids = state.get("kb_ids", [])
        request_user_question = state.get("question", ' ')
        if request_kb_ids:
            request_kb_set = set(request_kb_ids)

            if tool_name in ("search_documents", "list_documents"):
                # 强制修正 knowledge_base_ids
                llm_kb_ids = tool_args.get("knowledge_base_ids")
                if llm_kb_ids:
                    filtered = [kid for kid in llm_kb_ids if kid in request_kb_set]
                    # 如果过滤后为空，则使用全部允许的 ID
                    tool_args["knowledge_base_ids"] = filtered if filtered else request_kb_ids
                else:
                    # 如果 LLM 没传，则补全为全部允许的 ID
                    tool_args["knowledge_base_ids"] = request_kb_ids
                    tool_args["user_question"] = request_user_question

            if tool_name == "search_documents":
                llm_doc_ids = tool_args.get("doc_ids")
                if llm_doc_ids:
                    tool_args["allowed_kb_ids"] = request_kb_ids

            elif tool_name == "list_knowledge_bases":
                tool_args["allowed_kb_ids"] = request_kb_ids

            elif tool_name == "get_document_full_content":
                tool_args["allowed_kb_ids"] = request_kb_ids

        # --- 执行工具 ---
        # 此时 tool_args 包含修正后的 kb_ids 和 allowed_kb_ids，工具内部可正常校验
        if tool_name not in tool_map:
            result = f"未知工具: {tool_name}"
        else:
            try:
                if tool_name == "get_document_full_content":
                    real_tool_args = extract_tool_params(tool_args) #对参数进行预处理，主要用于提取文档的doc_id和知识库kb_id
                    result, references = tool_map[tool_name].invoke(real_tool_args)
                    depulicate_references(state, references)
                elif tool_name == "search_documents":
                    result, references = tool_map[tool_name].invoke(tool_args)
                    depulicate_references(state, references)
                else:
                    result = tool_map[tool_name].invoke(tool_args)
            except Exception as e:
                logging.exception(f"Tool {tool_name} execution error")
                result = f"工具 {tool_name} 执行出错: {str(e)}"

        # --- 生成 ToolMessage ---
        tool_messages.append(ToolMessage(
            content=str(result),
            tool_call_id=tc["id"],
            name=tool_name,
        ))

    return tool_messages


def should_continue(state: AgentState) -> str:
    """判断应该路由到 tool_node 还是 finish_node。

    检查最后一条消息：
    - 如果有 tool_calls → 路由到 "tools"
    - 如果是纯文本回答 → 路由到 "finish"
    - 如果超过最大迭代次数 → 路由到 "finish"
    """
    messages = state.get("messages", [])
    iteration = state.get("iteration", 0)

    if iteration >= MAX_ITERATIONS:
        return "finish"

    if messages:
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"

    return "finish"


def depulicate_references(state: AgentState, references: dict):
    if not state['references']['chunks'] and not state['references']['doc_aggs']:
        state["references"] = references
    else:
        if any(doc["doc_id"] == references["doc_aggs"][0]["doc_id"] for doc in state['references']['doc_aggs']):
            pass
        else:
            state["references"]["doc_aggs"].append(references["doc_aggs"][0])
            state["references"]["chunks"].extend(references["chunks"])


def extract_tool_params(data: Dict[str, Any]) -> Dict[str, Union[List[str], str]]:
    """
        提取 allowed_kb_ids 和 doc_id。
        如果数据已经是规范格式，则直接返回；如果是原始复杂格式，则进行提取。
        """
    # 1. 先判断数据是否已经是规范格式
    # 规范格式的特征：包含 'allowed_kb_ids' 和 'doc_id' 键，且不包含 'function' 键
    # 如果已经是规范格式，直接返回
    if isinstance(data, dict) and "allowed_kb_ids" in data and "doc_id" in data:
        return {
            "allowed_kb_ids": data["allowed_kb_ids"],
            "doc_id": str(data["doc_id"])
        }

    result = {}

    # 提取 allowed_kb_ids
    if "allowed_kb_ids" in data:
        result["allowed_kb_ids"] = data["allowed_kb_ids"]

    # 提取 doc_id
    doc_id = None
    if "function" in data and isinstance(data["function"], dict):
        args_str = data["function"].get("arguments")

        if isinstance(args_str, str):
            doc_id = parse_doc_id_from_arguments(args_str)

    # 兜底：在顶层查找 doc_id
    if not doc_id and "doc_id" in data:
        doc_id = data["doc_id"]

    if doc_id:
        result["doc_id"] = str(doc_id)

    return result


def parse_doc_id_from_arguments(args_str: str):
    """专门解析高度转义的 arguments 字符串"""
    try:
        # 1. 去除最外层引号
        s = args_str.strip()
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1]

        # 2. 多次替换转义字符
        s = s.replace('\\\\', '\\')  # 先把 \\\\ 变成 \
        s = s.replace('\\"', '"')  # 把 \" 变成 "
        s = s.replace('\\', '')  # 最后移除剩余的单 \

        # 3. 尝试 json 解析
        parsed = json.loads(s)
        if isinstance(parsed, dict) and "doc_id" in parsed:
            return parsed["doc_id"]

    except Exception:
        pass

    # 4. 正则兜底方案（非常可靠）
    # 匹配各种可能的 doc_id 写法
    patterns = [
        r'doc_id["\s]*[:=]["\s]*([^"\'\\,}\s]+)',
        r'"doc_id"\s*:\s*"([^"]+)"',
        r'doc_id["\']?\s*[:=]\s*["\']?([^"\'\\,}\s]+)'
    ]

    for pattern in patterns:
        match = re.search(pattern, args_str, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None
