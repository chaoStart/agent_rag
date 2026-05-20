import json
import logging
import threading
import time
from agent.state import AgentState
from typing import Generator
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import create_llm, execute_tools, MAX_ITERATIONS, build_messages_for_llm
from agent.tools import ALL_TOOLS
from agent.prompts import AGENT_SYSTEM_PROMPT


# def _collect_stream(llm_with_tools, messages) -> tuple[AIMessage, Generator]:
#     """流式调用 LLM，返回完整的 AIMessage 和 token 生成器。
#
#     使用 astream 的同步版本，收集所有 chunks 并合并为完整 AIMessage。
#     同时在流式过程中可以逐 token 输出。
#     """
#     chunks = []
#     for chunk in llm_with_tools.stream(messages):
#         chunks.append(chunk)
#         yield chunk
#
#     # 合并所有 chunks 为完整 AIMessage
#     # LangChain 的 AIMessageChunk 支持 __add__ 操作
#     if chunks:
#         full_message = chunks[0]
#         for c in chunks[1:]:
#             full_message = full_message + c
#         # 转为 AIMessage
#         yield AIMessage(
#             content=full_message.content,
#             tool_calls=full_message.tool_calls if hasattr(full_message, 'tool_calls') else [],
#             additional_kwargs=full_message.additional_kwargs,
#         )


def run_agent_graph(llm_model: object, model_config: dict, question: str,
                    kb_ids: list[str], history_messages: list[dict]) -> Generator:
    """运行 Agent 状态图，以生成器方式 yield SSE 事件。

    流程:
        START → agent_node ⇄ tool_node → finish_node → END

    所有轮次统一使用流式调用 (stream)，逐 token 检测输出内容：
    - 若检测到 tool_calls → 收集完整后执行工具
    - 若为纯文本回答 → 直接将流式 token 转发为 SSE 事件

    Args:
        llm_model: 数据库中的模型对象
        model_config: 模型配置
        question: 用户问题
        kb_ids: 知识库 ID 列表
        history_messages: 历史消息列表 [{"role": "user"/"assistant", "content": "..."}]

    Yields:
        dict: SSE 事件，格式为 {"type": "...", "data": "..."}
    """
    # 创建 LLM 实例并绑定工具
    llm = create_llm(llm_model, model_config)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # 构建初始消息
    messages = []
    for msg in history_messages:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    # 初始化状态
    state: AgentState = {
        "messages": messages,
        "question": question,
        "kb_ids": kb_ids,
        "references": {"chunks": [], "doc_aggs": []},
        "iteration": 0,
        "model_config": model_config,
        "total_tokens": 0,
    }

    # 迭代执行 Agent 循环
    iteration = 0
    is_first_round = True

    # 定义工具名称
    tools_names = {"list_knowledge_bases": "查询知识库工具",
                   "list_documents": "查询文档工具",
                   "search_documents": "查询知识块工具",
                   "get_document_full_content": "获取文档全文工具",
                   "get_current_datetime": "获取时间工具"}

    while iteration < MAX_ITERATIONS:
        iteration += 1
        state["iteration"] = iteration

        # 发送 agent_step 事件
        yield {"type": "agent_step", "data": f"智能体思考中... (第 {iteration} 轮)"}

        # Agent Node: 流式调用 LLM
        llm_messages = build_messages_for_llm(state)
        try:
            # 流式收集所有 chunks，同时检测 tool_calls 或纯文本
            all_chunks = []
            streamed_text = ""
            has_tool_calls = False
            generate_answer_sent = False

            for chunk in llm_with_tools.stream(llm_messages):
                all_chunks.append(chunk)

                # 流式输出文本 token
                if chunk.content:
                    # 判断是中间轮次还是最终回答
                    # 在流式过程中无法提前知道是否有 tool_calls
                    if not generate_answer_sent:
                        # 将生成答案的标志，设置为True
                        generate_answer_sent = True
                        # 发送 generate_answer 事件
                        yield {"type": "generate_answer", "data": "生成回答"}

                    # 输出最终回答文本
                    yield {"type": "text", "data": chunk.content}

                # 处理思考过程
                if hasattr(chunk, 'additional_kwargs') and chunk.additional_kwargs:
                    reasoning = chunk.additional_kwargs.get("reasoning_content", "")
                    if reasoning:
                        yield {"type": "thinking", "data": reasoning}

            # 流式结束，合并所有 chunks 为完整 AIMessage
            if all_chunks:
                full_message = all_chunks[0]
                for c in all_chunks[1:]:
                    full_message = full_message + c

                ai_message = AIMessage(
                    content=full_message.content if hasattr(full_message, 'content') else "",
                    tool_calls=full_message.tool_calls if hasattr(full_message, 'tool_calls') else [],
                    additional_kwargs=full_message.additional_kwargs if hasattr(full_message,
                                                                                'additional_kwargs') else {},
                )
            else:
                ai_message = AIMessage(content="")

            TOOL_INTENT_KEYWORDS = [
                "获取文档", "get_document_full_content", "调用", "接下来",
            ]
            if not ai_message.tool_calls and ai_message.content and ai_message.additional_kwargs:
                # 检测是否有未执行的工具意图
                has_intent = any(kw in ai_message.content for kw in TOOL_INTENT_KEYWORDS)
                if has_intent and iteration < MAX_ITERATIONS:
                    # 追加一条提示，强制模型调用工具
                    nudge = HumanMessage(content=f"请直接调用对应的工具{ai_message.additional_kwargs['tool_calls'][0]['function']['name']}，不要描述你的计划。")
                    state["messages"].append(ai_message)
                    state["messages"].append(nudge)
                    # 拼接工具（函数）所需要的参数和类型
                    for tool_args in ai_message.additional_kwargs['tool_calls']:
                        structure_tool_args = {"args": tool_args, "id": tool_args['id'], "name": tool_args['function']['name'], "type": "tool_call"}
                        ai_message.tool_calls.append(structure_tool_args)
                        # state["messages"].append(ai_message)

                        # 若ai_message中有额外工具参数，则继续执行额外工具获取数据
                        # # 发送工具调用参数
                        # for tc in ai_message.tool_calls:
                        #     # 使用 .get() 增加容错率
                        #     tool_key = tc.get("name")
                        #     # 使用字典的 .get() 方法，如果 key 不存在则返回默认值（这里默认返回 tool_key 本身）
                        #     display_name = tools_names.get(tool_key, tool_key)
                        #
                        #     yield {
                        #         "type": "tool_call",
                        #         "data": {
                        #             "tool_name": display_name,
                        #             "args": tc["args"],
                        #         }
                        #     }
                        #
                        # tool_messages = None
                        # for event in _execute_tools_with_heartbeat(ai_message.tool_calls, state):
                        #     if isinstance(event, dict):  # 心跳事件
                        #         yield event
                        #     else:
                        #         tool_messages = event  # 最后一次返回结果
                        #
                        # state["messages"].extend(tool_messages)
                    ai_message.additional_kwargs = {}  # 应该在最后置空
            # 判断是否有工具调用
            if ai_message.tool_calls:
                has_tool_calls = True

                # 第一轮 LLM 输出的文本内容作为执行计划
                if is_first_round and ai_message.content:
                    yield {
                        "type": "agent_plan",
                        "data": ai_message.content
                    }
                    is_first_round = False

                # 将 AIMessage 追加到 state
                state["messages"].append(ai_message)

                # 发送工具调用参数
                for tc in ai_message.tool_calls:
                    # 使用 .get() 增加容错率
                    tool_key = tc.get("name")
                    # 使用字典的 .get() 方法，如果 key 不存在则返回默认值（这里默认返回 tool_key 本身）
                    display_name = tools_names.get(tool_key, tool_key)

                    yield {
                        "type": "tool_call",
                        "data": {
                            "tool_name": display_name,
                            "args": tc["args"],
                        }
                    }

                # tool_messages = execute_tools(ai_message.tool_calls, state)
                #
                # # 将 ToolMessage 追加到 state
                # state["messages"].extend(tool_messages)

                # ==================== 关键优化：带心跳的工具执行 ====================
                tool_messages = None
                for event in _execute_tools_with_heartbeat(ai_message.tool_calls, state):
                    if isinstance(event, dict):          # 心跳事件
                        yield event
                    else:
                        tool_messages = event            # 最后一次返回结果

                state["messages"].extend(tool_messages)

            else:
                # 无工具调用，这是最终回答
                is_first_round = False

                # 将最终回答追加到 state
                state["messages"].append(ai_message)

                # 完成
                break

        except Exception as e:
            logging.exception("Agent node error")
            yield {
                "type": "text",
                "data": f"***智能体执行出错: {str(e)}，请重试或联系管理员***"
            }
            break

    else:
        # 达到最大迭代次数
        yield {
            "type": "text",
            "data": "智能体达到最大工具调用次数限制，请简化问题后重试。"
        }

    # Finish Node: 发送结束事件
    yield {
        "type": "finish",
        "data": {
            "references": state["references"]
        }
    }


def _execute_tools_with_heartbeat(tool_calls, state: AgentState, timeout: int = 30, heartbeat_interval: float = 4.0):
    """返回生成器 + 最终结果"""
    result = [None]
    exception = [None]
    done = threading.Event()

    def worker():
        try:
            result[0] = execute_tools(tool_calls, state)
        except Exception as e:
            exception[0] = e
        finally:
            done.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    start = time.time()
    last_beat = start

    while not done.is_set():
        time.sleep(0.2)
        now = time.time()

        if now - last_beat >= heartbeat_interval:
            elapsed = int(now - start)
            # 可以根据工具名称定制提示语
            yield {
                "type": "tool_progress",
                "data": f"正在执行搜索工具... (已用 {elapsed} 秒，请耐心等待)"
            }
            last_beat = now

    thread.join(timeout=1.0)

    if exception[0]:
        raise exception[0]

    yield result[0]