import json
import traceback
from flask import request, Response, Blueprint
from timeit import default_timer as timer

from utils.api_utils import get_data_error_result, validate_request, server_error_response
from service.model_service import ModelService
from db.db_models import close_connection
from agent.graph import run_agent_graph

manager = Blueprint('agent_conversation', __name__)


@manager.route('/completion', methods=['POST'])
@validate_request("chatId", "messages", "modelId")
def agent_completion():
    """
    智能体对话接口
    ---
    tags:
      - 智能体对话
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            chatId:
                type: string
                example: "用户聊天的ID编号"
            messages:
                type: array
                items:
                  type: object
                  properties:
                    question:
                      type: string
                    answer:
                      type: string
                example: [{"question": "试用期规定是什么", "answer": ""}]
            knowledgeBaseIds:
                type: string
                example: "知识库ID列表，多个用逗号分隔（选填）"
            modelId:
                type: string
                example: "大模型的ID编号"
            modelConfig:
                type: string
                example: "模型配置JSON字符串（选填）"
    responses:
      200:
        description: SSE 流式输出
    """
    req = request.json
    if not req:
        return get_data_error_result(message="请求数据格式错误")

    # (1) messages 字段处理
    messages = []
    for m in req["messages"]:
        if m.get("question"):
            if len(messages) == 0 or messages[-1]["role"] != "user":
                messages.append({"role": "user", "content": m["question"]})
        if m.get("answer"):
            messages.append({"role": "assistant", "content": m["answer"]})

    if not messages:
        return get_data_error_result(message="用户聊天信息为空")

    # (2) knowledgeBaseIds 字段处理（选填）
    kb_ids = []
    if req.get("knowledgeBaseIds") and isinstance(req["knowledgeBaseIds"], str):
        kb_ids = [kid.strip() for kid in req["knowledgeBaseIds"].split(",") if kid.strip()]

    docids = []
    # 安全获取 metadataListInfo，不存在则默认为空列表
    metadata_list = req.get("metadataListInfo", [])
    for item in metadata_list:
        docids.append(item["docId"])

    # (3) modelId 字段处理
    model_id = req["modelId"]
    if not model_id:
        return get_data_error_result(message="请选择对应的大模型，再进行提问")

    is_get_model, llm_model = ModelService.get_by_id(model_id)
    close_connection()
    if not is_get_model:
        return get_data_error_result(message="大模型不存在，请联系管理员检查大模型配置")

    # (4) modelConfig 字段处理
    model_config = {"max_tokens": 4096}
    model_config_list = ["temperature", "top_p", "frequency_penalty", "presence_penalty", "max_tokens"]

    if req.get("modelConfig"):
        try:
            req_model_config = eval(req["modelConfig"]) if isinstance(req["modelConfig"], str) else req["modelConfig"]
            for key in model_config_list:
                if key in req_model_config and req_model_config[key] is not None:
                    model_config[key] = req_model_config[key]
        except Exception:
            pass

    # 提取用户最新问题
    question = [m["content"] for m in messages if m["role"] == "user"][-1]

    try:
        def stream():
            retrieval_ts_start = timer()

            # 发送 before 事件
            yield "data:" + json.dumps({
                "code": 0, "message": "", "data": "before", "type": "before"
            }, ensure_ascii=False) + "\n\n"

            # 发送知识检索事件
            yield "data:" + json.dumps({
                "code": 0, "message": "", "data": "知识检索", "type": "knowledge_retrieval"
            }, ensure_ascii=False) + "\n\n"

            references = {"chunks": [], "doc_aggs": []}
            retrieval_ts_end = None
            chat_ts_start = timer()

            try:
                for event in run_agent_graph(
                    llm_model=llm_model,
                    model_config=model_config,
                    question=question,
                    kb_ids=kb_ids,
                    history_messages=messages,
                ):
                    event_type = event.get("type", "")
                    event_data = event.get("data", "")

                    if event_type == "agent_plan":
                        yield "data:" + json.dumps({
                            "code": 0, "message": "",
                            "data": event_data, "type": "agent_plan"
                        }, ensure_ascii=False) + "\n\n"

                    elif event_type == "agent_step":
                        yield "data:" + json.dumps({
                            "code": 0, "message": "",
                            "data": event_data, "type": "agent_step"
                        }, ensure_ascii=False) + "\n\n"

                    elif event_type == "tool_call":
                        yield "data:" + json.dumps({
                            "code": 0, "message": "",
                            "data": event_data, "type": "tool_call"
                        }, ensure_ascii=False) + "\n\n"

                    elif event_type == "generate_answer":
                        if retrieval_ts_end is None:
                            retrieval_ts_end = timer()
                        yield "data:" + json.dumps({
                            "code": 0, "message": "",
                            "data": "生成回答", "type": "generate_answer"
                        }, ensure_ascii=False) + "\n\n"
                        chat_ts_start = timer()

                    elif event_type == "text":
                        yield "data:" + json.dumps({
                            "code": 0, "message": "",
                            "data": event_data, "type": "text"
                        }, ensure_ascii=False) + "\n\n"

                    elif event_type == "thinking":
                        yield "data:" + json.dumps({
                            "code": 0, "message": "",
                            "data": event_data, "type": "thinking"
                        }, ensure_ascii=False) + "\n\n"

                    elif event_type == "finish":
                        if isinstance(event_data, dict) and "references" in event_data:
                            ref = event_data["references"]
                            if ref.get("chunks"):
                                references["chunks"] = ref["chunks"]
                            if ref.get("doc_aggs"):
                                references["doc_aggs"] = ref["doc_aggs"]

            except Exception as e:
                traceback.print_exc()
                yield "data:" + json.dumps({
                    "code": 0, "message": str(e),
                    "data": "***智能体执行出错，请重试或联系管理员***",
                    "type": "text"
                }, ensure_ascii=False) + "\n\n"

            # 计算耗时
            if retrieval_ts_end is None:
                retrieval_ts_end = timer()
            chat_ts_end = timer()

            # 发送最终事件（包含 references 和耗时）
            yield "data:" + json.dumps({
                "code": 0, "message": "", "data": True,
                "references": references,
                "retrieval": f"{(retrieval_ts_end - retrieval_ts_start):.4f}s",
                "generate": f"{(chat_ts_end - chat_ts_start):.4f}s",
            }, ensure_ascii=False) + "\n\n"

        resp = Response(stream(), mimetype="text/event-stream")
        resp.headers.add_header("Cache-control", "no-cache")
        resp.headers.add_header("Connection", "keep-alive")
        resp.headers.add_header("X-Accel-Buffering", "no")
        resp.headers.add_header("Content-Type", "text/event-stream; charset=utf-8")
        return resp
    except Exception as e:
        return server_error_response(e)
