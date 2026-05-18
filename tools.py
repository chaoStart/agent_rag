import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Union, Optional
from pydantic import Field
from langchain_core.tools import tool
from rapidfuzz import fuzz

from db.db_models import DB, Document, Knowledgebase, close_connection
from service.document_service import DocumentService
from service.knowledgebase_service import KnowledgebaseService
from service.models.embedding_service import embedding_model
from utils.settings import (
    retriever, llm_max_tokens,
    search_page, search_page_size,
    search_similarity_threshold, search_vector_similarity_weight, search_top
)
from rag.nlp.models_tokenizer_provider import Models_tokenizer_provider
from agent.utils import truncate_tool_result, format_chunks_for_display, format_doc_aggs_for_display, \
    truncate_text_by_tokens

from langgraph.prebuilt import InjectedState
from typing import Optional, Annotated


# --- 1. 定义一个辅助函数，用于从 State 中提取 allowed_kb_ids ---
# 这个函数会被 InjectedState 自动调用，将 state['allowed_kb_ids'] 注入到工具参数中
def get_current_kb_ids(state: Annotated[dict, InjectedState]) -> list[str]:
    """从 AgentState 中获取当前允许的 KB IDs"""
    return state.get("kb_ids", [])


def _fuzzy_match(keyword: str, candidates: list[dict], name_key: str = "name",
                 threshold: int = 60) -> list[dict]:
    """通用的模糊匹配逻辑：精确 → 子串包含 → rapidfuzz 模糊打分。

    Args:
        keyword: 搜索关键词
        candidates: 候选列表，每个元素是字典
        name_key: 字典中名称字段的 key
        threshold: rapidfuzz 模糊匹配的最低分数阈值

    Returns:
        匹配到的候选列表
    """
    if not keyword or not candidates:
        return candidates if not keyword else []

    keyword_lower = keyword.lower()

    # 第一阶段：精确匹配
    exact = [c for c in candidates if c[name_key].lower() == keyword_lower]
    if exact:
        return exact

    # 第二阶段：子串包含
    contains = [c for c in candidates if keyword_lower in c[name_key].lower()]
    if contains:
        return contains

    # 第三阶段：rapidfuzz 模糊打分
    scored = []
    for c in candidates:
        score = fuzz.partial_ratio(keyword_lower, c[name_key].lower())
        if score >= threshold:
            scored.append((c, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored]


def matching_kb_id(user_question: str, candidate_kb_id: Dict[str, str]) -> Union[str, List[str]]:
    """
    根据用户问题，从候选知识库字典中匹配最可能的知识库ID。
    匹配策略：精确匹配 -> 子串包含 -> rapidfuzz 模糊打分。

    Args:
        user_question: 用户的自然语言问题
        candidate_kb_id: 候选知识库字典，格式为 {kb_id: kb_name}

    Returns:
        匹配到的知识库ID，未找到返回 None
    """
    if not user_question or not candidate_kb_id:
        return None

    # 提取候选列表为字典格式，以便复用匹配逻辑
    candidates = [{"id": kb_id, "name": kb_name} for kb_id, kb_name in candidate_kb_id.items()]
    keyword_lower = user_question.lower()

    # --- 匹配逻辑开始 ---

    # 第一阶段：精确匹配（用户问题完全等于知识库名称）
    for item in candidates:
        if item["name"].lower() == keyword_lower:
            return [item["id"]]

    # 第二阶段：子串包含（知识库名称是用户问题的子串，或反之）
    # 通常我们检查知识库名是否在问题中，或者问题中的关键词是否在库名中
    for item in candidates:
        name_lower = item["name"].lower()
        # 情况1：知识库名出现在用户问题中 ("PC知识库" in "PC知识库里有哪些文件？")
        if name_lower in keyword_lower:
            return [item["id"]]
        # 情况2：用户问题中的某个词（去停用词后）完全匹配库名（简化版不做分词，直接看整体）
        # 这里主要处理情况1即可满足当前需求

    all_kb_id = []
    for item in candidates:
        all_kb_id.append(item["id"])
    return all_kb_id


@tool
def list_knowledge_bases(
        keyword: str = "",
        # 对 LLM 隐藏此参数 (include_in_schema=False)，防止模型看到并输出
        allowed_kb_ids: Annotated[list[str], Field(include_in_schema=False)] = None) -> str:
    """按名称模糊搜索知识库，返回匹配的知识库ID、名称、描述和文档数量。
    工具备注：支持精确匹配、子串包含和模糊打分三级匹配策略）。
    适用场景：用户提到知识库名称时定位其ID、了解可用知识库范围、在跨知识库查询之前定位知识库。
    核心约束：如果没有匹配结果，建议改用 search_documents 进行语义检索兜底。

    Args:
        keyword: 搜索关键词，按知识库名称匹配。为空时返回所有可访问的知识库
    """
    try:
        with DB.connection_context():
            query = Knowledgebase.select().where(Knowledgebase.deflag != '1')
            # 强制约束到允许的知识库范围
            if allowed_kb_ids:
                query = query.where(Knowledgebase.id.in_(allowed_kb_ids))
            kbs = list(query.dicts())
        close_connection()

        # 转为统一格式
        candidates = []
        for kb in kbs:
            candidates.append({
                "id": kb["id"],
                "name": kb.get("name", ""),
                "description": kb.get("description", ""),
                "doc_count": kb.get("doc_num", 0),
            })

        # 如果有关键词，进行模糊匹配
        if keyword.strip():
            candidates = _fuzzy_match(keyword, candidates, name_key="name")

        if not candidates:
            return f"未找到与「{keyword}」匹配的知识库。建议调用 search_documents 进行全量语义检索。"

        # result_lines = [f"找到 {len(candidates)} 个知识库:"]
        # for kb in candidates[:20]:
        #     result_lines.append(
        #         f"  - 名称: {kb['name']} | ID: {kb['id']} | 文档数: {kb['doc_count']} | 描述: {kb['description']}"
        #     )
        # return truncate_tool_result("\n".join(result_lines), llm_max_tokens)

        total = len(candidates)
        returned = min(total, 20)
        if total > 20:
            result_lines = [
                f"找到 {total} 个知识库（数量较多，仅返回前 {returned}个，请尝试缩小搜索范围以获取更精确的结果）:"]
        else:
            result_lines = [f"找到 {total} 个知识库:"]

        for kb in candidates[:20]:
            result_lines.append(
                f"  - 名称: {kb['name']} | ID: {kb['id']} | 文档数: {kb['doc_count']} | 描述: {kb['description']}"
            )

        return truncate_tool_result("\n".join(result_lines), llm_max_tokens)
    except Exception as e:
        logging.exception("list_knowledge_bases error")
        return f"查询知识库时出错: {str(e)}"


@tool
def list_documents(user_question: str,
                   keyword: str,
                   knowledge_base_ids: Optional[list[str]] = None
                   # 对 LLM 隐藏此参数 (include_in_schema=False)，防止模型看到并输出
                   # allowed_kb_ids: Annotated[Optional[list[str]], Field(exclude=True)] = None
                   ) -> str:
    """按文档名称模糊搜索，返回匹配文档的doc_id、名称、类型、大小和所属知识库ID。
    工具备注：支持精确匹配、子串包含和模糊打分三级匹配策略。
    适用场景：用户提到文档名称时定位doc_id，供 get_document_full_content 获取全文或 search_documents 做文档内检索。
    核心约束：
    1、如果按名称无匹配，说明用户描述的可能是文档内容而非文件名，应改用 search_documents 语义检索，从 doc_aggs 定位相关文档。
    2、返回多个候选时，将列表展示给用户选择，不要猜测。

    Args:
        user_question:用户查询的问题，辅助筛选知识库或文档名称
        keyword: 搜索关键词，按文档名称匹配
        knowledge_base_ids: 可选，限定搜索的知识库范围
    """

    # # 保证knowledge_base_ids的权限问题
    # # 将白名单转换为集合，降低时间复杂度
    # allowed_set = set(allowed_kb_ids)
    #
    # # 执行筛选
    # valid_ids = [kid for kid in knowledge_base_ids if kid in allowed_set]
    #
    # # 直接赋值 allowed_kb_ids 原列表，保持返回类型一致
    # knowledge_base_ids = valid_ids if valid_ids else allowed_kb_ids

    try:
        with DB.connection_context():
            kb_name_map: Dict[str, str] = {}
            # query = Document.select().where(Document.deflag != '1')
            query = Document.select().where(Document.status == '1')  # 启动的文档
            forbidden_query = Document.select().where(Document.status != '1')  # 禁用的文档
            if knowledge_base_ids:
                # 先提取一下知识库的名称name，再筛选去掉冗余的kb_id，获取真实知识库中的所有文档列表名称documents_list
                query_kbs = list(Knowledgebase.select(Knowledgebase.id, Knowledgebase.name)
                                 .where(Knowledgebase.id.in_(knowledge_base_ids)))

                for kb in query_kbs:
                    kb_name_map[kb.id] = kb.name

                knowledge_base_ids = matching_kb_id(user_question, kb_name_map)
                # 如果matching_kb_id()函数匹配到知识库id，则说明当前用户提问时指定知识库，想要查询知识库的信息（所有文件），此时keyword需要置空获取当前知识库中全部信息
                keyword = ""
                query = query.where(Document.knowledgebase_id.in_(knowledge_base_ids))
                forbidden_query = forbidden_query.where(Document.knowledgebase_id.in_(knowledge_base_ids))
            docs = list(query.dicts())
            forbidden_docs = list(forbidden_query.dicts())
        close_connection()

        # 转为统一格式
        candidates = []
        forbidden_candidates = []
        for doc in docs:
            candidates.append({
                "doc_id": doc["id"],
                "name": doc.get("name", ""),
                "type": doc.get("type", ""),
                "size": doc.get("size", 0),
                "kb_id": doc.get("knowledgebase_id", ""),
                "forbidden_doc": "No"
            })
        for doc in forbidden_docs:
            forbidden_candidates.append({
                "doc_id": doc["id"],
                "name": doc.get("name", ""),
                "type": doc.get("type", ""),
                "size": doc.get("size", 0),
                "kb_id": doc.get("knowledgebase_id", ""),
                "forbidden_doc": "Yes"
            })
        # 模糊匹配
        matched = _fuzzy_match(keyword, candidates, name_key="name")
        forbidden_matched = _fuzzy_match(keyword, forbidden_candidates, name_key="name")

        if not matched and not forbidden_matched:
            return f"未找到与「{keyword}」匹配的文档。建议调用 search_documents 进行语义检索，从 doc_aggs 中定位相关文档。"

        # result_lines = [f"找到 {len(matched)} 个匹配文档:"]
        # for doc in matched[:20]:  # 最多返回20个
        #     size_kb = doc['size'] / 1024 if doc['size'] else 0
        #     result_lines.append(
        #         f"  - 文档名: {doc['name']} | doc_id: {doc['doc_id']} | 类型: {doc['type']} | 大小: {size_kb:.1f}KB | 知识库ID: {doc['kb_id']}"
        #     )
        # return truncate_tool_result("\n".join(result_lines), llm_max_tokens)
        total = len(matched) + len(forbidden_matched)
        returned = min(total, 20)
        if total > 20:
            result_lines = [
                f"找到 {total} 个匹配文档（数量较多，仅返回前 {returned}个，请尝试缩小搜索范围以获取更精确的结果）:"]
        else:
            result_lines = [f"找到 {total} 个匹配文档:"]

        matched.extend(forbidden_matched)
        for doc in matched[:20]:
            size_kb = doc['size'] / 1024 if doc['size'] else 0

            result_lines.append(
                f"  - 文档名: {doc['name']} | doc_id: {doc['doc_id']} | 类型: {doc['type']} | 大小: {size_kb:.1f}KB | 知识库ID: {doc['kb_id']} |文档是否禁用{doc['forbidden_doc']}"
            )
        return truncate_tool_result("\n".join(result_lines), llm_max_tokens)
    except Exception as e:
        logging.exception("list_documents error")
        return f"查询文档时出错: {str(e)}"


@tool
def search_documents(query: str,
                     knowledge_base_ids: Optional[list[str]] = None,
                     doc_ids: Optional[list[str]] = None,
                     # 对 LLM 隐藏此参数 (include_in_schema=False)，防止模型看到并输出
                     allowed_kb_ids: Annotated[list[str], Field(include_in_schema=False)] = None) -> tuple[str, dict]:
    """核心检索工具。在知识库或指定文档中进行语义检索（chunk 级别）。
    工具备注：返回 chunks（匹配的文本片段，含文档名、内容、相似度）和 doc_aggs（按文档聚合的命中统计，可用于定位最相关的文档）。
    适用场景：事实问答直接检索回答、传 doc_ids 做文档内精准检索、从 doc_aggs 语义定位文档、多文档对比。
    核心约束：两个可选参数都不传时，自动在默认知识库范围内检索。

    Args:
        query: 检索问题，应具体明确
        knowledge_base_ids: 可选，限定检索的知识库范围
        doc_ids: 可选，限定检索的文档范围，可通过 list_documents 获取 doc_id
    """
    try:
        kb_ids = knowledge_base_ids or []
        doc_id_list = doc_ids or []

        # 校验 doc_ids 归属：只保留属于允许知识库的文档
        if doc_id_list and allowed_kb_ids:
            allowed_set = set(allowed_kb_ids)
            verified_doc_ids = []
            with DB.connection_context():
                for did in doc_id_list:
                    try:
                        found, doc = DocumentService.get_by_id(did)
                        if found and doc.knowledgebase_id in allowed_set:
                            verified_doc_ids.append(did)
                    except Exception:
                        pass
            close_connection()
            doc_id_list = verified_doc_ids
            if not doc_id_list:
                return "指定的文档不在当前允许访问的知识库范围内。"

        kbinfos = retriever.retrieval(
            query,
            embedding_model,
            kb_ids,
            search_page,
            search_page_size,
            search_similarity_threshold,
            search_vector_similarity_weight,
            top=search_top,
            doc_ids=doc_id_list if doc_id_list else None,
            aggs=True,  # agent 工具必须 aggs=True
        )

        chunks = kbinfos.get("chunks", [])
        doc_aggs = kbinfos.get("doc_aggs", [])
        references = {"chunks": chunks, "doc_aggs": doc_aggs}
        if not chunks:
            return "未检索到相关内容。请尝试调整查询关键词或扩大检索范围。", references

        # 格式化输出
        result_parts = []
        result_parts.append(f"共检索到 {len(chunks)} 个相关片段:")
        result_parts.append("")
        result_parts.append(format_chunks_for_display(chunks))
        result_parts.append("")
        result_parts.append(format_doc_aggs_for_display(doc_aggs))

        return truncate_tool_result("\n".join(result_parts), llm_max_tokens), references
    except Exception as e:
        logging.exception("search_documents error")
        return f"检索文档时出错: {str(e)}"


@tool
def get_document_full_content(doc_id: str,
                              # 对 LLM 隐藏此参数 (include_in_schema=False)，防止模型看到并输出
                              allowed_kb_ids: Annotated[list[str], Field(include_in_schema=False)] = None) -> tuple[
    str, list[dict]]:
    """获取指定文档的完整文本内容，用于文档总结、概览等需要全文的场景。
    核心约束：
    1、仅支持 PDF、DOCX、DOC 格式。文档过长时会自动截断并标记，截断部分可用 search_documents 补充检索。
    2、必须先通过 list_documents 获取 doc_id。此工具开销较大，简单问答应优先使用 search_documents。

    Args:
        doc_id: 文档ID，通过 list_documents 获取
    """
    try:
        # 1、连接ES
        from utils.settings import search_es_index
        from utils.es_conn_old import ESConnection
        es_conn = ESConnection()
        query_body = {
            "query": {
                "terms": {
                    "doc_id": [doc_id]
                }
            },
            "size": 200,
            "from": 0,
            "sort": [
                {
                    "create_timestamp_flt_double": {
                        "order": "asc"
                    }
                }
            ]
        }
        # 2、查询语句
        result = es_conn.search(query_body, search_es_index)
        # print("当前的单个文档的检索结果\n", result)
        full_text_list = []
        chunk_lists = result.body['hits']['hits']
        for item in chunk_lists:
            current_chunk = item["_source"]["content_with_weight"]
            full_text_list.append(current_chunk)
        full_text = '\n'.join(full_text_list)
        # Step 1: 获取文档信息
        is_found, doc = DocumentService.get_by_id(doc_id)
        close_connection()
        if not is_found:
            return f"未找到 doc_id={doc_id} 的文档。"

        # 校验文档归属于允许的知识库
        if allowed_kb_ids and doc.knowledgebase_id not in allowed_kb_ids:
            return f"文档「{doc.name}」不在当前允许访问的知识库范围内。"

        # 获取当前文档的知识库名称
        with DB.connection_context():
            query = Knowledgebase.select().where(Knowledgebase.deflag != '1')
            # 强制约束到允许的知识库范围
            if allowed_kb_ids:
                query = query.where(Knowledgebase.id.in_([doc.knowledgebase_id]))
            kbs = list(query.dicts())
        close_connection()

        # kb_names = []
        # for kb in kbs:
        #     kb_names.append(kb["name"])
        doc_name = doc.name
        file_type = doc.type

        doc_aggs = [{"doc_id": doc_id, "doc_name": doc_name}]

        # 构建references要求的chunks格式，
        reference_chunks = []
        for content in full_text_list:
            reference_chunks.append({
                "docnm_kwd": doc_name,
                "content_with_weight": content,
                "similarity": 1.0,
                "kb_id": kbs[0]["id"],
                "kb_name": kbs[0]["name"]
            })
        references = {"chunks": reference_chunks, "doc_aggs": doc_aggs}
        # # Step 2: 获取 MinIO 存储路径
        # bucket, file_path = DocumentService.get_storage_address(doc_id)
        # close_connection()
        # if not bucket or not file_path:
        #     return f"无法获取文档「{doc_name}」的存储地址。"
        #
        # # Step 3: 从 MinIO 下载文件
        # from utils.minio_conn import MinioDB
        # conn = MinioDB()
        # binary = conn.get(bucket, file_path)
        # if not binary:
        #     return f"无法从存储中下载文档「{doc_name}」。"
        #
        # # Step 4: 解析文件提取全文
        # from rag.dataProcessor.one import chunk
        #
        # # 从文件名或存储路径获取后缀
        # suffix = doc_name.split(".")[-1] if "." in doc_name else file_type
        # # 对于 doc/docx 且有 preview_link 的情况，实际文件是 PDF
        # if doc.preview_link and re.search(r"\.docx?$", doc_name, re.IGNORECASE):
        #     suffix = "pdf"
        # try:
        #     full_text = chunk(
        #         f"文档全文.{suffix}",
        #         binary=binary,
        #         callback=lambda *args, **kwargs: None  # agent 工具不需要进度回调
        #     )
        # except NotImplementedError:
        #     return f"文档「{doc_name}」的格式（{suffix}）暂不支持全文提取。仅支持 PDF、DOCX、DOC 格式。建议使用 search_documents 进行片段检索。"
        # except Exception as e:
        #     logging.exception("chunk parsing error")
        #     return f"解析文档「{doc_name}」时出错: {str(e)}。建议使用 search_documents 进行片段检索。"

        if not full_text:
            return f"文档「{doc_name}」解析后内容为空。"

        # Step 5: 基于 token 截断
        # max_content_tokens = int(llm_max_tokens * 0.6)
        max_content_tokens = int(llm_max_tokens * 2)
        truncated_text, was_truncated = truncate_text_by_tokens(full_text, max_content_tokens)

        result_parts = [
            f"文档名: {doc_name}",
            f"内容是否被截断: {'是' if was_truncated else '否'}",
            "",
            "--- 文档内容 ---",
            truncated_text
        ]

        if was_truncated:
            result_parts.append("")
            result_parts.append(
                "[注意: 文档内容已被截断。如需了解被截断部分的内容，可使用 search_documents 针对特定问题进行精准检索。]")

        # return "\n".join(result_parts), doc_aggs
        return "\n".join(result_parts), references
    except Exception as e:
        logging.exception("get_document_full_content error")
        return f"获取文档全文时出错: {str(e)}"


@tool
def get_current_datetime() -> str:
    """获取服务器当前的日期、时间和星期几。
    适用场景：用户提到相对时间（如"昨天"、"上周"、"最近三天"）时，应先调用此工具获取当前时间再做计算。"""
    now = datetime.now()
    weekday_map = {
        0: "星期一", 1: "星期二", 2: "星期三", 3: "星期四",
        4: "星期五", 5: "星期六", 6: "星期日"
    }
    weekday = weekday_map[now.weekday()]
    return f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')} {weekday}"


# 所有工具列表，供 agent 使用
ALL_TOOLS = [
    list_knowledge_bases,
    list_documents,
    search_documents,
    get_document_full_content,
    get_current_datetime,
]
