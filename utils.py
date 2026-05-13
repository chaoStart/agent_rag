from rag.nlp.models_tokenizer_provider import Models_tokenizer_provider


def truncate_text_by_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    """基于 token 数量截断文本。

    Args:
        text: 要截断的文本
        max_tokens: 最大 token 数

    Returns:
        (截断后的文本, 是否被截断)
    """
    token_count = Models_tokenizer_provider.num_tokens_from_string(text)
    if token_count <= max_tokens:
        return text, False

    # 按比例估算字符截断位置，然后微调
    ratio = max_tokens / token_count
    estimated_chars = int(len(text) * ratio * 0.95)  # 留 5% 余量
    truncated = text[:estimated_chars]

    # 微调确保不超过 max_tokens
    while Models_tokenizer_provider.num_tokens_from_string(truncated) > max_tokens:
        estimated_chars = int(estimated_chars * 0.9)
        truncated = text[:estimated_chars]

    return truncated + "\n\n[内容已截断]", True


def truncate_tool_result(result_text: str, llm_max_tokens: int) -> str:
    """截断工具返回结果，每个结果不超过 llm_max_tokens * 0.3 token。"""
    max_tokens = int(llm_max_tokens * 0.3)
    truncated, was_truncated = truncate_text_by_tokens(result_text, max_tokens)
    return truncated


def format_chunks_for_display(chunks: list, max_chunks: int = 10) -> str:
    """将检索到的 chunks 格式化为可读文本。"""
    lines = []
    for i, chunk in enumerate(chunks[:max_chunks]):
        doc_name = chunk.get("docnm_kwd", "未知文档")
        content = chunk.get("content_with_weight", "")
        similarity = chunk.get("similarity", 0)
        lines.append(f"[{i+1}] 文档: {doc_name} (相似度: {similarity:.4f})")
        lines.append(f"    内容: {content[:500]}")
        lines.append("")
    if len(chunks) > max_chunks:
        lines.append(f"... 共 {len(chunks)} 条结果，仅展示前 {max_chunks} 条")
    return "\n".join(lines)


def format_doc_aggs_for_display(doc_aggs: list) -> str:
    """将文档聚合结果格式化为可读文本。"""
    if not doc_aggs:
        return "无文档聚合结果"
    lines = ["文档命中统计:"]
    for agg in doc_aggs:
        lines.append(f"  - {agg['doc_name']} (doc_id: {agg['doc_id']}, 命中: {agg['count']} 次)")
    return "\n".join(lines)
