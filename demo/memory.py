"""上下文压缩：小模型摘要 + 滚动压缩（作为 LangGraph 节点调用）。

教学要点（对应我们之前的讨论）：
1. 【第2块压缩】多轮/多步历史超 token 阈值时，把「最早的步骤」滚动摘要，
   保留最近几步原文 + 一条历史摘要。触发时机 = 写回上下文后、下次决策前。
2. 长结果用小模型摘要：summarize_with_small_model 用 SMALL_MODEL（便宜快）。
   之前聊过：LangChain 的 memory 组件接受你传入的 llm 实例，模型由你指定。
"""
from typing import List
from langchain_core.messages import BaseMessage, SystemMessage

from config import Settings, get_llm

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    def _count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except Exception:
    def _count_tokens(text: str) -> int:
        return len(text) // 2  # fallback 粗估


def summarize_with_small_model(text: str, max_chars: int = 300) -> str:
    """长结果/历史摘要：用小模型把 raw 压成结论（对应讨论的「压缩用小模型」）。"""
    if _count_tokens(text) <= max_chars:
        return text
    llm = get_llm("small")
    prompt = f"请把下面内容压缩成不超过{max_chars}字的关键事实摘要，保留时间和数字：\n{text}"
    try:
        return llm.invoke(prompt).content
    except Exception:
        return text[:max_chars]


def compress_history(messages: List[BaseMessage], keep_recent: int = 6) -> List[BaseMessage]:
    """【滚动压缩节点】对超阈值的消息列表做滚动摘要。

    返回： [SystemMessage(历史摘要)] + 最近 keep_recent 条原文
    这是 LangGraph 图里的一个独立节点（对应讨论「压缩是硬编码逻辑点，
    框架不自动做，要你写在节点里」）。
    """
    total = sum(_count_tokens(m.content) for m in messages if hasattr(m, "content"))
    trigger = int(Settings.MAX_TOKEN_LIMIT * Settings.SUMMARY_TRIGGER_RATIO)
    if total <= trigger or len(messages) <= keep_recent:
        return messages  # 未超阈，不压

    summary = ""
    recent = list(messages)
    while total > trigger and len(recent) > keep_recent:
        oldest = recent.pop(0)
        summary = summarize_with_small_model(f"{summary}\n{oldest.content}", max_chars=200)
        total = sum(_count_tokens(m.content) for m in recent if hasattr(m, "content"))

    out: List[BaseMessage] = []
    if summary:
        out.append(SystemMessage(content=f"[历史摘要]\n{summary}"))
    out.extend(recent)
    return out
