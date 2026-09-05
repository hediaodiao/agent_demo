"""工作记忆压缩（上下文窗口层）。

职责边界（口径，务必分清）：
1. 【工作记忆 = 模型上下文窗口本身，易失，每轮现拼现用，不被"存储"】
   本文件只负责"压缩"这一变换：多轮历史超窗口阈值时，把更早的整轮对话滚动摘要，
   产出 [1 条历史摘要 SystemMessage] + [最近 N 轮原文]，由 compress_node
   用 graph.update_state 写回白板。
   触发时机 = compress_node 在每轮子 Agent 跑完后检查 token（总 token > 窗口 80%）。
   单位 = 「轮」（user+assistant 配对），不是零散 message 对象。
2. 【单轮内超窗（工具/检索返回过长）= 第1块压缩，在 tools.py 工具层压，不在本文件】
3. 【情景记忆 = checkpointer(PostgreSQL+pgvector) 按 thread_id 持久化的会话历史】
   ⚠️ 本文件**不再做任何落盘**：
   - 情景记忆落盘由框架在每个节点产出后自动完成，与压缩、与本文件完全解耦；
   - 压缩只改「窗口里的内容」，不是情景记忆的落库触发条件。
   摘要用 SMALL_MODEL（便宜快）：LangChain memory 接受你传入的 llm 实例。
"""
from typing import List, Sequence

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage

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


# ============================================================
# 轮次切分：把扁平的 messages 列表切成「对话轮」
# 一轮 = 从一条 HumanMessage 开始，到「下一条 HumanMessage 之前」结束
# （中间可能夹着 AIMessage / ToolMessage，都归到这一轮）
# ============================================================
def split_into_turns(messages: List[BaseMessage]) -> List[List[BaseMessage]]:
    """将消息列表按 HumanMessage 切分为若干「轮」。

    例：[H1, A1, T1, H2, A2] -> [[H1, A1, T1], [H2, A2]]
    最后若以非 Human 结尾（异常态）则并入上一轮。
    """
    turns: List[List[BaseMessage]] = []
    current: List[BaseMessage] = []
    for m in messages:
        if isinstance(m, HumanMessage) and current:
            turns.append(current)
            current = []
        current.append(m)
    if current:
        turns.append(current)
    return turns


def _turn_to_text(turn: List[BaseMessage]) -> str:
    """把一轮对话转成纯文本，方便小模型摘要（保留角色与内容）。"""
    parts = []
    for m in turn:
        role = ("用户" if isinstance(m, HumanMessage)
                else "助手" if isinstance(m, AIMessage)
                else "工具" if isinstance(m, ToolMessage)
                else "系统")
        content = getattr(m, "content", "") or ""
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def count_messages_tokens(messages: Sequence[BaseMessage]) -> int:
    """统计消息列表的总 token 数——压缩触发的**统一口径**。

    生产：压缩触发基于 token 计数（窗口 60%~80%），而不是字符数或轮数：
    - 字符数与真实 token 对不齐（中英文差异大）；
    - 轮数更不可靠（单轮可能极长或极短，与窗口上限对不齐）。
    graph.compress_node 与 compress_history 都用它，避免两套阈值口径。
    """
    return sum(_count_tokens(getattr(m, "content", "") or "")
               for m in messages if hasattr(m, "content"))


# ============================================================
# 核心：滚动压缩（按轮次）
# ============================================================
def compress_history(messages: Sequence[BaseMessage],
                     keep_recent_turns: int = 3) -> List[BaseMessage]:
    """【第2块压缩 · 按轮次】对超阈值的多轮历史做滚动摘要。

    行为：
      1. 把 messages 按「轮」切分；
      2. 保留最近 keep_recent_turns 轮原文（不压）；
      3. 更早的轮次 → 逐轮用小模型摘要；
      4. 返回 [SystemMessage(历史摘要)] + 最近 N 轮原文，供图内 update_state 替换白板。

    注意：压缩单位是「轮」，不是「消息条数」。一轮 = 一次 user 输入 + 对应 assistant 回复
    （子 Agent 只回结论后，一轮就是 Human + AI 结论，不再夹带 ToolMessage）。
    ⚠️ 本函数**只改工作记忆（窗口内容）**，不写任何数据库：
       情景记忆落盘由 checkpointer 在每个节点产出后自动完成，与压缩完全解耦。
    """
    turns = split_into_turns(messages)
    # 整段 token 数（统一口径：与 compress_node 共用 count_messages_tokens）
    total = count_messages_tokens(messages)
    trigger = int(Settings.MAX_TOKEN_LIMIT * Settings.SUMMARY_TRIGGER_RATIO)

    # 轮数本来就少（没超 keep_recent_turns），或总 token 未超阈值 → 不压
    if len(turns) <= keep_recent_turns or total <= trigger:
        return messages

    # 切分：old_turns = 要被压缩的更早轮次；recent_turns = 保留原文的近轮
    old_turns = turns[:-keep_recent_turns]
    recent_turns = turns[-keep_recent_turns:]

    summary_parts = []
    for turn in old_turns:
        turn_text = _turn_to_text(turn)
        # 逐轮摘要（可累加之前摘要，形成递进压缩）
        summary = summarize_with_small_model(
            (summary_parts[-1] + "\n" if summary_parts else "") + turn_text,
            max_chars=200,
        )
        summary_parts.append(summary)

    # 组装返回：[一条历史摘要 SystemMessage] + 最近 N 轮原文（摊平回消息列表）
    full_summary = "\n".join(f"[第{i}轮] {s}" for i, s in enumerate(summary_parts))
    out: List[BaseMessage] = []
    if full_summary:
        out.append(SystemMessage(content=f"[历史摘要]\n{full_summary}"))
    for turn in recent_turns:
        out.extend(turn)
    return out
