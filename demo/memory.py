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
from typing import List, Sequence, Tuple

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
    """把一轮对话转成纯文本，方便小模型摘要。

    只留问答正文，过滤工具类噪声（ToolMessage 是内部实现细节，不应进摘要污染、也省预算）。
    """
    parts = []
    for m in turn:
        if isinstance(m, ToolMessage):
            continue  # 工具返回是内部噪声，不进摘要
        role = ("用户" if isinstance(m, HumanMessage)
                else "助手" if isinstance(m, AIMessage)
                else "系统")
        content = getattr(m, "content", "") or ""
        if not content:
            continue
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
# 核心：滚动压缩（按轮次 · 两档触发 · 增量累积摘要）
# ============================================================
# 滚动摘要体量上限（约占窗口一小部分，避免摘要本身撑爆上下文）
SUMMARY_MAX_CHARS = 600


def compress_history(messages: Sequence[BaseMessage],
                     running_summary: str = "",
                     target_ratio: float = None) -> Tuple[List[BaseMessage], str]:
    """【第2块压缩 · 按轮次 · 两档触发】对超阈值多轮历史做滚动摘要。

    行为与主流（LangMem summarize）对齐：
      1. 按「轮」切分（一轮 = 一条 HumanMessage 起、到下一条 Human 前止）；
      2. 低于软触发线（窗口 60%）→ 原样返回，不压（摘要沿用旧值）；
      3. 软触发~硬顶（60%~80%）→ 把最老轮逐轮滚进滚动摘要，直到总量回落到软触发线以下
         （提前、少量、温和压缩，留大缓冲）；
      4. 超硬顶（>80%）→ 压到硬顶线以下；若压完保留区仍超硬顶（当前轮本身极大）→ 单轮兜底。

    滚动摘要 = 增量累积（旧 running_summary + 新滚入轮），非每次从零重压；
    窗口里始终 = [最近保留轮原文] + 一条由 state 承载的滚动摘要，不会长期堆几十轮。

    ⚠️ 本函数**只改工作记忆（窗口内容）**，不写任何数据库。
    返回 (压缩后消息列表, 更新后的滚动摘要)；摘要通过第 2 项传回，由调用方写入 state 字段，
    不再伪装成 messages 第 0 条的 SystemMessage。
    """
    turns = split_into_turns(messages)
    total = count_messages_tokens(messages)
    limit = Settings.MAX_TOKEN_LIMIT
    soft_limit = int(limit * Settings.SOFT_TRIGGER_RATIO)
    hard_limit = int(limit * Settings.HARD_LIMIT_RATIO)

    # 低于软触发线 → 不压（保留区 = 全部，摘要不变）
    if total <= soft_limit:
        return list(messages), running_summary

    # 目标水位：由调用方指定（软触发档→软触发线；硬顶档→硬顶线）
    target = int(limit * (target_ratio if target_ratio is not None else Settings.HARD_LIMIT_RATIO))

    # 从最新向最旧逐轮累计 token，预算内整轮保留原文（至少保留最新一轮），超出的轮滚进摘要
    recent_turns: List[List[BaseMessage]] = []
    budget = 0
    for turn in reversed(turns):
        t = count_messages_tokens(turn)
        if budget + t <= target or not recent_turns:  # 至少一个最新轮，避免返回空消息
            recent_turns.insert(0, turn)
            budget += t
        else:
            break
    old_turns = turns[: len(turns) - len(recent_turns)]

    # 超出的轮滚进增量累积滚动摘要
    summary = running_summary
    for turn in old_turns:
        summary = _roll_into_summary(summary, turn)

    # 单轮兜底：压完保留区仍超硬顶（当前轮本身极大）
    recent_tokens = count_messages_tokens([m for turn in recent_turns for m in turn])
    guard = 0
    while recent_tokens > hard_limit and guard < 10:
        guard += 1
        changed = False
        # ① 折叠最新一轮（进行中、可能多条子 Agent 结论）→ 合并成 1 条
        if recent_turns:
            collapsed = _collapse_turn(recent_turns[-1])
            if collapsed is not recent_turns[-1]:
                recent_turns[-1] = collapsed
                changed = True
        recent_tokens = count_messages_tokens([m for turn in recent_turns for m in turn])
        if recent_tokens <= hard_limit:
            break
        # ③ 把更前一轮（已规范成 [H,A] 的历史轮）整轮滚进摘要，向前扩展边界
        if len(recent_turns) > 1:
            summary = _roll_into_summary(summary, recent_turns.pop(0))
            changed = True
        recent_tokens = count_messages_tokens([m for turn in recent_turns for m in turn])
        if recent_tokens <= hard_limit:
            break
        if not changed:
            # ② 只剩最新一轮且仍超 → 对溢出部分硬截断（删溢出 token，非删完整结论）
            recent_turns[-1] = _truncate_turn(recent_turns[-1], hard_limit)
            recent_tokens = count_messages_tokens([m for turn in recent_turns for m in turn])
            break

    # 组装返回：只放最近保留轮原文；摘要通过第 2 项传出
    out: List[BaseMessage] = []
    for turn in recent_turns:
        out.extend(turn)
    return out, summary


def _roll_into_summary(summary: str, turn: List[BaseMessage]) -> str:
    """把一轮整轮滚进滚动摘要（增量累积：旧摘要 + 新轮）。"""
    turn_text = _turn_to_text(turn)
    text = (summary + "\n" + turn_text) if summary else turn_text
    return summarize_with_small_model(text, max_chars=SUMMARY_MAX_CHARS)


def _collapse_turn(turn: List[BaseMessage]) -> List[BaseMessage]:
    """折叠进行中轮：把轮内多条子 Agent 结论用 LLM 合并成 1 条 AIMessage（保留信息，非丢弃）。

    仅当轮内有多条 AI 结论时生效；单条 AI（已规范的一问一答）原样返回。
    """
    human = next((m for m in turn if isinstance(m, HumanMessage)), None)
    ai_msgs = [m for m in turn if isinstance(m, AIMessage)]
    if human is None or len(ai_msgs) <= 1:
        return turn
    merged_text = _turn_to_text(ai_msgs)
    merged = summarize_with_small_model(merged_text, max_chars=SUMMARY_MAX_CHARS)
    return [human, AIMessage(content=merged)]


def _truncate_turn(turn: List[BaseMessage], hard_limit: int) -> List[BaseMessage]:
    """极端兜底：把一轮内超出硬顶的部分硬截断（删溢出，非删完整结论）。"""
    out: List[BaseMessage] = []
    budget = 0
    for m in turn:
        t = count_messages_tokens([m])
        if budget + t <= hard_limit:
            out.append(m)
            budget += t
        else:
            room = max(0, hard_limit - budget)
            if room <= 0:
                break
            allowed_chars = room * 2  # 粗估 1 token≈2 字符
            content = getattr(m, "content", "") or ""
            if isinstance(m, (HumanMessage, AIMessage)) and content:
                out.append(type(m)(content=content[:allowed_chars] + "…[已截断]"))
                budget += room
            # 其他类型消息（如 SystemMessage）超预算则丢弃
            break
    return out
