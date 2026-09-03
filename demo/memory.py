"""上下文压缩（工作记忆层） + 情景记忆落盘（长期层，生产级）。

两层记忆职责（对应我们讨论，面试口径）：
1. 【工作记忆 / 短期 = checkpointer，本文件只管"压缩"这一变换】
   多轮历史超上下文窗口阈值时，把「更早的整轮对话」滚动摘要，
   保留最近 N 轮原文 + 一条历史摘要。压缩结果仍写回 checkpointer（messages 表），
   不负责持久化——持久化由 checkpointer(Postgres)每步落盘保证。
   触发时机 = compress_node 每轮子Agent跑完后检查 token（总 token > 窗口 80%）。
   单位 = 「轮」（user+assistant 配对），不是零散 message 对象。
2. 【单轮内超窗（工具/检索返回过长）= 第1块压缩，在 tools.py 工具层压，不在本文件】
3. 【情景记忆 / 长期 = store，跨 thread_id，按 user_id 命名空间】
   本文件的 _persist_episode / recall_episodic 仅模拟 store 的语义索引：
   - 生产：每轮 AI 答完，把本轮抽取的可复用事实 put 进 store（带 embedding），
     与"工作记忆是否压缩"解耦；不是"只有被压缩淘汰的才落盘"。
   - recall_episodic 用 query embedding 做语义检索 top-k，决策前注入上下文。
   ⚠️ 注意：工作记忆本身已由 checkpointer 持久化（重启不丢、无状态）；
     情景记忆是叠加的跨会话检索层，不要让它兼职"防丢数据"。
   摘要用 SMALL_MODEL（便宜快）：LangChain memory 接受你传入的 llm 实例。
"""
import json
import os
import time
from typing import List

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage

from config import Settings, get_llm

# 情景记忆「数据库」模拟目录：每个 thread_id 一个 jsonl 文件，每行一条被压掉的轮次摘要。
# 生产：把下面这些文件读写换成 PostgreSQL（pgvector 存摘要 embedding）或 Milvus。
_EPISODIC_DIR = os.path.join(os.path.dirname(__file__), "episodic_store")


def _ensure_store_dir():
    os.makedirs(_EPISODIC_DIR, exist_ok=True)


def _store_path(thread_id: str) -> str:
    # 生产：thread_id 即会话主键，对应 checkpointer 的 thread_id
    return os.path.join(_EPISODIC_DIR, f"{thread_id}.jsonl")


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


# ============================================================
# 情景记忆落盘：把被压缩掉的一轮摘要写入「数据库」（此处用本地 jsonl 模拟）
# ============================================================
def _persist_episode(thread_id: str, turn_index: int, summary: str):
    """把某一轮被压掉的历史写入情景记忆存储。

    生产：此处改为 INSERT INTO episodic_memory(thread_id, turn_index, summary, embedding, ts)
    并用 pgvector 存 summary 的 embedding，供后续语义检索。
    """
    _ensure_store_dir()
    record = {
        "thread_id": thread_id,
        "turn_index": turn_index,
        "summary": summary,
        "ts": time.time(),
    }
    with open(_store_path(thread_id), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def recall_episodic(thread_id: str, query: str = "", top_k: int = 5) -> List[str]:
    """从情景记忆存储召回历史（演示「被压掉的内容还能找回」）。

    生产：用 query 的 embedding 去 pgvector/Milvus 做相似度检索 top_k 条。
    此处 demo 简化为「返回该会话全部已落盘的历史摘要」（无 embedding 检索）。
    """
    path = _store_path(thread_id)
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.append(f"[第{rec['turn_index']}轮摘要] {rec['summary']}")
    return out[-top_k:]


# ============================================================
# 核心：滚动压缩（按轮次）
# ============================================================
def compress_history(messages: List[BaseMessage], thread_id: str = "default",
                     keep_recent_turns: int = 3) -> List[BaseMessage]:
    """【第2块压缩 · 按轮次】对超阈值的多轮历史做滚动摘要。

    行为：
      1. 把 messages 按「轮」切分；
      2. 保留最近 keep_recent_turns 轮原文（不压）；
      3. 更早的轮次 → 逐轮用小模型摘要，并【落盘到情景记忆存储】（不丢失）；
      4. 返回 [SystemMessage(历史摘要)] + 最近 N 轮原文，供图内 update_state 替换白板。

    注意：压缩单位是「轮」，不是「消息条数」。一轮 = 一次 user 输入 + 对应 assistant 回复
    （可能含中间 ToolMessage）。这样保留/压缩的粒度更符合人类对话直觉，面试也好讲。
    """
    turns = split_into_turns(messages)
    # 整段 token 数（用于判断是否超窗口）
    total = sum(_count_tokens(getattr(m, "content", "") or "")
                for m in messages if hasattr(m, "content"))
    trigger = int(Settings.MAX_TOKEN_LIMIT * Settings.SUMMARY_TRIGGER_RATIO)

    # 轮数本来就少（没超 keep_recent_turns），或总 token 未超阈值 → 不压
    if len(turns) <= keep_recent_turns or total <= trigger:
        return messages

    # 切分：old_turns = 要被压缩的更早轮次；recent_turns = 保留原文的近轮
    old_turns = turns[:-keep_recent_turns]
    recent_turns = turns[-keep_recent_turns:]

    summary_parts = []
    for idx, turn in enumerate(old_turns):
        turn_text = _turn_to_text(turn)
        # 逐轮摘要（可累加之前摘要，形成递进压缩）
        summary = summarize_with_small_model(
            (summary_parts[-1] + "\n" if summary_parts else "") + turn_text,
            max_chars=200,
        )
        summary_parts.append(summary)
        # 落盘：被压掉的这轮写入情景记忆（生产 = 写数据库，不丢失）
        _persist_episode(thread_id, idx, summary)

    # 组装返回：[一条历史摘要 SystemMessage] + 最近 N 轮原文（摊平回消息列表）
    full_summary = "\n".join(f"[第{i}轮] {s}" for i, s in enumerate(summary_parts))
    out: List[BaseMessage] = []
    if full_summary:
        out.append(SystemMessage(content=f"[历史摘要 · 已从情景记忆归档]\n{full_summary}"))
    for turn in recent_turns:
        out.extend(turn)
    return out
