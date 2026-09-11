"""情景记忆落库：PostgreSQL + pgvector（生产级，不使用内存介质）。

教学要点（对应标准 Q3/Q5/Q10）：
1. 为什么情景记忆用 PostgreSQL + pgvector，而不是 Milvus？
   - 情景记忆的查询以「时间范围 + 精确/JOIN」为主（约 80%），向量检索只占少量；
   - PostgreSQL 有表结构、时间索引、复杂 SQL；pgvector 顺带 cover 向量检索，一套搞定；
   - Milvus 是纯向量库、无表结构/无 SQL，适合语义记忆（90% 相似度检索）。
2. 原文与向量存「同一行」：不需要额外建关系，自动关联（见 schema.sql）。
3. 环境要求（由使用者自备，本模块不代劳）：
   - 启动 PostgreSQL 实例并建库；
   - 执行 `CREATE EXTENSION IF NOT EXISTS vector;`（pgvector 扩展）；
   - 执行 schema.sql 建表（checkpointer 表也可由 checkpointer.setup() 建立）；
   - 设置环境变量 POSTGRES_URI。

依赖：pip install "langgraph-checkpoint-postgres[psycopg]" psycopg[binary] pgvector
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence

from langgraph.checkpoint.postgres import PostgresSaver

from config import Settings

# 连接串走环境变量，不硬编码（生产禁止写死凭据）
POSTGRES_URI = os.getenv("POSTGRES_URI", "")


def get_postgres_uri() -> str:
    """读取连接串；缺失时给出明确的处置指引（fail fast）。"""
    if not POSTGRES_URI:
        raise RuntimeError(
            "未配置 POSTGRES_URI。生产前置步骤：\n"
            "  1) 启动 PostgreSQL 并建库；\n"
            "  2) 执行 CREATE EXTENSION IF NOT EXISTS vector;\n"
            "  3) 执行 schema.sql 建表；\n"
            "  4) set POSTGRES_URI=postgresql://user:pass@host:5432/db"
        )
    return POSTGRES_URI


def get_checkpointer() -> PostgresSaver:
    """生产 checkpointer = 情景记忆的持久化实现，按 thread_id 落盘会话状态。

    LangGraph 用它保证：下一轮接着聊、进程重启不丢、任意实例可恢复（无状态化）。
    落盘时机由框架控制（图中每个节点产出后），与 compress_node 的 token 阈值无关。

    框架表（checkpoints / writes / threads）由 setup() 幂等创建并随版本维护，
    **不在 schema.sql 里手写**——避免与框架结构冲突、也避免命名上和业务表混淆。
    需要物理隔离时改用：PostgresSaver.from_conn_string(uri, schema="langgraph")。
    """
    saver = PostgresSaver.from_conn_string(get_postgres_uri()) #返回的是 LangGraph 的检查点器实例（绑定到那个库
    saver.setup()  # 幂等：首次创建框架表（业务无需手写）
    return saver   # setup() 是框架（LangGraph）的方法，怎么"知道"要建什么表、结构是什么---表定义是写死在 LangGraph 库源码里的


def insert_episodic_memories(episodes: Sequence[dict],
                              thread_id: str = "default",
                              user_id: Optional[str] = None) -> None:
    """批量写入「蒸馏后的结构化情景记忆」（替代旧的逐轮原文归档）。

    C13：每轮 answer_node 产出最终回复后，由后台线程异步调用——把本轮对话经 LLM
    蒸馏成 0~N 条 episode 写库。**全量原文不再落库**。
    embedding 在此处统一用 vector_store.Embedder 计算（修掉之前 NULL 的问题）。

    :param episodes: list of dict，每条含
        signal_type(str), content(str), entities(dict, 可空), importance(int 1-5)
        可选覆盖 thread_id / user_id
    :param thread_id: 会话主键（缺省取 "default"）
    :param user_id: 来源用户（缺省取 Settings.DEFAULT_USER_ID）
    """
    if not episodes:
        return
    import psycopg
    from pgvector.psycopg import register_vector
    from vector_store import Embedder

    emb = Embedder()
    uid = user_id or Settings.DEFAULT_USER_ID
    rows = []
    for ep in episodes:
        content = (ep.get("content") or "").strip()
        if not content:
            continue
        vec = emb.embed(content)
        rows.append((
            ep.get("thread_id") or thread_id,
            ep.get("user_id") or uid,
            ep.get("signal_type"),
            content,
            json.dumps(ep.get("entities") or {}, ensure_ascii=False),
            int(ep.get("importance") or 3),
            vec,
        ))

    # 用与 checkpointer 相同的连接串开一条 PG 连接（上下文管理器，退出自动关闭）
    with psycopg.connect(get_postgres_uri()) as conn:
        # 注册 pgvector 类型适配器：让 psycopg 能正确收发 vector 列（embedding 字段）。
        # 不注册则 executemany 不认识 numpy/list 形式的向量，会报类型不支持错误。
        register_vector(conn)
        with conn.cursor() as cur:
            # 批量插入：一条参数化 SQL + rows 列表，一次写多条 episode（比逐条 INSERT 高效）
            cur.executemany(
                """
                INSERT INTO episodic_memory
                    (thread_id, user_id, signal_type, content, entities, importance, embedding)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s)
                """,
                rows,
            )
            # 显式提交事务（executemany 在上下文管理器内不会自动 commit，需提交才真正落库）
            conn.commit()


def _row_to_episode(row) -> dict:
    cols = ["signal_type", "content", "entities", "importance", "ts", "similarity"]
    ep = dict(zip(cols, row))
    if isinstance(ep.get("entities"), str):
        try:
            ep["entities"] = json.loads(ep["entities"])
        except json.JSONDecodeError:
            ep["entities"] = {}
    return ep


def search_episodic_by_vector(thread_id: str, query_embedding: Sequence[float],
                              top_k: Optional[int] = None) -> List[dict]:
    """按语义相似度召回本会话的历史情景记忆（pgvector 余弦距离，库内计算）。

    C13：返回完整 episode（content + entities + importance + similarity），供 recall_node
    注入上下文。按相似度降序取候选后，再按 (importance DESC, ts DESC) 排序（全量召回，
    不按 importance 过滤）。相似度在**数据库层**计算。

    :param query_embedding: 当前输入 embedding（同维度 1536）
    :param top_k: 默认 Settings.EPISODIC_TOP_K
    """
    import psycopg
    from pgvector.psycopg import register_vector

    top_k = top_k or Settings.EPISODIC_TOP_K
    with psycopg.connect(get_postgres_uri()) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT signal_type, content, entities, importance, ts,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM episodic_memory
                WHERE thread_id = %s AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (list(query_embedding), thread_id, list(query_embedding), top_k * 4),
            )
            cand = [_row_to_episode(r) for r in cur.fetchall()]
    # 全量召回排序：重要优先、时间新优先（不按 importance 过滤，仅排序加权）
    def _ts_key(e):
        ts = e.get("ts")
        return ts.timestamp() if ts else 0.0
    cand.sort(key=lambda e: (-int(e.get("importance") or 0), -_ts_key(e)))
    return cand[:top_k]


def search_episodic_by_entities(thread_id: str, entities: dict,
                                top_k: Optional[int] = None) -> List[dict]:
    """按 entities 字段做 JSONB 包含匹配（精确检索，Hybrid Search 的精确侧）。

    C13 触发条件①：用户提到订单号/工单号且 entities 命中即查，返回完整 episode。

    :param entities: 如 {"order_id": "A"}，用 `@>` 包含匹配
    """
    import psycopg
    from pgvector.psycopg import register_vector

    top_k = top_k or Settings.EPISODIC_TOP_K
    with psycopg.connect(get_postgres_uri()) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT signal_type, content, entities, importance, ts, NULL::float AS similarity
                FROM episodic_memory
                WHERE thread_id = %s AND entities @> %s::jsonb
                ORDER BY importance DESC, ts DESC
                LIMIT %s
                """,
                (thread_id, json.dumps(entities or {}, ensure_ascii=False), top_k),
            )
            return [_row_to_episode(r) for r in cur.fetchall()]
