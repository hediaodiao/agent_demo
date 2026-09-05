-- ============================================================
-- 记忆落库：PostgreSQL + pgvector（生产级）—— checkpointer=短期记忆，episodic_memory=蒸馏后的情景记忆(结构化片段)
--
-- 前置：
--   CREATE DATABASE agent_db;
--   \c agent_db
--   CREATE EXTENSION IF NOT EXISTS vector;   -- pgvector 扩展
--
-- 执行：
--   psql -f schema.sql "postgresql://user:pass@host:5432/agent_db"
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- ------------------------------------------------------------
-- 1) LangGraph checkpointer 表（checkpoints / writes / threads）
--
--    ⚠️ 重要澄清（避免命名混淆）：
--      - 「checkpointer」是 LangGraph 的【持久化组件/抽象】(本项目实例=PostgresSaver)，
--        负责把每一步的 state 快照写库，即 LangGraph 的【短期 / 工作记忆】持久化实现
--        （thread-scoped，管会话连续性 / 断点恢复 / 无状态化；按主流 agent taxonomy
--         它属于 short-term memory，不是严格 episodic memory）。
--      - 「checkpoints」是它写入的【物理表】——两者是同一持久化机制的
--        "组件层" 与 "存储层"，不是两张不同的表，更不是我们的业务表。
--      - 框架表名由 LangGraph 硬编码，【不能改名】；且应由 checkpointer.setup()
--        自动创建与版本维护（见 db.get_checkpointer），【不要】在此手写 CREATE TABLE，
--        否则既易与框架结构冲突，又和下面的业务表 episodic_memory 命名混在一起造成误解。
--    因此本文件只保留我们自己的业务表（见第 2 节）。
--    如需物理隔离：PostgresSaver.from_conn_string(uri, schema="langgraph") 会把
--    框架表放进独立 schema，业务表留在 public，互不干扰。
-- ------------------------------------------------------------

-- ------------------------------------------------------------
-- 2) 我们自己可控的「情景记忆：蒸馏后的结构化片段 + 向量同一行」
--
--    C13 改造：从「逐轮全量原文归档」改为「每轮异步蒸馏的结构化情景片段」。
--    原因：企业内部助手、不跨会话、无审计，按主流生产做法（Mem0 蒸馏 + Armalo
--    HWC 温层）应存「蒸馏摘要 + 重要信号」而非全量原文。全量原文不再落库。
--
--    不跨会话 → 无「开场预热预注入」；检索走「会话内按需触发」：
--      ① 实体触发：用户提到订单号/工单号，entities 命中即精确查；
--      ② 语义触发：用户说"之前/那个"且相似度 > 阈值时向量召回；
--      ③ 闲聊/无关：不查，只用 checkpointer 工作记忆。
--    对应 db.insert_episodic_memories / db.search_episodic_by_vector /
--        db.search_episodic_by_entities。
-- ------------------------------------------------------------
-- 说明：
--   - 每行 = 一条「蒸馏后的情景记忆」（发生了什么 + 结果），非逐字原文；
--   - signal_type 区分事件/偏好/失败/承诺/教训/异常；importance 排序加权；
--   - entities 是可检索标签（JSONB），供精确匹配（Hybrid Search 的精确侧）；
--   - embedding 供语义召回（Hybrid Search 的语义侧）。
CREATE TABLE IF NOT EXISTS episodic_memory (
    id          BIGSERIAL PRIMARY KEY,
    thread_id   TEXT        NOT NULL,   -- 会话主键（与 checkpointer 的 thread_id 同义，不跨会话）
    user_id     TEXT        NOT NULL DEFAULT '001',  -- 来源用户（上线改动态获取；留作未来跨会话扩展）
    signal_type TEXT        NOT NULL,   -- 6 类之一：见 config.SIGNAL_TYPE
    content     TEXT        NOT NULL,   -- 蒸馏后的"发生了什么 + 结果"（≤80字，非原文）
    entities    JSONB       NOT NULL DEFAULT '{}'::jsonb,  -- 可检索实体 {order_id, ticket_id, ...}
    importance  SMALLINT    NOT NULL DEFAULT 3 CHECK (importance BETWEEN 1 AND 5),  -- 1琐碎~5关键
    embedding   VECTOR(1536),           -- pgvector 列（text-embedding-3-small 维度）
    ts          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_signal_type CHECK (signal_type IN (
        'event_outcome', 'preference', 'failure', 'commitment', 'lesson', 'anomaly'
    ))
);

-- 时间范围查询索引（按 thread_id + 时间回看）
CREATE INDEX IF NOT EXISTS idx_episodic_thread_ts
    ON episodic_memory (thread_id, ts);

-- 实体精确检索索引（Hybrid Search 的精确侧：orders/工单号等命中）
CREATE INDEX IF NOT EXISTS idx_episodic_entities
    ON episodic_memory USING gin (entities);

-- 向量索引：HNSW（高性能，默认）；数据量到千万级可换 IVFFlat
CREATE INDEX IF NOT EXISTS idx_episodic_embedding
    ON episodic_memory USING hnsw (embedding vector_cosine_ops);

-- 实体精确召回（触发条件①：用户提到订单/工单号）：
--   SELECT content
--   FROM episodic_memory
--   WHERE thread_id = $1 AND entities @> '{"order_id":"A"}'::jsonb
--   ORDER BY importance DESC, ts DESC
--   LIMIT 5;
--
-- 语义召回（触发条件②：指代词 + 相似度 > 阈值，相似度在数据库层计算）：
--   SELECT content, 1 - (embedding <=> $1::vector) AS similarity
--   FROM episodic_memory
--   WHERE thread_id = $2 AND embedding IS NOT NULL
--   ORDER BY embedding <=> $1::vector
--   LIMIT 5;
