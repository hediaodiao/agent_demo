# Demo 记忆实现改造清单（对齐生产级 / 豆包·WorkBuddy·Cursor 形态）

> 目标：把当前 `agent_demo/demo`（方案 B：LangGraph）从"可跑 demo"改造为
> 贴合主流 Agent 生产做法的记忆架构。所有改动**待用户最终确认后执行**。
> 权威标准：`docs/memory_system_summary.md`（三层记忆）+ 记忆规则（ID 72006304）。
> 回答纪律（ID 40649107）：先查证成熟 agent 做法，禁止主观猜测 / 为迎合代码曲解主流。

---

## 一、目标架构（对齐标准三层 + 主流 Agent 形态）

| 层 | 定义 | 本 demo 落地 | 与豆包/WorkBuddy 对齐点 |
|---|---|---|---|
| **工作记忆** | 模型窗口内、易失、每轮现拼 | 每轮 `graph.invoke` 时由 checkpointer **load 该 thread_id 历史** + 本轮消息拼成；`compress_node` 超窗做 `[历史摘要]+[最近k轮]`。**注意：压缩只改工作记忆（窗口），不触发情景记忆落库** | 豆包同形态：单会话多轮共用，窗口超了就压 |
| **情景记忆** | 会话历史，按 thread_id，默认**不跨会话** | **PostgreSQL + pgvector**（不再用内存 MemorySaver）：checkpointer 逐节点落盘**清洗后的 user+AI 结论轮 + 压缩摘要**；不做审计/全量日志，采用"最近k轮原文 + 1条历史压缩摘要" | 豆包即此形态（不跨会话） |
| **语义记忆** | 用户级事实/画像，opt-in 跨会话 | **本 demo 不默认实现**（标准明确非默认）；仅留扩展注释 | Cursor project rules / ChatGPT Memory 才是跨会话，属 opt-in，不硬塞 |

关键修正：
- 子 Agent 内部 `tool_calls` / `ToolMessage` 不再回传主上下文，只回"结论" → 情景记忆存的是干净 `user+AI` 轮。
- **压缩（token 触发）= 工作记忆层面**；**情景记忆落盘 = 框架每节点产出自动完成，与 token 阈值无关**。
- 不做合规审计 / 历史回溯全量日志：情景记忆内容 = 最近 k 轮原文 + 之前压缩摘要（足够单会话多轮）。

---

## 二、当前实现与标准的出入（改造动因）

1. **层标签错配**：`graph.py` 把 `MemorySaver` checkpointer 称作"工作记忆/短期"，但按标准它按 thread_id 落盘全量会话 = **情景记忆**。工作记忆是窗口本身，不"存储"。
2. **情景记忆与压缩耦合**：旧 `memory.py` 的 `_persist_episode` 只在 `compress_history` 压缩淘汰时才写，违反标准"每轮 AI 答完就 put、与压缩解耦"（落库本应由 checkpointer 每步自动做）。
3. **`recall_episodic` 死代码**：从未被调用；跨会话召回属 opt-in 语义记忆，本 demo 不做 → 删除。
4. **持久化介质**：原 `MemorySaver` 是内存版，注释却写"重启不丢"——实际进程重启即丢。改为生产级 **PostgreSQL + pgvector**。
5. **触发口径不统一**：`compress_node` 用字符数 `len(m.content)`，而 `memory.py` 内部用 tiktoken token 数 → 统一为 token。
6. **子 Agent 跨会话泄漏**：`_make_subagent` 用固定 `thread_id="order/qa/ticket"` 会跨会话串味 → 改无状态。
7. **注释过度承诺**：`graph.py` 把 store 写成"跨 thread_id / 按 user_id namespace"，与标准冲突（跨会话属语义记忆、且非默认）。

---

## 三、PostgreSQL + pgvector 数据库 Schema（情景记忆落库）

> 用户要求：按生产环境 PostgreSQL + pgvector 写，不放内存；帮忙建表结构；环境自行搭建，代码不运行。

### 3.1 启用扩展 + LangGraph checkpointer 表（生产 情景记忆持久化）

`langgraph.checkpoint.postgres.PostgresSaver` 是生产标准 checkpointer，等价于"情景记忆持久化实现"。
它自带 `setup()` 自动建表，核心三张（结构示意，实际由库建）：

```sql
-- 启用向量扩展（一次）
CREATE EXTENSION IF NOT EXISTS vector;

-- checkpoints：每步会话快照（压缩后的 state 也落这里）
--   thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint(JSONB), metadata(JSONB)
-- writes：节点写出（channel 级增量）
--   thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob
-- threads：会话索引
--   thread_id, created_at, updated_at, metadata
```

### 3.2 我们可控的「情景记忆原文+向量同表」表（演示 pgvector，默认不接线召回）

原文与向量在**同一行**，自动关联（标准 Q3）；时间查询靠 `thread_id+ts` 索引，向量检索靠 `VECTOR` 列。

```sql
CREATE TABLE IF NOT EXISTS episodic_memory (
    id         BIGSERIAL PRIMARY KEY,
    thread_id  TEXT        NOT NULL,
    turn_index INT         NOT NULL,
    role       TEXT        NOT NULL,   -- 'user' / 'assistant'
    content    TEXT,                   -- 清洗后原文（用户原话 / AI 结论）
    summary    TEXT,                   -- 超窗后被压成的摘要（可选，与 content 二选一）
    embedding  VECTOR(1536),           -- pgvector 列（text-embedding-3-small 维度）
    ts         TIMESTAMPTZ DEFAULT NOW()
);
-- 时间范围查询索引（情景记忆主查询）
CREATE INDEX IF NOT EXISTS idx_episodic_thread ON episodic_memory (thread_id, ts);
-- 向量索引（HNSW 高性能；百万级可换 IVFFlat）
CREATE INDEX IF NOT EXISTS idx_episodic_embed ON episodic_memory
    USING hnsw (embedding vector_cosine_ops);

-- 向量相似度检索示例（召回默认不接线，仅作生产扩展位）：
-- SELECT content, 1 - (embedding <=> $1) AS sim
-- FROM episodic_memory WHERE thread_id = $2
-- ORDER BY embedding <=> $1 DESC LIMIT 5;
```

### 3.3 与 MySQL / Milvus 类比（帮助用户理解）

| 维度 | PostgreSQL + pgvector | 类比 MySQL | 类比 Milvus |
|---|---|---|---|
| 类型 | 关系库 + 向量扩展 | 关系库 | 纯向量库 |
| 表结构 | 有（行列 schema） | 有 | 无（collection） |
| 向量 | `VECTOR` 列，`<=>` 余弦，库内算 | 无官方扩展 | 原生，库内算 |
| 适用 | 情景记忆（时间+少量向量） | 纯关系 | 语义记忆（90% 相似度） |
| 一句话 | "MySQL + 一个向量列" | 同左无向量 | "纯向量专业户" |

---

## 四、修改点清单（M1–M9）—— 已完成，含「是否完成 / 备注」

| # | 时间 | 文件 | 改动内容 | 对齐理由 | 是否完成 | 备注 |
|---|---|---|---|---|:---:|---|
| **M1** | 2026-09-03 | `graph.py` 顶部 docstring | 重写"两层记忆"为正确**三层**：checkpointer 标为「情景记忆持久化实现」(非工作记忆)；删掉"store 跨 thread_id / 按 user_id namespace"过度承诺；标注语义记忆 opt-in 且本demo未实现 | 标准禁止①工作记忆说成落DB、②跨会话默认 | ✅ | 已写入「工作记忆(窗口,易失) / 情景记忆(checkpointer,单会话) / 语义记忆(opt-in,未实现)」三段 |
| **M2** | 2026-09-03 | `graph.py` `_make_subagent` / `_run_subagent` | 子Agent**无状态**（去掉自身 `MemorySaver`）；跑完**只回传最后一条 AI 结论**，不回 `tool_calls`/`ToolMessage`；新增 `_extract_conclusion` | 标准 Q23①"工具结果只留结论"；修跨会话泄漏 bug | ✅ | 顺带清理：移除未用的 `ToolMessage` 导入与 `agent_key` 参数；`order/qa/ticket_node` 同步改签名 |
| **M3** | 2026-09-03 | `graph.py` `compress_node` 182 | 触发口径从 `len(m.content)` 字符数 → **tiktoken token 计数**（`count_messages_tokens`），阈值 `MAX_TOKEN_LIMIT*0.8`；docstring 明确"**不触发**情景记忆落库" | 口径统一 + 厘清压缩(工作记忆) vs 落库(情景记忆) | ✅ | 与 `memory.py` 共用同一 `count_messages_tokens`，消除两套阈值 |
| **M4** | 2026-09-03 | `memory.py` 29-40、104-138、181 | **删除** `_persist_episode` / `recall_episodic` / `episodic_store` 整套；去掉 `compress_history` 内的落盘调用；清理 `os/json/time` 导入；签名改 `Sequence`；新增 `count_messages_tokens` | 死代码 + 与压缩耦合是反模式；落库由 checkpointer 每步自动负责 | ✅ | 保留 `summarize_with_small_model` / `split_into_turns` / `compress_history` |
| **M5a** | 2026-09-03 | `demo/schema.sql`（新增） | DDL：pgvector 扩展 + checkpointer 表 + `episodic_memory` 表（原文+向量同表）+ 时间索引 + HNSW 向量索引 + 召回 SQL 示例 | 标准 Q3/Q5/Q10 | ✅ | 路径为 `demo/schema.sql`（非计划原写的 `db/schema.sql`，避免无谓建包） |
| **M5b** | 2026-09-03 | `demo/db.py`（新增） | `get_checkpointer()`(PostgresSaver) + `insert_episodic_turn()` + `search_episodic_by_vector()`；`POSTGRES_URI` 缺失时 fail fast 打印指引 | 生产级情景记忆 | ✅ | 后两个函数按设计**默认不接线**（避免把跨会话做成默认），仅作扩展位 → 见 C4 调整 |
| **M5c** | 2026-09-03 | `graph.py` 231 `compile` | `MemorySaver` → `get_checkpointer()`（PostgresSaver） | 不使用内存介质 | ✅ | 注释写明落盘由框架每节点产出触发，与 token 阈值无关 |
| **M5d** | 2026-09-03 | `requirements.txt` | 新增 `langgraph-checkpoint-postgres` / `psycopg[binary]` / `pgvector` | 支撑 PG 落库 | ✅ | 同步移除已失效的"内存版替换点"注释 |
| **M6** | 2026-09-03 | `agent.py` 文件头 | 加遗留说明（早期手写版、import 已失效、保留不删除） | 需求 | ✅ | 未删除文件 |
| **M7** | 2026-09-03 | `session_store.py` 文件头 | 加遗留说明（graph.py 已用 checkpointer，本文件未引用，保留不删除） | 需求 | ✅ | 未删除文件 |
| **M8** | 2026-09-03 | `messages.py` 文件头 | 补 `# 保留不删除` | 需求 | ✅ | 原"请勿使用"保留 |
| **M9** | 2026-09-03 | `README.md` 多处 | ①技术栈加「情景记忆落库/语义记忆」行；②新增「前置：PostgreSQL+pgvector」小节；③对照表加"子Agent只回结论/情景记忆落库/无状态化改 db"；④新增「三层记忆」表；⑤生产替换点移除 `InMemorySessionStore`/`MemorySaver`；⑥文件分类标注遗留；⑦遗留清单补 `agent.py`/`session_store.py` | 需求 | ✅ | 全文相关段落已按"生产级、非 demo"口径重写 → 见 C6 再同步 |

### 完成情况汇总（M 系列）

- **已完成：M1–M9 共 13 个子项，✅ 13 / 13。**
- **未做运行时验证**：本机未安装 `langgraph` / `langchain` / `psycopg`，也未配置 PostgreSQL，故**未实跑**对话流程。仅用 `python -m py_compile` 对 `graph.py` / `memory.py` / `db.py` / `agent.py` / `session_store.py` / `messages.py` 做语法校验，**全部通过**。
- **lint 残留说明**：编辑器里仍有 `reportMissingImports`（依赖未安装）与 `reportImplicitRelativeImport`（本目录扁平 import 风格）告警，属**既有问题、非本次引入**；未改动项目既有 import 风格，以免扩大影响面。
- **环境由用户自理**：PostgreSQL 实例、`CREATE EXTENSION vector`、执行 `schema.sql`、配置 `POSTGRES_URI` 均需用户自行完成（步骤已写入 `README.md`）。

---

## 五、会删 / 会保留

- **会删**：`memory.py` 的 `episodic_store` 目录逻辑 + `_persist_episode` + `recall_episodic`（死代码/反模式，且不再需要）。
- **不删**：`agent.py` / `session_store.py` / `messages.py`（只加注释）。
- **不新增（语义记忆 stub）**：不硬塞半成品跨会话画像（标准非默认）。
- **不新增（审计 / 跨会话统计）**：按用户确认不做；但**保留期内逐轮全量存储**是基础能力，与审计是两回事（见第八节）。

---

## 六、拍板点（已确认）

1. **M4 删除 `recall_episodic`**：✅ 同意（checkpointer 已负责同会话召回；跨会话属 opt-in 语义记忆，本 demo 不做）。
2. **M2 子 Agent 无状态**：✅ 同意（修跨会话泄漏）。
3. **语义记忆最小 stub**：❌ 不做（标准非默认，豆包默认也不跨会话；仅留注释）。
4. **审计 / 跨会话统计**：❌ 不做；**全量逐轮存储**：✅ 要做（第八节 C 系列修正）。
5. **情景记忆落库介质**：✅ **PostgreSQL + pgvector**（不用内存）；建表结构见第三节；环境用户自搭。

---

## 七、执行状态（M 系列已全部完成）

- [x] M1 graph.py 顶层 docstring 重写（三层 + 去跨会话承诺）
- [x] M2 graph.py 子Agent只回结论 + 无状态
- [x] M3 graph.py compress_node 改 token 计数（仅工作记忆压缩，落库与之无关）
- [x] M4 memory.py 删除 episodic 死代码
- [x] M5a 新增 `demo/schema.sql`（扩展 + checkpointer 表 + episodic_memory 表 + 向量索引）
- [x] M5b 新增 `demo/db.py`（`get_checkpointer()` PostgresSaver + 另两个函数默认不接线）
- [x] M5c graph.py `compile` 换 `PostgresSaver`（env `POSTGRES_URI`）
- [x] M5d `requirements.txt` 加 `langgraph-checkpoint-postgres` / `pgvector` / `psycopg[binary]`
- [x] M6 agent.py 文件头遗留注释
- [x] M7 session_store.py 文件头遗留注释
- [x] M8 messages.py 文件头保留注释
- [x] M9 README.md 同步（含数据库搭建说明 + 三层记忆表）

> 校验方式：`python -m py_compile` 全部通过；未实跑（依赖与 PG 环境由用户自理）。

---

## 八、第二轮修正清单（C1–C6，待确认后执行）

> **起因**：复盘 M1 docstring 时发现一处**层归属错误**——把"最近 k 轮原文 + 1 条历史摘要"
> （这是**工作记忆 / 喂模型的窗口**产物）写成了**情景记忆的存储内容**。
> 由此引出三个连锁问题（用户提出，均已核实）：
> 1. 若情景记忆只存"k 轮 + 摘要"且不跨会话 → 时间戳、时间范围查询、分层存储**全部失去意义**；
> 2. "无状态"被误读为"模型只看本轮提问"；
> 3. "每条都存"被担心撑爆磁盘。
>
> **结论**：主流生产（豆包 / ChatGPT / Cursor）在保留期内**逐轮全量存储**（所以能打开旧会话逐轮回看），
> "k 轮 + 摘要"**只作用于喂模型的上下文**，压缩是上下文层的事、**不是存储裁剪**。

| # | 时间 | 文件 | 改动内容 | 对齐理由 | 是否完成 | 备注 |
|---|---|---|---|---|:---:|---|
| **C1** | 2026-09-04 | `graph.py` 情景记忆段 | 「存什么」改为**逐轮一条、全量、带 ts**（user + 最终回复）；**删掉**"落地形态 = 最近 k 轮 + 1 条摘要" | 主流可逐轮回看整段会话；标准情景记忆=存储所有历史对话 | ✅ | 核心层归属错误已修正：不再把工作记忆产物当存储内容 |
| **C2** | 2026-09-04 | `graph.py` 工作记忆段 | 明确「k 轮 + 摘要」是**工作记忆（喂模型窗口）**产物；补注"压缩只裁剪喂模型的内容，**不裁剪库里全量**" | 压缩是上下文层的事，不是存储裁剪 | ✅ | 与 C1 配套，一删一补 |
| **C3** | 2026-09-04 | `graph.py` 情景记忆段 | 澄清 checkpointer 是 **LangGraph 框架机制**（每节点一份 state 快照，断点恢复/无状态化），**不是另一套情景记忆、不属于记忆分层**；对比两种表的用途 | 避免把工程机制当记忆层（记忆规则禁止③） | ✅ | 情景记忆只有 `episodic_memory` 一张表；消除"两套存储"误读 |
| **C4** | 2026-09-04 | `db.py` + `graph.py` | `insert_episodic_turn` **接线**：`answer_node` 每轮写 2 行（user + 最终回复），与压缩解耦；签名**去掉 `turn_index`**；`search_episodic_by_vector` 保持不接线 | 情景记忆逐轮全量落库；跨会话召回属 opt-in（非默认） | ✅ | 不新增"从 episodic_memory 加载历史"逻辑，避免与 checkpointer 形成两个事实源 |
| **C5** | 2026-09-04 | `schema.sql` | **删除 `turn_index` 列**（排序靠自增 `id`、时间靠 `ts`）；补「按时间取最近 k 轮」SQL 示例 | 去掉冗余字段；让 ts/索引真正有用途 | ✅ | 后续可加按时间分区（热/温/冷）与 TTL 清理 |
| **C6** | 2026-09-04 | `README.md` | ①三层记忆表改为"逐轮全量"；②新增「无状态到底什么意思」小节；③新增「存储成本」小节；④对照表补汇总节点/情景记忆落库/checkpointer；⑤文件分类补 answer 节点 | 消除"无状态=只看本轮""全量=爆磁盘"两类误读 | ✅ | 与 C1/C2/C4/C7 同一口径 |
| **C7** | 2026-09-04 | `graph.py` | **新增汇总/作答节点 `answer_node`**（Q1 选 A·优化版）：单专家**透传**（零额外 LLM）；多专家**整合**成一条连贯回复（小模型）；0 结论走模板收尾语。它取代 `farewell_node`，是本轮**唯一最终回复出口** | 主流 Supervisor 汇总形态：子模块原始输出不直接给用户，由主模型整合；也落实"该存最终回复" | ✅ | `farewell_node` 已删除，其收尾语逻辑并入 `answer_node` 的 0 结论分支（收尾语仍入库，按用户要求） |
| **C7b** | 2026-09-04 | `graph.py` `supervisor` + `ROUTING_PROMPT` | **修复死循环缺陷**：调度台原只看 `last_human`，答完后会反复把同一诉求派给同一专家直到 `recursion_limit`。改为把「本轮已产出的专家结论」一并喂给调度台，并在 prompt 中加防重复派单规则 | 这是 C7 能收敛的**必要前提**——否则 `answer_node` 永远到不了 | ✅ | 属既有缺陷（非 M 系列引入）；已实现并在此报备，如不认可可回退 |

### C 系列补充说明

- **存储成本（回应"每条都存会不会爆"）**：
  - 算账：一条清洗后文本 ≈ 100 B ~ 几 KB；20 轮会话 ≈ 40 条 ≈ **20 KB**；100 万会话 ≈ **20 GB**（PG 压缩后更小，单表可承受）。
  - 真正会撑爆的是**长文档 / 图片 / 工具返回原始 JSON**——生产上一律**外置对象存储（S3 / MinIO），对话表只存引用**。本项目 M2 + `tools.py` 裁剪已保证工具返回不进历史。
  - 长期成本四招（标准 Q6）：分层存储(热/温/冷) + 定期清理(30 天) + 旧会话摘要化 + 大 payload 外置。
- **"无状态"正确口径**：指**服务实例不持有会话数据**（数据在 PostgreSQL，按 `thread_id`），
  任意实例可服务任意请求、重启不丢、可水平扩容；**不等于**模型只看本轮提问——
  每轮会先 `load(thread_id)` 取回历史，再组装 `[历史摘要] + [最近 k 轮] + [本轮提问]` 喂模型。

### 时间列说明

- **时间为会话日期（非精确时间戳）**：M 系列于 `2026-09-03` 会话内完成；C 系列于 `2026-09-04` 提出，**待用户确认后执行**。
- 图例：✅ 已完成 ｜ ⬜ 待确认 / 未执行（注：C8/C9/C10 见第十节）

---

## 九、执行状态（C 系列已全部完成）

- [x] C1 graph.py 情景记忆段改「逐轮全量带 ts」，删"k轮+摘要"
- [x] C2 graph.py 工作记忆段明确"压缩只裁剪上下文，不裁剪存储"
- [x] C3 graph.py 澄清 checkpointer 是框架机制，不是另一套情景记忆
- [x] C4 db.py + graph.py 接线 insert_episodic_turn（每轮写 user + 最终回复），去掉 turn_index
- [x] C5 schema.sql 删除 turn_index + 补「按时间取最近 k 轮」SQL 示例
- [x] C6 README.md 同步三层记忆表 + 无状态口径 + 存储成本小节
- [x] C7 graph.py 新增 `answer_node`（单专家透传 / 多专家整合 / 纯客套模板），取代 farewell_node
- [x] C7b graph.py 修复 supervisor 重复派单死循环（喂入本轮已产出结论 + 防重复规则）

> 校验：`python -m py_compile graph.py memory.py db.py` **全部通过**。
> 未实跑：依赖（langgraph/langchain/psycopg）与 PostgreSQL 环境由用户自理。

## 十一、框架表命名澄清（C11）

> **起因**：用户指出 `schema.sql` 里手写了一张 `checkpoints` 表，与 LangGraph 的
> 「checkpointer」概念命名撞车、易混淆，提议改表名。经查证：
> `checkpoints` 是 LangGraph 框架**硬编码**的物理表（checkpointer 组件的存储层），
> 不能改名；正确做法是框架表归 `PostgresSaver.setup()` 托管，业务文件只留 `episodic_memory`。

| # | 时间 | 文件 | 改动内容 | 对齐理由 | 是否完成 | 备注 |
|---|---|---|---|---|:---:|---|
| **C11** | 2026-09-04 | `schema.sql` + `db.py` | 删除 `schema.sql` 手写的框架表 DDL（checkpoints/writes/threads），改由 `PostgresSaver.setup()` 幂等建表；`get_checkpointer()` 内调用 `saver.setup()` | 框架表名硬编码不可改名；手写易与框架结构冲突且命名撞车。标准做法：框架表归 checkpointer 托管，业务 DDL 只留 `episodic_memory` | ✅ | 可选增强：用 `PostgresSaver.from_conn_string(uri, schema="langgraph")` 把框架表放进独立 schema，与业务表物理隔离 |

## 十二、护栏常量解耦（C12）

> **起因**：讨论 `graph.py:361` `recursion_limit` 时发现，`MAX_ITERATIONS=12` 被两处复用、
> 但语义不同——① `compile(recursion_limit=...)` 数的是「节点执行总步数」(防任意环死循环)；
> ② C10 护栏 `len(called_agents) >= MAX_ITERATIONS` 数的是「本轮不同专家派单次数」。
> 两者巧合共用 12，且由于 recursion_limit 物理上只容 ~3-4 次派单，C10 的「派单≥12」分支
> 实际在 recursion_limit 报错前就先触发，几乎不单独生效，语义混乱。

| # | 时间 | 文件 | 改动内容 | 对齐理由 | 是否完成 | 备注 |
|---|---|---|---|---|:---:|---|
| **C12** | 2026-09-05 | `config.py` + `graph.py` | 新增独立常量 `MAX_DISPATCHES=4`(默认)；C10 护栏由 `>= Settings.MAX_ITERATIONS` 改为 `>= Settings.MAX_DISPATCHES`；`MAX_ITERATIONS` 注释明确为「图步数硬护栏」；C10 注释说明两者语义不同 | 派单次数护栏(逻辑层,数不同专家派单数)与图步数护栏(框架层,数节点步数)是两种不同计数器，应各自独立常量，避免数字耦合导致语义误导 | ✅ | `recursion_limit=MAX_ITERATIONS=12` 不变(仍防环死循环)；C10 现在用 `MAX_DISPATCHES=4` 独立管控「本轮最多派 4 个不同专家」，可先于 recursion_limit 生效，语义干净 |

## 十三、情景记忆蒸馏化改造（C13）

> **起因**：经多轮查证与讨论，本系统定位为「企业内部助手、不跨会话、无审计」场景。
> 主流生产做法（Mem0 蒸馏 + Armalo HWC 温层）对情景记忆采用**蒸馏后的结构化片段**而非全量原文；
> 且因不跨会话，"开场预热预注入"不适用（同会话连续性已由 checkpointer 托管）。
> 故将 `episodic_memory` 从"逐轮全量原文归档"改为"每轮异步蒸馏的结构化情景片段"，
> 并在会话内按"实体 / 语义触发"按需召回，补回被压缩掉的精确细节。
> **已锁定细节**：① 召回取全量（不按 importance 过滤，仅排序）；② 语义触发阈值 0.78；
> ③ `user_id` 默认 `"001"`（上线改为动态获取）。

| # | 时间 | 文件 | 改动事项（一句话看懂改了什么） | 改动内容 | 对齐理由 | 是否完成 | 备注 |
|---|---|---|---|---|---|:---:|---|
| **C13-1** | 2026-09-05 | `schema.sql` | **把"逐轮全量对话记录表"改造成"结构化情景记忆表"**：不再整段存聊天原文，只存提炼后的关键信息，并给每条记忆标注「来源用户、记忆类型、重要程度、可检索标签」 | `episodic_memory` 表改造：删 `role`/`summary` 列；新增 `user_id`(TEXT, 默认'001')、`signal_type`(TEXT, 6类枚举+检查约束)、`entities`(JSONB)、`importance`(SMALLINT 1-5)；保留 `thread_id`/`content`/`embedding`/`ts`；注释改为"蒸馏后的情景记忆(同会话按需召回)"；保留 `idx_episodic_thread_ts`，新增 `entities` GIN 索引 | 无审计内部助手应存"蒸馏结构化摘要"而非全量原文（Mem0/HWC 主流）；`entities` 支持精确检索、`importance` 排序、`signal_type` 区类型 | ✅ | 6类枚举：event_outcome / preference / failure / commitment / lesson / anomaly |
| **C13-2** | 2026-09-05 | `db.py` | **把记忆写入从"存原文对话"改为"批量写入提炼后的结构化记忆并生成语义向量"**，同时补上「按标签精确查找」的能力，让记忆检索真正可用 | `insert_episodic_turn` → `insert_episodic_memories(episodes:list[dict])` 批量写，每条含 `signal_type/content/entities/importance/embedding`；复用 `vector_store.Embedder` 算 embedding（修掉之前 NULL）；`search_episodic_by_vector` 改为返回完整 episode(content+entities)、top-K 全量(按 `importance DESC, ts DESC`)；新增 `search_episodic_by_entities(thread_id, entities)` 做 JSONB 精确匹配 | 写入带向量才能真正检索；Hybrid Search=实体精确+向量语义；全量召回=不按 importance 过滤 | ✅ | 旧 `role/summary` 参数删除；⚠️ 演示 fake embed 维度已对齐 1536（坑一已修） |
| **C13-3** | 2026-09-05 | `graph.py` | **每轮回复用户后，在后台自动把本轮对话"总结提炼"成几条记忆入库**：不影响回复速度、不再落库原文 | `answer_node` 各分支产出最终回复后，启**后台线程 fire-and-forget** 调蒸馏：把"本轮 user 原话 + 主Agent最终回复 + 涉及专家"喂 small 模型(structured output)提炼 0~N 条 episode 写库；不阻塞返回 | 异步蒸馏=主流（每轮后提炼但不卡回复）；全量原文不再落库 | ✅ | 子 Agent 中间结论/工具调用不进；失败仅打日志不阻断 |
| **C13-4** | 2026-09-05 | `graph.py` | **新增"记忆召回"环节**：用户提到具体订单/工单或说"之前/那个"时，才从本会话历史里捞回相关记忆补进上下文；平时不查，避免无关信息干扰 | 新增 `recall_node`（插在 `START → supervisor` 之间）：读当前输入，①实体触发=输入命中 `entities` JSONB 即查；②语义触发=无实体命中时算输入 embedding 与同 thread 近期 episode 相似度 >0.78 且含指代词("之前/那个/刚才")才查；命中则捞 top-K 注入一条 `SystemMessage`"【相关历史记忆】\n..."；不命中返回 `{}` 不注入 | 触发性检索（非每轮无脑查）：避免延迟/噪声；不跨会话故无"开场预热" | ✅ | top-K=5；不触发时只靠 checkpointer 工作记忆 |
| **C13-5** | 2026-09-05 | `config.py` | **把记忆相关参数集中到配置文件**：默认用户、召回条数、触发阈值、记忆类型等都在此统一管理，方便上线调整 | 新增 `DEFAULT_USER_ID="001"`、`EPISODIC_TOP_K=5`、`SEMANTIC_TRIGGER_THRESHOLD=0.78`；定义 `SIGNAL_TYPE` 枚举 | 集中配置，避免硬编码 | ✅ | `user_id` 上线改为动态获取 |
| **C13-6** | 2026-09-05 | `graph.py`+`schema.sql`+`README.md` 注释 | **同步更新文档说明**：把记忆描述从"全量对话归档"改为"蒸馏后的结构化情景记忆"，消除文档与代码不一致 | 顶部"三层记忆"说明、`schema.sql` 注释、`README` 护栏行与记忆章节中"情景记忆=逐轮全量"改为"蒸馏结构化片段(同会话按需召回)"；`checkpointer` 仍为短期记忆持久化（C12 前已改） | 与实现一致，消除文档/代码错位 | ✅ | 语义记忆仍不实现 |

**下一轮主模型收到的信息（改造后）**：
1. 永远有：checkpointer 工作记忆 `[历史摘要]+[最近k轮 user+主Agent回复]` + 本轮用户输入；
2. 仅触发时有：`recall_node` 注入的 top-K 蒸馏 episode（实体/语义命中，补回被压缩掉的精确细节）；
3. 永远没有：跨会话历史、子 Agent `tool_calls`/`ToolMessage`、全量原文逐字稿。
<!-- write-probe -->


---

## 十四、第三轮：工作记忆压缩与主流对齐（D 系列，2026-09-08 提出，已确认待执行）

> **起因**：多轮讨论确认当前压缩实现与主流（LangMem 等）不匹配，需要调整：
> 压缩只在子 Agent 跑完后触发、纯客套轮永不压；触发阈值单一且偏高（80%）；
> 保留策略按固定"最近 3 轮"而非 token 预算；进行中轮爆表无兜底；
> 摘要无独立字段、靠第 0 条 SystemMessage 伪装成"伪轮"传递。
> **状态**：表 1 / 表 2 / 表 3 与关联 todo 均已确认，待按 c1→c7 执行。

### 关联 todo（codebuddy todo 面板 c1–c7）

| todo | 主题 | 对应表3编号 |
|---|---|:---:|
| c1 | 配置收口：config.py 单阈值拆两档 + 预算常量 | #1 |
| c2 | 入口压缩节点：graph.py 接线「召回→压缩→路由」 | #11 |
| c3 | 触发判定两档化：压缩函数三态判定 | #2 |
| c4 | 预算切分保留：固定最近 3 轮 → token 预算切分 | #3 |
| c5 | 单轮爆表兜底：折叠进行中轮 + 逐条裁剪 + 摘要净化 | #4/#5/#6 |
| c6 | 滚动摘要显式化：状态新增摘要字段 + 函数签名改造 | #8/#9 |
| c7 | 回归验证：多轮/多专家/长窗 demo 会话 | — |

### 表 1：目前代码的问题（与主流做法不匹配处）

| # | 问题（现状） | 与主流做法的差异 | 后果 |
|---|---|---|---|
| 1 | **压缩时机过窄**：只在子 Agent 跑完后检查；入口、路由首次调用前、纯客套轮（不派子 Agent）都不检查 | LangMem/主流是"每个模型调用前都有一道预算卡点"，尤其每轮入口必查一次 | 纯客套轮可以无限累积不触发压缩；某轮入口时窗口已超限，路由第一次调用就超窗 |
| 2 | **触发阈值太高**：窗口用到 80% 才动手压 | 主流用较低阈值"软触发"提前把最老历史滚进摘要，避免一次压太多 | 一次要压很多轮，摘要体量变大、损失更多细节 |
| 3 | **保留策略按"轮数"不看预算**：固定保留最近 3 轮原文，不量 token | 主流是"从最新往最旧累计 token，预算内保留、超出滚摘要" | 若近 3 轮本身很大（尤其当前轮结论多），压完仍然超窗——"压了个寂寞" |
| 4 | **单轮爆表无兜底**：进行中的轮（还没到每轮末尾的整合点）若自身就超预算，现逻辑仍把它整体当原文保留 | 主流对超预算部分一视同仁地滚进摘要/折叠 | 极端长轮（单轮多专家、结论冗长）压不动窗口 |
| 5 | **摘要没有独立承载字段**：历史摘要只能伪装成"摘要消息"放在列表第 0 位，靠字符串前缀识别，还会被切轮逻辑当成一个"伪轮"参与下次压缩 | 主流把"滚动摘要"作为状态的独立字段，能单独预算、单独传递 | 摘要语义靠前缀字符串维系、脆弱；"最近 k 轮"的 k 实际在漂移（= 上次压缩点至今，不是常数 3），无法解释也难以管理 |
| 6 | **触发过的压缩形态"不确定"**：下一轮 load 回的是否压缩态取决于上一轮有没有机会触发 | 主流不依赖"运气"，入口固定校准一次 | 与问题 1 同根，导致每次推理都要先猜窗口长什么样 |

> 值得强调的"非问题"（与主流一致、保留不动）：子 Agent 内部工具循环不进主窗（源头只回结论）；每个完整轮在末尾的整合节点被规范成"一问一答"；历史轮因此是干净的。

### 表 2：整体调整事项

| 事项 | 一句话说明 | 修的问题 |
|---|---|---|
| A. 压缩时机对齐"入口 + 节点后" | 新增**入口压缩**（召回后、路由前），子 Agent 后那道保留；纯客套轮也被入口兜住 | 问题 1、6 |
| B. 阈值分级 | 一个较低阈值做**软触发**（提前滚老历史），一个硬顶做**强制裁剪** | 问题 2 |
| C. 切分单位从"轮数"改"token 预算" | 从最新向最旧逐轮累计，预算内整轮保留原文，超出的轮滚进摘要 | 问题 3 |
| D. 单轮爆表兜底 | 进行中的轮若单独超预算 → 先折叠其内部多条子 Agent 结论，仍超再逐条裁 | 问题 4 |
| E. 摘要显式滚动 | 状态里新增**滚动摘要字段**；每次压缩用"旧摘要 + 新超预算轮"更新它，不再伪装成伪轮 | 问题 5 |
| F. 相关数值收进配置文件 | 阈值、保留预算等集中一处 | —（工程性） |

### 表 3：详细调整点（时间 = 会话日期，沿用 M/C 系列口径）

| 编号 | 位置 | 现状 | 要改成 | 对应事项 | 时间 | 改动内容 |
|---|---|---|---|---|---|---|
| #11 | 图结构（graph.py） | 入口为「召回 → 路由」直连 | 插入「召回 → 压缩 → 路由」，压缩函数超限才动、未超零成本返回 | A、问题 1/6 | 2026-09-08 | `builder.add_edge("recall", "supervisor")`（graph.py:467）改为先 `add_edge("recall","compress")` 再 `add_edge("compress","supervisor")`；直接复用 `compress_node`（graph.py:421-441） |
| #1 | 配置（config.py） | 单一高阈值（80%） | 拆成两档：软触发阈值（约窗口 60%，用于提前滚摘要）+ 硬顶（约 80%，压完必须低于它）；集中定义保留预算等常量 | B、问题 2 | 2026-09-08 | `Settings` 内 `MAX_TOKEN_LIMIT`(24行)/`SUMMARY_TRIGGER_RATIO`(25行) 处新增 `SOFT_TRIGGER_RATIO=0.6`、`HARD_LIMIT_RATIO=0.8`、保留预算与摘要体量上限常量；`SUMMARY_TRIGGER_RATIO` 改名/废弃 |
| #2 | 压缩函数触发判定（memory.py） | 只看是否超单阈值 | 按两档判定：低于软触发 → 原样返回；软触发~硬顶 → 滚动最老轮至预算内；超硬顶 → 强制裁到硬顶内 | B | 2026-09-08 | `compress_history` 触发判断段（memory.py:113-119）改三态；`compress_node` 的 `trigger` 计算（graph.py:435-437）同步用两档替换单阈值 |
| #3 | 压缩函数切分逻辑（memory.py） | 固定保留最近 3 轮原文，更早轮逐轮摘要 | 从最新向最旧逐轮累计 token，预算内整轮保留、其余轮并入摘要；摘要产物控制体量上限（约占窗口一小部分） | C、问题 3 | 2026-09-08 | 重写 `compress_history`（memory.py:112-142）：删 `keep_recent_turns=3` 固定切片（118、122-123 行），改 `reversed(turns)` 逐轮累计 token 至保留预算；摘要 `max_chars`(131行) 按窗口比例给 |
| #4 | 压缩函数兜底（memory.py） | 无（保留区可能仍超） | 预算内保留部分仍超硬顶时，对最新一轮内部逐条裁剪，直到低于硬顶 | C/D、问题 4 | 2026-09-08 | 压缩函数组返回值前（memory.py:142 附近）校验"保留区 ≤ 硬顶"，否则对最新一轮从较旧的 AI 结论起逐条裁，直到低于硬顶 |
| #5 | 压缩辅助函数（memory.py 新增） | 无（进行中的多结论轮从不折叠） | 新增"折叠进行中轮"：只留本轮提问 + 最新一条答复，把该轮内多条子 Agent 结论合并为一条 | D、问题 4 | 2026-09-08 | 新增 `_collapse_turn(turn)`：取第一条 HumanMessage + 最后一条 AIMessage、丢弃中间多条子 Agent 结论；保留区仍超硬顶时对最新一轮调用 |
| #6 | 轮转文本（memory.py） | 把轮内所有消息（含工具消息，若偶入）拼成文本再摘要 | 转换文本时跳过工具类消息，只留问答正文 | D（防御性） | 2026-09-08 | `_turn_to_text`（memory.py:69-79）循环里 `if isinstance(m, ToolMessage): continue`，只拼问答正文 |
| #8 | 状态结构（graph.py） | 无摘要字段 | 状态新增"滚动摘要"文本字段 | E、问题 5 | 2026-09-08 | `AgentState`（graph.py:88-91）新增 `summary: str`——覆盖式字段（不挂累加 reducer，否则会被翻倍） |
| #9 | 压缩函数签名与写回（memory.py + graph.py） | 摘要靠"摘要消息"第 0 位传递，被切轮当伪轮 | 压缩函数改为接收上次摘要作输入、产出更新后摘要写入字段；消息头部那条摘要提示由字段内容拼装，不再反向解析 | E、问题 5 | 2026-09-08 | `compress_history` 签名（memory.py:97）改 `(messages, running_summary="")`、返回 `(out, summary)`；`compress_node`（graph.py:438-440 update_state）把 `summary` 一并写入 state |

