# 生产级复杂工单客服 Agent（LangGraph 方案 B：纯 StateGraph 手写）

> **目的**：用**真实生产框架（LangGraph + LangChain）**实现一个复杂工单客服，
> 覆盖我们之前讨论过的所有实战要点，且代码带教学注释，**便于学习 / 应对 Agent 开发 **。
> 选方案 B（手写 StateGraph）而非 `create_supervisor`，因为手写才能学透底层机制
> （节点 / 边 / 状态 / 条件路由）， 时能讲清楚框架替你做了什么、你要硬编码什么。

## 技术栈

| 组件 | 框架/库 | 说明 |
|---|---|---|
| 编排 | **LangGraph** `StateGraph` | 手写 Supervisor 多 Agent 状态图 |
| 子 Agent | **LangChain** `create_react_agent` | 每个专家是一个 react Agent（含工具循环） |
| 工具 | **LangChain** `@tool` | 订单 / 知识库RAG / 工单 |
| 向量库（RAG） | `OpenAIEmbeddings` + 内存 `VectorStore` | 生产换 Milvus/Qdrant/pgvector |
| **情景记忆落库** | **PostgreSQL + pgvector**（`PostgresSaver`） | 按 `thread_id` 持久化会话，**非内存介质** |
| 语义记忆 | **未实现** | opt-in 跨会话画像（标准非默认，见「三层记忆」） |
| LLM | `ChatOpenAI` | 决策大模型 + 压缩小模型分离 |

## 运行

### 前置：情景记忆数据库（PostgreSQL + pgvector）

情景记忆**不使用内存介质**，必须先准备 PostgreSQL：

```bash
# 1) 安装 pgvector 扩展（依发行版而定），然后建库
createdb agent_db

# 2) 启用扩展 + 建表
psql -f schema.sql "postgresql://user:pass@localhost:5432/agent_db"

# 3) 配置连接串（生产禁止硬编码凭据）
set POSTGRES_URI=postgresql://user:pass@localhost:5432/agent_db
```

> 未配置 `POSTGRES_URI` 时 `db.get_postgres_uri()` 会 fail fast 并打印上述步骤。
> checkpointer 表也可由 `checkpointer.setup()` 建立（见 `schema.sql` 注释）。

### 启动

```bash
pip install -r requirements.txt
set OPENAI_API_KEY=你的key
set OPENAI_BASE_URL=https://your-endpoint   # 可选，兼容端点/自建 vLLM

python main.py                  # 交互式对话
python main.py --session user1  # 指定 thread_id（无状态化演示）
python test_demo.py             # 端到端冒烟测试（验证分流路由）
```

## 代码与「我们讨论的生产要点」对照

| 我们讨论的要点 | 在代码里的位置 |
|---|---|
| **多 Agent（Supervisor）** | `graph.py`：`supervisor` 节点做意图分类 + `add_conditional_edges` 分流到 order/qa/ticket；调度台会被告知「本轮已产出的专家结论」，避免重复派单死循环 |
| **Supervisor 汇总（多专家整合）** | `graph.py` `answer_node`：单专家**透传**（零额外 LLM）；多专家把中间结论**整合**成一条连贯回复；纯客套走模板收尾语。它是本轮**唯一最终回复出口** |
| **子 Agent 是独立实例** | `graph.py`：`order_agent/qa_agent/ticket_agent` 各用 `create_react_agent` 封装 |
| **工具返回压缩（第1块）** | `tools.py`：`query_orders` 用 `_format_orders` 裁剪；`rag_search` 过长用小模型摘要 |
| **工具/子Agent结果只留结论（第1块扩展）** | `graph.py` `_run_subagent` + `_extract_conclusion`：子 Agent 的 `tool_calls`/`ToolMessage` **不回传**主上下文，只回最后一条 AI 结论 |
| **上下文滚动摘要（第2块）** | `memory.py`：`compress_history` + `count_messages_tokens`（**token 计数**触发，压最早轮）；`graph.py` `compress_node` 在每轮子 Agent 后检查 |
| **单轮超长走 RAG** | `vector_store.py` + `tools.rag_search`：算「问题 vs 片段」相似度 |
| **无状态化 + 状态外置** | `db.get_checkpointer()`：`PostgresSaver` 按 `thread_id` 落盘（`session_store.py` 为遗留，未引用） |
| **情景记忆（蒸馏 + 按需召回）** | `graph.py` `answer_node`：每轮后台线程异步 LLM 蒸馏成 0~N 条结构化 episode 写 `episodic_memory`（全量原文不再落库）；`recall_node` 在会话开场按「实体/语义触发」按需召回 top-K 注入上下文，**不每轮无脑查** |
| **checkpointer = 短期 / 工作记忆持久化** | `db.get_checkpointer()`：`PostgresSaver` 按 `thread_id` 落 `checkpoints` 表（state 快照），即 LangGraph 的短期记忆（thread-scoped，管会话连续性 / 断点恢复 / 无状态化）；**不是记忆分层**，生产定期裁剪 |
| **步数封顶（硬护栏 / 图步数）** | `graph.py`：`compile(recursion_limit=Settings.MAX_ITERATIONS)` —— 单次 invoke 节点执行**总步数**上限，防任意环死循环 |
| **派单次数护栏（C10 / 逻辑层）** | `graph.py` `supervisor`：`len(called_agents) >= Settings.MAX_DISPATCHES`（默认 4）强制 `farewell` 收尾，杜绝重复派同一专家 / 无限派单。与图步数护栏 `MAX_ITERATIONS` **语义解耦**：前者数「本轮不同专家派单次数」，后者数「节点执行步数」，是两种不同计数器 |
| **降级 / 备用数据源** | `tools.query_orders`：主源挂了切 CSV 备胎 |
| **压缩用小模型** | `config.get_llm("small")`；`memory.summarize_with_small_model` 用它 |
| **循环骨架框架封装 vs 业务硬编码** | `create_react_agent` 给子 Agent 循环；工具压缩/`compress_history`/supervisor 路由是你写的 |

## 记忆（生产口径，对齐主流 agent taxonomy）

| 层 | 存什么 | 何时写 | 介质 | 作用域 |
|---|---|---|---|---|
| **短期记忆（short-term / working）** | 会话连续性所需的对话历史：checkpointer 持久化的 state 快照（含压缩后 `[摘要]+[最近k轮]`）；被 `load` 进窗口即充当「工作记忆」 | 每节点产出自动落 `checkpoints`；下轮 `load(thread_id)` 回 | checkpointer（`checkpoints` 表） | 单 `thread_id` |
| **情景记忆（蒸馏结构化片段 / episodic）** | 每轮 `answer_node` 后台异步蒸馏的 0~N 条 episode：`signal_type`(事件/偏好/失败/承诺/教训/异常) + `content`(发生了什么+结果) + `entities`(可检索标签) + `importance`(1-5)，带 `ts`+`embedding`；**全量原文不再落库** | 每轮 `answer_node` 后异步写（不阻塞回复，与压缩解耦）；`recall_node` 会话内按需召回（实体/语义触发） | `episodic_memory` 表（PostgreSQL + pgvector） | 单 `thread_id`，**默认不跨会话** |
| **语义记忆** | 用户级事实/画像（**未实现**） | opt-in：规则实时(60%) + 事件触发(5%) + 每日批量(35%)，绝不每轮调 LLM | 向量库 | 跨会话（**非默认**） |

⚠️ 易混点（面试高频）：

- **情景记忆（本项目 = `episodic_memory`，宽泛/生产常见口径）**：按时间存成结构化 episode（发生了什么+结果+标签），由 `recall_node` 按需召回注入上下文——本场景为企业内部助手、无审计，故只存蒸馏片段、不存全量原文，即主流 chatbot 的 episodic memory。**严格 agent 工程定义（episodic=带反思的任务轨迹、跨会话学经验）属进阶 opt-in，本 demo 未做；口语/宽泛认知定义下本表即情景记忆。**
- **checkpointer（`checkpoints` 表）是短期记忆的持久化实现**，不是"另一套情景记忆"；与 `episodic_memory` 是独立表、独立用途：前者给框架恢复+续聊（每节点一份快照），后者给会话内按需召回精确细节。
- **⚠️ 别把「checkpointer 的 load」和「情景记忆召回」混为一谈**：checkpointer 的 `load(thread_id)` 是**每轮框架自动恢复会话**（短期记忆/续聊连贯性，发生在 `recall_node` 之前）；情景记忆的召回是 `recall_node` **按需触发**（用户提订单/工单号或说"之前/那个"时，捞回 `episodic_memory` 蒸馏片段注入上下文）。这是**两套不同的「读」**——前者保"接着聊"，后者补"被压缩掉的精确细节"。
- **`[历史摘要] + [最近 k 轮]` 是短期记忆被 load 进窗口后的「工作态」**，其来源就是 `checkpoints` 表，不是凭空现拼；压缩只裁剪"喂给模型的内容"，不裁剪库里存的全量。
- 子 Agent 的 `tool_calls` / `ToolMessage` **不落盘**（只在内存里，invoke 完即弃）；多专家时的**中间结论**也**不入库**，只有整合后的最终回复入库。

## 「无状态」到底什么意思

**无状态 = 服务实例不持有会话数据，不是"模型只看本轮提问"。**

- ❌ 有状态：历史存在某进程内存里 → 请求必须路由回同一实例，重启/扩容就丢。
- ✅ 无状态：数据在 PostgreSQL 里按 `thread_id` 存着 → 任意实例可服务任意请求，重启不丢，可水平扩容。

每轮实际流程：

1. `load(thread_id)` 取回该会话历史；
2. 组装 `[历史摘要] + [最近 k 轮] + [本轮提问]`；
3. 喂给模型 —— **模型看到的是"历史 + 本轮"，不是孤零零一句提问**。

## 存储成本（每条都存会不会爆）

- 算账：一条清洗后文本 ≈ 100 B ~ 几 KB；20 轮会话 = 40 行 ≈ **20 KB**；100 万会话 ≈ **20 GB**（PG 压缩后更小，单表可承受）。
- 真正会撑爆的是**长文档 / 图片 / 工具返回原始 JSON** —— 生产一律**外置对象存储（S3 / MinIO），对话表只存引用**。本项目工具返回本就不进历史。
- 长期成本四招（标准 Q6）：分层存储(热/温/冷) + 定期清理(保留期) + 旧会话摘要化（已预留 `summary` 列）+ 大 payload 外置。
- `checkpoints` 表要单独清理：它是**每节点一份全量 state 拷贝**，生产按「每线程保留最近 K 份」或「超 N 天删除」裁剪（框架默认不自动清理，需自己写定时任务）。

##  复习路径（读代码顺序）

1. `config.py` —— LLM 工厂，big/small 分离（为什么压缩用小模型）
2. `tools.py` —— `@tool` 怎么写，工具返回压缩 + 降级（第1块压缩）
3. `memory.py` —— 滚动压缩原理（第2块压缩）+ 触发时机
4. `vector_store.py` —— RAG 切分/embed/检索（问题 vs 片段）
5. `graph.py` —— **核心**：State 定义、supervisor 路由、条件边、子 Agent 节点、压缩节点、recursion_limit 护栏

## 生产替换点

| Demo 里 | 生产替换为 |
|---|---|
| `VectorStore`（内存+假 embed） | Milvus / Qdrant / pgvector + 真实 embed |
| `PostgresSaver` 单点 | 连接池调优 + 读写分离 / 托管 PG（情景记忆主库） |
| `get_llm` 默认端点 | 自建 vLLM / 第三方网关（模型路由：简单用小模型） |
| 手写 supervisor | 若流变复杂可升 `langgraph-supervisor` 或加人工审核节点 |
| 语义记忆（未实现） | 如需跨会话画像：向量库 + 按 `user_id` namespace 的 opt-in 画像层 |

> 已移除的替换点：`InMemorySessionStore`（遗留，未被引用）、`MemorySaver`（已换 `PostgresSaver`）。

## 注意

- **必须先准备 PostgreSQL + pgvector**（见「前置：情景记忆数据库」），情景记忆不使用内存介质。
- 真正调 LLM 需要 `OPENAI_API_KEY`；无 key 时 import 正常、调用会报错（这是预期）。
- 遗留文件（保留不删除，当前方案未引用）：`agent.py`、`session_store.py`、`messages.py`，详见下方分类。



根据目录和文件内容，这一版（方案 B，LangGraph 框架版）的文件分类如下：

## 有用的文件（当前 1.0 框架版，需要看的）

| 文件 | 作用 |
|------|------|
| `requirements.txt` | 依赖声明（langgraph/langchain 等） |
| `config.py` | LLM 工厂，`get_llm("big"/"small")` 大小模型分离 |
| `vector_store.py` | RAG 向量库（切分/embed/检索、上传入库、删文件清向量） |
| `db.py` | **情景记忆落库**：`get_checkpointer()`(PostgresSaver) + `insert_episodic_turn` + `search_episodic_by_vector`（后两者默认不接线） |
| `schema.sql` | **数据库 DDL**：pgvector 扩展 + checkpointer 表 + `episodic_memory` 表（原文+向量同表）+ 向量索引 |
| `tools.py` | LangChain `@tool` 工具集（订单/问答/工单 + 工具返回压缩 + 降级） |
| `memory.py` | 小模型摘要 + `compress_history` 滚动压缩 + `count_messages_tokens` 统一 token 口径 |
| `graph.py` | **核心**：StateGraph 手写 Supervisor + 3 个子 Agent 节点 + 条件路由 + 压缩节点 + 汇总作答节点(`answer`) |
| `session_store.py` | ⚠️ 遗留（未被引用，见下方） |
| `agent.py` | ⚠️ 遗留（早期手写 ReAct 版，import 已失效，见下方） |
| `main.py` | 框架版入口 |
| `test_demo.py` | 测试脚本 |
| `README.md` | 教学说明，映射生产要点 |

## 不用管的文件（遗留，保留不删除）

- **`agent.py`** — 早期手写 ReAct 版（方案 B 之前）。它 import 的 `RollingSummaryMemory` / `call_tool` 在当前 `memory.py` / `tools.py` 中已不存在，**直接运行会 ImportError**。当前运行入口是 `graph.py`，本文件未被引用。
- **`session_store.py`** — 会话存储抽象（内存版 + Redis 注释版）。`graph.py` 已用 LangGraph checkpointer（`PostgresSaver`）实现状态外置，本文件**未被引用**，仅作对比示例。
- **`messages.py`** — 旧手搓版消息类残留，已被 `langchain_core.messages` 取代，请勿使用。
- **`_smoke_test.py`** — 临时冒烟测试文件（命名带下划线前缀），非核心交付物。

简单说：**这 4 个文件当前方案都不参与运行**，保留仅供对比学习；其余 `.py` + `README.md` + `requirements.txt` + `schema.sql` 都是当前版本有用的。


# 递归查找所有大文件
Get-ChildItem -Recurse -File | Where-Object { $_.Length -gt 100MB } | 
    Select-Object FullName, 
        @{Name="Size(MB)";Expression={[math]::Round($_.Length/1MB,2)}}

1.为什么必须：它是 AgentState 的字段，会随 checkpointer 落盘、跨轮保留。若不重置，第二轮用户再问"订单"时，"order" 可能还躺在上一轮累积的列表里 → 被 C10 硬护栏误判成"重复派单" → 强制 farewell → 答不出来。所以每轮开头清空，只在本轮内累计


白板 = 累积的 state；checkpointer = 给白板每步拍快照存盘的机制；它落到的 checkpoints 表，是 LangGraph 库为实现"按步存档/恢复"而写死表名与结构的一张专用表，不是我们的业务表、也不是你能改名的可配项

3.主流防死循环靠的是"终止规则 + 步数上限"，而不是"派单去重"这个是主流的做法
目前代码的不允许多次调用同一个子agent的做法不属于主流后续可以去掉；但是对于工具调用的死循环，一般会做限制，比如检测"连续调用 工具名+参数 一致" → 强制停

4.子 agent 的上下文看到的是 state["messages"] 全量主上下文——即 supervisor 看到的那一份：从会话一开始、所有用户提问 + 之前各专家已产出的结论（AIMessage），到当前这一步。所以和主 agent 一致，不是隔离的私有上下文
关于这个其实不是主流的做法吧 多agent的场景下是不是子agent有自己的上下文；自己的上下文一般总哪里来呢 

5.关于工作记录的持久化


6.多agent架构
那"多个 agent + 一个纯路由分类器"（1.0）主流吗？给个准确版图：
主流拓扑 ① Supervisor/Manager（最常见）：supervisor 本身就是个会调工具的 agent，持有对话、挂 handoff 工具（必要时还挂 memory 工具），由它决定委托谁、或直接答。这是 LangGraph create_supervisor 的默认形态。
拓扑 ② Router/Dispatcher（你 demo 用的）：轻量分类器把请求丢给自包含专家。生产里确实存在（尤其"先分类再分发"的客服流水线），但 router 不推理、不会用工具，所以没那么"agentic"，在多 agent 里不如 ① 常见。
拓扑 ③ Network（peer-to-peer 互转，无中心）

---

## 速答：主 agent 与子 agent 怎么交互（handoff 模式）

> 适用问题："你的多 agent 怎么协作？""supervisor 和子 agent 怎么通信？""handoff 是什么？"
> 目标：30~60 秒讲清机制 + 一句对比显深度，别陷进代码细节。

**一句话总览**
主流（LangGraph supervisor）里，主 agent 和子 agent 的交互是通过 **handoff（转交）工具** 实现的：子 agent 以"工具"的形式挂到主 agent 上，由主 agent 的 LLM **自己决定**何时转交给谁——本质还是工具调用，不是硬编码分发。

**handoff 到底是什么（本质）**
- 它**底层就是一个普通 tool**（有名字 + JSON schema，LLM 能调）。
- 和普通工具（如 search_memory）的唯一区别在**副作用**：普通工具返回"观察结果"；handoff 工具被调用后触发**图的控制权转移**（等价于 `Command(goto=子agent节点)`），框架把执行切到子 agent。
- 对 LLM 而言两者无差别——都是"该不该调这个工具"的决策；框架才把这次调用解释成"换人上"。

**一轮交互流程（讲这个最直观）**
1. 主 agent 收到用户问题，结合上下文推理；
2. 决定调用某个 handoff 工具（如 `transfer_to_qa`）；
3. 框架切到对应子 agent 节点，子 agent 用自己的业务工具跑完任务；
4. 子 agent 结果交回主 agent；
5. 主 agent 决定：再派别的专家 / 直接整合作答 / 或先调记忆工具。
→ 控制权始终能回到主 agent，所以多专家协作、汇总都由它收口。

**和本 demo（1.0 路由分类器）的区别（防追问关键）**
- demo 1.0：supervisor 是**分类器返回枚举** + `route()` **硬编码选边**分发——派单写死，LLM 不参与"派谁"；
- 主流：派单是 **LLM 决策的 handoff 工具调用**——更 agentic、更鲁棒（用户原话出现"订单"却问别的，枚举路由易误判，handoff 不会）。
→ 一句话点出这个差异，比只背概念显得真做过。

**两个易漏的重点（加分项）**
- **子 agent 应有自己的上下文，不是背全量主上下文**：主流子 agent 只拿和本任务相关的消息（handoff 传入的 payload / 裁剪后上下文），而非整份 state。省 token、做隔离。⚠️ 本 demo 1.0 让子 agent 看全量 `state["messages"]`，是和主流不符的点，真实项目要切。
- **主 agent 持有对话与决策，子 agent 是无状态专家**：子 agent 跑完即弃，不持有跨轮状态；跨轮连续性在主 agent / checkpointer。

**收尾的一句话**
"所以是「主 agent 编排 + handoff 工具转交，LLM 驱动而非写死枚举」；子 agent 是无状态专家、跑完交回主 agent 做整合——协作靠工具调用，不是写死的 if/else 路由。"