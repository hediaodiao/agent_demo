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
| 向量库 | `OpenAIEmbeddings` + 内存 `VectorStore` | 生产换 Milvus/Qdrant/pgvector |
| 会话存储 | 内存 `InMemorySessionStore` | 生产换 Redis（带 TTL） |
| LLM | `ChatOpenAI` | 决策大模型 + 压缩小模型分离 |

## 运行

```bash
pip install -r requirements.txt
set OPENAI_API_KEY=你的key
set OPENAI_BASE_URL=https://your-endpoint   # 可选，兼容端点/自建 vLLM

python main.py                  # 交互式对话
python main.py --session user1  # 指定 session_id（无状态化演示）
python test_demo.py             # 端到端冒烟测试（验证分流路由）
```

## 代码与「我们讨论的生产要点」对照

| 我们讨论的要点 | 在代码里的位置 |
|---|---|
| **多 Agent（Supervisor）** | `graph.py`：`supervisor` 节点做意图分类 + `add_conditional_edges` 分流到 order/qa/ticket |
| **子 Agent 是独立实例** | `graph.py`：`order_agent/qa_agent/ticket_agent` 各用 `create_react_agent` 封装 |
| **工具返回压缩（第1块）** | `tools.py`：`query_orders` 用 `_format_orders` 裁剪；`rag_search` 过长用小模型摘要 |
| **上下文滚动摘要（第2块）** | `memory.py`：`compress_history`（token 计数触发，压最早步骤）；`chat()` 入口示意调用 |
| **单轮超长走 RAG** | `vector_store.py` + `tools.rag_search`：算「问题 vs 片段」相似度 |
| **无状态化 + 状态外置** | `session_store.py`；LangGraph `MemorySaver` 用 `thread_id` 管理历史 |
| **步数封顶（硬护栏）** | `graph.py`：`compile(recursion_limit=Settings.MAX_ITERATIONS)` |
| **降级 / 备用数据源** | `tools.query_orders`：主源挂了切 CSV 备胎 |
| **压缩用小模型** | `config.get_llm("small")`；`memory.summarize_with_small_model` 用它 |
| **循环骨架框架封装 vs 业务硬编码** | `create_react_agent` 给子 Agent 循环；工具压缩/`compress_history`/supervisor 路由是你写的 |

##  复习路径（读代码顺序）

1. `config.py` —— LLM 工厂，big/small 分离（为什么压缩用小模型）
2. `tools.py` —— `@tool` 怎么写，工具返回压缩 + 降级（第1块压缩）
3. `memory.py` —— 滚动压缩原理（第2块压缩）+ 触发时机
4. `vector_store.py` —— RAG 切分/embed/检索（问题 vs 片段）
5. `graph.py` —— **核心**：State 定义、supervisor 路由、条件边、子 Agent 节点、压缩节点、recursion_limit 护栏

## 生产替换点

| Demo 里 | 生产替换为 |
|---|---|
| `InMemorySessionStore` | Redis（带 TTL）/ PostgreSQL + JSON |
| `VectorStore`（内存+假 embed） | Milvus / Qdrant / pgvector + 真实 embed |
| `MemorySaver` checkpointer | Redis/Postgres checkpointer（持久化、可恢复） |
| `get_llm` 默认端点 | 自建 vLLM / 第三方网关（模型路由：简单用小模型） |
| 手写 supervisor | 若流变复杂可升 `langgraph-supervisor` 或加人工审核节点 |

## 注意

- `messages.py` 为旧手搓版遗留，已弃用，统一用 `langchain_core.messages`。
- 本 demo 默认**内存可跑结构**，但**真正调 LLM 需要 OPENAI_API_KEY**（无 key 时 import 正常、调用会报错，这是预期）。



根据目录和文件内容，这一版（方案 B，LangGraph 框架版）的文件分类如下：

## 有用的文件（当前 1.0 框架版，需要看的）

| 文件 | 作用 |
|------|------|
| `requirements.txt` | 依赖声明（langgraph/langchain 等） |
| `config.py` | LLM 工厂，`get_llm("big"/"small")` 大小模型分离 |
| `vector_store.py` | RAG 向量库（切分/embed/检索、上传入库、删文件清向量） |
| `session_store.py` | 会话状态外置（内存版 + Redis 生产版注释） |
| `tools.py` | LangChain `@tool` 工具集（订单/问答/工单 + 工具返回压缩 + 降级） |
| `memory.py` | 小模型摘要 + `compress_history` 滚动压缩逻辑 |
| `graph.py` | **核心**：StateGraph 手写 Supervisor + 3 个子 Agent 节点 + 条件路由 |
| `agent.py` | 子 Agent 轻量 ReAct 循环骨架 |
| `main.py` | 框架版入口 |
| `test_demo.py` | 测试脚本 |
| `README.md` | 教学说明，映射生产要点 |

## 不用管的文件（1.0 可忽略）

- **`messages.py`** — 旧手搓版残留，文件第 1 行已写明：`"此文件为「手搓版」遗留，已被 langchain_core.messages 取代，请勿使用。"` 框架版统一用 `langchain_core.messages`，不会 import 它，可直接无视（也可删掉，不影响运行）。
- **`_smoke_test.py`** — 临时冒烟测试文件（命名带下划线前缀），非核心交付物，仅供快速验证语法/跑通用，可不用管。

简单说：**`messages.py` 明确弃用不用看；其余 `.py` + `README.md` + `requirements.txt` 都是当前版本有用的**。`_smoke_test.py` 是辅助测试，非必读。


# 递归查找所有大文件
Get-ChildItem -Recurse -File | Where-Object { $_.Length -gt 100MB } | 
    Select-Object FullName, 
        @{Name="Size(MB)";Expression={[math]::Round($_.Length/1MB,2)}}
