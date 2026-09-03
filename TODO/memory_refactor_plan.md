# Demo 记忆实现改造清单（对齐生产级 / 豆包·WorkBuddy·Cursor 形态）

> 目标：把当前 `agent_demo/demo`（方案 B：LangGraph）从"可跑 demo"改造为
> 贴合主流 Agent 生产做法的记忆架构。所有改动**待用户确认后执行**。
> 权威标准：`docs/memory_system_summary.md`（三层记忆）+ 记忆规则（ID 72006304）。

---

## 一、目标架构（对齐标准三层 + 主流 Agent 形态）

| 层 | 定义 | 本 demo 落地 | 与豆包/WorkBuddy 对齐点 |
|---|---|---|---|
| **工作记忆** | 模型窗口内、易失、每轮现拼 | 每轮 `graph.invoke` 时由 checkpointer **load 该 thread_id 历史** + 本轮消息拼成；`compress_node` 超窗做 `[历史摘要]+[最近k轮]` | 豆包同形态：单会话多轮共用，窗口超了就压 |
| **情景记忆** | 会话历史，按 thread_id，默认**不跨会话** | LangGraph `checkpointer`（demo 用 `MemorySaver`，生产换 `PostgresSaver`）逐节点落盘**清洗后的 user+AI 结论轮** | 豆包即此形态（不跨会话） |
| **语义记忆** | 用户级事实/画像，opt-in 跨会话 | **本 demo 不默认实现**（标准明确非默认）；仅留扩展注释 | Cursor project rules / ChatGPT Memory 才是跨会话，属 opt-in，不硬塞 |

关键修正（回应需求第 1 点）：**子 Agent 内部 tool_calls / ToolMessage 不再回传主上下文**，
只回"结论"，这样落进情景记忆的就是干净的 `user+AI` 轮，不是全量工具循环噪声。

---

## 二、当前实现与标准的出入（改造动因）

1. **层标签错配**：`graph.py` 把 `MemorySaver` checkpointer 称作"工作记忆/短期"，
   但按标准它按 thread_id 落盘全量会话 = **情景记忆**。工作记忆是窗口本身，不"存储"。
2. **情景记忆与压缩耦合**：`memory.py` 的 `_persist_episode` 只在 `compress_history` 压缩淘汰时才写，
   违反标准"每轮 AI 答完就 put、与压缩解耦"。
3. **`recall_episodic` 死代码**：从未被调用，既没跨会话召回、也没会话内注入。
4. **持久化误导**：`MemorySaver` 是内存版，注释却写"重启不丢"——实际进程重启即丢。
5. **触发口径不统一**：`compress_node` 用字符数 `len(m.content)`，而 `memory.py` 内部用 tiktoken token 数。
6. **子 Agent 跨会话泄漏**：`_make_subagent` 用固定 `thread_id="order/qa/ticket"`，会跨会话串味。
7. **注释过度承诺**：`graph.py` 把 store 写成"跨 thread_id / 按 user_id namespace"，与标准冲突
   （跨会话属语义记忆、且非默认）。

---

## 三、修改点清单（M1–M9，确认后执行）

| # | 文件 | 行范围 | 改动内容 | 对齐理由 |
|---|---|---|---|---|
| **M1** | `graph.py` | 顶部 docstring 12-49 | 重写"两层记忆"为正确**三层**：checkpointer 标为「情景记忆持久化实现」(非工作记忆)；删掉"store 跨 thread_id / 按 user_id namespace"的过度承诺；标注语义记忆 opt-in 且本demo未实现 | 标准禁止①把工作记忆说成落DB、②跨会话默认 |
| **M2** | `graph.py` | `_run_subagent` 137-143；`_make_subagent` 75-79 | 子Agent跑完**只回传最后一条 AI 结论**（`[AIMessage(结论)]`），不回 tool_calls/ToolMessage；子Agent改**无状态**（去掉自身 `MemorySaver`，避免跨会话泄漏） | 标准 Q23①"工具结果只留结论"；需求第1点"子agent给结论"；修跨会话bug |
| **M3** | `graph.py` | `compress_node` 182 | 触发口径从 `len(m.content)` 字符数 → **tiktoken token 计数**（与 `memory.py` 一致），阈值 `MAX_TOKEN_LIMIT*0.8` | 口径统一 |
| **M4** | `memory.py` | 29-40、104-138、181 | **删除** `_persist_episode` / `recall_episodic` / `episodic_store` 整套；去掉 `compress_history` 内对 `_persist_episode` 的调用；清理 `os/json/time` 导入；保留 `summarize_with_small_model`/`split_into_turns`/`compress_history` | 死代码 + 与压缩耦合是反模式；checkpointer 已负责持久化与召回 |
| **M5** | `graph.py` | `compile` 231 及注释 | 把"重启不丢"类表述改为"demo 用内存 `MemorySaver` 可跑；**生产替换为 `PostgresSaver`(PostgreSQL+pgvector)** 才真正持久化"，附清晰生产替换注释 | 消除误导 |
| **M6** | `agent.py` | 文件头 | 加注释：`# 遗留代码：运行入口为 graph.py（方案B）；本文件为早期手写 ReAct 版，import 符号已不存在，当前未使用，保留不删除` | 需求第3点 |
| **M7** | `session_store.py` | 文件头 | 加注释：`# 遗留代码：graph.py 已用 LangGraph checkpointer 实现状态外置，本文件未被引用，保留不删除` | 需求第3点 |
| **M8** | `messages.py` | 文件头 | 补一句 `# 保留不删除`（原有"请勿使用"保留） | 需求第3点 |
| **M9** | `README.md` | 多处 | 同步：①"不用管的文件"补 `agent.py`/`session_store.py`/`messages.py`；②"生产要点对照表"改：压缩触发=token、子Agent只回结论、checkpointer=情景记忆持久化；③"生产替换点"标 `MemorySaver→PostgresSaver`、session_store 标遗留；④"两层记忆"描述改为正确三层 | 需求第3、4点 |

---

## 四、会删 / 会保留

- **会删**：`memory.py` 里的 `episodic_store` 目录逻辑 + `_persist_episode` + `recall_episodic`（纯死代码/反模式）。
- **不删**：`agent.py` / `session_store.py` / `messages.py`（按需求只加注释，不删除）。
- **不新增**：不硬塞半成品"语义记忆"（标准是 opt-in 非默认，硬加反而违反主流形态）。

---

## 五、待用户拍板的 3 个点

1. **M4 删除 `recall_episodic` 是否同意？** 判断：checkpointer 已负责情景记忆持久化与同会话召回（load thread_id），
   `recall_episodic` 冗余死代码；跨会话召回属 opt-in 语义记忆，不在本 demo 范围。→ 建议删。
2. **M2 子 Agent 改"无状态"是否同意？** 当前子 Agent 用固定 `thread_id="order/qa/ticket"` 会跨会话串味；
   改为无状态（每次用主上下文重新跑）更生产正确。→ 建议改。
3. **语义记忆要不要做最小 stub？** 建议**不做**（标准非默认，豆包默认也不跨会话）；
   只在注释留"生产可接向量库做 opt-in 跨会话画像"扩展说明。若面试想演示跨会话，可加最小 `user_profile_store` stub——待定。

---

## 六、执行状态（执行时更新）

- [ ] M1 graph.py 顶层 docstring 重写
- [ ] M2 graph.py 子Agent只回结论 + 无状态
- [ ] M3 graph.py compress_node 改 token 计数
- [ ] M4 memory.py 删除 episodic 死代码
- [ ] M5 graph.py 持久化注释修正
- [ ] M6 agent.py 文件头遗留注释
- [ ] M7 session_store.py 文件头遗留注释
- [ ] M8 messages.py 文件头保留注释
- [ ] M9 README.md 同步
- [ ] 第五节 3 个拍板点确认
