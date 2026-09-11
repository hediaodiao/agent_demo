# 复习笔记 · 2026.09.07

> 格式：按主题归并的多轮提问，每条 Q 对应一整段讨论。
> 代码引用：`文件:行号`（如 graph.py:489）。

---

## 一、LangGraph Checkpointer 与 checkpoints 表机制（db.py / graph.py）

### Q1：节点 `return` / `graph.update_state(...)` 为什么会写 `checkpoints` 表？这个"关联"体现在代码哪里？
**A：** 关联不是写在节点里的，而是在**编译图时把 checkpointer 挂上去**那一行一次性绑定的；之后由 LangGraph 框架在每个节点产出后**自动落盘**。

- 核心关联行：`graph.py:489` —— `graph = builder.compile(checkpointer=get_checkpointer(), recursion_limit=...)`
  - 不传 `checkpointer` → 图纯内存、无状态、啥都不落库；传了 → 运行时每走一个节点自动写一份全量 state 快照。
- 落地实现：`db.py:44-56` `get_checkpointer()` 返回 `PostgresSaver`（LangGraph 的 PG 检查点器）。
- 分桶 key：调用时 `config={"configurable":{"thread_id": session_id}}`，checkpoints 按 `thread_id` 区分会话。
- 运行时链路（节点里看不到 INSERT 的原因）：
  ```
  节点 return {...} → 框架走 reducer 合并 state → checkpointer.put_writes(thread_id, state) 写 checkpoints 表
  下轮 graph.invoke(config) → checkpointer.load(thread_id) 读回上一份快照（短期记忆恢复）
  ```
- `update_state(config, {...})` 走**同一条路径**也能落盘；所以 `compress_node` 用 `update_state` 把白板替换成 `[摘要]+[最近k轮]` 后，压缩结果真的落了库，下轮 load 回来的是压缩态（压缩没白做）。
- 文件顶部注释明确写了这点（graph.py:45-48、graph.py:483-485）："落盘由框架在每个节点产出后自动完成，与 compress_node 的 token 阈值无关"。

> 一句话：节点只 `return`，框架通过 `compile(checkpointer=...)` 绑定的 `PostgresSaver` 自动把全量 state 写进 `checkpoints` 表。

---

### Q2：`checkpoints` / `writes` / `threads` 是什么表？为什么不用手写建表？`schema` 怎么隔离？
**A：** 这三张是 LangGraph 的**框架表**（存全量 state、不存业务数据），由 `setup()` 自动建，与你的业务表解耦。

- 三张表各自职责：
  | 表 | 存什么 | 主键维度 |
  |---|---|---|
  | `threads` | 会话元数据（每 `thread_id` 一行：`created_at`/`updated_at`/`metadata`） | `thread_id` |
  | `checkpoints` | **全量 state 快照**：每节点一行 | `(thread_id, checkpoint_ns, checkpoint_id)` |
  | `writes` | 节点 return 时各 channel 的增量写记录（框架做精确断点恢复用） | `(thread_id, checkpoint_ns, checkpoint_id, task_id, channel)` |
- `setup()`（db.py:55）是 **LangGraph `PostgresSaver` 的方法**，内部硬编码 `CREATE TABLE IF NOT EXISTS` 语句，**幂等**（有就跳过、没有才建），表结构**随 LangGraph 版本维护**——所以你项目里看不到手写建表 SQL。
- 为什么"不在 `schema.sql` 手写"（db.py:50-52）：
  1. 避免和你手写的业务表结构冲突（框架升版改了 `checkpoints` 列会报错）；
  2. 避免命名混淆（框架表和业务表混在同一份手写 schema 里分不清）。
- schema 隔离：当前 `db.py:54` 没传 `schema` → 默认 `public`，框架表与业务表**同库同 `public` schema**；需要物理隔离时改用 `PostgresSaver.from_conn_string(uri, schema="langgraph")`，即同库内换到独立 `langgraph` schema（仍是同一个数据库，只是换个命名空间）。

> 一句话：`setup()` 把库源码里写死的建表 SQL 跑一遍，自动建出三张框架表；表结构由 LangGraph 版本决定，不该手写。

---

### Q3：`from_conn_string` 是 LangGraph 的方法吗？连接串怎么理解？`setup` 什么时候才真的连库？类比 Milvus 的 `default`？
**A：**

- **`from_conn_string` 是 LangGraph 的方法**（db.py:23 导入 `from langgraph.checkpoint.postgres import PostgresSaver`），参数是标准 **PostgreSQL 连接 URI**：`postgresql://user:password@host:port/dbname`。同类还有 `AsyncPostgresSaver` / `SqliteSaver` 的 `from_conn_string`。
- **连接时机（懒连接）**：
  - `db.py:54` 只 `new` 一个 `PostgresSaver` 对象、把连接串存进去，**此时未连库**；
  - `db.py:55` `saver.setup()` 才**真正连库 + 建框架表**；
  - `db.py:56` `return saver` 把这个 checkpointer 交给 `compile(checkpointer=...)` 用。
  - 运行时也是走**连接池懒连接**：每次 invoke 临时拿连接、用完归还，非常驻长连接。
- 返回的是 **`PostgresSaver` checkpointer 实例**（框架对象），**不是**裸的 `psycopg` 连接，也不是"数据库实例"本身。
- **连接串 vs Milvus `default`**：
  - Milvus 的 `"default"` 只是**数据库名**；连接时还要另外传 host/port/auth。
  - PG 连接串把"连哪台 + 账号 + 用哪个库"全压进一条：`host:port`+`user:pass` 类比 Milvus 的 host/port/auth，`/dbname`（如 `/ob_detection`）才类比 Milvus 的 `"default"` 数据库名。
  - 即：整个连接串 ≈ "Milvus 的 host/port/auth + default 数据库名"合在一起；不能把它只理解成"连哪个实例"。

> 一句话：第 54 行按连接串 new 一个 checkpointer（认准那个库）；第 55 行才真连库建表；连接串里 `/dbname` 对应 Milvus 的 `default` 库名。

---

## 二、主图调度逻辑疑问（graph.py，同会话 earlier 部分）

### Q4：`compress_node` 用 `update_state` 把白板整体替换成 `[摘要]+[最近k轮]`，这个 k 轮怎么算？checkpoints 存全量 vs 压缩是不是矛盾？
**A：**
- **k 轮 = 用户和 AI 各算一轮**（一轮 = 一次 `graph.invoke`：用户一条输入 → AI 给出最终回复）。`[最近k轮]` 指最近 k 个这样的"用户↔AI"来回。
- **不矛盾**：`checkpoints` 表存的是**全量 state 快照**（压缩只发生在"喂给模型的窗口/工作记忆层"，不裁剪库）；`compress_node` 用 `update_state` 改的是**工作记忆（窗口内容）**，库里仍然保留全量历史，压缩 ≠ 存储删除。续聊时 load 回来的是已被压缩的窗口态。

### Q5：`graph.py:147` 附近 `range` 三个参数、`graph.py:163` 的 `or ""`、`graph.py:171` 派单次数怎么算？
**A：**
- `range(start, end, step)`：起点、终点（不含）、步长；步长为负表示倒序。`range(末尾, 0, -1)` 是从末尾往前遍历（配合早已倒序的列表不会"双重倒序成正序"，要看原列表是否已倒序）。
- `or ""`：是**取默认值**的表达（`x or ""` 当 `x` 为假时用空串），不是赋值；放在判断条件里只是给缺失值兜底，没有它逻辑上某些情况会拿到 `None` 而报错——是否需要取决于上游是否保证有值。
- **派单次数**：`MAX_DISPATCHES` 是"**本轮最多派给几个不同专家**"（不是每专家最多几次）。每轮 `called_agents` 重置为空；派一个专家就追加进列表；护栏 `if next_agent in called_agents or len(called_agents) >= MAX_DISPATCHES: next_agent = "farewell"`——所以同一专家整轮只能派 1 次（连不连续都不行），上限 = 可派专家数（当前 3 个：order/qa/ticket）。

### Q6：`with_structured_output` 是什么？`farewell` 是节点吗？`recursion_limit=12` 走得到吗？
**A：**
- `with_structured_output(RoutingDecision)`：把"自由文本输出"的模型绑定成"**必须按给定 TypedDict schema 返回**"的模型（底层走 function-calling / JSON mode + schema 约束），调用后直接拿到解析好的对象而非要自己解析的文字。
- `farewell` **不是节点**，只是 `RoutingDecision` 枚举里的一个**哨兵字符串**（终止信号）。条件边映射 `{"order":"order","qa":"qa","ticket":"ticket","farewell":"answer","__end__":END}` 把它翻译成去 `answer` 节点收尾。
- 一轮最多 3 次子 Agent 调用（3 个专家各 1 次）。`recursion_limit=12` 是**物理步数天花板**（数节点执行次数），3 专家全派的最满情况步数也远小于 12，所以走得到、且是兜底防死循环的安全网（去重太狠最坏只是提前 `farewell`，不会卡死）。这套"不重复派 + 上限"是 supervisor 路由主流基线，代价是"同专家需多次调用"的复杂意图会被牺牲。

---

##  速记（Checkpointer 主线）

1. **落库触发点** = `compile(checkpointer=get_checkpointer())` 一处绑定，节点 `return`/`update_state` 经框架自动写 `checkpoints`。
2. **三张框架表** = `threads`(会话) / `checkpoints`(全量快照) / `writes`(增量写)；由 `setup()` 内置 SQL 幂等建，不手写。
3. **连接** = `from_conn_string(uri)` 只存串（不连）；`setup()` 才真连+建表；走连接池懒连接。
4. **隔离** = 默认同库同 `public`；`schema="langgraph"` 同库内另起 schema。
5. **thread_id** = checkpoints 按会话分桶的 key（无状态化 + 续聊恢复）。

---

## 三、LangGraph 节点编写约定（graph.py）

### Q7：节点的输入参数和 return 一般有什么讲究？
**A：**
- **输入（签名约定）**：第 1 个参数永远是 `state`（整张状态）；框架再按签名**自动注入**可选参数：
  - 第 2 个 `config`（`RunnableConfig`，含 `configurable:{thread_id,...}`）—— 用到才声明；
  - 第 3 个 `store`（跨线程持久记忆 `BaseStore`）—— 可选新特性。
  - 印证：
    ```166:166:agent_demo/demo/graph.py
    def supervisor(state: AgentState) -> dict:          # 只用 state
    ```
    ```226:228:agent_demo/demo/graph.py
    def order_node(state):  return _run_subagent(order_agent, state)   # 只用 state
    ```
    ```376:376:agent_demo/demo/graph.py
    def recall_node(state: AgentState, config: dict) -> dict:          # state + config
    ```
- **return（返回值约定）**：
  | 返回 | 含义 | 注意 |
  |---|---|---|
  | `dict` | 部分更新，key 须是 state 字段，按 reducer 合并 | `messages` 等 `Annotated` 累加通道会**追加**；普通通道覆盖 |
  | `Command(update=..., goto=...)` | 同时更新状态 + 指定下一步（节点内分支） | 本图未用，路由走 `add_conditional_edges` |
  | `None` | 不改状态 | — |
- **最大讲究（已踩坑并写注释）**：`messages` 用累加 reducer，节点若 `return {"messages": 完整列表}` 会把整份列表**再追加一遍、历史翻倍**，整体替换某通道要用 `graph.update_state`（覆盖，不走 reducer）：

    ```425:428:agent_demo/demo/graph.py
    #   - State.messages 用 Annotated 累加 reducer，节点若 `return {"messages": 完整列表}`
    #     会把完整列表「再追加」到白板，导致历史翻倍。所以**不能用普通返回值写回完整列表**。
    #   - 正确做法：用 graph.update_state(...) 把整个 messages 字段「整体替换」为压缩结果
    ```

> 一句话：节点签名 `state`(必有)+`config`/`store`(用到才写)；return 用 dict 走 reducer 合并，messages 通道别 return 完整列表否则翻倍。

---

### Q8：`recall_node` 需要 config，为什么定义/注册时不写？写不写必须和 `order_node` 一样吗？
**A：** 把两件事分开：
- **注册时（`add_node`）永远不传 config**——`builder.add_node("recall", recall_node)`（graph.py:456）传的是**函数引用**，不是调用；config 是运行时由框架注入的，所以那行本来就不能写 config。
- **函数定义里写不写 config，看 body 用不用**：
  - `recall_node` **写了** `config`（graph.py:376），因为 body 要取 `thread_id` 限定召回哪个会话的记忆：
    ```390:390:agent_demo/demo/graph.py
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")
    ```
  - `order_node` **没写** `config`（graph.py:226），因为它只把 `state` 丢给子 Agent 跑，用不到 `thread_id`。
- **不必和 `order_node` 一样**：config 写不写完全取决于"本节点函数体要不要用到它"。`order_node` 只是"不需要 config"的样例，不是模板；为了和 `order_node` 一致而删掉 `recall_node` 的 config，会导致取不到 `thread_id`、记忆召回跨会话。

> 一句话：`add_node` 不传 config 是注册机制决定；函数定义里 config 写不写是 body 用不用决定，recall_node 用了所以写、order_node 没用所以不写，二者都合法、不必一致。

---

### Q9：自定义业务参数（如 `agent=order_agent`）必须预绑定吗？有哪几种写法？
**A：** **必须预绑定。** LangGraph 运行时只调 `node(state[, config])`，没有任何机制让你 `graph.invoke` 时给某节点单独传 `agent`。所以 `_run_subagent(agent, state)` 这种"带自定义参数"的函数**不能直接 `add_node("order", _run_subagent)`**，必须把 `agent` 在注册前用闭包固定成"只吃 state"的可调用对象。四种等价写法：

1. **手写 wrapper 函数（本图采用，最直白主流）**：
   ```225:225:agent_demo/demo/graph.py
   def order_node(state):  return _run_subagent(order_agent, state)
   ```
   `order_agent` 从模块作用域闭包进来，一行透传；可读、可单独加注释/断点。
2. **`functools.partial`（最简洁，免写 3 个薄函数）**：
   ```python
   from functools import partial
   builder.add_node("order", partial(_run_subagent, order_agent))
   ```
   与 `order_node` 完全等价。
3. **lambda（能跑但不推荐）**：`builder.add_node("order", lambda state: _run_subagent(order_agent, state))`——可读性差、难调试。
4. **闭包工厂（专家多时优雅）**：
   ```python
   def make_agent_node(agent):
       def node(state): return _run_subagent(agent, state)
       return node
   builder.add_node("order", make_agent_node(order_agent))
   ```

> 一句话：自定义参数不能当节点运行时参数传（框架只给 state/config），必须在注册前用闭包绑定；手写 wrapper 最主流，`partial` 最简洁，二者等价。

---

##  速记（节点编写）

1. **节点签名**：第 1 参 `state`（必有），第 2 参 `config`（用到才写，框架按签名注入），第 3 参 `store`（跨线程记忆，可选）。
2. **return**：`dict`=部分更新（走 reducer 合并）；`Command`=更新+路由；`None`=不改。`messages` 是累加 reducer，`return` 完整列表会翻倍，整体替换用 `update_state`。
3. **config**：`add_node` 那行不传（只注册函数）；函数定义里写不写看 body 用不用——`recall_node` 取 `thread_id` 所以写，`order_node` 纯跑子 Agent 所以不写，二者都合法、不必一致。
4. **自定义参数必须预绑定**：LangGraph 只给 `state`/`config`，不能运行时传 `agent`。绑法：手写 wrapper（主流）/ `functools.partial`（最简洁）/ lambda（不推荐）/ 闭包工厂（专家多时）。
