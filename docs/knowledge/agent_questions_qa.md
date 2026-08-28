# Agent 核心疑惑 Q&A 总结

> 整理自 agent 学习过程中的问答，聚焦"什么是 Agent / 框架做什么 / 企业怎么落地"。

---

## Q1. 直接调大模型 API、不框架，算不算 Agent？只是单轮对话吗？

**A：** 关键不在"用没用框架"，而在"有没有自主循环 + 工具调用"。

- 只调 API 做单轮问答（发一句、回一句、结束）→ **不是 Agent**，是单轮对话。
- 但如果你**自己写代码**实现 `调模型 → 解析出"要调工具" → 执行工具 → 结果拼回 → 再调模型 → … → 模型说答完了`，即使零框架，它**也是 Agent**（如 `react-agent-from-scratch.ipynb`，纯手写循环但它是标准 ReAct Agent）。

> **结论：Agent 的本质 = 模型自主决定调工具 + 观察结果 + 循环直到完成。框架只是把这套机制封装好，让你不用手写循环。**

---

## Q2. 用 OpenAI / LangChain 框架定义 Agent + 工具，就会自动 思考-行动-观察 循环吗？

**A：** 对，这正是框架的价值。定义好 `Agent`（提示词）+ `tools`（工具）+ `handoffs`（交接），调用 `Runner.run(...)`，框架内部自动跑：

```
思考(模型决定下一步) → 行动(调工具/交接) → 观察(拿到结果) → 再思考 → … → 结束
```

你完全不用手写这个 while 循环（参考 `customer_service/main.py`：只定义业务，循环交给 `Runner.run`）。

---

## Q3. 用框架 vs 手写，边界怎么划分？

| 做法 | 是 Agent 吗 | 说明 |
|---|---|---|
| 直接调 API，单轮问答 | ❌ 不是 | 无循环、无工具 |
| 直接调 API，但自己手写 思考-行动-观察 循环 + 工具 | ✅ 是 | 没框架也是 Agent |
| 用 OpenAI/LangChain 框架定义 Agent+工具 | ✅ 是 | 框架替你写循环，最省事 |

---

## Q4. 我之前的"测试用例项目"（RAG + 调 API + 结果塞提示词）算 Agent 吗？

**A：** 处于灰色地带，取决于有没有"自主决策循环"：
- 如果是**固定流程**（先检索 → 结果塞提示词 → 问答）→ 是 **RAG 应用，不是真正自主 Agent**。
- 如果模型能**自己决定要不要检索、调哪个工具** → 已是 Agent 雏形。

你之前判断"第三个项目不算 Agent，因为只是调 API + RAG，没有编排/工具调用/思考行动观察"——**判断准确**。严谨说它是"LLM 应用（RAG + 多模型封装）"，不是 Agent。

---

## Q5. 企业实际落地用什么 Agent 框架？中小公司呢？

**A：**
- **大型 / 复杂业务**（状态机、长流程、强可控）：**LangGraph** 最主流（图状态机、checkpointer、人在环、可观测）；大厂多自研（字节 Coze、阿里百炼、百度千帆、腾讯元器等）。
- **中小型 / 快速出产品**：**OpenAI Agents SDK**（轻量、内置 Handoff/Session/MCP）、**Dify / Coze 低代码平台**（拖拽上线最快）、**CrewAI**（多角色协作）、**LlamaIndex**（偏 RAG 知识库 Agent）。

---

## Q6. `handoffs`（OpenAI 框架）里的 `handoff` 和 `tool_name_override` 是什么？

**A：**
- `handoff` 是 OpenAI 封装好的"把对话移交给另一个 Agent"的能力，模型自己决定交给谁。
- 在 SDK 里 **Handoff 底层就是把"转给另一个 Agent"伪装成一个特殊工具**给模型选。`tool_name_override="transfer_to_faq_agent"` 是**自定义这个交接工具显示的名字**，让模型更清楚用途、更可控可读。

---

## Q7. `with trace(...)` 是什么？

**A：** Python 上下文管理器（同 `with open()`）。`trace("Customer service", group_id=...)` 是 OpenAI 的**链路追踪**：把整轮对话（模型调用、工具、交接）记录到追踪后台，便于可视化监控耗时/花费。**纯可观测性辅助，去掉也不影响运行。**

---

## Q8. `Runner.run(...)` 没看到实现，它做什么？入参是什么？

**A：** 是 OpenAI 封装好的执行入口，内部跑完整 ReAct 循环。

入参（本示例传了三个）：
- `current_agent`：从哪个 Agent 开始跑（首轮是分诊台）。
- `input_items`：对话历史（含之前的 user/assistant/tool 消息）。
- `context=context`：业务状态实例，框架自动传给工具和钩子。

内部流程：
1. 历史 + 提示词 + 工具列表发给模型；
2. 模型返回文本（结束）或"我要调工具 X"；
3. 要调工具 → 框架自动执行该函数 → 结果喂回模型；
4. 要 Handoff → 框架自动切换目标 Agent；
5. 重复直到模型输出最终文本；
6. 返回 `result`（含 `new_items` / `to_input_list()` / `last_agent`）。

---

## Q9. 钩子（hook，如 `on_seat_booking_handoff`）的作用？为什么自动生成航班号？

**A：** 钩子 = **交接发生时自动触发的副作用代码**，用来做"进入某业务前的预处理"。

示例中 `update_seat` 工具断言 `flight_number is not None`（改座必须知道航班号），但用户不会主动报航班号。于是设计：一旦决定转给"订座专员"，钩子立刻随机生成航班号塞进 Context，保证专员调工具时数据就绪。

**真实落地场景**：转订单 Agent 前自动按用户 ID 从数据库查出订单列表预填；转退款 Agent 前自动校验登录态/权限。本 demo 航班号是随机造的（无真实库），真实项目从会话/订单系统取。

---

## Q10. 为什么企业落地代码里看不到手写 ReAct 循环？

**A：** 因为那个循环（模型输出→解析→调工具→塞回→再问）**已被 `Runner.run` 封装在框架内部**。企业落地只关心业务层：定义 Agent、工具、交接、写主循环收用户输入。手写循环版（`react-agent-from-scratch.ipynb`）和框架版（`customer_service/main.py`）是**同一件事的两层**——前者教本质，后者教落地。

---

## Q11. `customer_service` 示例项目解决什么业务问题？

**A：** 解决"多业务线客服咨询，如何让系统自动把用户引导到正确处理流程"。采用**分诊制**：
1. 分诊台 Agent（Triage）：不干活，只判断该找谁，Handoff 给专员；
2. FAQ 专员：只查常见问题（`faq_lookup_tool`）；
3. 订座专员：只改座位（`update_seat`，写回订单状态）；
4. 专员处理完或遇管不了的问题，可转回分诊台，形成闭环。

用 Context 跨轮次记住订单信息；用 Handoff 实现自动路由。这是标准"企业多 Agent 客服落地模板"。

---

## Q12. 我本地已下载的"现成企业落地代码"有哪些？

**A：** 均在 `09_OpenAI官方框架-openai-agents/openai-agents-python/examples/` 下：
1. `customer_service/main.py` — 单文件完整客服落地（Context+Tools+Handoff+循环）⭐首推
2. `research_bot/` — 多文件研究助手（规划+并行搜索）
3. `financial_research_agent/` — 多文件（子 Agent 工具 + 验证环节）
4. `handoffs/` — 多 Agent 交接最小落地
5. `memory/` — 会话/长期记忆落地（15 个 py）
6. `mcp/` — MCP 接外部系统落地（17 个 py）

另有 LangGraph 落地对照版：`agent/handoffs/handoffs.py`。

---

## Q13. Function Calling、Tool、MCP 三者区别是什么？

**A：**
- **Function Calling（函数调用）**：大模型本身的底层能力（不是框架功能）。模型不能直接执行函数，只能输出结构化 JSON：`{"name":"search","arguments":{"q":"北京天气"}}`，表示"我要调这个函数"。真正解析 JSON、执行函数、把结果回灌模型，由你的代码或框架（`Runner.run`）负责。**它是"思考-行动-观察"里"行动"的底层机制。**
- **Tool（工具，即 demo 里的 `@tool`）**：Function Calling 的**上层封装**。把一个 Python 函数自动翻译成模型能懂的"能力说明书"+注册进框架。`@tool` 从函数签名自动生成函数名、参数 schema、描述（description），模型靠 description 判断何时调用。
  - 即：**Tool ≈ Function Calling 的方便写法**，本质一样，Tool 省去手搓 JSON schema。
- **MCP（Model Context Protocol）**：解决"工具从哪来、怎么共享"的标准协议（类似 USB 接口）。把提供能力的一方做成 **MCP Server**，用能力的一方（Agent）做成 **MCP Client**，双方遵守协议即可即插即用，不用为每个系统单独写适配。MCP Server 暴露的能力，底层仍通过 Function Calling 调用。

三者层级：
```
MCP（标准化"工具从哪来"，跨项目/跨系统复用、即插即用）
  ↓ MCP Server 暴露的工具底层还是 Function Calling
Function Calling（模型能力：输出"调谁+参数"的 JSON）
  ↓
Tool（@tool 把 Python 函数自动变成 Function Calling 格式）
  ↓
你的代码（真正干活的函数，如 faq_lookup_tool / update_seat）
```

**适用场景：**
- 工具只给自己项目用、逻辑简单 → 直接写 `@tool`（如 demo）最省事。
- 工具要被多个 Agent/项目共用，或连外部系统（数据库/CRM/文件系统）→ 用 MCP 做成 Server 统一接入。本地 `09_openai-agents/examples/mcp/` 是现成落地示例。

**历史演进（旧模型没有原生 Function Calling）：**
- Function Calling 不是"模型凭空多了执行函数的能力"——模型永远只输出 token。区别在于有没有**结构化的 tool_calls 机制**。
- **早期模型（GPT-3.5/GPT-4 的 0301/0314 版本，2023 年初）**：**没有原生 Function Calling**。要靠 prompt 工程约定 JSON 格式（system prompt 写"要查天气就输出 `{"action":"get_weather","city":"..."}`"），再自己写正则/JSON 解析文本 → 不稳定、易出错。这也是早期 ReAct 论文（2022）要手写 "Thought/Action/Observation" 文本解析的原因。
- **2023-06 起（gpt-3.5-turbo-0613 / gpt-4-0613）**：OpenAI 推出 **native Function Calling**，API 新增 `tools`/`functions` 参数，模型结构化返回 `tool_calls`，框架可直接接住。
- **之后所有主流模型**（GPT-4o 系列、Claude 3+、Gemini、Qwen、DeepSeek 等）都原生支持，字段名略不同但机制一致。
- 一句话：**Function Calling = "模型能力 + API 协议"共同提供。旧模型 + 旧 API 没有这套结构化机制，只能靠 prompt 模拟；新模型则在训练时对齐了"何时输出工具调用"的格式，稳定性和可解析性大幅提升。**

---

## Q14. 当前的 demo（`@tool` 写法）算实现了什么？如果改成 MCP 怎么改？

**A：**
当前 demo 的 `faq_lookup_tool` / `update_seat` 是**本地、写死在当前文件里的 Tool（= Function Calling 封装）**。模型能调它们，但函数代码就在项目内，无法被别的项目复用，也无法连远程系统。

**如何改成 MCP（思路，不改动文件）：**

1. **抽出能力做成 MCP Server（独立进程/服务）**
   - 把 `faq_lookup_tool`、`update_seat` 的逻辑，用 MCP SDK（如 `mcp` 库的 `@mcp.tool()`）注册成一个 MCP Server，对外暴露这两个工具。
   - 这个 Server 可以跑在另一台机器，通过 stdio 或 HTTP/SSE 通信。

2. **demo 里的 Agent 改成 MCP Client**
   - 删掉本地的 `@tool` 函数定义。
   - 用 OpenAI Agents SDK 的 MCP 接入方式（如 `MCPServerStdio` / `MCPServerSse`）连接那个 Server，把返回的工具列表挂到 Agent 上（`agents=mcp_servers=[...]` 或类似的 `tools` 接入）。
   - 模型调用工具时，框架自动通过 MCP 协议把"调谁+参数"发给 Server，Server 执行后把结果回传。

3. **Context（业务状态）的处理**
   - 注意：`update_seat` 现在靠 `RunContextWrapper` 读写本地 `AirlineAgentContext`。改成 MCP 后，状态要么保留在 Client 侧（Agent 把参数传过去，Server 只做纯计算/查库），要么 Server 自己管状态（如连数据库存订单）。真实落地通常让 Server 连真实数据库，Client 不再持有业务状态。

**改动收益：** 这个 MCP Server 可以同时被你别的 Agent、别人的项目复用；换数据库/系统只改 Server，Agent 代码不动。

**一句话：** 当前 demo = 本地 Tool（Function Calling）；改 MCP = 把工具搬进独立 MCP Server，Agent 当 Client 即插即用。

---

## Q15. 把 demo 改造成 MCP 的示例代码（Server + Client）

**A：** 示意代码（不写入文件），基于现有 `customer_service` 改造。

**① MCP Server（独立服务，用官方 `mcp` 库暴露工具）：**
```python
# mcp_server.py —— 独立进程运行，例如: python mcp_server.py
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("airline-service")

@mcp.tool()
def faq_lookup_tool(question: str) -> str:
    """Lookup frequently asked questions about the airline."""
    q = question.lower()
    if any(k in q for k in ["bag", "baggage", "luggage", "carry-on"]):
        return "You are allowed to bring one bag on the plane. Under 50 pounds and 22x14x9 inches."
    elif any(k in q for k in ["seat", "seats", "seating", "plane"]):
        return "There are 120 seats: 22 business, 98 economy. Exit rows 4 and 16."
    elif any(k in q for k in ["wifi", "internet", "wireless", "network", "online"]):
        return "We have free wifi on the plane, join Airline-Wifi"
    return "I'm sorry, I don't know the answer to that question."

@mcp.tool()
def update_seat(confirmation_number: str, new_seat: str, flight_number: str) -> str:
    """Update the seat for a given confirmation number and flight."""
    # 真实场景连数据库更新订单；demo 直接返回
    return f"Updated seat to {new_seat} for confirmation {confirmation_number} on flight {flight_number}"

if __name__ == "__main__":
    mcp.run()  # 默认 stdio；也可 mcp.run(transport="sse") 走 HTTP
```
> 注意：原 `update_seat` 靠 `RunContextWrapper` 读写本地 `AirlineAgentContext`，改 MCP 后状态不再由 Client 持有——`flight_number` 改为由调用方（Agent）作参数传入，Server 只做纯处理/查库。

**② 改造后的 Client（Agent 当 MCP Client 接入）：**
```python
# customer_service_mcp.py
from agents import Agent, Runner
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from agents.mcp import MCPServerStdio

async def main():
    server = MCPServerStdio(
        name="airline-service",
        params={"command": "python", "args": ["mcp_server.py"]},
    )
    await server.connect()  # 建立连接，框架自动拉取 Server 暴露的工具列表

    faq_agent = Agent(
        name="FAQ Agent",
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
        You are an FAQ agent. Use the faq_lookup_tool to answer questions.
        If you cannot answer, transfer back to the triage agent.""",
        mcp_servers=[server],
    )
    seat_booking_agent = Agent(
        name="Seat Booking Agent",
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
        Ask for confirmation number, seat, and flight number, then use the update_seat tool.""",
        mcp_servers=[server],
    )
    triage_agent = Agent(
        name="Triage Agent",
        instructions=f"{RECOMMENDED_PROMPT_PREFIX} You delegate to FAQ or Seat Booking agents.",
        handoffs=[faq_agent, seat_booking_agent],
        mcp_servers=[server],
    )
    result = await Runner.run(triage_agent, [{"role": "user", "content": "What's your wifi policy?"}])
    print(result.final_output)
    await server.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

**改造前后对比：**
| 维度 | 原 demo（本地 `@tool`） | 改造后（MCP） |
|---|---|---|
| 工具定义位置 | 写死在当前文件 | 独立 `mcp_server.py` |
| 工具复用 | 仅本项目 | 任意 Agent/项目即插即用 |
| 通信 | 进程内函数调用 | 通过 MCP 协议（stdio/HTTP） |
| 业务状态 | Client 持有 `AirlineAgentContext` | 状态由 Server 管（连库）或参数传入 |
| 底层机制 | Function Calling | Function Calling（经 MCP 封装） |

现成参考：`09_openai-agents/openai-agents-python/examples/mcp/`（17 个真实 MCP 落地文件）。

---

## Q16. MCP 接入的精确理解（容易踩的坑）

**A：** 正确流程：写 MCP Server（工具在里定义）→ 起服务 → Agent 用连接配置对象接入 Server → 框架运行时自动拉取工具给模型用。

**关键校准：**
- **不是只传"服务名"字符串**。`name="airline-service"` 只是给人看的标签/别名，Agent 不靠它找服务。
- **真正建立连接的是连接参数**：stdio 方式传"用 python 启动哪个文件"；远程 Server（HTTP/SSE）传 URL。
  ```python
  server = MCPServerStdio(
      name="airline-service",                     # 标签
      params={"command": "python", "args": ["mcp_server.py"]},  # 真正怎么连
  )
  await server.connect()   # 必须先 connect，框架才去拉取工具列表
  ```
- Agent 实例化时传的是**这个 server 对象**：`Agent(name="FAQ Agent", mcp_servers=[server])`。
- **工具列表是运行时动态拉取的**：Agent 并不知道 Server 上有哪几个工具，直到 `connect()` 后框架去问 Server "你暴露了啥"再动态加载。所以换 Server、加工具，Agent 代码一行都不用改——这正是 MCP 即插即用的核心价值。

---

## Q17. 除了 MCP / Function Calling / Skills / Harness / ReAct / ToC，Agent 开发还有哪些新概念要掌握？

**A：** 按"必掌握 / 进阶 / 工程化生产"三层梳理（假设你已懂 MCP、Function Calling、ReAct 基础）。

### 一、必掌握（与已学概念直接衔接）
- **Agent Loop / Orchestration（编排）**：比 ReAct 更上层的"如何串联多步骤/多 Agent"。对应 `customer_service` 的 Handoff 编排、LangGraph 的图状态机。
- **Memory（记忆）**：短期（会话内 messages）+ 长期（跨会话，存数据库/向量库）。`handoffs.py` 的 `SummarizationMiddleware` 即记忆策略，对照 `memory/` 示例。
- **Tool Use 设计原则**：描述怎么写、参数怎么约束、失败怎么重试（Function Calling 之上）。
- **Planning（规划）**：ReAct 是"走一步看一步"，Planning 是"先列计划再执行"（`plan-and-execute` 范式），Agent 从玩具到能干活的关键跃迁。
- **Multi-Agent（多智能体）**：多 Agent 协作/辩论/分工（CrewAI、AutoGen 思想）。`customer_service` 已是多 Agent 雏形。

### 二、进阶（大厂/复杂场景常考）
- **Agentic RAG**：让 Agent 自己决定"是否检索、检索哪路"，而非固定流程塞提示词（你 RAG 背景强，此为优势方向）。
- **Reflection / Self-Correction（反思与自我修正）**：跑完自己检查对错再改（本地 `reflection/`、`reflexion/`）。
- **Human-in-the-loop（人在环）**：关键步骤暂停等人确认（本地 `human_in_the_loop/`），生产必备。
- **Guardrails / Safety（护栏）**：输入输出校验、越权拦截（OpenAI Agents 内置 Guardrails）。
- **Context Engineering（上下文工程）**：2025 最热概念——动态组织/裁剪/压缩喂给模型的上下文（你的摘要机制就是其中一块）。
- **Structured Output（结构化输出）**：强制模型按 schema 返回（已下 `react-agent-structured-output.ipynb`）。
- **Reasoning Models（推理模型调用）**：何时用推理模型、如何控制思考预算。

### 三、工程化 / 生产落地
- **Observability（可观测）**：tracing、LangSmith、评估链路（`customer_service` 的 `trace()` 即此）。
- **Evaluation（评测）**：量化 Agent 好坏（轨迹评估、任务成功率）。
- **State / Checkpointer（状态持久化）**：`InMemorySaver` 内存版 → 生产换 PostgreSQL/Redis。
- **Streaming（流式输出）**：SSE/逐字返回。
- **A2A（Agent-to-Agent 协议）**：Google 提出的 Agent 间通信标准，与 MCP（Agent↔工具）互补，2025 新热点。
- **Subgraph / 嵌套编排**：复杂流程拆子图（本地 `subgraph.ipynb`）。
- **Agent Deployment（部署）**：容器化、并发、会话隔离、成本控制。

### 四、清单中几个词的正名（避免混淆）
- **Skills**：非通用 Agent 术语，是特定平台（Claude Code / CodeBuddy 等）的"可复用能力包"——把 prompt + 脚本 + 工作流打包成插件。作为使用者了解即可，非底层必学。
- **Harness**：跑 Agent 的外壳/运行环境（如某 CLI、IDE 插件），是工程概念非算法概念。
- **ToC**：待确认具体所指（可能是 Tool-Calling 笔误，即 Function Calling 同义；或某编排范式缩写）。需用户澄清原词后再补准。

### 五、学习优先级建议（基于"已会 RAG/工具/MCP 概念，缺落地"背景）
```
1. Memory（长期记忆 + 摘要）    ← 对应 handoffs.py，最短路径
2. Planning（规划范式）         ← 能力跃迁关键
3. Human-in-the-loop           ← 生产必备
4. Context Engineering         ← 2025 热点，与摘要机制相关
5. A2A + Observability + Eval  ←加分，进阶
```

---

## Q18. 之前提到的"tool_call 阶段"是什么？函数体为什么"不会被真正执行业务逻辑"？

**A：** 用 OpenAI 兼容 API 时，一次带工具的对话实际是**两次模型交互**：

```
第 1 次（模型决策）：
  你发 messages + tools 定义 → 模型返回 tool_calls: [{name:"select_skill", arguments:{skill_key:"finance_policy"}}]
  （模型此时只"说"要调哪个工具、传什么参，还没真正执行任何代码）

[你这边的代码/框架拦截]  ← 这就是 "tool_call 阶段"
  看到 tool_calls → 决定要不要真跑函数体。
  在 Skill 路由场景：框架直接在这里截获 skill_key，去加载 content 注入 prompt，
  然后假装函数返回了 "已选择：finance_policy"。

第 2 次（模型执行）：
  你把 tool 结果塞回 messages（role:"tool", content:"已选择..."）
  → 模型基于新上下文继续生成最终回答。
```

**"tool_call 阶段" = 第 1 次模型返回、还没真正执行函数之前的拦截点。**

函数体"不会被真正执行业务逻辑"是因为：框架在上面的拦截点就直接用 `skill_key` 去加载 content 注入 prompt 了，根本不会去 call 你写的那个 `return f"已选择..."`。函数体存在只是为了满足 OpenAI API 协议要求（tool 必须能返回一个结果塞回 messages）。

---

## Q19. Tool 式 Router，200 个技能的描述要全写进 tool 的 description 吗？企业会优化吗？

**A：** 用 Tool 方式做 Router，这 200 条 `(key, description)` 确实会**全部塞进 `description_override`**。这正是 Tool 式 Router 的硬伤：

- description 变几百上千字，占每轮 input tokens；
- 模型从一长串里挑，长上下文下挑选准确率下降；
- 每次对话都带这 200 条，浪费且笨重。

**企业一定会优化**，主流做法：

| 方案 | 做法 | 适用 |
|---|---|---|
| A. Tool 式（原样） | 全量 description 塞进 tool desc | 技能数 < 20 |
| B. 分类预筛 + Tool | 先按业务域分 8~10 大类，tool desc 只列大类；选中后再从该类挑 | 中等规模 |
| C. 向量检索预筛 + Tool | 用户问题先 embedding，从 200 个里召回 top-5 写进 desc | 最精准，desc 永远短 |
| D. Router Agent + RAG | 轻量 Agent 做路由，目录走检索召回 | 平台级 |

**真实企业里 C / D 最常见**——几乎没人把 200 条硬塞进工具描述。你的直觉"工具描述一般字数不多"是对的，Tool 式只适合技能数少的场景。

---

## Q20. Tool 式 vs Agent 式 Router 怎么写？instructions 是 system prompt 吗？

**A：**

### Tool 式（之前示例）
即 `@tool(name_override="select_skill", description_override=...)` + `select_skill` 函数，挂到主 Agent 的 `tools=[...]`。函数体空跑、框架在 tool_call 阶段截获 key。

### Agent 式（改造写法）
**不是"把那段 JSON 放进 instructions"**，而是单独起一个轻量 Router Agent，把目录作为上下文写进它的 `instructions`：

```python
SKILL_CATALOG = [  # 从存储加载，不是手写
    {"key": "finance_policy", "description": "咨询报销标准、财务合规、预算审批等财务制度时使用"},
    {"key": "data_security", "description": "问数据脱敏、权限、隐私合规时使用"},
    # ...200 个
]

router_instructions = (
    "你是技能路由专员。根据用户问题，从下列技能目录中选择最匹配的 skill_key。\n"
    "只返回 skill_key，不要解释。\n技能目录：\n" + format_catalog(SKILL_CATALOG) +
    "\n若都不匹配，返回 'none'。"
)

router_agent = Agent(name="SkillRouter", model="gpt-4o-mini", instructions=router_instructions)

result = await Runner.run(router_agent, user_msg)
skill_key = result.final_output.strip()        # 模型直接输出 key
if skill_key != "none":
    content = load_skill_content(skill_key)     # 加载 content
    executor = Agent(
        name="Executor",
        instructions=BASE_INSTRUCTION + "\n\n# 本次启用技能\n" + content,
    )
    await Runner.run(executor, user_msg)
```

> Agent 式同样有"200 条塞进 instructions 太长"的问题，所以生产里通常配合向量检索：用户问题 embedding → 召回 top-5 再拼进 `router_instructions`（即 Router + RAG）。

### instructions 是 system prompt 吗？
**是的。** `Agent(instructions=...)` 最终拼成对话的第一条 `{"role": "system", "content": ...}`，等价于 system prompt。区别：
- `instructions` 可以是字符串，也可以是动态函数 `instructions=callable(ctx)->str`（每次 run 现算，便于注入 skill content）；
- system prompt 是底层 API 的称呼，`instructions` 是框架封装。

---

## Q21. Skills 在自建 Agent 里怎么落地？（合并：框架封装 / Tool式 vs Agent式 / 规模化优化 / 与 MCP 关系）

> 把 Skills 相关追问合并，便于复习。核心结论：**Skills 必须自己用 Tool 或 Agent 实现，框架没现成封装，且与 MCP 互补、不互相取代。**

### 21.1 框架里有现成的 `@skill` 封装吗？
**没有。** Skills 是特定平台（Claude Code / CodeBuddy 这类 harness）的概念，不是通用 Agent 框架（OpenAI Agents SDK / LangGraph / LangChain）的一等公民。框架里有 `@tool`、`Agent`、`handoff`、`MCPServer` 这类原生封装，但**没有**把"skill（触发条件 + description + content）"直接绑模型的原生装饰器。
→ 自建 Agent 时，Skills 必须**自己用 Tool 或 Agent 方式实现**（路由 + 注入 content）。`@tool` 是框架给的；Skills 机制是平台给的、你自己复刻的。

### 21.2 两种落地方式
- **Tool 式**：`@tool(name_override="select_skill", description_override=目录)` + `select_skill` 函数挂到主 Agent 的 `tools=[...]`。函数体空跑，框架在 tool_call 阶段截获 `skill_key` 去加载 content 注入 prompt（见 Q18）。
- **Agent 式**：单独起轻量 Router Agent，把目录写进它的 `instructions`，模型直接输出 `skill_key`；Executor Agent 再 `instructions = BASE + content` 注入干活（见 Q20 代码）。

> **C 与 D 是平行路线，不是"D = C + 多一步 Agent"**：
> - C（向量检索预筛 + Tool）：用户问题 → 你的代码离线 embedding 召回 top-5 → 拼进 Tool 的 description → **主 Agent 调 `select_skill` 从 5 个里选 1 个**。
> - D（Router Agent + RAG）：用户问题 → 离线召回 top-5 → 拼进 **Router Agent 的 instructions** → **Router Agent 输出 key** → Executor 注入。
> 差别是"选择者"不同：C 选者=主 Agent；D 选者=独立 Router Agent（可用便宜小模型，路由/执行解耦、好测试）。

### 21.3 易混点：向量检索是 Tool 吗？
**不是。** 向量检索（用户问题 → embed → 查向量库 top-5）是**你代码里在 `Runner.run` 之前预先跑的离线预处理**，结果直接拼进 Tool desc / Router instructions 的文本里。只有 `select_skill` 是 Tool。若把检索也做成 Tool 让模型自己决定"先检索再选"，那是更进阶的 **Agentic RAG**，非方案 C 标准做法。

### 21.4 规模化：200 个 Skills 怎么处理？
Tool 式会把 200 条全塞进 description（占 token、长列表挑选率降），企业必优化：
- B 分类预筛（按域分大类，desc 只列大类）；
- C 向量检索召回 top-5 写进 desc（最精准，desc 永远短）；
- D Router Agent + RAG（平台级）。
**C / D 最常见**，Tool 式只适合 < 20 个技能。

### 21.5 Skills 会取代 MCP 吗？
**不会，互补不替代：**
- **MCP** = 模型**调用外部能力/数据**的协议（Agent ↔ 工具/数据源），解决"怎么执行操作、拿结果"。
- **Skills** = 给模型**注入领域知识/步骤**的文本包（不执行，只"教模型怎么做"），解决"懂不懂这个领域的规矩"。
典型配合：Skill 的 content 里指导模型"此时应调哪个 MCP 工具"。Skill 决定注入什么知识，知识里引用 MCP 工具去干活。

---

## Q22. Harness 专题（合并：定义 / 实现 / 与框架·Skills 关系 / 为什么火 / 解决什么 / interview 应答）

> 把 Harness 相关追问合并，便于复习。核心结论：**Harness 是运行 Agent 的产品化外壳，范围远大于 Agent 本身。**

### 22.1 Harness 是什么
Harness（原意"马具"）在 Agent 语境指**运行 Agent 的外壳/承载环境**——把模型、工具、记忆、上下文管理、循环调度、交互层（UI/CLI）、权限、日志等**打包在一起跑起来的那一层**。
- 类比：模型 = 引擎；Agent 逻辑（ReAct 循环、tool 定义）= 传动系统；**Harness = 整车外壳 + 驾驶舱**（用户通过它与 Agent 交互，它管会话状态、文件权限等）。
- 常见 harness：Claude Code（终端 CLI）、CodeBuddy（IDE 插件）、ChatGPT 对话界面、Dify / Coze 等平台。

### 22.2 具体如何实现
Harness 不是某个算法，而是一组工程能力的组合：
- **会话/进程管理**：维护多轮 messages、跨会话 session 状态（内存或持久化）。
- **循环调度器**：调 `Runner.run` / 手写循环，驱动"模型→工具→观察"反复跑。
- **上下文组装（Context Engineering）**：动态拼 system prompt、注入 Skills 内容、裁剪/压缩历史、插入 RAG 结果。
- **工具/权限网关**：决定 Agent 能调哪些工具、能否写文件、要不要人确认（Human-in-the-loop）。
- **交互层（UI/CLI）**：接收输入、流式展示输出、渲染工具调用过程。
- **可观测/日志**：trace、记录 token/耗时/工具轨迹。
- **扩展挂载点**：Skills 目录、MCP 配置、插件系统，让第三方能力即插即用。
本质就是**一个普通应用程序**（Python/Node 服务 + 前端）把这些模块串起来。

### 22.3 与框架、Skills 的关系
- **与框架**：通用 Agent 框架（OpenAI Agents SDK / LangGraph）只提供 Agent 逻辑层（Agent/Runner/tools），**不提供交互外壳/UI/会话产品化**——这部分框架不管。所以"产品级 harness"不在框架里。但框架提供了 harness 需要的"零件"；AutoGen / CrewAI 比纯 SDK 更靠近 harness，而 Claude Code / CodeBuddy / Dify 这种带 UI、权限、Skills、MCP 的才是完整 harness。
- **与 Skills**：Skills 是 harness 的一个**挂载能力**（平台通过目录加载 Skill：触发条件+描述+内容，运行时路由并注入 prompt）。通用框架**没有**原生 `@skill` 封装（同 Q21），要自己用 Tool / Router Agent 复刻；harness 产品才把 Skills 作为一等能力内置。
- **澄清"和 Skills 一样框架没封装吗"**：不同。Skills 是框架确实缺（需自己复刻）；Harness 是**本来就不属框架职责范围**——框架给引擎和传动，harness 是别人盖的整车。

### 22.4 CLI 与终端的区别（易混）
- **终端（Terminal）**：软件窗口/程序，用来输入命令、看输出（如 PowerShell、macOS Terminal），本身不执行逻辑，只把按键传给 shell。
- **CLI（Command-Line Interface）**：一种**交互方式/程序形态**——通过文字命令操作软件，相对 GUI。如 `git`、`npm`、`claude` 都是 CLI 程序。
- 关系：**CLI 程序通常在终端里运行**。终端是容器，CLI 是里面跑的应用。Claude Code 是 CLI 程序，通过终端启动使用。不能说"终端就是 CLI"。

### 22.5 范围层级（Harness ⊃ Agent）
```
Harness（整车产品层）
 ├─ UI / 交互层（终端 CLI、IDE 插件、Web 界面）
 ├─ 会话/状态管理（多轮、跨会话记忆）
 ├─ 权限/安全网关（写文件？人确认？）
 ├─ Skills 挂载（目录加载、路由、注入）
 ├─ MCP 接入（连外部工具/数据源）
 ├─ 上下文工程（动态拼 prompt、摘要、RAG）
 ├─ 可观测/日志（trace、token 统计）
 └─ Agent（核心引擎层，只占一块：模型调用 + ReAct 循环 + Tools/Handoff）
```
→ **Agent 回答"怎么思考和执行"，Harness 回答"怎么让它作为一个产品被安全、好用地用起来"。** 一个只跑 `Runner.run` 的脚本是最小 harness；Claude Code 这类是成熟 harness。

### 22.6 为什么 2025 年 Harness 这么火
- **纯框架跑不出能用的东西**：2023-24 卷框架，但 demo 多是脚本里跑通循环，离真正给用户用差很远（无记忆持久化、无权限、交互差、长任务崩）。瓶颈不在"能不能调工具"，在"怎么稳定产品化"。
- **模型能力拉平**：推理模型、长上下文、原生工具调用普及后，框架层差异化被削弱，差异化转移到外围工程（上下文、记忆、权限、Skills、MCP）——正是 harness 的事。
- **Agent 产品成主战场**：Claude Code / Cursor / Manus / Devin 等直接面向用户的 Agent 产品爆发，核心竞争力是 harness（交互、权限、长任务稳定性），不是底层循环。
- 本质：行业从"能不能做 Agent"进阶到"怎么让 Agent 真正可用、可商用"，最难最值钱的就是 harness 这段整合。它无神秘算法，价值在**工程整合与产品化**。

### 22.7 Harness 解决了之前 Agent 的什么问题
| 纯 Agent 脚本痛点 | Harness 解法 |
|---|---|
| 无持久记忆（重启清零、长任务爆上下文） | 会话管理 + 摘要压缩（Context Engineering） |
| 无权限边界（乱写文件、乱调工具） | 权限网关 + Human-in-the-loop |
| 交互差（只能 print） | UI/CLI 层，流式输出、可中断 |
| 能力零散（工具写死、换场景重写） | Skills 挂载 + MCP 接入，即插即用 |
| 不可观测（出错不知哪步挂） | trace / 日志 / token 统计 |
| 上下文弱（长对话超 token） | 动态拼 prompt、检索注入、历史裁剪 |
| 难给普通用户用 | 产品化外壳降门槛 |
→ **把实验室原型补齐成生产系统，解决可靠性、安全性、可用性、可维护性。**

### 22.8 interview 应答要点（position：Agent 开发 / 大模型应用开发）
- **Q：什么是 Harness，和 Agent/框架关系？** 答：Harness 是运行 Agent 的产品化外壳，包含 UI/CLI、会话状态、权限网关、Skills 挂载、MCP 接入、可观测。Agent 是核心引擎，框架给引擎零件，Harness 是整车。举 Claude Code / CodeBuddy 例子。
- **Q：为什么 2025 火？** 答：模型能力拉平→差异化转向外围工程；Agent 产品成主战场；火的不是算法是工程整合。
- **Q：解决什么痛点？** 答：记忆/权限/交互/能力复用/可观测/上下文/降门槛 七点，本质补齐生产四性（可靠、安全、可用、可维护）。
- **Q：从零搭 harness？** 答：分层——底座 Agent 框架+工具注册 → 会话管理+摘要 → 权限网关+人在环 → Skills+MCP → 交互层 → 可观测。先做最小可用再补外围；Skill 规模化用向量检索召回 top-k。
- **加分项**：区分算法层（ReAct/Planning）vs 工程层（Harness）；多举产品名；提 Context Engineering / Human-in-the-loop / Observability / MCP；收尾"框架给零件，Harness 盖整车"。
