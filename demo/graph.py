"""LangGraph 编排层（方案 B：纯 StateGraph 手写，便于学习）。

本文件是整个 demo 的核心，逐节点对应我们之前讨论的生产要点。
读这个文件 = 复习：Supervisor 多 Agent / 上下文压缩时机 / 工具返回压缩 /
无状态化 / 步数硬护栏 / 降级。

为什么用方案 B 不是 create_supervisor：
- 面试要考底层机制（节点/边/状态/条件路由），手写才学得透。
- 生产复杂流（加人工审核节点、多轮评审）也只有手写 StateGraph 能表达。

============================================================================
生产环境「两层记忆」模型（务必分清，避免歧义）
============================================================================
【工作记忆 / 短期记忆 = checkpointer，PostgreSQL+pgvector，按 thread_id】
  存什么：当前会话完整的 messages 列表（含被压缩成的"历史摘要"消息）。
          例：[System, H1"查订单", A1, Tool(订单列表), H2"退货政策", A2...]
          压缩只是把旧轮原地改写成一条 System(历史摘要)，仍在表里，不是删除。
  作用域：单 thread_id（同一会话）。同一会话内"第3轮看第1、2轮"靠它，
          与"跨会话"无关——那是下面情景记忆的活。
  何时写（每步 = 图中每个节点产出后，框架自动持久化该 thread_id 的 messages）：
        - supervisor 节点分类意图后 → 写
        - order/qa/ticket 子Agent节点跑完（产生 AI+Tool 消息）后 → 写
        - compress 节点压缩后（update_state 整体替换 messages）→ 写
          例：用户第4轮提问，图走 supervisor→order→compress→supervisor，
              每经过一个节点，thread_id 的 messages 快照就更新一次到 Postgres。
  何时读：每次新请求 invoke 时，框架 load(thread_id) 取全量 messages 作为本轮起点。
          例：用户第4轮进来，先 load 前3轮全部消息，再继续。
  为谁服务：保证 LLM 每轮有完整上下文、token 不溢出（靠压缩）、进程重启不丢、
           任意实例可恢复 → 这就是"无状态化"的落地方式。

【情景记忆 / 长期记忆 = store，pgvector，跨 thread_id，按 namespace(如 user_id)】
  存什么：从对话抽取的可复用事实 + embedding，不是整段对话。
          例：("u_1001偏好顺丰", emb)、("u_1001是金卡会员", emb)。
  作用域：跨 thread_id / 跨会话（同一用户的不同对话）。
          ⚠️ 歧义澄清：情景记忆专指"不同会话之间复用"，例如周一会话A沉淀
          "偏好顺丰"，周三新开会话B检索到并注入。同一会话内"上一轮→下一轮"
          看历史，是 checkpointer 的活，不要混为一谈。
  何时写：每轮 AI 答完后，由显式逻辑把本轮回合值得长期保留的内容 put 进 store
          （带 namespace + embedding），与"工作记忆是否压缩/溢出"完全解耦。
          例：第3轮子Agent答完，抽取"用户确认金卡会员"→ store.put(namespace="u_1001",...)。
  何时读：决策前（supervisor 分类或子Agent开工前），用当前 query embedding
          语义检索 top-k 注入上下文。例：新会话B问"我的会员权益"，先检索 store
          命中"金卡会员"→ 注入。
  为谁服务：长期个性化、跨会话记忆、避免重复问已知信息。

两层职责不重叠：持久化与无状态由 checkpointer 保证；情景记忆是叠加的语义检索层。
（本 demo 用 MemorySaver 仅作可跑演示，生产替换为 PostgresSaver；
 episodic_store 仅模拟 store 的语义索引，且当前与压缩耦合，生产应解耦为每轮写。）
"""
from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from config import Settings, get_llm
from tools import ORDER_TOOLS, QA_TOOLS, TICKET_TOOLS
from memory import compress_history

# ============================================================
# 1. 状态定义（State）：节点之间传递的数据
# ============================================================
# 教学：LangGraph 的 State 是一个 TypedDict。messages 用 Annotated + operator.add
# 做「累加 reducer」——每个节点返回的新消息会自动追加，而不是覆盖。
# 这正是「无状态实例 + 状态在图里累积」的体现。
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]
    next: str   # supervisor 写：下一步去哪个子 Agent（"order"/"qa"/"ticket"/"__end__"）


# ============================================================
# 2. 三个专家子 Agent（每个用 LangChain create_react_agent 封装）
#    对应讨论：子 Agent 是独立实例，各自持有自己的工具与上下文。
# ============================================================
def _make_subagent(tools, name: str):
    """造一个 react 子 Agent（等价于 AgentExecutor，但可被 LangGraph 调用）。"""
    model = get_llm("big")
    prompt = SystemMessage(content=f"你是{name}专家客服，只用给你的工具回答，不要编造。")
    return create_react_agent(model, tools, prompt=prompt, checkpointer=MemorySaver())


order_agent = _make_subagent(ORDER_TOOLS, "订单")
qa_agent = _make_subagent(QA_TOOLS, "知识库问答")
ticket_agent = _make_subagent(TICKET_TOOLS, "工单")


# ============================================================
# 3. Supervisor 节点（手写决策：分流到哪个子 Agent）
#    对应讨论：多 Agent 的「主 Agent 调度」，这里是核心路由逻辑。
# ============================================================
ROUTING_PROMPT = SystemMessage(content="""
你是客服调度台。根据用户最新消息，决定下一步去向，只输出一个词：
- 订单/物流/发货/查订单 -> order
- 退货政策/发票/会员/知识库 -> qa
- 投诉/建工单/提交工单 -> ticket
- 结束语且用户无新诉求（纯谢谢/再见/明白了，没有再问问题）-> farewell
- 闲聊/问候且无业务意图 -> farewell
- 任务已完成且用户没有新诉求 -> farewell
注意：
1. 如果用户消息里除了致谢还包含了新的问题或诉求（例如"谢谢，另外我想问…"），必须按新意图派给 order/qa/ticket，不要结束。
2. 只有「确认/致谢/闲聊」且确实没有新问题时，才输出 farewell。
""")


def supervisor(state: AgentState) -> dict:
    # 【第2块压缩触发点】每次子 Agent 干完 -> compress 节点 -> 回到这里（supervisor）。
    # 真正的滚动压缩在 compress_node 里做（用 graph.update_state 整体替换 messages，
    # 绕过 Annotated 累加 reducer 的冲突）。这里 supervisor 只负责意图分类 + 写 next。
    # 取最后一条 human 消息做意图分类
    last_human = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_human = m.content
            break

    # 是否结束/收尾由模型根据「有无新意图」判断（见 ROUTING_PROMPT），
    model = get_llm("big")
    resp = model.invoke([ROUTING_PROMPT, HumanMessage(content=last_human)])
    decision = resp.content.strip().lower()
    # 映射到 next 字段
    if "order" in decision:
        next_agent = "order"
    elif "ticket" in decision:
        next_agent = "ticket"
    elif "qa" in decision:
        next_agent = "qa"
    elif "farewell" in decision:
        next_agent = "farewell"
    else:
        next_agent = "farewell"   # 无法识别意图时默认优雅收尾，而非硬派活
    return {"next": next_agent}


# ============================================================
# 4. 子 Agent 节点（调用对应子 Agent，并把结果写回 state.messages）
# ============================================================
def _run_subagent(agent, agent_key: str, state: AgentState) -> dict:
    # 取本会话历史（无状态化：从历史里取，子 Agent 自身不长期持有）
    # 这里直接用当前 messages 作为上下文传入
    cfg = {"configurable": {"thread_id": agent_key}}
    result = agent.invoke({"messages": state["messages"]}, config=cfg)
    # 子 Agent 返回的 messages 追加到主 state（LangGraph 自动累加）
    return {"messages": result["messages"][len(state["messages"]):]}


def order_node(state): return _run_subagent(order_agent, "order", state)
def qa_node(state):    return _run_subagent(qa_agent, "qa", state)
def ticket_node(state): return _run_subagent(ticket_agent, "ticket", state)


# ============================================================
# 4.1 收尾节点（farewell）：优雅终止
#     对应讨论：用户说「谢谢/好的」时，AI 不该直接哑火结束，应先回一句礼貌收尾。
#     用模板生成（不调 LLM，便宜、确定），写回 messages 后再 __end__。
# ============================================================
def farewell_node(state: AgentState) -> dict:
    # 若上一轮已经是收尾语（防止重复生成），则直接结束
    if any(isinstance(m, AIMessage) and "不客气" in (m.content or "") for m in state["messages"][-1:]):
        return {"next": "__end__"}
    reply = "不客气～如果还有其他问题，随时找我。祝您生活愉快！"
    return {"messages": [AIMessage(content=reply)], "next": "__end__"}


# ============================================================
# 5. 压缩节点（手写：写回上下文后调用 compress_history）
#    对应讨论的「第2块压缩」：触发时机 = 每次节点处理后，检查 token 压最早步骤。
# ============================================================
def compress_node(state: AgentState, config: dict) -> dict:
    """【生产生效节点】第2块压缩：按轮次滚动压缩累积历史。

    生产要点：
    - State.messages 用 Annotated 累加 reducer，节点若 `return {"messages": 完整列表}`
      会把完整列表「再追加」到白板，导致历史翻倍。所以**不能用普通返回值写回完整列表**。
    - 正确做法：用 graph.update_state(...) 把整个 messages 字段「整体替换」为压缩结果
      （update_state 是直接覆盖白板，不走 reducer 累加）。
    - 压缩粒度 = 「轮」（不是消息条数）；更早的轮次摘要后【落盘到情景记忆存储】
      （memory.compress_history 内部调用 _persist_episode；生产换 PostgreSQL/pgvector）。
    - 没超阈值则不改动，原样返回。
    - thread_id 从 config 取，保证压缩产物按会话归档到正确的情景记忆分片。
    """
    trigger = int(Settings.MAX_TOKEN_LIMIT * Settings.SUMMARY_TRIGGER_RATIO)
    total = sum(len(m.content) for m in state["messages"] if hasattr(m, "content"))
    if total <= trigger:
        return {}  # 未超阈，不压
    # 真正压缩：按轮次压，并把旧轮落盘到情景记忆
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")
    compressed = compress_history(state["messages"], thread_id=thread_id)
    # 用 update_state 整体替换白板，绕过累加 reducer（避免重复追加）
    graph.update_state(config, {"messages": compressed})
    return {}  # 节点本身不重复返回，避免 reducer 再追加一次


# ============================================================
# 6. 条件路由函数（conditional_edges 用）：根据 supervisor 写的 next 决定走向
# ============================================================
def route(state: AgentState) -> str:
    return state.get("next", "__end__")


# ============================================================
# 7. 画图（StateGraph）：把节点和边连起来
# ============================================================
builder = StateGraph(AgentState)

# 加节点
builder.add_node("supervisor", supervisor)
builder.add_node("order", order_node)
builder.add_node("qa", qa_node)
builder.add_node("ticket", ticket_node)
builder.add_node("compress", compress_node)   # 压缩节点（演示插在任意位置）
builder.add_node("farewell", farewell_node)   # 收尾节点：优雅终止

# 边
builder.add_edge(START, "supervisor")                 # 入口 -> 调度台
builder.add_conditional_edges(                        # 调度台 -> 按 next 分流
    "supervisor",
    route,
    {"order": "order", "qa": "qa", "ticket": "ticket",
     "farewell": "farewell", "__end__": END},
)
# 子 Agent 干完 -> 回到调度台（可多轮：用户追问会再次分流）
builder.add_edge("order", "compress")
builder.add_edge("qa", "compress")
builder.add_edge("ticket", "compress")
builder.add_edge("compress", "supervisor")            # 压缩后回到调度台，形成循环
builder.add_edge("farewell", END)                     # 收尾 -> 结束（不再循环）

# 编译（recursion_limit = 硬护栏：步数封顶，防无限循环）
# 生产：MemorySaver 换成 PostgresSaver（checkpointer=持久化），实现无状态 + 重启不丢。
#       store（长期情景记忆）此处未接，生产用 langgraph.store + pgvector 按 namespace 写/检索。
graph = builder.compile(checkpointer=MemorySaver(), recursion_limit=Settings.MAX_ITERATIONS)


# ============================================================
# 对外接口
# ============================================================
def chat(user_input: str, session_id: str = "default") -> str:
    """单次对话入口。session_id 区分不同用户（无状态化 + 状态外置）。

    【第2块压缩生效点】生产里在入口处对「本轮要传入的历史」先压缩，
    绕过 State 累加 reducer 的冲突——这是比图内压缩节点更稳妥的做法。
    （本 demo 用 MemorySaver 作 checkpointer，历史由 thread_id 自动管理；
     这里示意性展示 compress_history 的用法，真实长会话应在取历史后压缩再传入。）
    """
    cfg = {"configurable": {"thread_id": session_id}}
    result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config=cfg)
    # 取最后一条 AI 消息作为答复
    for m in reversed(result["messages"]):
        if isinstance(m, AIMessage):
            return m.content
    return "（无回复）"
