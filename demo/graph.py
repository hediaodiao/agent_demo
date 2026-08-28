"""LangGraph 编排层（方案 B：纯 StateGraph 手写，便于学习）。

本文件是整个 demo 的核心，逐节点对应我们之前讨论的生产要点。
读这个文件 = 复习：Supervisor 多 Agent / 上下文压缩时机 / 工具返回压缩 /
无状态化 / 步数硬护栏 / 降级。

为什么用方案 B 不是 create_supervisor：
- inter要考底层机制（节点/边/状态/条件路由），手写才学得透。
- 生产复杂流（加人工审核节点、多轮评审）也只有手写 StateGraph 能表达。
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
你是客服调度台。根据用户最新问题，决定交给哪个专家处理，只输出一个词：
- 订单/物流/发货/查订单 -> order
- 退货政策/发票/会员/知识库 -> qa
- 投诉/建工单/提交工单 -> ticket
- 闲聊/问候 -> __end__
""")


def supervisor(state: AgentState) -> dict:
    # 【第2块压缩触发点】每次回到调度台时，先对累积历史做滚动压缩
    # （写回上下文后、下次决策前检查 token —— 对应我们讨论的压缩时机）
    # 注意：这里 compress_history 返回压缩后的「完整消息列表」，
    # 我们用它替换 state 的 messages。由于 State 用 Annotated 累加 reducer，
    # 直接返回完整列表会造成重复追加；因此生产里更稳妥是在子 Agent 内部用
    # memory 或在 chat() 入口压缩。为演示「压缩节点」概念，这里仅更新 next，
    # 真正压缩放在 compress_node 里对「刚产生的子 Agent 结果」做轻量处理。
    # 取最后一条 human 消息做意图分类
    last_human = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_human = m.content
            break
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
    else:
        next_agent = "__end__"
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
# 5. 压缩节点（手写：写回上下文后调用 compress_history）
#    对应讨论的「第2块压缩」：触发时机 = 每次节点处理后，检查 token 压最早步骤。
# ============================================================
def compress_node(state: AgentState) -> dict:
    """【演示节点】压缩时机示意。

    说明：LangGraph 的 State.messages 用 Annotated 累加 reducer，节点直接返回
    完整列表会与累加冲突（重复追加）。生产里压缩更稳妥的做法是：
      (a) 在子 Agent 内部配 ConversationSummaryBufferMemory（自动压历史）；或
      (b) 在 chat() 入口对传入的 session 历史先做 compress_history 再传图。
    本节点保留为「教学示意」：仅打印当前 token 占用，不做破坏性替换。
    真正生效的压缩见 tools.py（工具返回压缩）与 memory.compress_history()。
    """
    total = sum(len(m.content) for m in state["messages"] if hasattr(m, "content"))
    # 仅做触发判断示意（不修改 state，避免 reducer 冲突）
    trigger = int(Settings.MAX_TOKEN_LIMIT * Settings.SUMMARY_TRIGGER_RATIO)
    if total > trigger:
        # 生产在这里应把历史摘要后回写；演示版略过写入以保图正确运行
        pass
    return {}  # 空返回 = 不改 state


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

# 边
builder.add_edge(START, "supervisor")                 # 入口 -> 调度台
builder.add_conditional_edges(                        # 调度台 -> 按 next 分流
    "supervisor",
    route,
    {"order": "order", "qa": "qa", "ticket": "ticket", "__end__": END},
)
# 子 Agent 干完 -> 回到调度台（可多轮：用户追问会再次分流）
builder.add_edge("order", "compress")
builder.add_edge("qa", "compress")
builder.add_edge("ticket", "compress")
builder.add_edge("compress", "supervisor")            # 压缩后回到调度台，形成循环

# 编译（recursion_limit = 硬护栏：步数封顶，防无限循环）
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
