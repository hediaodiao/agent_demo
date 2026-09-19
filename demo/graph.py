"""LangGraph 编排层（方案 B：纯 StateGraph 手写，便于学习）。

本文件是整个 demo 的核心，逐节点对应我们之前讨论的生产要点。
读这个文件 = 复习：Supervisor 多 Agent / 上下文压缩时机 / 工具返回压缩 /
无状态化 / 步数硬护栏 / 降级。

为什么用方案 B 不是 create_supervisor：
- 要考底层机制（节点/边/状态/条件路由），手写才学得透。
- 生产复杂流（加人工审核节点、多轮评审）也只有手写 StateGraph 能表达。

============================================================================
生产环境「三层记忆」模型（务必分清，避免歧义）
============================================================================
【工作记忆 Working Memory = 模型上下文窗口本身，易失，每轮现拼现用，不被"存储"】
  是什么：每轮 invoke 时拼进模型的 messages（窗口内容），推理完即弃。工作记忆
          本身不落库——使「续聊」成立的短期记忆由 checkpointer 落库（见下情景记忆段
          的 checkpointer 说明），情景记忆(episodic_memory) 另行蒸馏落库。
  内容组成：[短期记忆(checkpointer) load 回来的该会话历史（已压成 1 条历史摘要 + 最近 k 轮原文）]
            + [本轮 HumanMessage] + [本轮子 Agent 回传的 AIMessage 结论]。
      ⚠️「1 条历史摘要 + 最近 k 轮原文」是**工作记忆（喂模型的窗口）**的产物，
      不是数据库的存储形态——库里(checkpoints)存的是全量 state 快照（压缩只发生在
      工作记忆层，不裁剪库）；情景记忆(episodic_memory) 另存蒸馏片段，**不存全量原文**。
  不进窗口：子 Agent 的 tool_calls / ToolMessage 原始循环（工具结果只留结论，
            见 _run_subagent；这是标准「Agent 循环超窗」第①块）。
  何时更新：每轮入口 load 短期记忆(checkpointer) 作基底 → 各节点产出追加 → compress_node
            超阈（token > 窗口 80%）把最早轮滚成摘要，保留 [摘要]+[最近k轮]。
  为谁服务：保证 LLM 每轮看到完整且 token 不溢出的上下文。
  ⚠️ 压缩只裁剪「喂给模型的内容」，**不裁剪库里存的全量**（压缩≠存储删除）。

【情景记忆 Episodic Memory = 蒸馏后的结构化情景片段，带时间戳，默认不跨会话】
  C13 改造：从「逐轮全量原文归档」改为「每轮异步蒸馏的结构化片段」（对齐 Mem0/HWC
          主流生产做法；本场景为企业内部助手、无审计，不需全量原文）。
  存什么：**每轮 0~N 条**由 `answer_node` 后台异步蒸馏产出的结构化 episode——
          `signal_type`(事件/偏好/失败/承诺/教训/异常) + `content`(发生了什么+结果)
          + `entities`(可检索标签) + `importance`(1-5)，带 ts 与 embedding。
          全量原文**不再落库**（豆包/ChatGPT 的"逐字回看"在本场景不需要）。
  不存什么：子 Agent 的 tool_calls / ToolMessage 原始循环；子 Agent 的**中间结论**
            （多专家时它们是中间产物，整合后的最终回复才参与蒸馏）。
  介质：`episodic_memory` 表（PostgreSQL + pgvector），一行一条 episode。
  作用域：单 thread_id（同一会话）。⚠️ 默认**不跨会话**——这是豆包等主流形态；
          跨会话属于下面语义记忆（opt-in）的活，不要混为一谈。
  何时写/读：**每轮** answer_node 产出最终回复后，后台线程蒸馏写库（不阻塞回复）；
          读取由主 Agent 的 `search_memory` 工具在会话内**按需 agentic 触发**（实体/语义），不每轮无脑查。
          与压缩完全解耦。
  ⚠️ 关于 checkpointer：LangGraph 的 checkpointer（`checkpoints` 表，每个节点存
     一份全量 state 快照）是**框架机制**，用于断点恢复 / 无状态化，
     **不是"另一套情景记忆"，不属于记忆分层**。两者是独立的表、独立的用途：
       - checkpoints  → 给框架恢复/续聊用（每节点一份全量拷贝，生产会定期裁剪）；
       - episodic_memory → 给会话内按需召回注入 / 按时间语义查询用（蒸馏片段，保留期内全留）。
       一次 invoke（用户说一句话）会跑很多个节点，每个节点 = 一个图步 = 一条 checkpoint 行；
       一个用户轮 = 多个图步 = 多条 checkpoint 行； 压缩落点：压缩结果写在"其中一条（update_state 那次）"里，
       最新一条即压缩态； 读取：只认最新一条；存储：全部累积、无自动删 → 需生产自建清理
  （以上 checkpoints 何时读：每次新请求 invoke 时，框架 load(thread_id) 取回该会话 state 作为本轮起点；
   checkpoints 为谁服务：下一轮接着聊、重启不丢、任意实例可恢复（无状态化的落地方式）。）

【语义记忆 Semantic Memory = opt-in 跨会话用户画像，本 demo 不实现】
  存什么（生产）：用户级事实/偏好（"u_1001 金卡会员""偏好顺丰"），向量库，
                  按 user_id namespace，无时间戳。
  状态：标准明确其为**可选(opt-in)、非默认**（Cursor project rules / ChatGPT Memory
        形态；豆包默认也不跨会话）。本 demo 不做，也不写任何跨会话 store
        ——避免把"非默认"当"默认"。
  何时写（若生产接入）：规则实时提取(60%) + 关键事件触发(5%) + 每日批量(35%)，
                        绝不每轮调大模型。

三层职责：工作记忆=窗口(易失，checkpointer 供续聊)，情景记忆=蒸馏结构化片段落库(单会话、按需召回)，
          语义记忆=跨会话画像(opt-in，未实现)。不使用内存版介质。
"""
from typing import Annotated, TypedDict, Sequence, List
# 注：主上下文不再处理 ToolMessage（子 Agent 只回结论，见 _run_subagent），故不导入
import threading
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool, InjectedState
from db import (get_checkpointer, insert_episodic_memories,
              search_episodic_by_vector, search_episodic_by_entities)

from config import Settings, get_llm
from vector_store import Embedder
from tools import ORDER_TOOLS, QA_TOOLS, TICKET_TOOLS
from memory import compress_history, count_messages_tokens

# ============================================================
# 1. 状态定义（State）：节点之间传递的数据
# ============================================================
# 教学：LangGraph 的 State 是一个 TypedDict。messages 用 Annotated + operator.add
# 做「累加 reducer」——每个节点返回的新消息会自动追加，而不是覆盖。
# 这正是「无状态实例 + 状态在图里累积」的体现。
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]
    summary: str   # 【D系列】滚动摘要（覆盖式字段，非 reducer）：压缩节点写入，入口压缩时传给 compress_history；不再靠第0条 SystemMessage 伪装
    thread_id: str  # 会话主键（= thread_id）：供主 Agent 的记忆检索工具按会话隔离读取（十五 #34）
    delegated_request: str  # 十六 #41 补全：主 Agent 分解后委派给子 Agent 的具体子任务（handoff 工具经 Command.update 写入，_run_subagent 读取作为子 Agent 上下文）


# 主 Agent 子图的状态：消息用标准 add_messages reducer 累加，并携带 thread_id。
# thread_id 不交给 LLM，仅用于 LangGraph 通过 InjectedState 注入 search_memory 工具，
# 实现「按会话隔离检索情景记忆」而不污染模型可见入参（十五 #34）。
class SupervisorState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    thread_id: str


# ============================================================
# 2. 三个专家子 Agent（每个用 LangChain create_react_agent 封装）
#    对应讨论：子 Agent 是独立实例，各自持有自己的工具与上下文。
# ============================================================
def _make_subagent(tools, name: str):
    """造一个 react 子 Agent（等价于 AgentExecutor，但可被 LangGraph 调用）。

    ⚠️ 生产要点：子 Agent **无状态**（不传 checkpointer）。
    - 每轮都由主上下文重新 invoke，本身不需要跨调用记忆；
    - 若给它传固定 thread_id 的 checkpointer（旧实现是 thread_id="order"），
      会造成跨会话串味：A 用户的上下文被 B 用户的会话读到。
    """
    model = get_llm("big")
    prompt = SystemMessage(content=f"你是{name}专家客服，只用给你的工具回答，不要编造。")
    return create_react_agent(model, tools, prompt=prompt)


order_agent = _make_subagent(ORDER_TOOLS, "订单")
qa_agent = _make_subagent(QA_TOOLS, "知识库问答")
ticket_agent = _make_subagent(TICKET_TOOLS, "工单")


# ============================================================
# 3. Supervisor = 真正的主 Agent（create_react_agent + 工具）
#    十五 #31/#35：supervisor 不再是纯分类器，而是会调工具的「主 Agent」；
#    通过 handoff 工具（transfer_to_*）委派子 Agent，通过 search_memory 工具
#    按需召回情景记忆——派单与记忆召回都由 LLM 自决（agentic），而非写死枚举。
# ============================================================
def _last_human_index(messages) -> int:
    """返回最后一条 HumanMessage 的下标；没有则返回 -1。"""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return i
    return -1


def _conclusions_since_last_human(messages) -> list:
    """取「最后一条 HumanMessage 之后」、由子 Agent 写回的结论（带 metadata 标记）。

    只收带 type=subagent_conclusion 的结论，排除主 Agent 自身的推理/工具调用噪声。
    单专家时它本身就是最终回复（直接透传）；多专家时由 answer_node 整合成一条。
    """
    idx = _last_human_index(messages)
    tail = messages[idx + 1:] if idx >= 0 else messages
    return [m for m in tail
            if isinstance(m, AIMessage) and (m.content or "")
            and (m.metadata or {}).get("type") == "subagent_conclusion"]


# ---- 十五 #33：实体抽取从正则换 LLM（与蒸馏侧实体口径一致）----
class EntityExtract(TypedDict):
    entities: dict  # 如 {"order_id":"A","ticket_id":"5521"}，无则 {}


def extract_entities_llm(text: str) -> dict:
    """用大模型从用户语句抽取可检索实体（取代原正则 _extract_entities_from_text）。"""
    model = get_llm("small").with_structured_output(EntityExtract)
    try:
        res = model.invoke([
            SystemMessage(content="从用户语句中抽取订单号/工单号等可检索实体，"
                                  "返回 entities dict；没有则返回 {}。只输出结构化结果。"),
            HumanMessage(content=text),
        ])
        return (res or {}).get("entities") or {}
    except Exception:
        return {}


# ---- 十五 #34：记忆检索工具（agentic 召回，取代 recall_node 固定闸门）----
def _search_memory_impl(thread_id: str, query: str) -> str:
    """实体（LLM 抽取）+ 语义（向量）混合召回；由主 Agent 决定何时调用。"""
    emb = Embedder()
    # ① 实体召回（LLM 抽取实体，精确匹配，直接返回不卡阈值）
    ents = extract_entities_llm(query)
    hits = search_episodic_by_entities(thread_id, ents) if ents else []
    if hits:
        return _build_recall_message(hits).content
    # ② 语义召回（向量相似度，需超阈值）
    try:
        qv = emb.embed(query)
        hits = search_episodic_by_vector(thread_id, qv)
    except Exception:
        hits = []
    if hits and (hits[0].get("similarity") or 0) >= Settings.SEMANTIC_TRIGGER_THRESHOLD:
        return _build_recall_message(hits).content
    return "（无相关历史记忆）"


# search_memory 作为工具被主 Agent 调用；thread_id 由 LangGraph 通过 InjectedState
# 从 SupervisorState 注入，LLM 不可见该参数（只暴露 query）。
@tool
def search_memory(query: str, thread_id: Annotated[str, InjectedState("thread_id")]) -> str:
    """检索当前会话的历史情景记忆，找回被压缩掉的精确细节（订单号/工单号/过往承诺等）。
    当用户问题涉及"之前/上次/刚才/那个"或需要历史上下文时调用。"""
    return _search_memory_impl(thread_id, query)


# ---- 十五 #35：handoff 工具（把子 Agent 暴露为主 Agent 可调用工具）----
def _make_handoff_tool(node: str, label: str, scenarios: str):
    """生成一个 handoff 工具：被调用即把控制权转交到对应子 Agent 节点（跨图 Command）。

    十六 #41 补全「主 Agent 委派指令」：工具新增 request 参数，由 supervisor LLM 填入
    本次分解出的具体子任务（含必要实体），经 Command.update 写入父图状态 delegated_request，
    供 _run_subagent 作为子 Agent 的独立上下文——而非退化成「最新一条用户来信」。
    scenarios 为该专家负责的场景说明，写入工具 description，由 LLM 按需选择
    （替代原 SUPERVISOR_PROMPT 里的路由枚举，避免 prompt 随 skill 增多而膨胀）。
    """
    @tool(f"transfer_to_{node}",
          description=f"将【一个具体子任务】委派给{label}专家子 Agent 处理。"
                      f"{label}专家负责的场景：{scenarios}。"
                      f"request 必须写成分解后的明确指令（含订单号/工单号等必要实体），"
                      f"不要照抄用户原话。")
    def _handoff(request: str) -> Command:
        return Command(goto=node, graph=Command.PARENT,
                       update={"delegated_request": request})
    return _handoff


transfer_to_order = _make_handoff_tool("order", "订单",
                                       "订单查询、物流/发货状态、配送改派")
transfer_to_qa = _make_handoff_tool("qa", "知识库问答",
                                    "退货政策、发票、会员权益、产品知识库问答")
transfer_to_ticket = _make_handoff_tool("ticket", "工单",
                                        "投诉受理、建工单、售后问题升级处理")


@tool
def finish() -> Command:
    """当已收集到足够信息可以回答用户，或用户仅致谢/闲聊无业务意图时调用，进入最终回复整合。"""
    return Command(goto="answer", graph=Command.PARENT)


SUPERVISOR_PROMPT = SystemMessage(content="""
你是客服总台（主 Agent），负责协调多个专家子 Agent 并给出最终回复。
- 若用户问题需要历史上下文（涉及"之前/上次/刚才/那个"，或需要过往订单/工单信息），
  先调用 search_memory 检索相关记忆。
- 需要某位专家处理时，调用对应的 transfer_to_* 工具委派（各工具的 description 已说明其负责场景，
  按用户诉求匹配对应专家即可，无需记忆固定映射）。
- 调用 transfer_to_* 时，request 参数务必写成【分解后的具体子任务】——明确要查什么、
  并带上订单号/工单号等必要实体；不要整句照抄用户原话。同一用户诉求含多个子任务时，
  分别委派、各传各自的具体指令（例如"查订单A物流"与"查工单B进度"各传各的）。
- 当你已获得足够信息（或无需专家、纯致谢闲聊）可以回答用户时，调用 finish 工具结束本轮。
- 不要自己编造订单/工单数据，数据交给对应专家的工具去查。
- 同一诉求不要重复委派给已处理过的专家；确无新诉求就调用 finish。
""")


# 主 Agent：持 handoff 工具 + 记忆检索工具，派单与记忆召回皆由 LLM 自决（agentic）。
# 用 SupervisorState 作 state_schema，使 thread_id 进入子图状态，供 InjectedState 注入。
supervisor_agent = create_react_agent(
    get_llm("big"),
    [transfer_to_order, transfer_to_qa, transfer_to_ticket, search_memory, finish],
    prompt=SUPERVISOR_PROMPT,
    state_schema=SupervisorState,
)


# ============================================================
# 4. 子 Agent 节点（调用对应子 Agent，并把结果写回 state.messages）
# ============================================================
def _extract_conclusion(new_msgs) -> AIMessage:
    """从子 Agent 产出的消息里提取「最终结论」。

    子 Agent 的 ReAct 循环消息形如：
        [AIMessage(tool_calls), ToolMessage, ..., AIMessage(最终回答)]
    取最后一条 AIMessage 作为结论；若它仍带 tool_calls（异常/步数打满），
    仍取其 content 兜底，避免主上下文收到空白。
    """
    for m in reversed(new_msgs):
        if isinstance(m, AIMessage):
            return AIMessage(content=m.content or "")
    tail = new_msgs[-1] if new_msgs else None
    return AIMessage(content=getattr(tail, "content", "") or "（子 Agent 未产出结论）")


# ============================================================
# 4.1 子 Agent 节点（十六 #41/#42/#43：独立上下文）
#    子 Agent 只接收「当前用户最新问题」作为任务输入，不看全量主上下文，
#    实现上下文隔离、省 token；且只把结论写回主 state（带 metadata 标记）。
# ============================================================
def _scope_for_subagent(state: AgentState, agent_name: str) -> list:
    """十六 #41：构造子 Agent 的「独立上下文」。

    优先级：① 主 Agent 分解后写入的委派指令 delegated_request（具体子任务，含必要实体）；
            ② 退化为最新一条用户消息（兜底，避免委派指令缺失时子 Agent 无输入）。
    不再把整段主上下文丢给子 Agent——隔离 + 省 token + 避免多专家互相污染。
    """
    req = (state.get("delegated_request") or "").strip()
    if req:
        return [HumanMessage(content=req)]
    msgs = list(state["messages"])
    idx = _last_human_index(msgs)
    if idx < 0:
        return []
    return [msgs[idx]]


def _run_subagent(agent, state: AgentState, agent_name: str = "") -> dict:
    """十六 #41/#42/#43：只喂「独立上下文」给子 Agent，且只把结论写回主 state。
    结论带 metadata 标记，便于 answer_node 精准收集（不被主 Agent 推理噪声干扰）。"""
    scoped = _scope_for_subagent(state, agent_name)
    result = agent.invoke({"messages": scoped})
    new_msgs = result["messages"][len(scoped):]
    conclusion = _extract_conclusion(new_msgs)
    conclusion.metadata = {**(conclusion.metadata or {}), "type": "subagent_conclusion"}
    return {"messages": [conclusion]}


def order_node(state):  return _run_subagent(order_agent, state, "order")
def qa_node(state):     return _run_subagent(qa_agent, state, "qa")
def ticket_node(state): return _run_subagent(ticket_agent, state, "ticket")


# ============================================================
# 4.1 汇总/作答节点（answer）—— 本轮唯一「最终回复」的出口
#
#     为什么要有这一层（生产主流形态：Supervisor 汇总）：
#     - 单专家：子 Agent 的结论本身就是最终回复 → **直接透传**（不调 LLM，零额外成本）；
#     - 多专家：一轮内多个子 Agent 的结论是**中间产物**，直接堆给用户是碎片，
#       应由本节点整合成一条连贯回复再给出（主流 agent 均如此：
#       工具/子模块的原始输出不直接给用户，由主模型整合）。
#     - 0 个结论（纯致谢/闲聊）：走模板收尾语（即原 farewell 的内容），也入库。
#
#     ⚠️ 只有本节点产出的最终回复才写 episodic_memory；
#        子 Agent 的中间结论不入库（它们留在 state/checkpointer 里）。
# ============================================================
FAREWELL_REPLY = "不客气～如果还有其他问题，随时找我。祝您生活愉快！"

ANSWER_PROMPT = SystemMessage(content="""
你是客服总台。下面是本轮各位专家分别给出的结论，请把它们整合成**一条**连贯、
自然的回复给用户：去重、按用户提问顺序组织、不要暴露"专家A/专家B"等内部角色，
也不要遗漏任何一条结论里的关键信息（订单号、时效、金额等数字要保留）。
只输出整合后的回复正文，不要加前后缀说明。
""")


def _integrate_conclusions(conclusions: list) -> str:
    """多专家时：用小模型把多条结论整合成一条连贯回复（成本可控）。"""
    joined = "\n\n".join(f"[{i+1}] {m.content}" for i, m in enumerate(conclusions))
    try:
        resp = get_llm("small").invoke([ANSWER_PROMPT, HumanMessage(content=joined)])
        return (resp.content or "").strip() or conclusions[-1].content
    except Exception:
        # 降级：整合失败也不能丢答案，退化为按序拼接（宁可朴素，不可丢失）
        return "\n\n".join(m.content for m in conclusions)


# ============================================================
# 4.2 情景记忆蒸馏（C13）：每轮异步提炼成结构化片段，全量原文不再落库
# ============================================================
DISTILL_PROMPT = SystemMessage(content="""你是记忆提炼器。阅读本轮客服对话，提取有后续参考价值的「情景记忆」。
规则：
- 只提取带具体事件/结果/偏好/失败/承诺/教训/异常的内容；纯客套、闲聊、套话不提取。
- 每条给：
  signal_type（枚举：event_outcome 事件结果 / preference 用户偏好 /
    failure 失败异常 / commitment 承诺 / lesson 经验教训 / anomaly 异常越权）；
  content（≤80字，包含"发生了什么 + 结果"）；
  entities（可检索实体 dict，如 {"order_id":"A","ticket_id":"5521"}，无则 {}）；
  importance（1-5，1琐碎，5关键承诺/失败/越权）。
- 返回 episodes 列表；本轮无价值则返回空列表。
""")


class Episode(TypedDict):
    signal_type: str
    content: str
    entities: dict
    importance: int


class DistillResult(TypedDict):
    episodes: List[Episode]


def _extract_episodes(user_text: str, final_reply: str) -> List[dict]:
    """用小模型把本轮对话蒸馏成 0~N 条结构化情景记忆。"""
    model = get_llm("small").with_structured_output(DistillResult)
    res = model.invoke([DISTILL_PROMPT,
                        HumanMessage(content=f"用户：{user_text}\n助手：{final_reply}")])
    return res.get("episodes") or []


def _distill_worker(thread_id: str, user_text: str, final_reply: str) -> None:
    """后台线程：蒸馏 + 落库（fire-and-forget，失败仅告警不阻断对话）。"""
    try:
        episodes = _extract_episodes(user_text, final_reply)
        if episodes:
            insert_episodic_memories(episodes, thread_id=thread_id)
    except Exception as e:  # 蒸馏/落库失败不影响主对话
        print(f"[warn] 情景记忆蒸馏/落库失败（thread_id={thread_id}）: {e}")


def _schedule_distill(config: dict, messages, final_reply: str) -> None:
    """answer_node 调用：启后台线程异步蒸馏（不阻塞回复）。"""
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")
    idx = _last_human_index(messages)
    user_text = messages[idx].content if idx >= 0 else ""
    threading.Thread(target=_distill_worker,
                     args=(thread_id, user_text, final_reply),
                     daemon=True).start()


def answer_node(state: AgentState, config: dict) -> dict:
    """产出本轮唯一最终回复 + 异步蒸馏情景记忆（0 / 1 / N 条结论三种分支）。

    C13：全量原文不再落库，改为后台线程蒸馏成结构化片段写入 episodic_memory。
    """
    msgs = list(state["messages"])
    concluded = _conclusions_since_last_human(msgs)

    # 分支 1：没有子 Agent 结论（纯致谢 / 闲聊）→ 模板收尾语，直接追加
    if not concluded:
        _schedule_distill(config, msgs, FAREWELL_REPLY)
        return {"messages": [AIMessage(content=FAREWELL_REPLY)]}

    # 分支 2：单专家 → 结论即最终回复，直接透传（不调 LLM、不追加、不改写）
    if len(concluded) == 1:
        _schedule_distill(config, msgs, concluded[0].content)
        return {}

    # 分支 3：多专家 → 整合后【替换】掉这些中间结论，只留一条最终回复
    final_reply = _integrate_conclusions(concluded)
    idx = _last_human_index(msgs)
    new_msgs = msgs[: idx + 1] + [AIMessage(content=final_reply)]
    # 整体替换白板（同 compress_node）：避免 reducer 把完整列表再追加一次
    graph.update_state(config, {"messages": new_msgs})
    _schedule_distill(config, msgs, final_reply)
    return {}


# ============================================================
# ============================================================
# 4.3 记忆召回（十五 #32/#34）：不再有入口固定闸门节点
#
#   原 recall_node（实体/指代词硬触发）已移除；记忆召回改为「agentic」：
#   由主 Agent 在推理时按需调用 search_memory 工具（见第 3 节），
#   LLM 自己决定要不要查、查什么——避免"没说指代词但指代旧上下文"时漏召。
#   召回实现（实体 LLM 抽取 + 向量语义）封装在 _search_memory_impl / search_memory 工具中，
#   复用下方 _build_recall_message 拼装注入内容。
# ============================================================
def _build_recall_message(hits: List[dict]) -> SystemMessage:
    lines = []
    for h in hits:
        ent = h.get("entities") or {}
        ent_str = " " + str(ent) if ent else ""
        lines.append(f"- [{h.get('signal_type')}]{ent_str} {h.get('content')}")
    return SystemMessage(content="【相关历史记忆】\n" + "\n".join(lines))


# ============================================================
# 5. 压缩节点（手写：写回上下文后调用 compress_history）
#    对应讨论的「第2块压缩」：触发时机 = 每次节点处理后，检查 token 压最早步骤。
# ============================================================
def compress_node(state: AgentState, config: dict) -> dict:
    """【生产生效节点 · 两档触发】滚动压缩累积历史 —— **只作用于工作记忆（上下文窗口）**。

    生产要点：
    - State.messages 用 Annotated 累加 reducer，节点若 `return {"messages": 完整列表}`
      会把完整列表「再追加」到白板，导致历史翻倍。所以**不能用普通返回值写回完整列表**。
    - 正确做法：用 graph.update_state(...) 把 messages 与 summary 字段「整体替换/覆盖」
      （update_state 直接覆盖白板，不走 reducer 累加）。
    - 触发口径 = **token 计数 · 两档**（软触发 60% 提前温和压、硬顶 80% 强制压到硬顶下），
      与 compress_history 共用 memory.count_messages_tokens；不用字符数，也不用轮数粗判。
    - 压缩粒度 = 「轮」（不是消息条数）；压缩手段 = 小模型 LLM 摘要（与 LangMem 一致）。
    - 滚动摘要通过 state.summary 字段承载（覆盖式），由本节点写回；不再伪装成 messages 第0条。
    ⚠️ 关键澄清：本节点**不触发情景记忆落库**。情景记忆由 checkpointer 在图中
       每个节点产出后自动落盘，与这里的 token 阈值无关。
    """
    soft_limit = int(Settings.MAX_TOKEN_LIMIT * Settings.SOFT_TRIGGER_RATIO)
    if count_messages_tokens(state["messages"]) <= soft_limit:
        return {}  # 未到软触发线，不压（零成本；只压工作记忆/窗口，不涉及落库）
    # 目标水位：已冲过硬顶则压到硬顶线以下，否则压到软触发线以下（留更大缓冲）
    total = count_messages_tokens(state["messages"])
    hard_limit = int(Settings.MAX_TOKEN_LIMIT * Settings.HARD_LIMIT_RATIO)
    target_ratio = Settings.HARD_LIMIT_RATIO if total > hard_limit else Settings.SOFT_TRIGGER_RATIO
    out, summary = compress_history(
        state["messages"],
        running_summary=state.get("summary", ""),
        target_ratio=target_ratio,
    )
    # 用 update_state 整体替换白板（messages 整体替换 + summary 覆盖），绕过累加 reducer
    graph.update_state(config, {"messages": out, "summary": summary})
    return {}  # 节点本身不重复返回，避免 reducer 再追加一次


# ============================================================
# 6. 路由说明（十五 #31/#36）
#    主 Agent（supervisor）通过 handoff/finish 工具返回 Command(goto=...)，
#    父图据此跳转；无 Command 时走下方默认边到 answer。无需旧的 route 枚举分发。
# ============================================================

# ============================================================
# 7. 画图（StateGraph）：把节点和边连起来
# ============================================================
builder = StateGraph(AgentState)

# 加节点
builder.add_node("supervisor", supervisor_agent)   # 主 Agent 子图（持有对话 + handoff/记忆工具）
builder.add_node("order", order_node)
builder.add_node("qa", qa_node)
builder.add_node("ticket", ticket_node)
builder.add_node("compress", compress_node)   # 压缩节点（子Agent跑完后检查 token）
builder.add_node("answer", answer_node)       # 汇总/作答节点：本轮唯一最终回复出口

# 边
builder.add_edge(START, "compress")                   # 入口 -> 压缩（校准窗口，未超零成本返回）
builder.add_edge("compress", "supervisor")           # 压缩后 -> 主 Agent
# 主 Agent 调用 handoff/finish 工具 → Command(goto) 跳转对应节点；
# 未返回 Command（正常结束）则走默认边到 answer
builder.add_edge("supervisor", "answer")
# 子 Agent 干完 -> 压缩 -> 回到主 Agent（可多轮：用户有多个诉求会再次委派）
builder.add_edge("order", "compress")
builder.add_edge("qa", "compress")
builder.add_edge("ticket", "compress")
builder.add_edge("compress", "supervisor")            # 压缩后回到主 Agent，形成循环
builder.add_edge("answer", END)

# 编译（recursion_limit = 硬护栏：步数封顶，防无限循环）
# checkpointer = 短期记忆的持久化实现（PostgreSQL + pgvector），按 thread_id 落盘：
#   - 保证「下一轮接着聊 / 进程重启不丢 / 任意实例可恢复」（无状态化）；
#   - 落盘由框架在每个节点产出后自动完成，与 compress_node 的 token 阈值无关。
# 注：语义记忆（跨会话用户画像）是 opt-in，标准里非默认，本 demo 不接。
# 注：情景记忆（蒸馏结构化片段）由 episodic_memory 表承担，主 Agent 的 search_memory
#     工具按需 agentic 召回，与 checkpointer 是两张独立表、独立用途（详见顶部三层记忆说明）。
graph = builder.compile(checkpointer=get_checkpointer(),
                        recursion_limit=Settings.MAX_ITERATIONS)


# ============================================================
# 对外接口
# ============================================================
def chat(user_input: str, session_id: str = "default") -> str:
    """单次对话入口。session_id = thread_id，区分不同会话（无状态化 + 状态外置）。

    一次 invoke 的「工作记忆」组装过程（对应三层记忆的第 1 层 短期记忆）：
      1. 框架用 thread_id 从 checkpointer(PostgreSQL) load 回该会话历史
         = **短期记忆（续聊）恢复**，不是情景记忆召回；若历史已超窗，
         取回的已是 [历史摘要] + [最近 k 轮]；未超窗时 load 回的是原始逐轮、无摘要
      2. 把本轮 HumanMessage 追加进去（即「取回的短期记忆 + 本轮」）；
      3. 图内循环：compress(校准窗口) → 主 Agent(按需调 search_memory 召回记忆、
         调 transfer_to_* 委派子 Agent) → 子 Agent(只回结论) → compress → 主 Agent…
         直到主 Agent 调 finish 结束本轮；
      4. answer_node 产出**本轮唯一最终回复**（单专家透传 / 多专家整合 /
         纯客套走模板收尾语），并启后台线程异步蒸馏成结构化情景记忆写入 episodic_memory
         （全量原文不再落库，蒸馏失败不影响本轮返回）；
      5. 每个节点产出后框架自动落盘 checkpoints（与压缩、与 answer 均无关）。

    情景记忆（第 2 层）的「写 & 读」与本过程的关系：
      - 写：第 4 步后台线程蒸馏写库（每轮一次，异步）；
      - 读：第 3 步由主 Agent 的 search_memory 工具**按需 agentic 召回**（取代原 recall_node
        固定闸门）——用户提订单/工单号或说"之前/那个"时，LLM 自决调用 search_memory 把同会话
        相关 episode 注入上下文，补回被压缩掉的精确细节；平时不查。
      ⚠️ 不要把「第 1 步 checkpointer 恢复」和「第 3 步情景记忆召回」混为一谈：
        前者是短期记忆（续聊连贯性），后者是蒸馏结构化片段（找回精确细节），两张表、两套用途。

    ⚠️ 无状态 ≠ 模型只看本轮：模型看到的是「历史 + 本轮」，
       只是这份历史不在进程内存里，而在 PostgreSQL 中按 thread_id 存着。
    """
    cfg = {"configurable": {"thread_id": session_id}, "recursion_limit": Settings.MAX_ITERATIONS}
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_input)],
         "thread_id": session_id, "summary": "", "delegated_request": ""},
        config=cfg,
    )
    # 取最后一条 AI 消息作为答复
    for m in reversed(result["messages"]):
        if isinstance(m, AIMessage):
            return m.content
    return "（无回复）"
