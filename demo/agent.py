# 遗留代码（保留不删除）：当前运行入口是 graph.py（方案 B：LangGraph 手写 StateGraph）。
# 本文件是方案 B 之前的早期手写 ReAct 版，import 的 RollingSummaryMemory / call_tool
# 等符号在当前 memory.py / tools.py 中已不存在，直接运行会 ImportError。
# 仅作对比学习用，请勿 import 到生产路径。
"""Agent 编排层。

生产实践要点（对应我们讨论的）：
1. 循环骨架由框架封装：这里手写轻量 ReAct 循环（等价于 LangChain AgentExecutor），
   展示「模型只决策调工具，工具返回处理/上下文压缩是硬编码逻辑」。
2. 硬编码逻辑点：
   - 工具返回后 -> 结果已自带压缩（tools.py 里 format/truncate）
   - 每步写回上下文后 -> RollingSummaryMemory.maybe_compress() 检查 token 压早步
3. 硬护栏：max_iter 步数封顶，防无限循环（对应 graph.py 的
   recursion_limit=Settings.MAX_ITERATIONS——图步数护栏）。
   注：本文件是单 Agent ReAct 循环，无「子 Agent 派单」概念，
   故只有步数护栏，没有 graph.py 里 C10 的派单次数护栏（MAX_DISPATCHES）。
4. 无状态化：历史从 session_store 按 session_id 取，Agent 本身不持有。
5. Supervisor 多 Agent：主 Agent 分流到专家子 Agent（订单/问答/工单）。
"""
from messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from config import Settings, get_llm
from memory import RollingSummaryMemory
from session_store import InMemorySessionStore
from tools import ORDER_TOOLS, QA_TOOLS, TICKET_TOOLS, ALL_TOOLS, call_tool

store = InMemorySessionStore()

SYSTEM_PROMPT = (
    "你是企业工单客服 Agent。请使用工具回答问题。"
    "能查数据的用工具查，不要编造；工具返回已压缩，直接基于结论回答用户。"
)


class AgentLoop:
    """轻量 ReAct 循环（等价于 AgentExecutor 的封装）。"""

    def __init__(self, tools, role: str = "decider", max_iter: int = None):
        self.tools = tools
        self.llm = get_llm("big", role=role)
        self.role = role
        self.max_iter = max_iter or Settings.MAX_ITERATIONS
        self.mem = RollingSummaryMemory()

    def _build_messages(self):
        return [SystemMessage(content=SYSTEM_PROMPT)] + self.mem.to_messages()

    def _run_once(self, user_input: str, session_id: str) -> str:
        # 无状态化：从外部存储取历史
        history = store.get(session_id)
        self.mem.reset()
        for m in history:
            self.mem.add(m)

        self.mem.add(HumanMessage(content=user_input))

        for step in range(self.max_iter):
            ai = self.llm.invoke(self._build_messages())
            self.mem.add(ai)

            if not getattr(ai, "tool_calls", []):
                store.set(session_id, self.mem.recent)
                return ai.content

            for tc in ai.tool_calls:
                obs = call_tool(tc.name, tc.args)
                self.mem.add(ToolMessage(content=str(obs), tool_call_id=tc.id))
                # add 内已触发 maybe_compress（写回后压早步）

        store.set(session_id, self.mem.recent)
        return "（已触发步数护栏）我需要更多信息或请稍后重试，当前无法自动完成。"

    def chat(self, user_input: str, session_id: str = "default") -> str:
        return self._run_once(user_input, session_id)


class SupervisorAgent:
    """Supervisor 多 Agent：主 Agent 分流，调用子 Agent（普通 AgentLoop）。"""

    def __init__(self):
        self.order_agent = AgentLoop(ORDER_TOOLS, role="order")
        self.qa_agent = AgentLoop(QA_TOOLS, role="qa")
        self.ticket_agent = AgentLoop(TICKET_TOOLS, role="ticket")
        self.super = get_llm("big", role="supervisor")

    def _dispatch(self, agent_name: str, text: str, sid: str) -> str:
        return getattr(self, agent_name).chat(text, session_id=sid + ":" + agent_name)

    def chat(self, user_input: str, session_id: str = "default") -> str:
        msgs = [
            SystemMessage(content="你是调度台，判断问题类型并 handoff 到对应专家："
                                   "订单/物流->order_agent，政策/知识->qa_agent，"
                                   "投诉/建工单->ticket_agent。"),
            HumanMessage(content=user_input),
        ]
        decision = self.super.invoke(msgs)
        if getattr(decision, "tool_calls", []):
            target = decision.tool_calls[0].args.get("agent", "qa_agent")
            return self._dispatch(target, user_input, session_id)
        return decision.content
