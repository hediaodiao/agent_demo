import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from agent import AgentLoop, SupervisorAgent
from tools import ALL_TOOLS

print("=== 单 Agent 模式 ===")
a = AgentLoop(ALL_TOOLS, role="decider")
qs = ["帮我查一下我的订单", "你们的退货政策是什么", "我要投诉，帮我建个工单"]
for q in qs:
    print("用户>", q)
    print("客服>", a.chat(q, session_id="demo_single"))
    print()

print("=== Supervisor 多 Agent 模式 ===")
s = SupervisorAgent()
for q in qs:
    print("用户>", q)
    print("客服>", s.chat(q, session_id="demo_multi"))
    print()
