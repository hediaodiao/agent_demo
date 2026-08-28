"""端到端冒烟测试（生产级框架版）。

验证：Supervisor 分流是否正确（订单/问答/工单各自走对应子 Agent）。
运行：python test_demo.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from graph import chat

qs = [
    "帮我查一下我的订单",       # -> order_agent
    "你们的退货政策是什么",     # -> qa_agent
    "我要投诉，帮我建个工单",   # -> ticket_agent
]

for q in qs:
    print("用户>", q)
    try:
        print("客服>", chat(q, session_id="test"))
    except Exception as e:
        print("客服> [需配置 OPENAI_API_KEY]", e)
    print()
