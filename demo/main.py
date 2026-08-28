"""CLI 演示入口（生产级框架版：LangGraph + LangChain）。

运行：
    python main.py                  # 交互式对话
    python main.py --session user1  # 指定 session_id（无状态化演示）

依赖：pip install -r requirements.txt，并设置 OPENAI_API_KEY。
无 key 时仍能 import，但真正调用 LLM 会报错（这是预期，生产必须有 key）。
"""
import argparse
from graph import chat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="user_001")
    args = ap.parse_args()

    print("=== 生产级工单客服（LangGraph 方案 B：纯 StateGraph）===")
    print("输入 exit 退出。\n")
    while True:
        try:
            u = input("用户> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if u.lower() in ("exit", "quit"):
            break
        if not u:
            continue
        try:
            resp = chat(u, session_id=args.session)
        except Exception as e:
            resp = f"[调用出错，请检查 OPENAI_API_KEY / 网络] {e}"
        print("客服>", resp, "\n")


if __name__ == "__main__":
    main()
