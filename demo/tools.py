"""工具层：订单查询 / 知识库RAG / 工单（LangChain @tool 原生写法）。

教学要点（对应我们之前的讨论）：
1. 【工具返回压缩 - 第1块】长结果不原样进上下文，先结构化裁剪 / 小模型摘要。
   之前聊过：LangChain 的 @tool 里你写压缩逻辑，框架不管这块。
2. 【降级 / 备用数据源】主数据源(DB)挂了 -> 切 CSV 备胎（"有备用数据源就切换"）。
3. RAG 走「问题 vs 内容片段」相似度（见 vector_store）。
"""
from typing import List
from langchain_core.tools import tool

from vector_store import VectorStore
from memory import summarize_with_small_model

# ---- 模拟数据源 ----
_PRIMARY_ORDERS = [
    {"order_id": "O-001", "user_id": "u_1001", "item": "无线耳机", "status": "已发货", "amount": 299},
    {"order_id": "O-002", "user_id": "u_1001", "item": "机械键盘", "status": "待付款", "amount": 459},
    {"order_id": "O-003", "user_id": "u_1001", "item": "显示器", "status": "已签收", "amount": 1299},
    {"order_id": "O-004", "user_id": "u_1001", "item": "鼠标", "status": "运输中", "amount": 129},
    {"order_id": "O-005", "user_id": "u_1001", "item": "摄像头", "status": "已发货", "amount": 199},
]
_BACKUP_ORDERS_CSV = [
    {"order_id": "O-001", "user_id": "u_1001", "item": "无线耳机", "status": "已发货(备胎)", "amount": 299},
    {"order_id": "O-002", "user_id": "u_1001", "item": "机械键盘", "status": "待付款(备胎)", "amount": 459},
]

KB_TEXT = """
公司退货政策：商品签收后7天内可申请无理由退货，需保持包装完整。
物流时效：普通快递3-5天，顺丰次日达。偏远地区加1-2天。
会员权益：金卡会员享免费上门取件退货，银卡会员运费自理。
发票说明：电子发票在发货后24小时内发送至注册邮箱。
投诉处理：工单提交后承诺24小时内首次响应，复杂问题不超过3个工作日。
"""


def _format_orders(orders: List[dict]) -> str:
    """【返回压缩】结构化裁剪：只取关键字段、限制条数，避免长结果撑爆上下文。"""
    lines = [f"{o['order_id']} {o['item']} [{o['status']}] RMB{o['amount']}" for o in orders[:5]]
    return "订单列表(前{}条):\n".format(len(lines)) + "\n".join(lines)


@tool
def query_orders(user_id: str) -> str:
    """查询用户订单。返回前做结构化裁剪（只留结论）。"""
    try:
        rows = [o for o in _PRIMARY_ORDERS if o["user_id"] == user_id]
        if not rows:
            return "未查询到该用户的订单。"
        return _format_orders(rows)
    except Exception:
        # 【降级】主数据源挂了 -> 切备用数据源（CSV 备胎）
        rows = [o for o in _BACKUP_ORDERS_CSV if o["user_id"] == user_id]
        return "[降级-备胎数据] " + _format_orders(rows)


# 知识库向量库（懒初始化，模拟「上传即入库」；生产是异步预处理）
_kb_store = None


@tool
def rag_search(query: str) -> str:
    """知识库问答检索：先确保知识库已入库，再检索 top-K 片段（问题 vs 片段相似度）。"""
    global _kb_store
    if _kb_store is None:
        _kb_store = VectorStore()
        _kb_store.ingest("kb_policy", KB_TEXT, meta={"source": "policy"})
    chunks = _kb_store.similarity_search(query, top_k=3)
    joined = "\n---\n".join(chunks)
    # 【返回压缩】片段拼接过长时用小模型摘要收口
    if len(joined) > 600:
        joined = summarize_with_small_model(joined, max_chars=400)
    return "[知识库检索]\n" + joined


@tool
def create_ticket(title: str, detail: str) -> str:
    """提交工单。返回只留工单号与状态（不返回原始长 detail）。"""
    tid = "T-" + str(abs(hash(title)) % 9000 + 1000)
    return f"工单已创建：编号 {tid}，状态=待处理，承诺24h内首次响应。"


# 按业务分组供不同子 Agent 使用（对应 Supervisor 分流）
ORDER_TOOLS = [query_orders]
QA_TOOLS = [rag_search]
TICKET_TOOLS = [create_ticket, query_orders]
ALL_TOOLS = [query_orders, rag_search, create_ticket]
