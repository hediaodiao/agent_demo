# 遗留代码（保留不删除）：graph.py 已用 LangGraph checkpointer（PostgreSQL + pgvector）
# 实现状态外置 / 无状态化，本文件在当前方案中**未被引用**。
# 仅保留作为「会话存储抽象」的对比示例，生产无需再接这一套。
"""会话存储（状态外置，对应讨论的「无状态化」）。

教学要点：
- Agent / 子图实例本身不持有会话历史；历史按 session_id 存外部。
- 任意实例都能用 session_id 取到历史 -> 可水平扩展、可恢复。
- 生产用 Redis（带 TTL）或 PostgreSQL + JSON；这里用内存 dict 默认可跑，
  接口留好替换点（见 RedisSessionStore 注释）。
"""
from typing import List
from langchain_core.messages import BaseMessage


class InMemorySessionStore:
    """内存版（演示/单机）。生产替换为下面 RedisSessionStore。"""

    def __init__(self):
        self._data: dict = {}

    def get(self, session_id: str) -> List[BaseMessage]:
        return self._data.get(session_id, [])

    def append(self, session_id: str, msg: BaseMessage):
        self._data.setdefault(session_id, []).append(msg)

    def set(self, session_id: str, msgs: List[BaseMessage]):
        self._data[session_id] = msgs

    def clear(self, session_id: str):
        self._data.pop(session_id, None)


"""
# ===== 生产版（取消注释并 pip install redis 即可启用） =====
import redis, json
from langchain_core.messages import BaseMessage, human_message, ai_message  # 需序列化

class RedisSessionStore:
    def __init__(self, url="redis://localhost:6379", ttl=3600):
        self.r = redis.Redis.from_url(url)
        self.ttl = ttl

    def get(self, session_id):
        raw = self.r.get(f"session:{session_id}")
        return deserialize(raw) if raw else []   # 需自己写消息序列化

    def append(self, session_id, msg):
        msgs = self.get(session_id)
        msgs.append(msg)
        self.r.set(f"session:{session_id}", serialize(msgs), ex=self.ttl)

    # ... 其余同理
"""
