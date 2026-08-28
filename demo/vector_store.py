"""向量库（RAG 检索）。

教学要点（对应我们之前的讨论）：
1. 单轮超长 / 知识库问答都走 RAG：内容先「切分」成小片段再 embed 入库。
2. 检索算的是「用户问题向量」vs「每个内容片段向量」的相似度
   （不是长内容自比，也不是和历史对话比）。
3. 生产用 Milvus / Qdrant / pgvector + 真实 embed 模型；
   这里用内存版 + OpenAI embed（或本地假 embed）默认可跑，接口留好替换点。

面试考点：为什么切分？—— 整段语义被稀释、检索不精，且整段塞不进上下文。
"""
import hashlib
import math
import re
from typing import List, Dict, Optional

from langchain_openai import OpenAIEmbeddings
from config import Settings


def _fake_embed(text: str) -> List[float]:
    """本地假 embed（无 API 时演示用）。生产替换为 OpenAIEmbeddings。"""
    dim = 64
    vec = [0.0] * dim
    for tok in re.findall(r"[\w\u4e00-\u9fff]+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class Embedder:
    """embed 抽象：生产用 OpenAIEmbeddings，无 key 时降级假 embed。"""

    def __init__(self):
        if Settings.has_real_key():
            self._emb = OpenAIEmbeddings(model="text-embedding-3-small",
                                         base_url=Settings.BASE_URL or None,
                                         api_key=Settings.API_KEY or None)
            self._fake = False
        else:
            self._fake = True

    def embed(self, text: str) -> List[float]:
        if self._fake:
            return _fake_embed(text)
        return self._emb.embed_query(text)


class VectorStore:
    """内存向量库（接口对齐 Milvus/pgvector，生产直接换实现）。"""

    def __init__(self, embedder: Optional[Embedder] = None, chunk_size: int = None):
        self.embedder = embedder or Embedder()
        self.chunk_size = chunk_size or Settings.CHUNK_SIZE
        self._chunks: List[Dict] = []

    # ---- 切分（对应讨论：中文按标点递归切） ----
    def _split(self, text: str) -> List[str]:
        pieces = re.split(r"(?<=[。！？；\n])", text)
        chunks, buf = [], ""
        for p in pieces:
            if len(buf) + len(p) <= self.chunk_size:
                buf += p
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return chunks or [text]

    def ingest(self, doc_id: str, text: str, meta: Dict = None):
        """上传即入库：切分 + embed + 写库（对应讨论的「上传文件异步预处理」）。"""
        for i, chunk in enumerate(self._split(text)):
            self._chunks.append({
                "text": chunk,
                "vector": self.embedder.embed(chunk),
                "meta": {**(meta or {}), "doc_id": doc_id, "chunk_idx": i},
            })

    def delete_by_doc(self, doc_id: str):
        """删文件联动清向量（对应讨论的「删文件即删向量」）。"""
        self._chunks = [c for c in self._chunks if c["meta"].get("doc_id") != doc_id]

    def similarity_search(self, query: str, top_k: int = None) -> List[str]:
        """问题 vs 片段 相似度检索，返回 top-K 片段文本。"""
        qv = self.embedder.embed(query)
        top_k = top_k or Settings.TOP_K
        scored = [(self._cos(qv, c["vector"]), c["text"]) for c in self._chunks]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:top_k]]

    @staticmethod
    def _cos(a, b):
        return sum(x * y for x, y in zip(a, b))
