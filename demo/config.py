"""全局配置 + LLM 工厂（生产级，基于 langchain-openai）。

教学要点（对应我们之前的讨论）：
- 决策用「大模型」(BIG_MODEL)，压缩/摘要用「小模型」(SMALL_MODEL)。
  之前聊过：压缩调用的模型一般用小模型，LangChain 的 memory/tool 都接受
  你传入的 llm 实例 —— 这里 get_llm("small") 就是给压缩组件用的。
- 支持兼容端点（OPENAI_BASE_URL），生产常接自建 vLLM / 第三方网关。
"""
import os
from langchain_openai import ChatOpenAI


class Settings:
    # 模型（生产按成本/质量选；小模型做压缩性价比最高）
    BIG_MODEL = os.getenv("BIG_MODEL", "gpt-4o-mini")
    SMALL_MODEL = os.getenv("SMALL_MODEL", "gpt-4o-mini")

    BASE_URL = os.getenv("OPENAI_BASE_URL", None)
    API_KEY = os.getenv("OPENAI_API_KEY", "")

    # 护栏 / 压缩阈值
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "12"))   # 递归上限（硬护栏）
    MAX_TOKEN_LIMIT = int(os.getenv("MAX_TOKEN_LIMIT", "4000"))  # 上下文压缩触发上限
    SUMMARY_TRIGGER_RATIO = 0.8   # 达到窗口 80% 开始压最早步骤

    # RAG
    CHUNK_SIZE = 400
    TOP_K = 5

    @classmethod
    def has_real_key(cls) -> bool:
        return bool(cls.API_KEY) and cls.API_KEY != "sk-xxx"


def _make(model: str, temperature: float = 0.2, max_tokens: int = 1024) -> ChatOpenAI:
    kwargs = {"model": model, "temperature": temperature, "max_tokens": max_tokens}
    if Settings.BASE_URL:
        kwargs["base_url"] = Settings.BASE_URL
    if Settings.API_KEY:
        kwargs["api_key"] = Settings.API_KEY
    return ChatOpenAI(**kwargs)


def get_llm(role: str = "big") -> ChatOpenAI:
    """LLM 工厂：role='big' 决策大模型，role='small' 压缩小模型。

    考点：为什么分两个？——压缩是高频、低推理需求的活，
    用小模型便宜 10 倍、快很多，且质量足够（把对话浓缩成事实）。
    """
    if role == "small":
        return _make(Settings.SMALL_MODEL, temperature=0.0, max_tokens=512)
    return _make(Settings.BIG_MODEL)
