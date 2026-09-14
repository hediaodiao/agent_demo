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
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "12"))   # 图步数硬护栏（recursion_limit）：单次 invoke 节点执行总步数上限，防任意环死循环
    MAX_DISPATCHES = int(os.getenv("MAX_DISPATCHES", "4"))    # 派单次数护栏（C10）：本轮最多派给几个不同专家，超了强制 farewell 收尾
    MAX_TOKEN_LIMIT = int(os.getenv("MAX_TOKEN_LIMIT", "4000"))  # 上下文窗口 token 上限（硬顶基准）
    SUMMARY_TRIGGER_RATIO = 0.8   # 【已废弃】旧单阈值；仅作兼容保留，新逻辑用下方两档
    SOFT_TRIGGER_RATIO = 0.6  # 软触发线：窗口用到 60% 即提前温和压缩，留大缓冲（压得少、损失小）
    HARD_LIMIT_RATIO = 0.8    # 硬顶线：窗口不可超 80%，压完必须回落到硬顶以下（极端时连带单轮兜底）

    # RAG
    CHUNK_SIZE = 400
    TOP_K = 5

    # 情景记忆（C13：蒸馏结构化片段，同会话按需召回）
    DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "001")          # 上线改为从请求上下文动态获取
    EPISODIC_TOP_K = int(os.getenv("EPISODIC_TOP_K", "5"))         # 召回注入条数
    SEMANTIC_TRIGGER_THRESHOLD = float(os.getenv("SEMANTIC_TRIGGER_THRESHOLD", "0.78"))  # 语义触发相似度下限

    @classmethod
    def has_real_key(cls) -> bool:
        return bool(cls.API_KEY) and cls.API_KEY != "sk-xxx"


class SIGNAL_TYPE:
    """情景记忆的 6 类信号（对齐 Mem0/HWC 生产做法）。"""
    EVENT_OUTCOME = "event_outcome"   # 具体事件 + 结果
    PREFERENCE = "preference"         # 用户偏好 / 习惯
    FAILURE = "failure"               # 失败 / 异常 / 降级
    COMMITMENT = "commitment"         # 承诺 / 待办
    LESSON = "lesson"                 # 经验教训 / 处理方式
    ANOMALY = "anomaly"               # 异常 / 边界 / 越权
    CHOICES = [EVENT_OUTCOME, PREFERENCE, FAILURE, COMMITMENT, LESSON, ANOMALY]


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
