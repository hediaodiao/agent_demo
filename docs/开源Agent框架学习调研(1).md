# 开源 Agent 框架学习调研报告

> 调研日期：2026-08-08 ｜ 数据来源：GitHub Star 排名、多个 2026 框架横评（the-agent-report / deepyard / presenc.ai / awesomeagents / 中文社区）
> 说明：Star 数为 2026 年中近似值，框架迭代很快，仅供量级参考。所有链接均为官方 GitHub 仓库。

---

## 一、结论先行：按你的目标选

| 你想做的事 | 首选 | 备选 | 一句话理由 |
|---|---|---|---|
| **系统搞懂 Agent 底层原理**（推荐零基础第一站） | [hello-agents](https://github.com/datawhalechina/hello-agents) | [learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) | 中文、从零手写代码讲清 ReAct/Memory/MCP，不是调包 |
| **20 行跑通多智能体协作 Demo** | [CrewAI](https://github.com/crewAIInc/crewAI) | [Agno](https://github.com/agno-agi/agno) | 角色/目标/工具心智模型最直观，上手最快 |
| **生产级、复杂、有状态、可中断恢复** | [LangGraph](https://github.com/langchain-ai/langgraph) | [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) | 图模型 + Checkpoint，业界落地最多 |
| **最小代码 + 强类型 + 结构化输出** | [Pydantic AI](https://github.com/pydantic/pydantic-ai) | [smolagents](https://github.com/huggingface/smolagents) | 类型即控制层，易测试、易 CI |
| **官方原生、轻量 handoff** | [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) | [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) | Swarm 正统后继，代码极少 |
| **可视化 / 低代码搭 Bot** | [Dify](https://github.com/langgenius/dify) | [n8n](https://github.com/n8n-io/n8n) / [Flowise](https://github.com/FlowiseAI/Flowise) | 拖拽即上线，非工程师友好 |
| **做 coding / 电脑操控 Agent** | [OpenHands](https://github.com/All-Hands-AI/OpenHands) | [Open Interpreter](https://github.com/OpenInterpreter/open-interpreter) | 沙箱里真跑代码/终端/浏览器 |

---

## 二、先分清两类：教程型 vs 框架型

新手最容易踩的坑是「一上来就啃重型框架」（比如直接上 LangChain），结果只会调 API、拼流程，没理解 Agent 核心运转逻辑。建议先读**教程型仓库**，再进框架。

### 📘 教程型（适合系统学习，强推）
| 项目 | GitHub | Star(约) | 亮点 |
|---|---|---|---|
| **hello-agents**（Datawhale 出品） | https://github.com/datawhalechina/hello-agents | 46.7k | 中文、16 章 5 阶段：原理→手写代码→框架实战→真实项目；覆盖 ReAct / Memory / MCP / A2A |
| **learn-claude-code**（shareAI-lab） | https://github.com/shareAI-lab/learn-claude-code | 54.7k | 几百行代码讲透「One loop & Bash is all you need」；12 节课渐进式，全中文 |

### ⚙️ 框架型（工程落地用）
下面按「编排范式」分组，范式决定了你写代码的心智模型。

---

## 三、框架逐类对比（含源码地址）

### 范式 A：图编排（Graph-based，最精确、可控、学习曲线陡）
| 框架 | GitHub | Star(约) | 语言 | 定位 / 学习曲线 |
|---|---|---|---|---|
| **LangGraph** | https://github.com/langchain-ai/langgraph | 33.9k | Py / JS | 状态图 + Checkpoint + 人工介入，生产级首选；曲线：高 |
| **Microsoft Agent Framework** | https://github.com/microsoft/agent-framework | 60k* | Py / .NET | AutoGen+Semantic Kernel 合并后继，Azure 原生；曲线：中高 |
| **Semantic Kernel** | https://github.com/microsoft/semantic-kernel | 27.9k | C# / Py | 微软 SDK，.NET/Azure 团队首选；曲线：中 |
| **Mastra**（TS） | https://github.com/mastra-ai/mastra | 23.9k | TypeScript | TS 原生工作流，前端栈友好；曲线：中 |
| **Vercel AI SDK**（TS） | https://github.com/vercel/ai | 24.2k | TypeScript | 流式/React hooks/Edge，Web 集成强；曲线：中 |

> *Microsoft Agent Framework 与 AutoGen 共用 Star 统计口径，约为合并后的社区量级。

### 范式 B：角色编排（Role-based，最直观、上手快）
| 框架 | GitHub | Star(约) | 语言 | 定位 / 学习曲线 |
|---|---|---|---|---|
| **CrewAI** | https://github.com/crewAIInc/crewAI | 52.8k | Python | 角色/目标/任务，20 行出 Demo；曲线：低。生产规模略吃力 |
| **Agno**（原 Phidata） | https://github.com/agno-agi/agno | 40.1k | Python | 主打「快」+ 内置 session/记忆，性价比高；曲线：低-中 |
| **AutoGen**（⚠️维护模式） | https://github.com/microsoft/autogen | 58.7k | Py / .NET | 对话式多智能体开创者，2025 末进维护模式；社区分叉 **AG2** https://github.com/ag2ai/ag2 继续开发 |

### 范式 C：类型安全的工具循环（Loop/Handoff，最小抽象）
| 框架 | GitHub | Star(约) | 语言 | 定位 / 学习曲线 |
|---|---|---|---|---|
| **Pydantic AI** | https://github.com/pydantic/pydantic-ai | 17.1k | Python | 类型即控制层，结构化输出 + 易测试；曲线：中 |
| **OpenAI Agents SDK** | https://github.com/openai/openai-agents-python | 26.9k | Python | Swarm 正统后继，handoff + guardrails + tracing；曲线：低。与 OpenAI 耦合 |
| **Claude Agent SDK** | https://github.com/anthropics/claude-agent-sdk-python | 6.9k | Python | Anthropic 原生，内置 MCP/工具；曲线：低 |
| **smolagents**（HF） | https://github.com/huggingface/smolagents | 27.3k | Python | ~1000 行极简，CodeAgent 心智；曲线：低-中 |

### 范式 D：RAG / 数据 / 记忆
| 框架 | GitHub | Star(约) | 语言 | 定位 |
|---|---|---|---|---|
| **LlamaIndex** | https://github.com/run-llama/llama_index | 49.4k | Python | 数据/RAG 之王，私有知识库问答 |
| **Haystack** | https://github.com/deepset-ai/haystack | 25.2k | Python | 模块化 pipeline，生产搜索系统 |
| **Letta**（原 MemGPT） | https://github.com/letta-ai/letta | 22.7k | Python | 长期记忆 / 有状态 Agent |
| **DSPy** | https://github.com/stanfordnlp/dspy | 34.4k | Python | 「编程而非提示」LLM，科研向 |

### 范式 E：多智能体 / 自动化公司（偏研究/产品化）
| 框架 | GitHub | Star(约) | 语言 | 定位 |
|---|---|---|---|---|
| **MetaGPT** | https://github.com/FoundationAgents/MetaGPT | 65k | Python | 「第一个 AI 软件公司」，多角色流水线 |
| **AutoGPT** | https://github.com/Significant-Gravitas/AutoGPT | 184k | Python | 自主任务 Agent 鼻祖，体量最大之一 |

### 范式 F：低代码 / 可视化平台
| 框架 | GitHub | Star(约) | 语言 | 定位 |
|---|---|---|---|---|
| **Dify** | https://github.com/langgenius/dify | 144k | Python | 全栈 LLM 平台，可视化工作流+知识库 |
| **n8n** | https://github.com/n8n-io/n8n | 187.8k | TypeScript | 工作流自动化，400+ 集成，执行层首选 |
| **Flowise** | https://github.com/FlowiseAI/Flowise | 52.8k | TS/Node | 拖拽式 Agent |
| **Langflow** | https://github.com/langflow-ai/langflow | 145k | Python | 可视化搭建部署 Agent |

### 范式 G：编程 / 电脑操控 Agent
| 框架 | GitHub | Star(约) | 语言 | 定位 |
|---|---|---|---|---|
| **OpenHands**（原 OpenDevin） | https://github.com/All-Hands-AI/OpenHands | 39k | Python | 沙箱里写/跑/测/调代码 |
| **Open Interpreter** | https://github.com/OpenInterpreter/open-interpreter | 63.8k | Python | 自然语言直接操控电脑 |
| **Aider** | https://github.com/paul-gauthier/aider | 44.8k | Python | 终端配对编程 Agent |
| **Cline** | https://github.com/cline/cline | 61.8k | TS | IDE 内编码 Agent |

---

## 四、Star 排名速览（Top 15，2026 中）
| 排名 | 仓库 | Star |
|---|---|---|
| 1 | [n8n-io/n8n](https://github.com/n8n-io/n8n) | 187.8k |
| 2 | [Significant-Gravitas/AutoGPT](https://github.com/Significant-Gravitas/AutoGPT) | 184.3k |
| 3 | [langchain-ai/langchain](https://github.com/langchain-ai/langchain) | 136.7k |
| 4 | [browser-use/browser-use](https://github.com/browser-use/browser-use) | 93.9k |
| 5 | [FlowiseAI/Flowise](https://github.com/FlowiseAI/Flowise) | 52.8k |
| 6 | [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) | 51.4k |
| 7 | [run-llama/llama_index](https://github.com/run-llama/llama_index) | 49.4k |
| 8 | [BerriAI/litellm](https://github.com/BerriAI/litellm) | 46.9k |
| 9 | [paul-gauthier/aider](https://github.com/paul-gauthier/aider) | 44.8k |
| 10 | [agno-agi/agno](https://github.com/agno-agi/agno) | 40.1k |
| 11 | [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) | 34.4k |
| 12 | [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | 33.0k |
| 13 | [microsoft/semantic-kernel](https://github.com/microsoft/semantic-kernel) | 27.9k |
| 14 | [huggingface/smolagents](https://github.com/huggingface/smolagents) | 27.3k |
| 15 | [openai/openai-agents-python](https://github.com/openai/openai-agents-python) | 26.9k |

---

## 五、推荐学习路径（由浅入深）
1. **原理祛魅**：先读 `hello-agents`（中文系统教程）→ `learn-claude-code`（看懂核心循环）。
2. **最小跑通**：用 `smolagents` 或 `CrewAI` 写一个会调用工具的小 Agent，拿到正反馈。
3. **类型与工程**：用 `Pydantic AI` 写结构化输出 Agent，理解依赖注入与测试。
4. **生产编排**：用 `LangGraph` 建模有状态 / 可恢复 / 人工介入的流程。
5. **可视化落地（可选）**：用 `Dify` / `n8n` 做前端与自动化执行层，框架只管智能核心。

> MCP 已成标配：2026 年起主流框架均原生支持 MCP 工具协议，工具层不再锁定框架，可 CrewAI 原型 → LangGraph 加固，模型随时替换。

---

## 六、避坑提示
- ⚠️ **AutoGen 已进入维护模式**（微软重定向到 Microsoft Agent Framework），新项目别选；要微软系多智能体改看 MAF，或社区分叉 **AG2**。
- ⚠️ **OpenAI Agents SDK 与 OpenAI API 强耦合**，换模型需 LiteLLM 适配；介意厂商锁定就选 LangGraph / Pydantic AI（provider-agnostic）。
- ⚠️ **LangChain 体量最大但偏 RAG/链**，复杂 Agent 现在更推荐它的子项目 **LangGraph**。
- ⚠️ **Star 高 ≠ 适合学习**：Dify/n8n/AutoGPT 体量大但是平台/产品，新手当「用」而非「学原理」。
- 你之前关注的 **Multica / Paperclip** 属「开箱即用 harness」类（拿来即用、少写码），与上面「需写代码的框架」是互补关系，不是替代。

---

## 七、基于 LangGraph 实现的开源 Agent 项目（重点补充）

> 以下均为**确属 LangGraph 实现**的开源项目，按「读源码学习价值」排序。Star 为 2026 年中近似量级。

### 7.1 LangChain 官方预构建库（最适合精读源码，理解图编排范式）
| 项目 | GitHub | Star(约) | 学什么 |
|---|---|---|---|
| **open_deep_research** | https://github.com/langchain-ai/open_deep_research | 1.6k | 迭代式网络研究 + 报告撰写，官方多智能体研究范例 |
| **langgraph-supervisor** | https://github.com/langchain-ai/langgraph-supervisor-py | 45.7k(下载) | 「主管-下属」多智能体编排，生产常用模式 |
| **langgraph-swarm** | https://github.com/langchain-ai/langgraph-swarm-py | 5.8k | 「集群式」去中心化多智能体协作 |
| **langgraph-reflection** | https://github.com/langchain-ai/langgraph-reflection | 1.4k | 在图里插入「反思/自评」步骤 |
| **langgraph-codeact** | https://github.com/langchain-ai/langgraph-codeact | 0.5k | CodeAct 范式：让 Agent 生成并执行代码而非调工具 |
| **langmem** | https://github.com/langchain-ai/langmem | 13.7k(下载) | 让 Agent 从交互中持续学习/记忆（长期记忆） |
| **langchain-mcp-adapters** | https://github.com/langchain-ai/langchain-mcp-adapters | 91.2k(下载) | 把 MCP 工具桥接进 LangGraph 代理 |

### 7.2 社区明星 Agent 项目（基于 LangGraph，体量/实战价值高）
| 项目 | GitHub | Star(约) | 定位 / 学习点 |
|---|---|---|---|
| **TradingAgents** | https://github.com/TauricResearch/TradingAgents | 82k+ | 多智能体金融交易：分析师/研究员多空辩论→交易员→风控，LangGraph 工具节点 + 双 LLM + 反思记忆。架构极适合学「有状态多智能体编排」 |
| **gpt-researcher** | https://github.com/assafelovic/gpt-researcher | 27k+ | 自主深度研究 Agent，planner/execution/publisher 多角色，官方明确「multi-agent built with LangGraph」，支持 MCP、本地文档、Deep Research 树状探索 |
| **khoj** | https://github.com/khoj-ai/khoj | 30k+ | 自托管「第二大脑」，文档/网页问答 + Agent，后端基于 LangGraph |

### 7.3 学习资源 / awesome 清单
| 资源 | GitHub | 用途 |
|---|---|---|
| **awesome-LangGraph**（生态索引） | https://github.com/gtesei/awesome-LangGraph | 概念 / 项目 / 模板 / 案例一站索引，含 Uber、Klarna、Elastic 等落地案例 |
| **awesome-langgraphjs** | https://github.com/eherrador/awesome-langgraphjs | 专收 LangGraph.js 开源项目、模板与视频（Open Agent Platform、Open Canvas、Social Media Agent 等） |

> 建议读法：先读 `open_deep_research` 或 `langgraph-supervisor` 的源码（官方范例、注释最全），再横向看 `TradingAgents` / `gpt-researcher` 如何把图编排用到真实业务，最后用 `awesome-LangGraph` 按领域找案例。

---

## 八、附：Memories 之外你可能想顺手看的
- 协议层：[Model Context Protocol (MCP)](https://github.com/modelcontextprotocol) ｜ [Anthropic Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)
- 路由网关：[LiteLLM](https://github.com/BerriAI/litellm)（统一 200+ 模型 API）
- 记忆层：[Mem0](https://github.com/mem0ai/mem0)（给 Agent 装长期记忆）
