# 🏥 AI 智能健康管家 (AI Health Agent)

基于 LangGraph 与多模态大模型应用架构与智能体（Agent）开发的企业级全栈实训项目。本项目构建了一个支持多模态视觉识别、RAG 检索膳食指南检索以及多工具协同的健康管理智能体，致力于为用户提供科学的饮食分析、营养估算与运动恢复建议。

## ✨ 核心特性 (Core Features)

- **多模态视觉识别 (Multi-modal Vision)**
  - 接入智谱 GLM-4.6V-Flash 视觉大模型，支持用户上传食物图片，自动精准识别食材内容并无缝接入 Agent 推理流。
- **ReAct 智能体架构 (LangGraph Agent)**
  - 摒弃了传统的单向链式调用，采用 LangGraph 的 `StateGraph` 构建具备“感知-规划-行动”能力的循环状态机。
  - 挂载 3 大自定义工具（Tool）：`calculate_nutrition`（营养估算）、`evaluate_recovery_score`（恢复评估）、`search_health_guidelines`（文献检索）。
- **RAG 检索增强生成 (Retrieval-Augmented Generation)**
  - 集成 ChromaDB 向量数据库与 `BAAI/bge-small-zh-v1.5` 轻量级中文嵌入模型。
  - 基于《中国居民膳食指南》等权威文献，解决大模型在专业健康领域的“幻觉”问题（LLM as a Judge）。
- **长期记忆与会话隔离 (Memory & Session Management)**
  - 后端基于 `MemorySaver` 实现 `thread_id` 级别的多会话状态隔离。
  - 前端基于 HTML5 `sessionStorage` 实现聊天记录与用户长期目标的“刷新防丢”及“短期阅后即焚”。
- **企业级全栈工程化 (Enterprise Engineering)**
  - **后端**：FastAPI + SSE (Server-Sent Events) 实现极速流式输出。
  - **前端**：Tailwind CSS 纯原生无框架构建，支持 Markdown 实时渲染、动态图片 Lightbox 缩放及全套响应式交互。
  - **部署**：提供标准的 `Dockerfile`，支持一键容器化交付。

## 🏗️ 系统架构设计 (Architecture)

系统采用经典的**“前端 UI 层 - 后端 API 层 - 大模型推理层”**三层架构：

- **前端 UI**：Tailwind CSS + HTML5 (SessionStorage 状态流转)
- **后端服务**：FastAPI + Pydantic
- **AI 推理层**：DeepSeek-V4-Pro (主脑) + GLM-4.6V-Flash (视觉) + BAAI/bge-small-zh (Embedding)
- **编排与存储**：LangChain + LangGraph + ChromaDB

```text
[ 用户端 Browser ]
   │  ▲
   │  │ (HTTP / SSE 流式响应)
   ▼  │
[ FastAPI 后端服务 ] ──(Base64)──> [ 视觉大模型 GLM-4V ] (图片识别拦截)
   │  ▲
   │  │ (状态更新与编排)
   ▼  │
[ LangGraph 智能体 (DeepSeek V4) ]
   ├──> Tool 1: 营养估算器
   ├──> Tool 2: 恢复评分器
   └──> Tool 3: RAG 检索器 <====> [ ChromaDB 向量库 ]

```
## 🛠️ 快速启动 (Quick Start)

1. 环境准备
项目根目录需新建 .env 文件并配置相关模型密钥：

# DeepSeek API (主脑 LLM)
```ini
DEEPSEEK_API_KEY=your_deepseek_key_here
# 智谱 API (视觉多模态)
ZHIPU_API_KEY=your_zhipu_key_here

2. 知识库构建 (RAG)
首次运行前，需初始化本地向量数据库：

python rag_builder.py --build

3. 本地开发运行 (Local Development)

pip install -r requirements.txt
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

访问 http://127.0.0.1:8000 即可体验。

4. Docker 容器化部署 (Docker Deployment)
项目已内置标准的 Debian-slim 基础环境配置，支持一键打包与运行：

# 构建镜像
docker build -t health-agent .

# 启动容器
docker run -d -p 8000:8000 --name my-health-agent health-agent
```
📁 目录结构 (Directory Structure):

详见根目录文件 Structure.md


🛡️ 安全与边界控制
系统在 Agent 层面加入了严格的防越狱（Jailbreak）机制。针对非健康/运动/营养相关的用户提问（如编程、政治、金融等），Agent 将触发系统级防御并拒绝回答，确保垂直应用场景的专业性与合规性。

### 💡 常见问题 (FAQ)

**Q: Docker 容器启动后，构建知识库卡住或报错网络连接失败？**
A: 本项目首次构建知识库时，会下载 `BAAI/bge-small-zh-v1.5` 嵌入模型。国内用户如果遇到 Hugging Face 连通性问题，请在宿主机或容器内设置镜像源环境变量后重试：
`export HF_ENDPOINT=https://hf-mirror.com`
*(如果是在 Windows PowerShell，请使用 `$env:HF_ENDPOINT="https://hf-mirror.com"`)*
