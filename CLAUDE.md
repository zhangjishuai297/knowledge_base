# CLAUDE.md

本文档指导 Claude Code（claude.ai/code）在此仓库中工作时遵循的项目约定和上下文。

## 项目概述

**掌柜智库** — 基于 RAG（检索增强生成）的知识库问答系统。功能分为两大流程：

1. **导入流程**：上传 PDF 产品手册 → 解析为 Markdown → 分块 → 生成向量嵌入 → 存入 Milvus 向量库
2. **查询流程**：用户自然语言提问 → 多路检索（稠密+稀疏混合检索、HyDE 假设文档检索、MCP 网络搜索）→ RRF 融合 → BGE 重排序 → LLM 生成答案

## 技术栈

- **Python 3.11**，包管理器：`uv`
- **FastAPI** + **Uvicorn**（两个独立 API 服务）
- **LangGraph 0.2.50** — 两个 StateGraph 工作流（导入管线 & 查询管线）
- **LangChain 0.3.23** — ChatOpenAI 客户端（DashScope/Qwen 后端）
- **LLM**：阿里云 DashScope（`qwen3.6-27b` 文本，`qwen3.7-plus` 视觉）
- **Embedding 模型**：BGE-M3（稠密+稀疏，1024维），本地加载
- **Reranker**：BGE Reranker Large（FlagEmbedding）
- **向量数据库**：Milvus（pymilvus 3.0.0），两个集合：`kb_chunks`（文档块）、`kb_item_names`（产品名向量）
- **对象存储**：MinIO（上传文件、提取的图片）
- **聊天历史**：MongoDB（`kb002.chat_message` 集合）
- **PDF 解析**：MinerU API（阿里云/openXlab REST API）
- **网络搜索**：阿里云百炼 MCP（Model Context Protocol，SSE 协议 WebSearch 工具）
- **日志**：Loguru（控制台 + 按天轮转文件）
- **流式响应**：FastAPI StreamingResponse（SSE）

## 项目结构

```
app/
├── clients/                  # 数据库/存储客户端（Milvus, MinIO, MongoDB, Neo4j）
├── conf/                     # 配置数据类（.env → dataclass 自动加载）
├── core/                     # 日志器、提示词模板加载器
├── import_process/           # 文档导入管线（LangGraph）
│   ├── agent/
│   │   ├── main_graph.py     # 7节点 StateGraph 定义
│   │   ├── state.py          # ImportGraphState TypedDict
│   │   └── nodes/            # 每节点一个文件（入口, PDF→MD, 图片处理, 分块, 产品名识别, Embedding, 导入Milvus）
│   └── api/
│       └── file_import_service.py  # FastAPI 应用：文件上传 + 状态轮询
├── query_process/            # 查询/RAG 管线（LangGraph）
│   ├── agent/
│   │   ├── main_graph.py     # 多路检索 StateGraph（混合搜索 + HyDE + MCP网络搜索）
│   │   ├── state.py          # QueryGraphState TypedDict
│   │   └── node/             # 每节点一个文件（产品名确认, 向量搜索, HyDE, 网络搜索, KG, RRF, Rerank, 答案生成）
│   └── api/
│       └── query_service.py  # FastAPI 应用：/query 端点, SSE 流式, 聊天历史
├── lm/                       # LLM/Embedding/Reranker 工具函数
├── tool/                     # 模型下载脚本（BGE-M3, Reranker）
└── utils/                    # 杂项工具（字符串转义, 限流, SSE, 路径, 格式化等）
prompts/                      # LLM 提示词模板文件（.prompt 格式）
```

## 两个 LangGraph 工作流

### 导入管线（线性 + 条件分支）

```
node_entry（验证文件类型 PDF/MD）
  → node_pdf_to_md（MinerU PDF→MD 转换）[或] node_md_img（图片上传 + VLM 描述）
  → node_document_split（递归标题感知分块）
  → node_item_name_recognition（LLM 提取产品名 → 写入 Milvus）
  → node_bge_embedding（BGE-M3 稠密+稀疏向量生成）
  → node_import_milvus（导入 kb_chunks 集合）
```

### 查询管线（扇出/扇入）

```
node_item_name_confirm（LLM 识别产品名 + Milvus 向量匹配）
  →（命中缓存答案？）→ node_answer_output [提前退出]
  → 多路并行检索：
      ├── node_search_embedding（稠密+稀疏混合检索）
      ├── node_search_embedding_hyde（HyDE 假设文档检索）
      └── node_web_search_mcp（百炼 MCP 网络搜索）
  → node_rrf（互惠排名融合 — 暂为占位桩）
  → node_rerank（BGE 交叉编码器重排序 — 暂为占位桩）
  → node_answer_output（LLM 最终答案生成）
```

**注意**：`node_rrf`、`node_rerank`、`node_answer_output`、`node_query_kg` 目前为占位桩（仅 `time.sleep(1)`），需实现真实逻辑。

## 如何运行

### 启动导入服务（上传界面位于 /）
```bash
uv run python app/import_process/api/file_import_service.py
```
FastAPI 监听 `http://127.0.0.1:8000`

### 启动查询服务（聊天界面位于 /）
```bash
uv run python app/query_process/api/query_service.py
```
FastAPI 监听 `http://0.0.0.0:8000`

### 运行单个节点测试（各节点文件自带 __main__ 块）
```bash
uv run python app/import_process/agent/nodes/node_entry.py
uv run python app/query_process/agent/node/node_search_embedding.py
```

### 运行端到端管线测试
```bash
uv run python app/import_process/agent/main_graph.py
uv run python app/query_process/agent/main_graph.py
```

### 安装/同步依赖
```bash
uv sync
```

## 配置说明

所有配置位于项目根目录 `.env`（已 gitignore）。`app/conf/` 中的配置数据类通过 `python-dotenv` 自动加载。关键配置项：
- DashScope API Key（LLM + VLM）
- Milvus 连接地址与集合名
- MinIO 端点/凭证/存储桶
- MongoDB 连接串与数据库名
- MinerU PDF 解析 API Token
- BGE-M3 模型路径与运行设备
- MCP/百炼网络搜索配置

## 开发注意事项

- **语言**：所有提示词、日志、注释、UI 均为中文。LLM 提示词模板位于 `prompts/*.prompt`，通过 `app/core/load_prompt.py` 加载。
- **测试**：无 pytest/unittest 框架。每个节点文件和两个 `main_graph.py` 均包含 `if __name__ == "__main__":` 测试块。`test/` 目录下的测试脚本已被 gitignore。
- **占位节点**：`node_rrf`、`node_rerank`、`node_answer_output`、`node_query_kg` 尚未实现，当前仅含 `time.sleep(1)`。
- **已知重复**：`app/import_process/config/mineru_config.py` 与 `app/conf/mineru_config.py` 功能重复（字段名不一致：`api_key` vs `api_token`）；`mongo_history_utils_new.py` 是 `mongo_history_utils.py` 的扩展版本。
- **模型权重**：BGE-M3 和 BGE Reranker 通过 `app/tool/` 下的 ModelScope 脚本下载，本地路径在 `.env` 中配置。
- **LangGraph 版本固定**：`langgraph==0.2.50`，升级时注意 API 兼容性。
