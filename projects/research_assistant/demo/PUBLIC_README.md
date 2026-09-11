# 岗位需求 RAG 分析助手

一个有来源、可评测、能保存评估结果的 Python RAG 项目，包含岗位检索、评估后端与原生 PDF 证据问答扩展。

## 先看演示

下载并解压后，用浏览器打开根目录的 [OPEN_DEMO.html](OPEN_DEMO.html)。页面可以单独打开，**无需安装、无需服务器、无需 API Key**。

选择案例后，可以查看处理结果、展开来源、回看运行过程。前三例由本机代码实际执行后保存，PDF 例来自已有真实模型记录。演示不接受任意新问题；所有日期与范围均有说明。

## 项目包含什么

| 部分 | 代码入口 | 能力 |
|---|---|---|
| 检索与回答 | `cli.py`、`service_factory.py`、`grounded_service.py` | BM25 词法、BGE 向量、RRF 混合、Cross-Encoder 重排、引用检查与拒答 |
| 工作流 | `rag_workflow.py`、`langgraph_workflow.py` | 状态、有限重试、确认与安全结束；手写版与图版对照 |
| 岗位后端 | `backend_api.py`、`job_backend/` | SQLAlchemy/PostgreSQL、版本快照、事务、幂等重放和乐观锁复查 |
| PDF 扩展 | `pdf_cli.py`、`pdf_api.py`、`pdf_module/` | 原生文字、部分表格结构、页码坐标、范围检索与原文数值检查 |
| 评测 | `rag_evaluation.py`、`rerank_evaluation.py` | 引用/拒答回归、排序指标及候选池失败归因 |

以上代码位于 `projects/research_assistant/`。本公开整理包保留相关原创代码，未包含整套学习笔记、其他复现项目或个人求职材料。完整研究数据不随包分发，因此旧研究集指标不能仅凭这几条演示材料复算。

## 想自己运行

以下步骤是可选的；只看演示无需安装。建议 Python 3.12，在项目目录创建虚拟环境，再按需要安装依赖。

```bash
cd projects/research_assistant
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements-backend.txt
python -m unittest test_demo_replay
python demo_replay.py
```

这会用附带的少量摘录重新生成前三个案例；模型记录保持为原有历史结果，不产生新的模型调用。默认生成的 SQLite 仅用于演示，不是 PostgreSQL 生产部署验证。

### 导入自己的岗位或文本资料

自行准备有权使用的资料，不必沿用原研究数据。将 CSV 转为 Chunk JSONL 的示例：

```python
from document import load_job_csv_documents, chunk_document, write_chunks_jsonl
docs = load_job_csv_documents("data/local/jobs.csv")
chunks = [c for d in docs for c in chunk_document(d)]
write_chunks_jsonl(chunks, "data/local/chunks.jsonl")
```

CSV 可包含 `id,job_title,company,city,skills_summary,duties_summary,source_url` 等字段，字段含义见 `document.py`。普通 `.md` / `.txt` 可用 `load_text_documents`。输出包含来源和正文，应保存在自己的本地数据目录。

先用无需模型的词法与原文摘录模式运行：

```bash
python cli.py "你的问题" --strategy lexical --generator extractive --chunks-path data/local/chunks.jsonl
```

启用向量检索需要安装 Embedding 依赖，首次下载模型需要网络、磁盘和内存：

```bash
python -m pip install -r requirements-embedding.txt
python job_vector_cli.py --index-path data/index/my_vectors.json build --chunks-path data/local/chunks.jsonl
python cli.py "你的问题" --strategy vector --chunks-path data/local/chunks.jsonl --index-path data/index/my_vectors.json
```

`hybrid`、`vector_rerank`、`hybrid_rerank` 为可选策略。首次重排也需要下载模型。启用 LangGraph 对照另安装 `requirements-workflow.txt`。本包不预置模型权重或研究向量索引。

### 导入自己的 PDF

```bash
python -m pip install -r requirements-pdf.txt
python pdf_cli.py ingest path/to/your.pdf
python pdf_cli.py list
python pdf_cli.py ask --document 返回的文档ID --question "你的问题"
```

默认是原文摘录，不调用模型。当前模块不包含扫描 OCR、通用图形理解和跨页表格合并，不能把非空输出当作全文解析正确。

### 模型回答与后端

真实模型功能由使用者自行配置：通过本地环境变量设置 `MODEL_API_KEY`、`MODEL_ENDPOINT`（完整 HTTPS Chat Completions 地址）、`MODEL_NAME`，在岗位 CLI 显式使用 `--generator model`，或在 PDF CLI 显式使用 `--mode model`。API Key 不写进代码或网页；外部模型按供应商规则计费，本项目不会提供共享额度。

岗位后端需要 `DATABASE_URL` 与至少 32 个 ASCII 字符的 `BACKEND_API_TOKEN`。先运行 `python backend_cli.py migrate`，再以 `uvicorn backend_api:create_app --factory --host 127.0.0.1 --port 8011` 启动。表结构、请求模型及接口见 `job_backend/contracts.py` 和 `backend_api.py`。它与原 `/ask`、PDF API 是独立入口，尚未声称整合成一个线上平台。

## 证据与边界

- Demo 用关键词检索与确定性摘录，不能表述为实时大模型或语义向量效果。
- PDF 记录是一条历史真实调用，不表示所有问题都能答对；引用、原文和数字检查不等于语义验证。
- 后端案例包含实际写入、重复请求、复查冲突和关闭连接后重新读取，但仍是本机流程。
- 完整源码可以阅读和扩展；历史研究集没有分发，外部使用者需提供自己的数据与评测标签。
- [演示说明](projects/research_assistant/demo/README.md)与 [可读结果](projects/research_assistant/demo/replay.json)列明公开内容。

## 许可与资料

本次公开整理的原创代码按 MIT 许可提供，详见 LICENSE。第三方依赖继续适用其各自许可。两条招聘摘要及厂商说明节选仅用于展示项目证据链，权利归各来源权利人，不因本项目代码许可获得第三方材料的再许可。
