# 岗位需求 RAG 分析助手

一个证据优先、可评测、会在证据不足时拒答的 Python RAG 项目。它覆盖岗位数据导入、BM25 / 向量 / 混合检索、Cross-Encoder 重排、真实模型回答、引用校验、评测后端，以及原生 PDF 证据问答。

公开包采用“离线回放 + 自备数据的真实链路”两层交付：可以先零配置查看结果，再决定是否安装模型依赖和调用自己的付费 API。

## 30 秒体验

下载并解压后，直接用浏览器打开根目录的 [OPEN_DEMO.html](OPEN_DEMO.html)。

- 无需安装 Python、无需服务器、无需 API Key。
- 可以查看四个固定案例、来源摘录和执行步骤。
- 前三个案例由本机代码实际执行后保存；PDF 案例来自一条历史真实模型记录。
- 这是固定回放，不接受任意新问题，也不代表岗位当前仍开放。

## 项目亮点

| 能力 | 实现 | 证据边界 |
|---|---|---|
| 多路检索 | BM25 词法、BGE 向量、RRF 混合 | BM25 仅是词法基线；公开包不附完整研究索引 |
| 二阶段排序 | Cross-Encoder ReRank | 只对候选池重排，不能找回首阶段漏掉的文档 |
| Grounded 回答 | 引用编号、来源回查、无证据拒答 | 引用格式正确不等于结论天然正确 |
| 本地网页 | FastAPI + 原生 HTML，懒加载 RAG 服务 | 需要使用者自己的数据、索引和模型配置 |
| 受控真实验收 | 最多四个付费 RAG 场景，零自动重试 | 必须显式确认付费参数；不作为生产稳定性证明 |
| 可复现评测 | Recall@K、MRR、nDCG、引用/拒答回归 | 固定回归集结果不能外推为开放世界准确率 |
| 持久化后端 | SQLAlchemy、迁移、幂等重放、乐观锁复查 | 演示使用 SQLite；不能冒充 PostgreSQL 生产验证 |
| PDF 证据问答 | 页码、坐标、范围检索、原文数值检查 | 不含扫描 OCR、通用图形理解或跨页表格合并 |

核心回答链路：

```text
问题 -> BM25 / BGE -> RRF -> Cross-Encoder ReRank
     -> 证据上下文 -> OpenAI-compatible 模型 -> 引用与截断检查
     -> 带来源回答 / 安全拒答
```

## 代码位置

主要源码位于 `projects/research_assistant/`：

| 入口 | 用途 |
|---|---|
| `personal_live_web.py` | 启动最新的本地真实网页链路 |
| `personal_live_cli.py` | 在终端中连续提问，复用已加载模型 |
| `run_live_web_scenarios.py` | 显式付费确认后的受控真实场景验收 |
| `cli.py` | 词法、向量、混合和重排策略的通用命令行 |
| `service_factory.py` | 装配检索、重排、生成和市场统计服务 |
| `grounded_service.py` | Grounded 生成、引用合同、拒答和截断处理 |
| `rag_evaluation.py` | 回答、引用与拒答评测 |
| `backend_api.py`、`job_backend/` | 版本快照、保存评估、幂等与复查冲突 |
| `pdf_cli.py`、`pdf_api.py`、`pdf_module/` | 原生 PDF 证据问答扩展 |

本包只保留公开交付所需的原创代码、少量来源摘录和测试。完整岗位库、第三方 PDF、向量索引、模型缓存、日志、数据库、个人资料和密钥均未包含。

## 用自己的资料运行

只看离线演示可以跳过这一节。以下示例建议使用 Python 3.12。

### 1. 创建环境

```bash
cd projects/research_assistant
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements-live.txt
```

`requirements-live.txt` 包含网页、后端测试、Embedding 和 ReRank 所需依赖。首次使用 BGE 或重排模型时，可能需要网络、磁盘空间和较多内存。

### 2. 准备自己的岗位数据

请只使用自己有权处理的资料。CSV 可包含 `id,job_title,company,city,skills_summary,duties_summary,source_url` 等字段：

```python
from document import load_job_csv_documents, chunk_document, write_chunks_jsonl

docs = load_job_csv_documents("data/local/jobs.csv")
chunks = [chunk for document in docs for chunk in chunk_document(document)]
write_chunks_jsonl(chunks, "data/processed/jobs_enriched_chunks.jsonl")
```

普通 `.md` / `.txt` 可以使用 `load_text_documents`。然后构建默认向量索引：

```bash
python job_vector_cli.py build \
  --chunks-path data/processed/jobs_enriched_chunks.jsonl \
  --index-path data/index/jobs_enriched_vectors.json
```

如果只想先验证数据与引用，可以跳过向量索引，运行不调用模型的词法摘录模式：

```bash
python cli.py "你的问题" \
  --strategy lexical \
  --generator extractive \
  --chunks-path data/processed/jobs_enriched_chunks.jsonl
```

### 3. 配置自己的模型

复制 `projects/research_assistant/.env.example` 为同目录下的 `.env`，再填写 OpenAI-compatible Chat Completions 服务的 Key、完整 HTTPS endpoint 和模型名。也可以在当前进程中设置同名环境变量。

```dotenv
MODEL_API_KEY=
MODEL_ENDPOINT=https://your-provider.example/v1/chat/completions
MODEL_NAME=your-model-name
```

真实 Key 不应提交到 Git。项目的 `.gitignore` 会忽略 `.env`、日志、数据、索引和输出目录。

### 4. 启动网页

Windows 可以从仓库根目录运行 `START_LIVE_RAG_WEB.cmd`。其他平台或终端用户可以在项目目录运行：

```bash
python personal_live_web.py
```

默认网页地址为 `http://127.0.0.1:8000/`。服务只绑定本机回环地址；首次真正提问才加载检索模型并调用外部生成模型。个人入口固定零自动重试、单次最多输出 1500 tokens，但供应商仍可能产生费用。

仅检查配置而不加载模型、不启动服务、不调用 API：

```bash
python personal_live_web.py --check-config
```

## 受控真实场景验收

先启动网页服务，再在另一个终端运行：

```bash
python run_live_web_scenarios.py \
  --execute-real-api \
  --base-url http://127.0.0.1:8000
```

没有 `--execute-real-api` 时，运行器会直接拒绝执行。完整套件最多发起四个付费 RAG 请求，自动重试为 0，另有一个不调用模型的确定性市场统计场景。报告写入被 Git 忽略的 `output/`。

2026-09-26 的一次真实环境验收中，三个 RAG 场景生成了完整带引答案，一个复杂 Top-5 比较因输出达到上限而安全失败；确定性市场统计没有模型 usage。完整过程、费用口径和未验证边界见 [真实场景验收记录](docs/LIVE_WEB_VALIDATION_2026-09-26.md)。这不是开放世界准确率、并发、生产 SLO 或长期费用证明。

## 测试与净室验证

在项目目录运行公开包单元测试：

```bash
python -m unittest \
  test_demo_replay \
  test_personal_live_cli \
  test_personal_live_web \
  test_live_web_scenarios
```

重新生成离线演示不会发起网络连接或新的模型调用：

```bash
python demo_replay.py
```

从仓库根目录可以重新导出一个只含白名单文件的目录与 ZIP，并在独立临时目录中验证清单哈希、Python 语法、README 链接、测试、演示重建、导入、自备 CSV 路径和词法 CLI：

```bash
python tools/export_replay_project.py output/public-release
python tools/validate_replay_export.py output/public-release.zip
```

净室验证不会安装新依赖、不会打开浏览器、不会调用真实模型。发布前仍应在目标机器上按依赖文件重新安装并验证。

## PDF 扩展

```bash
python -m pip install -r requirements-pdf.txt
python pdf_cli.py ingest path/to/your.pdf
python pdf_cli.py list
python pdf_cli.py ask --document 返回的文档ID --question "你的问题"
```

默认使用原文摘录，不调用模型。需要模型回答时显式使用 `--mode model`。非空输出不能当作 PDF 全文解析正确的证明。

## 后端扩展

岗位后端需要 `DATABASE_URL` 与至少 32 个 ASCII 字符的 `BACKEND_API_TOKEN`：

```bash
python backend_cli.py migrate
uvicorn backend_api:create_app --factory --host 127.0.0.1 --port 8011
```

它与只读 `/ask`、PDF API 是独立入口，尚未整合为一个公网多用户平台。

## 许可与第三方资料

本公开包中的原创代码按 [MIT License](LICENSE) 提供。第三方依赖继续适用各自许可。演示中的两条招聘摘要和厂商说明节选只用于展示证据链，权利归各来源权利人；MIT 许可不构成对第三方材料的再许可。
