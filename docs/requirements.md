# 需求分析：news-auto-agent 技术博客自动生成

## 1. 背景与目标

从每日科技 RSS 聚合 JSON 中自动选出 1 个「最便于分享」的主题，扩写成一篇中文技术博客文章（Markdown 文件）。

输入样例：`sample/2026-09-03_tech_rss.json`（8 个来源、172 条，结构为 `date` / `meta` / `sources[]` / `sources[].items[]`，每条含 `title`、`link`、`source`、`published_at`、`summary`、`category`；Hacker News 条目的 `summary` 中含 `Points` 与 `# Comments` 热度信息，arXiv 条目为论文摘要）。

## 2. 总体架构

使用 Deep Agents 框架（`create_deep_agent`），组织为 **1 个主 Agent + 1 个写作 SubAgent**：

```
main.py
  ├── 代码层预处理：加载 RSS JSON，为每条提取 title/link/source/published_at/
  │   summary 前 200 字/HN Points/HN Comments（arXiv 条目先剥掉 "arXiv:... Abstract:"
  │   元数据前缀再截取），按 HN Points 降序取 top-30，生成紧凑候选列表
  │   （纯 Python，不走 LLM；30 × ~400 字符 ≈ 12KB，适配 8k 上下文）
  └── 主 Agent（选题，SELECT_MODEL）
        ├── 基于候选列表按 热度 × 普适性 × 可展开性 评分，选 1 个主题
        ├── 输出结构化选题结论（SelectionResult Pydantic schema，见 §3 FR1）
        └── 通过内置 task 工具委派写作 SubAgent（WRITE_MODEL）
              ├── web_search / web_fetch 联网补充素材（本地 SearXNG + Crawl4AI）
              └── write_file 落盘 Markdown 博客
```

两个 Agent 均使用本地 Ollama 模型，通过 `langchain-ollama` 的 `ChatOllama` 接入。模型名以本机 `ollama list` 实际存在的 tag 为准（当前为 `qwen3.5:9b`、`qwen3.8:latest`），通过 `.env` 配置。

预处理必须在代码层完成：原始 JSON 约 100KB（>32k token），不能整文件喂给模型。

## 3. 功能需求

- **FR1 选题**：主 Agent 基于预处理后的候选列表（按 HN Points 降序取 top-30），按 热度（HN points/comments，权重最高）× 普适性（面向开发者大众；纯学术论文条目降权，除非有大众化解读角度）× 可展开性（有公开素材可写成文章）综合评估，只选 1 个主题。输出为结构化 `SelectionResult`（Pydantic）：
  ```python
  class SelectionResult(BaseModel):
      topic_title: str        # 选中条目标题
      link: str               # 原文链接
      source: str             # 来源
      published_at: str       # 发布时间
      selection_reason: str   # 选题理由
      outline: list[str] = Field(min_length=5, max_length=8)  # 文章大纲（5-8 个要点）
  ```
  实现方式：优先使用 `create_deep_agent` 支持的 `response_format=SelectionResult`；若所用 deepagents 版本不支持，则由主 Agent 输出 JSON 字符串，代码层用 `SelectionResult.model_validate_json` 校验，失败时把校验错误回喂给主 Agent 重试一次，仍失败则退出（非零退出码）。
- **FR2 委派**：主 Agent 通过内置 `task` 工具把「主题 + 大纲 + 原文链接」交给写作 SubAgent；写作完成后主 Agent 向用户汇报选题结论与文件路径。
- **FR3 写作**：写作 SubAgent 将选题展开为中文技术博客。素材优先级：原文内容（`web_fetch` 原文链接）> 联网搜索（`web_search`）> RSS 摘要。素材降级策略：`web_search` 失败仅 warning 并继续；`web_fetch` 失败也仅 warning 并继续；两者全部失败时仅当 RSS 摘要非空才允许生成，且文首必须加警告行 `[警告：联网工具不可用，本文仅基于 RSS 摘要生成]`。
- **FR4 输出**：生成 Markdown 文件到 `OUTPUT_DIR`，命名规则：`<date>-<slug>.md`。
  - `date` 取 RSS JSON 顶层 `date` 字段；缺失时取当天日期。
  - `slug`：主题英文译名（或拼音），小写字母/数字/短横线，3–40 字符；同日期冲突时追加 `-2`、`-3`。
  - `OUTPUT_DIR` 相对项目根目录，自动 `mkdir -p`；文件已存在时覆盖写。

**文章质量验收 checklist**（写作 SubAgent 的系统提示须明确要求，验收时逐条核对）：
- [ ] 中文，目标字数 ≥ 1500 字
- [ ] 结构：H1 标题、引言（为什么值得关注）、≥ 2 个 H2 核心章节、影响与评价、参考链接
- [ ] 文末参考链接 ≥ 3 条（含原文链接），以 Markdown 列表给出
- [ ] 技术事实须来自素材，不得编造具体数字/版本/引语

## 4. 配置（全部通过 .env）

实现时提供 `.env.example`；`.env` 缺失时**使用代码内置默认值（与 .env.example 一致）**，不加载 `.env.example` 本身。`.env` 已加入 `.gitignore`。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | 本地 Ollama 地址 |
| `LOCAL_WEB_BASE_URL` | `http://127.0.0.1:8000` | 本地联网封装层（SearXNG + Crawl4AI），`web_search`/`web_fetch` 都打这里 |`r`n| `LOCAL_WEB_COMPOSE_FILE` | searxng-crawl4ai compose 路径 | 供 `cli.py` 自动 `docker compose up -d` / `stop` |`r`n| `LOCAL_WEB_AUTOSTART` | `1` | 1=自动启停容器；0=直接用现有服务 |
| `SELECT_MODEL` | `qwen3.5:9b` | 选题 Agent 模型（`SELECT_PROVIDER=ollama` 时生效） |
| `WRITE_MODEL` | `qwen3.8:latest` | 写作 SubAgent 模型（`WRITE_PROVIDER=ollama` 时生效） |
| `SELECT_PROVIDER` / `WRITE_PROVIDER` | `ollama` | 模型提供商，可选 `ollama`（本地）或 `deepseek`（OpenAI 兼容 API），两个 Agent 可独立配置 |
| `DEEPSEEK_API_KEY` | 空 | 任一 provider 为 `deepseek` 时必填 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | DeepSeek API 地址 |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | DeepSeek 模型名（`deepseek-v4-flash` / `deepseek-chat` / `deepseek-reasoner` 等，以账号可用为准）；云端模型上下文充足，`NUM_CTX` 不适用，`NUM_PREDICT` 映射为 `max_tokens` |
| `SELECT_NUM_CTX` | `16384` | 选题模型上下文。实测 system 提示 + 8 个工具 schema + top-30 候选 ≈ 7.4k tokens（pi 的 N1 估算只算了候选，漏了工具 schema），8k 只剩 ~800 token 生成余量会截断输出，故定为 16384 |
| `WRITE_NUM_CTX` | `32768` | 写作模型上下文（联网搜索结果较长，≥32k） |
| `SELECT_TEMPERATURE` | `0.2` | 选题温度（稳定） |
| `WRITE_TEMPERATURE` | `0.7` | 写作温度（创造性） |
| `NUM_PREDICT` | `4096` | 写作最大生成 token 数 |
| `RSS_FILE` | `sample/2026-09-03_tech_rss.json` | 输入文件，相对项目根目录；不存在则报错退出（非零退出码） |
| `OUTPUT_DIR` | `output` | 输出目录，相对项目根目录，自动创建 |
| `THREAD_ID` | `news-blog` | deepagents checkpointer 的 thread_id |

可选：`LANGSMITH_API_KEY` 有值即启用 LangSmith 追踪；`LANGSMITH_PROJECT` 可选设置项目名（不设置则用 default）。两者均无值时跳过追踪。

## 5. 技术要点与依赖

- 依赖（写入 `pyproject.toml`）：`deepagents`、`langchain`（>=1.0,<2.0）、`langchain-core`（>=1.0,<2.0）、`langsmith`（>=0.3.0）、`langchain-ollama`、`langchain-openai`（DeepSeek 等 OpenAI 兼容 API）、`ollama`、`python-dotenv`。
- 模型提供商可配置（`SELECT_PROVIDER` / `WRITE_PROVIDER`）：`ollama` 走本地 `ChatOllama`；`deepseek` 走 `ChatOpenAI`（base_url 指向 DeepSeek 的 OpenAI 兼容端点）。联网搜索与抓取由本地 SearXNG + Crawl4AI 封装层提供（`LOCAL_WEB_*`），与 LLM 提供商无关；`cli.py` 在运行前后自动启停 docker 容器。
- 本地封装层的 `/api/web_search`、`/api/web_fetch`（与 Ollama 同形）包装为 LangChain tool：返回值序列化为字符串（截断至 ≤4000 字符）；失败返回 `[web_search 失败]` / `[web_fetch 失败]` 前缀，由 FR3 兜底。
- 主 Agent 配置 `FilesystemBackend(root_dir=PROJECT_ROOT, virtual_mode=True)` 与 `MemorySaver` checkpointer，`thread_id` 取 `THREAD_ID`。
- **路径基准**：所有相对路径（`RSS_FILE`、`OUTPUT_DIR`、`FilesystemBackend.root_dir`）均以项目根目录（`main.py` 所在目录，`Path(__file__).parent` 解析）为基准解析，不依赖运行时的 cwd。
- 代码组织：入口 `main.py`，agent 与工具定义放 `src/` 下的模块（用 `uv` 脚本方式运行，无需打包配置；`main.py` 直接 import `src` 下模块或用 flat 包结构，由实施者按 uv 惯例处理）。
- 错误处理：web 工具异常统一捕获并 warning（含原因）；LLM 输出非法 JSON 按 FR1 重试一次；RSS 文件缺失/JSON 解析失败 → 报错退出（非零退出码）。

## 6. 验收标准

1. `uv run main.py` 可端到端跑通，生成 `output/2026-09-03-<slug>.md`，内容满足 §3 文章质量 checklist（含字数、章节、参考链接逐条核对）。
2. 本地联网服务不可用（Docker 未启动/容器异常）时程序不崩溃：web 工具调用失败被捕获，按 FR3 降级策略继续生成（若 RSS 摘要可用），stdout 打印 warning。
3. `.env` 不存在时使用内置默认值，行为与有 `.env` 时一致。
4. `RSS_FILE` 指向不存在的文件时，程序清晰报错并以非零退出码退出。
