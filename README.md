# news-auto-agent

每日科技新闻 Deep Agents 应用，包含两条流水线：

- **RSS 博客 Agent**（`--type tech`）：主 Agent 从 RSS 候选中选定 `count` 个主题，各委派写作 SubAgent 联网取材并落盘一篇中文技术博客。
- **热点解读 Agent**（`--type news`）：主 Agent 从热点新闻文件中选出最有价值的 `count` 则新闻，各委派一次写作 SubAgent 生成一篇中文热点解读。

```bash
uv run cli.py --type tech --count 1   # 技术博客：选 1 题 -> 1 篇博客
uv run cli.py --type news --count 2   # 热点解读：选 2 则 -> 2 篇解读
```

## 工作流程

RSS 博客 Agent：

1. 读取 RSS 聚合文件（`--file` 或 `RSS_FILE`），预处理为候选列表。
2. 主 Agent（选题模型）选定 `count` 个主题并输出结构化选题结论（JSON，每个主题含大纲）。
3. 写作 SubAgent（写作模型）使用 `web_search` / `web_fetch` 联网取材，通过 `resolve_output_path` 规划路径并写入 `OUTPUT_DIR`。
4. 代码层兜底：强制 `<date>-` 文件名前缀；联网工具全部不可用时在文章中补降级警告与参考链接。

热点解读 Agent：

1. 读取热点新闻文件（`--file` 或 `HOTSPOTS_FILE`），按来源限流（每来源至多 5 条、合计 top-24）生成候选列表。
2. 主 Agent（选题模型）选出恰好 `count` 则最有价值的热点新闻（时效性、影响力、AI/科技相关性、大众关注度）。
3. 对每个选题分别委托 `hotspot_writer` 写作 SubAgent：有原文链接用 `web_fetch` 抓取后总结，无链接或抓取失败用 `web_search` 搜索补充，产出热点解读 Markdown。
4. 主 Agent 输出结构化选稿结论（`{"picks": [...]}`，数量必须等于 `count`，不符自动重试一次）；代码层兜底与 RSS 博客 Agent 相同（日期前缀、frontmatter、联网降级标记）。

## 环境要求

- Python >= 3.13，使用 [uv](https://docs.astral.sh/uv/) 管理依赖
- 本地 [Ollama](https://ollama.com/) 服务运行中（默认 `http://localhost:11434`），且已 `ollama pull` 配置中指定的模型（`SELECT_MODEL` / `WRITE_MODEL`）
- 本地联网服务：[searxng-crawl4ai](D:/project/private/new_life/code/local_flow/searxng-crawl4ai)（SearXNG + Crawl4AI），
  默认 `http://127.0.0.1:8000`；`cli.py` 会自动 `docker compose up -d` / `stop`，无需手工启停
  （需 Docker Desktop 在运行；不再需要 ollama.com 账号或 `OLLAMA_API_KEY`）
- 可选：DeepSeek API key（当 `SELECT_PROVIDER` / `WRITE_PROVIDER` 设为 `deepseek` 时必填）
- 可选：LangSmith API key（启用追踪）

## 快速开始

```bash
# 1. 安装依赖（自动创建 .venv）
uv sync

# 2. 准备配置
cp .env.example .env   # Windows: copy .env.example .env
# 编辑 .env，按需填写 DEEPSEEK_API_KEY / LOCAL_WEB_* 等

# 3. 确认输入文件存在（默认 RSS：sample/2026-09-03_tech_rss.json，热点：sample/2026-09-03_news_hotspots.json，
#    可用 --file 或 RSS_FILE / HOTSPOTS_FILE 修改）

# 4. 运行（统一 CLI）
uv run cli.py --type tech --count 1              # 技术博客：选 1 题 -> 1 篇博客
uv run cli.py --type news --count 2              # 热点解读：选 2 则 -> 2 篇热点解读
uv run cli.py --type 新闻 --file sample/2026-09-03_news_hotspots.json --count 3

# main.py / main_hotspots.py 仍可用，等价于上面前两条命令
```

`--type` 支持别名：`tech`/`技术`/`blog`，`news`/`新闻`/`热点`；`--count` 缺省为 tech=1、news=2；`--file` 缺省取 `RSS_FILE` / `HOTSPOTS_FILE`。

运行成功后，终端打印选题结论与文章路径，文章写入 `output/<date>-<主题slug>.md`（重名自动追加 `-2`、`-3`）。退出码：0 成功，1 失败（错误日志见终端输出；选题解析失败时对话记录转储到 `output/debug_transcript.txt`）。writer 产出不足 `--count` 个文件即报错退出。

## 配置说明（.env）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | 本地 Ollama 服务地址 |
| `OLLAMA_API_KEY` | 空 | 已不再用于联网取材；仅当 `WEB_TOOLS_OLLAMA_FALLBACK=1` 时作为兜底 |
| `LOCAL_WEB_BASE_URL` | `http://127.0.0.1:8000` | 本地联网封装层地址（web_search / web_fetch 都打这里） |
| `LOCAL_WEB_COMPOSE_FILE` | searxng-crawl4ai 的 compose 绝对路径 | `cli.py` 用它自动启动/停止容器 |
| `LOCAL_WEB_AUTOSTART` | `1` | 1=运行前起容器、跑完停容器；0=不管容器 |
| `LOCAL_WEB_KEEPALIVE` | `0` | 1=跑完不停容器（调试用） |
| `LOCAL_WEB_API_TOKEN` | 空 | 封装层设了 `API_TOKEN` 时填同值 |
| `LOCAL_WEB_START_TIMEOUT` | `150` | 等待 `/health` 就绪的秒数 |
| `WEB_TOOLS_OLLAMA_FALLBACK` | `0` | 本地服务失败时是否回退 Ollama 云端联网（额度已用完，默认关） |
| `SELECT_MODEL` / `WRITE_MODEL` | `qwen3.5:9b` / `qwen3.8:latest` | 选题 / 写作模型（provider=ollama 时生效，需本地已拉取） |
| `SELECT_PROVIDER` / `WRITE_PROVIDER` | `ollama` | 模型提供商：`ollama`（本地）或 `deepseek`（云端） |
| `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL` | — | provider=deepseek 时使用 |
| `SELECT_NUM_CTX` / `WRITE_NUM_CTX` | `16384` / `32768` | 模型上下文长度 |
| `SELECT_TOP_N` | `20` | 送给选题模型的 RSS 候选条数（越小越省 token） |
| `SELECT_SUMMARY_LIMIT` | `120` | 每条候选摘要截断字数 |
| `SELECT_HOTSPOT_TOP_N` | `18` | 送给选题模型的热点候选条数 |
| `SELECT_TEMPERATURE` / `WRITE_TEMPERATURE` | `0.2` / `0.7` | 选题 / 写作温度 |
| `NUM_PREDICT` | `4096` | 写作最大生成 token 数 |
| `RSS_FILE` | `sample/2026-09-03_tech_rss.json` | RSS 聚合文件（相对项目根目录） |
| `HOTSPOTS_FILE` | `sample/2026-09-03_news_hotspots.json` | 热点新闻文件（热点 Agent 输入，相对项目根目录） |
| `OUTPUT_DIR` | `output` | 博客输出目录 |
| `THREAD_ID` | `news-blog` | deepagents thread_id |
| `HOTSPOT_THREAD_ID` | `news-hotspots` | 热点 Agent 的 deepagents thread_id |
| `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` | 空 | 有值则启用 LangSmith 追踪 |

## DeepSeek token 优化

DeepSeek 只负责**选题**，每天 **2 次调用**（技术 1 次 + 热点 1 次），写作全部由本地 Ollama 完成。

1. **直连 JSON 输出**：`src/selection.py` 单次 `ChatOpenAI` + `response_format=json_object`，
   不走 deepagents，省掉系统提示词、文件系统/task 工具 schema 和多轮 tool 往返。
2. **关闭 thinking**：选题只需排序 + 写理由，默认 `reasoning_effort=none`；输出 token 从 1,478 降到 298。
   需要更深的判断时在 `.env` 设 `SELECT_REASONING_EFFORT=medium`。
3. **候选压缩**：RSS 30→20 条、热点 24→18 条，摘要 200→120 字、标题 160→120 字；
   prompt 从约 1.5 万字符降到约 6.8k（`SELECT_TOP_N` / `SELECT_SUMMARY_LIMIT` 可调）。
4. **结果回填**：选完按链接/标题回填 title/link/source/published_at，避免模型改写字段触发校验重试。
5. **写作不经过 DeepSeek**：`run_writer` 直接调本地 writer Agent，写作结果不再回传给主 Agent。
6. **兜底**：DeepSeek 调用失败时自动回退本地模型 Agent 流程，保证当天仍能出稿。

实测（2026-11-11，同一批数据）：

| 指标 | 优化前 | 优化后 |
|---|---|---|
| 技术选题 tokens | 18,132（输入 16,078 / 输出 2,054，2 次调用） | **3,126**（输入 2,828 / 输出 298，1 次调用） |
| 热点选题 tokens | 未单独计数 | **2,670**（输入 2,405 / 输出 265，1 次调用） |
| 一天合计 | 约 3 万（按同结构推算） | **5,796** |
| 技术选题 + 生成耗时 | 3m45s | 约 2m |
| 送给模型的候选 | 30 条 × ~400 字 | 20 条 × ~300 字 |
## 项目结构

```
cli.py             # 统一 CLI：--type tech|news --file <json> --count N
main.py            # 等价于 cli.py --type tech --count 1
main_hotspots.py   # 等价于 cli.py --type news --count 2
src/
  config.py        # .env 加载与默认值（Settings）
  preprocess.py    # RSS / 热点文件加载与候选预处理（load_rss / load_hotspots）
  agents.py        # Deep Agents 主 Agent / 写作 SubAgent 构建、结论解析、数量校验与重试
  tools.py         # web_search / web_fetch（本地服务）/ resolve_output_path 工具
  local_web.py     # 本地联网客户端 + docker compose 自动启停
  prompts.py       # 主 Agent 与写作 SubAgent 系统提示词（博客 / 热点两套，按 count 生成）
  models.py        # 选题结论、热点选稿等数据模型（picks 数量由运行层 count 校验）
  output.py        # 输出路径规划、文件名兜底、降级标记
sample/            # RSS / 热点示例输入
output/            # 生成的博客文章与热点解读
tests/             # pytest 测试（联网类测试需本地 Ollama 在线）
```

## 常见问题

- **`pull model manifest: 404`**：本地没有拉取配置的模型，执行 `ollama pull <SELECT_MODEL/WRITE_MODEL>`。
- **联网搜索失败/降级**：先确认 Docker Desktop 在运行，再手动跑 `docker compose -f "D:\project\private\new_life\code\local_flow\searxng-crawl4ai\docker-compose.yml" up -d`，浏览器打开 `http://127.0.0.1:8000/health` 应返回 `status: ok`；`docker compose logs -f api` 看具体报错。降级触发时文章会带降级警告行，运行日志有 WARNING。
- **两次尝试均未获得合法 BlogSelectionResult / HotspotSelectionResult**：本地小模型输出不稳定，查看 `output/debug_transcript.txt` 排查；可换更大模型或调低 `SELECT_TEMPERATURE`。
- **picks 数量与要求 N 不一致**：主 Agent 返回的选稿数与 `--count` 不符（重试一次后仍不符则退出），可调低 `--count` 或换更大选题模型。
- **`writer 应产出 N 个 Markdown 文件，实际找到 M 个`**：写作 SubAgent 未全部落盘，查看上文 WARNING 日志与 `output/debug_transcript.txt`。
- **`DEEPSEEK_API_KEY 未设置`**：任一 provider 设为 `deepseek` 时必须填写。
- **运行测试**：`PYTHONPATH=. uv run pytest tests/`（联网类测试需本地 Ollama 在线）。
