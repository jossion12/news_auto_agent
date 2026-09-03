# news-auto-agent

每日科技 RSS 自动选题并生成中文技术博客的 Deep Agents 应用：主 Agent 从 RSS 候选中选定一个主题，委派写作 SubAgent 联网取材并落盘 Markdown 文章。

## 工作流程

1. 读取 RSS 聚合文件（`RSS_FILE`），预处理为候选列表。
2. 主 Agent（选题模型）选定主题并输出结构化选题结论（JSON，含大纲）。
3. 写作 SubAgent（写作模型）使用 `web_search` / `web_fetch` 联网取材，通过 `resolve_output_path` 规划路径并写入 `OUTPUT_DIR`。
4. 代码层兜底：强制 `<date>-` 文件名前缀；联网工具全部不可用时在文章中补降级警告与参考链接。

## 环境要求

- Python >= 3.13，使用 [uv](https://docs.astral.sh/uv/) 管理依赖
- 本地 [Ollama](https://ollama.com/) 服务运行中（默认 `http://localhost:11434`），且已 `ollama pull` 配置中指定的模型（`SELECT_MODEL` / `WRITE_MODEL`）
- 联网搜索（可选二选一）：
  - ollama.com 账号 API key（见下文配置）；或
  - 本机 Ollama 应用已登录账号（自动降级到本地 `/api/experimental` 端点，无需 key）
- 可选：DeepSeek API key（当 `SELECT_PROVIDER` / `WRITE_PROVIDER` 设为 `deepseek` 时必填）
- 可选：LangSmith API key（启用追踪）

## 快速开始

```bash
# 1. 安装依赖（自动创建 .venv）
uv sync

# 2. 准备配置
cp .env.example .env   # Windows: copy .env.example .env
# 编辑 .env，按需填写 OLLAMA_API_KEY / DEEPSEEK_API_KEY 等

# 3. 确认 RSS 输入文件存在（默认 sample/2026-09-03_tech_rss.json，可用 RSS_FILE 修改）

# 4. 运行
uv run main.py
```

运行成功后，终端打印选题结论与文章路径，文章写入 `output/<rss日期>_<主题slug>.md`（重名自动追加 `-2`、`-3`）。退出码：0 成功，1 失败（错误日志见终端输出；选题解析失败时对话记录转储到 `output/debug_transcript.txt`）。

## 配置说明（.env）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | 本地 Ollama 服务地址 |
| `OLLAMA_API_KEY` | 空 | ollama.com 账号 API key；不填或失效时自动降级到本地 experimental 端点（需本机 Ollama 已登录账号） |
| `SELECT_MODEL` / `WRITE_MODEL` | `qwen3.5:9b` / `qwen3.6:27b` | 选题 / 写作模型（provider=ollama 时生效，需本地已拉取） |
| `SELECT_PROVIDER` / `WRITE_PROVIDER` | `ollama` | 模型提供商：`ollama`（本地）或 `deepseek`（云端） |
| `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL` | — | provider=deepseek 时使用 |
| `SELECT_NUM_CTX` / `WRITE_NUM_CTX` | `16384` / `32768` | 模型上下文长度 |
| `SELECT_TEMPERATURE` / `WRITE_TEMPERATURE` | `0.2` / `0.7` | 选题 / 写作温度 |
| `NUM_PREDICT` | `4096` | 写作最大生成 token 数 |
| `RSS_FILE` | `sample/2026-09-03_tech_rss.json` | RSS 聚合文件（相对项目根目录） |
| `OUTPUT_DIR` | `output` | 博客输出目录 |
| `THREAD_ID` | `news-blog` | deepagents thread_id |
| `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` | 空 | 有值则启用 LangSmith 追踪 |

## 项目结构

```
main.py            # 入口：配置加载 -> RSS 预处理 -> Agent 运行 -> 兜底与报告
src/
  config.py        # .env 加载与默认值（Settings）
  preprocess.py    # RSS 聚合文件加载与候选预处理
  agents.py        # Deep Agents 主 Agent / 写作 SubAgent 构建、选题解析与重试
  tools.py         # web_search / web_fetch / resolve_output_path 工具
  prompts.py       # 主 Agent 与写作 SubAgent 系统提示词
  models.py        # 选题结论等数据模型
  output.py        # 输出路径规划、文件名兜底、降级标记
sample/            # RSS 示例输入
output/            # 生成的博客文章
```

## 常见问题

- **`pull model manifest: 404`**：本地没有拉取配置的模型，执行 `ollama pull <SELECT_MODEL/WRITE_MODEL>`。
- **联网搜索失败/降级**：检查 `OLLAMA_API_KEY` 是否有效（ollama.com/settings/keys 生成），或确认本机 Ollama 应用已登录账号。降级触发时文章会带降级警告行，运行日志有 WARNING。
- **两次尝试均未获得合法 SelectionResult**：本地小模型输出不稳定，查看 `output/debug_transcript.txt` 排查；可换更大模型或调低 `SELECT_TEMPERATURE`。
- **`DEEPSEEK_API_KEY 未设置`**：任一 provider 设为 `deepseek` 时必须填写。
