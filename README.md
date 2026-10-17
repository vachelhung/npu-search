# npu-search

Local semantic search over your own documents and code, with embeddings computed on an
**Intel NPU** (Intel AI Boost) via OpenVINO. Search in Chinese or English by meaning, from the
command line or as an **MCP server** so an AI client (Claude Code / Claude Desktop) can call it.

在 Intel AI Boost NPU 上跑的本機語意搜尋：用中文或英文、照「意思」搜尋自己的文件和程式碼，
也可以包成 MCP 工具讓 AI 用戶端直接呼叫。

- Embedding model: [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small), OpenVINO FP16
- Index: SQLite (`index.db`), vectors stored as float16
- Formats: PDF, DOCX, XLSX, PPTX, Markdown, text, CSV, RTF, notebooks, source code, KiCad files
- Fully local: the code makes no network calls. The model is loaded from disk.

## Why this might be useful to you

The NPU driver 32.0.100.4023 (2025/04) ships a compiler that produces **wrong results** for this
BERT-style model (different inputs give almost identical vectors). `npusearch/embed.py` works around it:

- `NPU_COMPILER_TYPE=PLUGIN` makes the NPU use OpenVINO's bundled compiler (results match CPU, cos = 1.0).
- Inputs are fed as `int32` because int64 inputs are read incorrectly on this driver.
- The model is reshaped to a static `[8, 256]` shape, which the NPU requires.

Newer drivers may not need these; try removing them after updating.

## Setup

Requires Windows with an Intel NPU, and Python 3.12.

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install "optimum-intel[openvino]==2.2.0"   # only for the export below

# Export the embedding model (not included in this repo, ~235 MB)
.venv\Scripts\optimum-cli export openvino --model intfloat/multilingual-e5-small --weight-format fp16 models\me5-small

copy config.example.json config.json    # then edit "roots" to the folders you want indexed
```

If `tokenizer.json` is missing from `models\me5-small`, download it from the model's Hugging Face page.

## Usage

```powershell
.venv\Scripts\python.exe -m npusearch.cli index                   # incremental index (new/changed files only)
.venv\Scripts\python.exe -m npusearch.cli search "USB-C power delivery design"
.venv\Scripts\python.exe -m npusearch.cli search "I2C" --path DataSheet -k 10
.venv\Scripts\python.exe -m npusearch.cli status
```

`--device GPU` uses the integrated GPU instead (about 5x faster than the NPU, but uses more power).
The first NPU compile takes ~15 s; later runs use the cache in `models/me5-small/cache`.

### As an MCP server

```powershell
claude mcp add --scope user npu-search -- C:\path\to\npu-search\.venv\Scripts\python.exe C:\path\to\npu-search\mcp_server.py
```

Tools: `semantic_search(query, k, path_contains)`, `index_status()`, `reindex()`.

## Privacy notes

- `index.db` contains the **plain text of your documents, unencrypted**. It is git-ignored; never share or upload it.
- When an AI client calls `semantic_search`, the returned passages become part of that conversation
  and are sent to the AI provider. Only index folders you are comfortable with that.
- `.gitignore` is a whitelist (everything is ignored except the source files) so `config.json`,
  the index and the model cannot be committed by accident.

## Known limitations

- Legacy Office formats (`.doc`, `.xls`, `.ppt`) are not supported.
- OneDrive cloud-only placeholder files are skipped, to avoid triggering downloads.
- New folders are not picked up automatically; add them to `roots` in `config.json`, then re-index.

## 繁體中文補充

- 設定檔：複製 `config.example.json` 為 `config.json`，把 `roots` 改成要索引的資料夾。
- 模型不含在倉庫內（約 235 MB），請依上方指令用 `optimum-cli` 匯出到 `models\me5-small`。
- `index.db` 存放文件內容的純文字且未加密，已列入 `.gitignore`，請勿分享。
- 透過 AI 用戶端搜尋時，回傳的片段會進入對話並送到服務供應商，請只索引你能接受的資料夾。
