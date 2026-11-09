# MADBench

MADBench measures the overhead that Model Context Protocol (MCP) database servers add to SQL workloads, the "Protocol Tax". Track A sends fixed TPC-H queries (Q1, Q3, Q6, Q9, Q21) through three MCP servers (mcp-alchemy, dbhub, anthropic-pg) on PostgreSQL and MySQL, plus an in-process DuckDB reference, traces each interaction with OpenTelemetry, and splits its wall time into protocol phases. Track B has LLM agents answer 50 natural-language TPC-H questions through the same servers and scores each answer against the result of a gold SQL query.

## Requirements

- Linux or macOS. Disk: about 100 GB for TPC-H SF1, SF3 and SF10 across all databases, about 7 GB for SF1 alone.
- Docker with the compose plugin (PostgreSQL 16.14 and MySQL 8.4.9 images).
- git, make and a C compiler (to build TPC-H dbgen).
- Python 3.11 (installed through uv by the setup scripts).
- Node.js 22 (installed through nvm by the setup scripts if missing).
- Track B only: an OpenRouter API key in `OPENROUTER_API_KEY`.

Pinned versions: mcp-alchemy 2025.8.15.91819, @bytebase/dbhub 0.22.6, @modelcontextprotocol/server-postgres 0.6.2 (set in `scripts/setup/_lib.sh`). Python dependencies are locked in `uv.lock`.

## Setup

```
bash scripts/setup/setup_all.sh
```

This runs the numbered steps in `scripts/setup/` in order: install uv and Python, create `.venv` from `uv.lock`, install the three MCP servers, build TPC-H data (SF1, SF3, SF10), start PostgreSQL (port 55432) and MySQL (port 53306) with `docker-compose.yml`, load the data into all three databases, prepare the dbhub PostgreSQL databases and roles, and add TPC-H keys and indexes. Each step can be run on its own; `SFS=1,3` limits the scale factors. TPC-H data goes to `$SCRATCH_DIR` (default `/tmp/madbench-$USER` on Linux).

Then check the installation:

```
make smoke                      # Track A and Track B, costs a few cents
SKIP_TRACK_B=1 make smoke       # Track A only
make test                       # unit tests
```

The smoke run executes Track A for mcp-alchemy on PostgreSQL, Q6, SF1, 2 reps, and Track B for one question, one model and one coupling, 1 rep. It stops with a message naming any missing prerequisite (database, Node.js, MCP server, API key).

## Track A

One script per server. Each runs its full matrix by default (5 queries x 3 scale factors x cold and warm x 30 reps):

```
.venv/bin/python scripts/run_track_a_mcp_alchemy.py     # PostgreSQL, MySQL, DuckDB
.venv/bin/python scripts/run_track_a_dbhub.py           # PostgreSQL, MySQL
.venv/bin/python scripts/run_track_a_anthropic_pg.py    # PostgreSQL
```

Use `--backends`, `--queries`, `--sfs`, `--reps`, `--cold-warm` to run a subset. A warm cell reuses one server process and connection for all reps; a cold cell starts a new server process and connection for every rep. Output goes to `results/track_a/<server>/`: raw traces, `csv/per_trace_metrics.csv` and per-cell summaries.

## Track B

```
export OPENROUTER_API_KEY=...
.venv/bin/python scripts/track_b/capture_gold_results.py --sf 1
.venv/bin/python scripts/track_b/capture_gold_results.py --sf 3
bash scripts/track_b/run_track_b.sh
.venv/bin/python scripts/track_b/rescore.py --output results/track_b/rescored.csv results/track_b
.venv/bin/python scripts/track_b/build_corpus.py --output results/track_b/corpus.csv results/track_b/rescored.csv
```

`run_track_b.sh` runs every model x coupling x scale factor with 3 reps of every question, and resumes where it stopped. Override with `MODELS`, `SFS`, `REPS`, `QUERY_IDS`, `ONLY_COUPLING=server:backend` and `BUDGET_USD`. The default models are `deepseek/deepseek-v4-flash`, `z-ai/glm-5.3-flash`, `openai/gpt-6-luna` and `anthropic/claude-haiku-5.5`. The agent system prompt is `SYSTEM_PROMPT` in `scripts/track_b/_attempt.py`; agents may make at most 10 tool calls per question. dbhub on PostgreSQL logs in as a role that can read only the schema of the scale factor under test (`DBHUB_SEARCH_PATH_CONFIG=1`, the default in `run_track_b.sh`).

## Figures

The four scripts in `figures/` rebuild every published figure from `data/`:

```
make figures
```

With the pinned pandas and matplotlib versions in the `Makefile`, the PDFs are byte-identical on every run. Each script also prints the numbers it plots.

## Reproducibility

Figures regenerate exactly from `data/`. A new benchmark run does not reproduce the same numbers: timings depend on hardware and system load, and LLM outputs vary between calls even at temperature 0. A rerun gives comparable distributions, not identical values. `data/track_b/corpus.csv` stores outcomes and diagnostics but not the agents' submitted answers, so `rescore.py` needs the raw traces and gold results of a run.

## Data dictionary

`data/track_a/traces.csv`, one row per measured Track A interaction (5,378 rows):

- `cell_id`, `server`, `backend`, `query`, `scale_factor`, `rep_idx`, `cold_warm`: the cell and repetition. Rows with backend `duckdb` are the in-process DuckDB reference.
- `T_disc_ms`, `T_life_ms`, `T_hand_ms`, `T_ser_ms`, `T_parse_ms`, `T_xport_ms`: protocol phase times (discovery, lifecycle, handover, serialization, parsing, transport) in ms.
- `T_db_ms`: database execution time. `T_total_ms`: wall time of the interaction.
- `T_teardown_ms`: time from the last parse span to the end of the interaction; present for dbhub on PostgreSQL and anthropic-pg. Where it is empty, that time is part of `epsilon_ms`.
- `epsilon_ms`: residual, `T_total - T_db - phases - T_teardown`.
- `PTax_ms`: Protocol Tax, `T_total_ms - T_db_ms` (every phase, teardown and residual included). `PTR`: `PTax_ms / T_db_ms`.
- `B_mcp`, `tau_disc`, `n_life`, `R_retry_ms`, `M_peak_mb`: bytes on the MCP wire, discovery tokens, lifecycle handshakes, retry time, peak server memory (MiB). `phi`, `gamma`, `S_ser`, `rho`, `T_pool_ms` are not measured in Track A and are empty.

`data/track_b/corpus.csv`, one row per Track B attempt (7,200 rows: 4 models x 6 couplings x 2 scale factors x 50 questions x 3 reps):

- `model`, `server`, `backend`, `scale_factor`, `query`, `category`, `rep_idx`: the attempt. `category` is the SQLyzr complexity level of the question (c2, c4, c5, c6).
- `logged_outcome`: outcome recorded during the run. `outcome`: outcome after re-scoring with `rescore.py`. Values: `correct`, `incorrect`, `unparseable`, `call_limit_exceeded`, `agent_error`.
- `n_tool_calls`, `total_input_tokens`, `total_output_tokens`, `cost_usd`, `wall_clock_ms`: effort and cost of the attempt.
- `diagnostic`: the verifier's or the error's message.

`data/track_b/gold_rows.csv`: row count of each gold result per question and scale factor.

## Use with your own setup

**Questions.** Add a YAML file in the format of `workloads/track_b/questions.yaml`: top-level keys are levels (`c1` to `c6`), each holding a list of entries with `id`, `target_subcat` (optional), `question` and `gold_sql`. Pass it with `QUERY_FILES="workloads/track_b/questions.yaml my_questions.yaml"` (or `--query-file` to the Python scripts) and capture its gold results with `capture_gold_results.py --query-file my_questions.yaml`. Ids must be unique across files.

**Databases.** Connection settings are constants: `BACKEND_DSNS` and `SF_TO_NAMESPACE` at the top of each `scripts/run_track_a_*.py` (Track A), and `_PG_*` / `_MYSQL_*` in `scripts/track_b/_mcp_launch.py` and `_PG_DEFAULT` / `_MYSQL_DEFAULT` / `_DUCKDB_FILE_TEMPLATE` in `scripts/track_b/_backends.py` (Track B). PostgreSQL data is expected in schemas `sf<N>` of one database, MySQL data in databases `tpch_sf<N>`, DuckDB data in one file per scale factor. Load your own TPC-H data with `scripts/load_tpch.py`.

**MCP servers.** Track A: implement `mcp_dbs_bench.MCPServerPlugin` (see `src/mcp_dbs_bench/servers/dbhub.py` for a subprocess-backed example), register it under the `mcp_dbs_bench.servers` entry point in `pyproject.toml`, check it with `mcp_dbs_bench.validate_plugin`, and copy a `run_track_a_*.py` script with `SERVER` set to its name. Track B: add a launch function to `scripts/track_b/_mcp_launch.py` that returns an `McpLaunchSpec` (binary, arguments, environment), list the name in `SUPPORTED_SERVERS` and dispatch to it in `build_launch_spec_for`.

**Models.** Any OpenRouter model with tool calling works: `MODELS="vendor/model-a vendor/model-b" bash scripts/track_b/run_track_b.sh`, or `--model` on `run_track_b_sweep.py`.

## License

Apache-2.0, see `LICENSE`. TPC-H queries are reproduced from the TPC Benchmark H specification; dbgen is downloaded by `scripts/build_tpch.sh` and is subject to the TPC end-user license.
