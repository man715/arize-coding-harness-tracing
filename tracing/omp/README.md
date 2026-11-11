# omp Tracing

Automatic [OpenInference](https://github.com/Arize-ai/openinference) tracing for [Oh My Pi (omp)](https://github.com/can1357/oh-my-pi) terminal coding sessions. Spans are exported to [Arize AX](https://arize.com) or [Phoenix](https://github.com/Arize-ai/phoenix).

## What gets traced

One trace per **agent run** — one user prompt → the agent's internal turn/tool-use loop → its final answer. A single agent run usually contains several model calls (omp re-prompts the model after each round of tool calls), so one trace covers all of them. Each trace is a tree:

| Span | Kind | Notes |
|------|------|-------|
| `Turn` | CHAIN | Root span. `input.value` is the user prompt (from `before_agent_start`); `output.value` is the final assistant message's text (or its reasoning text when the run produced no visible text). One per agent run. |
| `LLM: <model>` | LLM | Child of `Turn`. One per `turn_end` (one per model call in the loop). Carries `llm.model_name`, `llm.provider`, prompt/completion/reasoning token counts, cache read/write tokens, `llm.cost`, `llm.reasoning` (the assistant's `thinking` content items, queryable as an attribute), and `llm.output_messages` — the turn's thinking and text parts as OpenInference message contents. When a turn has no visible text (thinking + tool calls only) `output.value` also falls back to the reasoning text. |
| `<tool>` | TOOL | Child of `Turn`. One per `ToolResultMessage` in a `turn_end`, paired with its originating `ToolCall` by id. Records `tool.name`, redacted input args + output, and tool-specific attributes. Errors are recorded with span status. |

Reasoning **text** is always available, and it is exported twice: as the `llm.reasoning` attribute (queryable/filterable), and as `message_content` items of type `reasoning` inside `llm.output_messages`, in omp's own block order — ahead of the visible answer. The message form is what makes the LLM span's Output panel show a collapsible **Reasoning** row: Phoenix keys that row on `message_content.type == "reasoning"`, and an ordinary attribute has no slot there. `thinkingSignature` (an opaque provider token, not text) is never exported. Both copies are gated by `logging.thinking` / `ARIZE_LOG_THINKING`, which defaults to the prompt-logging flag. Reasoning **token counts** depend on the provider: the count reaches a span only as `usage.reasoningTokens`, which omp populates from `usage.completion_tokens_details.reasoning_tokens` on OpenAI-compatible responses. Providers that report the count elsewhere — e.g. SGLang returns `usage.reasoning_tokens` as a top-level member — lose it inside omp before the hook sees the message, so the `llm.token_count.completion_details.reasoning` attribute is simply omitted while the reasoning text still exports.

Token usage **is** captured — omp surfaces cumulative `usage` (input/output tokens, cache read/write, and cost) inline on each assistant message, unlike vendors that withhold it from local surfaces.

Timestamps come from omp's own millisecond clocks where present (`AssistantMessage.timestamp` / `duration`, `turn_start.timestamp`, `ToolResultMessage.timestamp`), falling back to wall-clock time on the tracing process when absent.

## Architecture

omp loads its extensions **in-process** inside its Bun runtime ([hook docs](https://omp.sh/docs/hooks)) — there is no per-event subprocess and no host-spawned hook. Unlike opencode, omp exposes rich, **once-fired** lifecycle events that already carry final, structured data, so the integration is a straightforward stateful event-forward (no snapshot reconciliation, no dedup). It is split into two pieces:

1. **TypeScript hook shim** (`~/.omp/extensions/arize-tracing.ts`). A dumb bridge. On a small whitelist of lifecycle events — `before_agent_start` (carries the prompt), `turn_end` (carries the completed `AssistantMessage` with inline token usage + model, plus that turn's `toolResults`), `agent_end` (run finished), and `session_shutdown` — it spawns `arize-hook-omp` detached and pipes the event payload to stdin. The shim contains no tracing logic and never blocks omp's event loop.
2. **Python event handler** (`arize-hook-omp`). A small state machine keyed by session id. It dispatches on `payload["type"]`, accumulates per-session state, and emits `Turn`/`LLM`/`TOOL` spans on receipt. Pairs each `ToolCall` with its `ToolResultMessage` by id to build TOOL spans with both input args and output.

Because omp's lifecycle events fire exactly once each and carry final data, there is no message-id or callID dedup — spans are emitted as events arrive.

## Setup

The installer prompts for your backend (Phoenix or Arize AX) and project name, writes credentials to `~/.arize/harness/config.json`, copies the hook shim into `~/.omp/extensions/arize-tracing.ts`, **and** registers the shim's absolute path in the `extensions` array of `~/.omp/agent/settings.json`. omp does **not** auto-discover an extensions directory ([extension loading docs](https://omp.sh/docs/hooks)) — explicit registration is required, and the installer handles it. Spans are sent directly to the backend from the handler — no separate buffer/collector service is required.

Pass `--with-skills` to also symlink the `manage-omp-tracing` skill into the current directory's `.agents/skills/` so coding agents in this workspace can help manage omp tracing configuration.

### Remote setup

#### macOS / Linux

Install:

```bash
curl -sSL https://raw.githubusercontent.com/man715/arize-coding-harness-tracing/main/install.sh | bash -s -- omp
```

Uninstall:

```bash
curl -sSL https://raw.githubusercontent.com/man715/arize-coding-harness-tracing/main/install.sh | bash -s -- uninstall omp
```

#### Windows (PowerShell)

Install:

```powershell
iwr -useb https://raw.githubusercontent.com/man715/arize-coding-harness-tracing/main/install.bat -OutFile $env:TEMP\install.bat
& $env:TEMP\install.bat omp
```

Uninstall:

```powershell
iwr -useb https://raw.githubusercontent.com/man715/arize-coding-harness-tracing/main/install.bat -OutFile $env:TEMP\install.bat
& $env:TEMP\install.bat uninstall omp
```

### Local setup

```bash
git clone https://github.com/man715/arize-coding-harness-tracing.git
cd coding-harness-tracing
```

**macOS / Linux**

Install:

```bash
./install.sh omp
```

Uninstall:

```bash
./install.sh uninstall omp
```

**Windows (PowerShell)**

Install:

```powershell
install.bat omp
```

Uninstall:

```powershell
install.bat uninstall omp
```

Uninstall removes the shim's path from the `extensions` array in `~/.omp/agent/settings.json`, deletes the hook file at `~/.omp/extensions/arize-tracing.ts` (only if it carries the Arize header marker — your own extensions are left alone), and removes the `harnesses.omp` block from `~/.arize/harness/config.json`.

## Default Settings

| Setting | Default |
|---------|---------|
| Harness key | `omp` |
| Project name | `omp` |
| Phoenix endpoint | `http://localhost:6006` |
| Arize AX endpoint | `otlp.arize.com:443` |
| Hook file | `~/.omp/extensions/arize-tracing.ts` |
| Registration | absolute path in `extensions` array of `~/.omp/agent/settings.json` |
| Lifecycle events forwarded | `before_agent_start`, `turn_end`, `agent_end`, `session_shutdown` |
| Span tree | `Turn` (CHAIN) → `LLM` / `TOOL` |
| Trace granularity | one trace per agent run |
| State directory | `~/.arize/harness/state/omp/` |
| Log file | `~/.arize/harness/logs/omp.log` |

## Verifying tracing

Run any omp session as you normally would. omp loads the registered extension on startup; the shim forwards lifecycle events to the Python handler.

- Errors and handler stderr land in `~/.arize/harness/logs/omp.log` always (the adapter redirects Python stderr there via `ARIZE_LOG_FILE`); set `export ARIZE_VERBOSE=true` before launching omp to also see routine handler activity (event dispatch, span emits, state transitions).
- Confirm spans appear in your configured project in Arize AX or Phoenix.
- Open an `LLM: <model>` span: its Output panel lists the turn's content parts, with a collapsible **Reasoning** row holding the thinking text (only when the model emitted thinking and prompt/thinking logging is on).
- Set `ARIZE_TRACE_DEBUG=true` to dump the raw event payloads under `~/.arize/harness/state/debug/` (files are named `omp_before_agent_start_<ts>.json`, `omp_turn_end_<ts>.json`, `omp_agent_end_<ts>.json`, `omp_session_shutdown_<ts>.json`) for inspection.

See the [main README's Environment variables section](../../README.md#environment-variables) for the full list of runtime overrides (`ARIZE_TRACE_ENABLED`, `ARIZE_DRY_RUN`, `ARIZE_USER_ID`, `ARIZE_PROJECT_NAME`, `ARIZE_VERBOSE`, `ARIZE_TRACE_DEBUG`, etc.).
