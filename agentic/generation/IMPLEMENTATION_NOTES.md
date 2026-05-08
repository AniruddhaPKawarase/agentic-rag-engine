# v3.1 Generation Chain — Implementation Notes (Phase P3)

This package implements the answer-shaping subsystem that runs **after** the
ReAct retrieval agent (`agentic/core/agent.py:run_agent`). It turns verbose
agent output into 1-2 line, human, ChatGPT/Claude-style replies, and
re-expresses cached answers so they don't feel like canned playbacks.

## Modules

| Module | Role | Public surface |
|---|---|---|
| `llm_client.py` | Provider-agnostic generate() w/ Anthropic+OpenAI routing & fallback | `generate(...)` |
| `synthesizer.py` | Agent 4 — compresses raw ReAct answer to shortest faithful reply | `synthesize(...)` |
| `stylist.py` | Agent 5 — polishes draft / Agent 5b — re-expresses cached answer | `stylize(...)`, `reexpress_cached(...)` |

## New dependency

`anthropic>=0.40.0,<1.0.0` added to `_versions/v3.1-generation-chain/requirements.txt`
**only**. The root `requirements.txt` is unchanged. Install:

```bash
pip install -r _versions/v3.1-generation-chain/requirements.txt
```

## Environment variables

| Var | Default | Used by | Purpose |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | *(required if claude-* model in use)* | `llm_client` | Anthropic auth |
| `OPENAI_API_KEY` | *(required if gpt-* model in use)* | `llm_client` | OpenAI auth |
| `SYNTHESIZER_MODEL` | `claude-haiku-4-5` | `synthesizer` | Primary model |
| `SYNTHESIZER_MODEL_FALLBACK` | `gpt-4o-mini` | `synthesizer` | Outage fallback |
| `STYLIST_MODEL` | `claude-haiku-4-5` | `stylist` | Primary model (both fns) |
| `STYLIST_MODEL_FALLBACK` | `gpt-4o-mini` | `stylist` | Outage fallback |
| `STYLE_SKIP_THRESHOLD_CHARS` | `300` | `stylist.stylize` | Below this length, polish only if jargon-heavy |

Loading: `agentic/config.py` already calls `load_dotenv()` at import; this
package relies on that and does **not** load `.env` itself.

## Model selection logic

`llm_client._provider_for_model()`:

- `claude*` → Anthropic
- `gpt*`, `o1*`, `o3*` → OpenAI
- Anything else → OpenAI (logged as a warning; safest given existing infra)

The `fallback_model` is intended to be of the *opposite* provider for true
cross-provider redundancy (e.g. primary `claude-haiku-4-5`, fallback
`gpt-4o-mini`). The router doesn't enforce this — same-provider fallbacks
work too, just with less outage protection.

## Skip-rule rationale (latency-first)

We skip the LLM call when there's no work to do. Latency budget per request
matters far more than "always call the model."

- **`synthesize`** skips when `len(raw_answer) < 200` **and** no OCR-garble
  markers (`▯`, `□`, `???`). Rationale: short raw answers from the ReAct
  agent are already concise; compressing them wastes a round-trip and risks
  losing nuance. OCR garble markers override the skip — the LLM can clean
  those up even on short inputs.
- **`stylize`** skips when the draft is short (< `STYLE_SKIP_THRESHOLD_CHARS`)
  **and** not jargon-dense (no two ALL-CAPS technical tokens within 60 chars
  of each other). Rationale: a single acronym like "RFI" or "API" doesn't
  warrant a polish; clusters like "RCP HVAC VAV CFM AHU" do.
- **`reexpress_cached`** skips when there's no session context at all
  (`last_assistant_turn is None and rolling_summary is None and not is_followup`).
  Rationale: re-expressing a cached answer with zero conversational anchor is
  pure entropy — phrasing variation buys nothing if there's nothing to weave
  it against. The replay is identical to a fresh first-turn answer.

## Streaming protocol

Every public function accepts `stream: bool = False`.

- `stream=False` → returns a `str`.
- `stream=True` → returns an `Iterator[str]` that yields chunks as they
  arrive from the provider.

**Critical:** when a function takes the *skip* path under `stream=True`, it
still returns an iterator — a single-chunk generator that yields the
unchanged input once. This preserves the caller contract: callers can always
`for chunk in generate(...)` regardless of whether the skip rule fired or the
LLM was actually invoked.

The Anthropic streaming path uses the SDK's `messages.stream(...)` context
manager and consumes `text_stream`. The OpenAI path uses
`stream=True` on chat completions and walks `event.choices[0].delta.content`.

## Retry & fallback policy

`llm_client.generate()`:

1. Up to 2 attempts on the primary provider with backoff `0.75 * attempt`.
2. Retries are skipped for non-429 4xx errors (those are caller bugs, not
   transient — retrying won't help).
3. On primary exhaustion, the same 2-attempt cycle runs against
   `fallback_model` (typically the other provider).
4. After both providers fail, the last exception is re-raised.

Retryable signals: HTTP `429`, `5xx`, plus class-name heuristics matching
`ratelimit`, `apiconnection`, `apitimeout`, `internalserver`,
`serviceunavailable`, `overloaded` (covers both SDKs without a hard
import-time dependency on either error class).

## Prompt design notes (synthesizer & stylist)

### Synthesizer system prompt

Optimized for the client's #1 complaint: "answers too long, too verbatim."
Hard rules:

- **Length-by-query-type:** factual → 1-2 sentences; explanatory → ≤6
  sentences; bullets only when listing 3+ items.
- **Citation preservation is mandatory** — every `[drawing pX]` from the raw
  answer must survive to the output.
- **Forbidden phrases enumerated explicitly:** "Direct Answer:", "Summary:",
  "Conclusion:", "Based on the documents…", "According to the drawings…",
  closing wrap-ups. Listing the exact bad strings is more effective than
  abstract negative instructions.
- **`temperature=0.2`** — synthesis is a faithfulness task, not a creative
  one.

### Stylist system prompts

The two stylist prompts share the "no headers / no greetings / no sign-offs /
preserve citations" core, then diverge on intent:

- **`stylize`** (`temperature=0.4`): polishes a one-shot draft. Adds the
  continuity rule when `last_assistant_turn` is present — *one short clause*,
  not a full recap. Example: "Yes, and on top of that, the framing plan
  shows…".
- **`reexpress_cached`** (`temperature=0.5`): higher temp because the goal is
  phrasing variety. Two distinctive constraints:
  1. "Never reveal that the answer is cached" — explicit, because LLMs love
     to hedge with "as I mentioned before…" which exposes the cache.
  2. "Only the phrasing should change" — facts and citations are immutable.

`max_tokens=400` everywhere keeps the output bounded for the sub-second
latency target.

## Logging

Every module uses `logging.getLogger("agentic_rag.generation.<modname>")`.
Token usage from both providers is logged at INFO when available
(`anthropic.tokens` and `openai.tokens` keys for easy filtering).

## What's NOT done in P3 (explicit non-goals)

- **No wiring into `gateway/orchestrator.py`.** That happens in a follow-up
  phase so this PR stays reviewable. The orchestrator integration point is
  after `run_agent()` returns and before `_extract_source_documents` builds
  the response — the synthesizer replaces `result.answer` in-place and the
  stylist polishes that.
- **No conversation-state plumbing.** `rolling_summary` and
  `last_assistant_turn` are accepted but it's the caller's job to populate
  them from the existing session store.
- **No streaming through to FastAPI.** The streaming protocol is implemented
  at the generation layer; SSE/chunked-transfer wiring is a downstream task.

## Tests

`tests/v3_1/test_generation_layer.py` — 14 tests, all mock-only.

Run from the worktree root:

```bash
cd _versions/v3.1-generation-chain
pytest tests/v3_1/test_generation_layer.py -q
```

Coverage (manual):

- `llm_client`: routing (claude/gpt), fallback on 5xx, no retry on 4xx,
  streaming.
- `synthesizer`: skip on short clean input, compress on long input, citation
  preservation, OCR-garble override, streaming-skip contract.
- `stylist.stylize`: skip on short clean draft, polish on long draft, polish
  on jargon-heavy short draft, streaming.
- `stylist.reexpress_cached`: skip when no session context, fact preservation
  on full context, follow-up-only path.
