# v3.1 Generation Chain Eval Harness — Phase P5 (Path A)

Pure local evaluation harness. Imports `gateway.generation_chain` directly,
toggles env flags between v3.0 baseline (all flags off) and v3.1 variant
(synthesizer + stylist on), and grades the outputs on length, citation
preservation, multi-turn coherence, latency, and cost.

## Files

```
eval/
├── run_v31_eval.py   — CLI runner with 4 modes + pair list (50 multi-turn pairs)
├── llm_judge.py      — gpt-4o-as-judge for human-tone + multi-turn coherence
└── README.md         — this file
```

## Required environment

The harness reads keys from your shell env (or a `.env` file at the
worktree root if you load it before running). Required:

| Variable | Used by |
|----------|---------|
| `OPENAI_API_KEY` | LLM client + judge |
| `ANTHROPIC_API_KEY` | LLM client (Claude models) |
| `MONGODB_URI` | Live retrieval (already configured for v3.1) |

Optional model overrides (the harness sets these per `--model-set`):

```
SYNTHESIZER_MODEL, SYNTHESIZER_MODEL_FALLBACK
STYLIST_MODEL,     STYLIST_MODEL_FALLBACK
RECALL_MODEL, REWRITER_MODEL
```

## Modes

### single_turn — v3.1 vs v3.0 head-to-head

```bash
python eval/run_v31_eval.py \
    --mode single_turn \
    --questions "C:/path/to/test_questions_for_all_projects.xlsx" \
    --project-id 7325 --set-id 4987 \
    --sample 30 \
    --output-dir eval_results/single_turn \
    --model-set haiku
```

Outputs:
- `single_turn_baseline.csv` — v3.0 (flags off) per-question metrics
- `single_turn_variant.csv` — v3.1 (flags on) per-question metrics
- `single_turn_summary.json` — aggregates: median/p95 length, citation
  preservation %, latency delta, cost delta, win-rate

### multi_turn — Reference resolution coherence

```bash
python eval/run_v31_eval.py \
    --mode multi_turn \
    --project-id 7325 --set-id 4987 \
    --output-dir eval_results/multi_turn
```

Runs all 50 hand-crafted pairs (15 pronoun, 15 trade-followup, 10 implicit
count, 10 continuation) through the variant with memory + rewriter ON.
Each pair is scored 1-5 by gpt-4o on whether turn 2 used turn-1 context.

Outputs: `multi_turn_results.csv`, `multi_turn_summary.json`.

### model_ab — Pick the best model set

```bash
python eval/run_v31_eval.py \
    --mode model_ab \
    --questions "<xlsx>" \
    --project-id 7325 --set-id 4987 \
    --sample 30 \
    --output-dir eval_results/model_ab
```

Runs the variant 3 times (`haiku`, `sonnet`, `gpt4o-mini`) on the same
question set, then gpt-4o judges each answer 1-5 on human tone.

Outputs: `model_ab_haiku.csv`, `model_ab_sonnet.csv`,
`model_ab_gpt4o-mini.csv`, `model_ab_results.csv` (judge scores side-by-side),
and `model_ab_summary.json` with per-set averages.

### all — Full report

```bash
python eval/run_v31_eval.py \
    --mode all \
    --questions "<xlsx>" \
    --project-id 7325 --set-id 4987 \
    --sample 30 \
    --output-dir eval_results/full
```

Runs single_turn -> multi_turn -> model_ab and writes
`EVAL_REPORT.md` summarising deltas and target-met flags.

## Useful flags

- `--baseline-only` — only run v3.0 baseline rows (amortise cost across reruns)
- `--variant-only` — reuse a previously-written `single_turn_baseline.csv`
- `--no-banner` — skip the 5-second cost-confirmation pause
- `--max-workers N` — thread-pool size (default 3; don't go above 5
  unless you've raised your Anthropic rate limit)

## Cost estimates (rough)

| Mode | Sample | Estimated cost | Estimated runtime |
|------|-------:|---------------:|------------------:|
| single_turn | 30 | $0.30 ($0.20–$0.50) | ~10 min |
| multi_turn  | 50 pairs | $1.25 ($0.80–$2.10) | ~15 min |
| model_ab    | 30 × 3 | $1.20 ($0.80–$2.00) | ~25 min |
| **all**     | 30 + 50 + 30 | **~$2.75 ($1.80–$4.60)** | ~50 min |

Costs are real-LLM costs only — no infra costs since this is Path A
(in-process, no deployed service).

## Implementation notes

- **No edits outside the worktree.** The harness is a pure consumer of
  the existing chain modules.
- **Errors are non-fatal.** A bad question writes an `error` column and
  continues — we never abort the whole run.
- **Concurrency.** `ThreadPoolExecutor(max_workers=3)` by default; each
  thread wraps `run_generation_chain` in `asyncio.run(...)`.
- **Env-flag isolation.** Every chain call uses a `env_flags(...)`
  context manager so no test-state leaks into the surrounding process.
