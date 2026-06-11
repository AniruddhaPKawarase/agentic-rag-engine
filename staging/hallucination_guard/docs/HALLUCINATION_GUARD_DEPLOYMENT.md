# Hallucination Guard — Deployment Runbook

**Roadmap reference:** `HYBRID_RAG_v32_ROADMAP.md` §5.21 (Tactical Patch v2)
**Target:** 8001 PROD on sandbox VM, app dir `/home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent-v31/`
**Scope:** Pillars 1 + 4 + 6 + 7 + 11 (additive, env-flag-gated)
**Total effort:** ~4 working days (eval-first ordering)
**Reversibility:** Master flag flip = 30-second rollback
**Compatibility:** Coexists with `ENABLE_SPECIFICATIONS=false` (drawings-only mode)

## Pre-conditions (must be true before Day 0)

- [x] User has approved Tactical Patch v2 scope (per HARD RULE 1)
- [x] Sandbox SSH key valid (`ai_assistant_sandbox.pem`); NOT using PROD key (`ai_assistant.pem`)
- [x] Snapshot space available (≥ 5 GB free)
- [x] VCSAI not running active jobs at moment of any service bounce
- [x] No `drawings_v2` / `specifications_v2` references in code touched by this patch (these are READ-ONLY forever per HARD RULE)

## Day 0 — Pre-flight + Baseline Snapshot (~2 hrs)

### Step 0.1 — SCP staging tree to sandbox VM

From your workstation:

```bash
# From the staging dir (this workspace)
cd "c:/Users/ANIRUDDHA ASUS/Downloads/projects/hybrid-rag-v32/implementations/hallucination_guard"

# SCP scripts + tests to a staging dir on the VM (NOT yet in app dir)
scp -i ~/path/to/ai_assistant_sandbox.pem -r scripts tests docs \
    ubuntu@<sandbox-vm-host>:/home/ubuntu/staging/hallucination_guard/

# (The agentic/ tree stays staged — it goes into app dir AFTER baseline)
scp -i ~/path/to/ai_assistant_sandbox.pem -r agentic \
    ubuntu@<sandbox-vm-host>:/home/ubuntu/staging/hallucination_guard/
```

### Step 0.2 — Run pre-flight check on the VM

SSH in and run:

```bash
ssh -i ~/path/to/ai_assistant_sandbox.pem ubuntu@<sandbox-vm-host>
cd /home/ubuntu/staging/hallucination_guard/scripts
chmod +x preflight_check.sh snapshot_baseline.sh rollback.sh
./preflight_check.sh
```

**Expected exit codes:**
- `0` = SAFE TO PROCEED
- `2` = WARNINGS (review and decide; usually safe)
- `1` = STOP, investigate failures

If exit code 1: do NOT continue. Send me the output; we fix root cause first.

### Step 0.3 — Create baseline snapshot

```bash
./snapshot_baseline.sh
# Note the printed tag, e.g. hg_v2_baseline_20260603T214500Z
```

This:
- Copies `.env`, `registry.py`, synthesizer files, agentic/ tree to `/home/ubuntu/backups/hallucination_guard/<tag>/`
- Creates a full app tarball (~30-50 MB)
- Writes SHA256 checksums for tamper detection
- Updates LATEST symlink

### Step 0.4 — Verify rollback procedure (DRY RUN)

Before any code change, confirm rollback works:

```bash
# DRY-RUN ONLY — do not type YES-ROLLBACK; just verify the script reads
# the snapshot and lists what it would restore.
./rollback.sh
# When prompted "Type 'YES-ROLLBACK' to proceed:", type anything else and press Enter
```

You should see the list of files that would be restored. This confirms the snapshot is readable and the rollback path is functional. If something looks wrong, STOP.

### Step 0.5 — Verify ENABLE_SPECIFICATIONS still in expected state

```bash
grep -E "^(ENABLE_SPECIFICATIONS|MASTER_SPECS_OFF)" /home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent-v31/.env
```

Expected: `ENABLE_SPECIFICATIONS=false` (per session memory, drawings-only mode).

**Day 0 complete.** No production change yet.

---

## Day 1A — Baseline Eval (~6 hrs)

### Step 1A.1 — Install eval suite

```bash
# On sandbox VM
cd /home/ubuntu/staging/hallucination_guard/tests
pip install requests anthropic    # anthropic is optional, only for LLM-judge escalation
```

### Step 1A.2 — Verify env: HG flags all OFF for baseline

```bash
grep -E "^HALLUCINATION_GUARD" /home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent-v31/.env
# Should output NOTHING (no flag set) — defaults to OFF.
# Or if explicit: HALLUCINATION_GUARD_ENABLED=false
```

If any HG flag is `true`, set it to `false` BEFORE baseline (we want to measure 8001's current behavior, not a half-deployed state).

### Step 1A.3 — Run baseline

```bash
cd /home/ubuntu/staging/hallucination_guard/tests
python run_eval.py \
    --gold adversarial_gold_set.jsonl \
    --endpoint http://127.0.0.1:8001/query \
    --tenant hg-baseline-eval \
    --output baseline_responses_$(date -u +%Y%m%dT%H%M%SZ).jsonl
```

Expect ~10-30 minutes runtime for 50 multi-turn tests (some have 3-4 turns each). The CLI prints per-test progress.

### Step 1A.4 — Grade baseline

```bash
python grader.py \
    --gold adversarial_gold_set.jsonl \
    --responses baseline_responses_<ts>.jsonl \
    --output baseline_grades_<ts>.json \
    --run-id baseline_pre_HG
```

The grader prints a per-pillar and per-category pass-rate summary.

**Expected baseline (no HG)**: 40-60% overall pass rate. This is the NUMBER YOU'RE TRYING TO BEAT.

### Step 1A.5 — Save baseline as the comparison anchor

```bash
mv baseline_responses_<ts>.jsonl baseline_grades_<ts>.json \
   /home/ubuntu/staging/hallucination_guard/baselines/
```

**Day 1A complete.** You have a quantified baseline.

---

## Day 1B — Ship Pillar 6 (~2 hrs)

Lowest-risk pillar (prompt-only, additive).

### Step 1B.1 — Copy module into app dir

```bash
ssh -i ~/path/to/ai_assistant_sandbox.pem ubuntu@<sandbox-vm-host>
cp -r /home/ubuntu/staging/hallucination_guard/agentic/hallucination_guard \
      /home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent-v31/agentic/

# Verify
ls -la /home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent-v31/agentic/hallucination_guard/
# Should show: __init__.py, flags.py, pillar_6_anti_anchor.py, test_pillar_6.py
```

### Step 1B.2 — Run unit tests in place

```bash
cd /home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent-v31
python -m pytest agentic/hallucination_guard/test_pillar_6.py -v
```

All tests must pass before you wire into synthesizer. If any test fails: STOP. Inspect.

### Step 1B.3 — Wire Pillar 6 into the synthesizer

See `PILLAR_6_INTEGRATION.md` for the exact code change. Summary: ONE LINE in your synthesizer's system-prompt assembly. Example:

```python
# Before:
system_prompt = build_system_prompt(persona, context)

# After:
from agentic.hallucination_guard.pillar_6_anti_anchor import compose_system_prompt
system_prompt = compose_system_prompt(build_system_prompt(persona, context))
```

When the flag is OFF (default), `compose_system_prompt` is a no-op. Safe to merge with flag off.

### Step 1B.4 — VCSAI safety check + restart service

```bash
cd /home/ubuntu/staging/hallucination_guard/scripts
./preflight_check.sh    # confirm [3] VCSAI in-process jobs is PASS

# Restart
sudo systemctl restart rag-agent.service
sleep 5
systemctl is-active rag-agent.service    # must be 'active'

# Smoke test
curl -sf http://127.0.0.1:8001/health || curl -sf http://127.0.0.1:8001/
```

If the service fails to start: `journalctl -u rag-agent.service -n 100` to inspect. Most likely cause: import error. Fix the import; restart. If it's bad: rollback (Step 0.4 procedure for real this time).

### Step 1B.5 — Verify flag-off behavior (no change vs baseline)

```bash
# Flag-off: send one query; verify no behavior change
curl -X POST http://127.0.0.1:8001/query -H "Content-Type: application/json" \
  -d '{"query":"How many FCUs on L3 of project 7224?","tenant_id":"hg-pillar-6-flagoff"}'
```

The response should look identical to pre-deployment. If it doesn't, your integration changed behavior with the flag OFF — that's a bug. Fix or rollback.

### Step 1B.6 — Enable Pillar 6

Append to `.env`:

```bash
sudo tee -a /home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent-v31/.env <<'EOF'

# Hallucination Guard v2 — Tactical Patch
HALLUCINATION_GUARD_ENABLED=true
HG_PILLAR_6=true
EOF
```

VCSAI safety check + restart:

```bash
./preflight_check.sh
sudo systemctl restart rag-agent.service
sleep 5
systemctl is-active rag-agent.service
```

### Step 1B.7 — Re-run eval with Pillar 6 ON

```bash
cd /home/ubuntu/staging/hallucination_guard/tests
python run_eval.py \
    --gold adversarial_gold_set.jsonl \
    --endpoint http://127.0.0.1:8001/query \
    --tenant hg-pillar-6-on \
    --output pillar6_responses_$(date -u +%Y%m%dT%H%M%SZ).jsonl

python grader.py \
    --gold adversarial_gold_set.jsonl \
    --responses pillar6_responses_<ts>.jsonl \
    --output pillar6_grades_<ts>.json \
    --run-id pillar_6_only
```

### Step 1B.8 — Compare baseline vs Pillar 6

Expected delta: **-5 to -10% hallucination on anti-anchor category** (the category Pillar 6 targets). Other categories should be roughly equal.

If delta is negative (hallucination INCREASED): rollback immediately. Either the prompt is fighting your existing system prompt, or the integration introduced a regression.

```bash
# Compare aggregates
python -c "
import json
b = json.load(open('baseline_grades_<ts>.json'))['aggregate']
p = json.load(open('pillar6_grades_<ts>.json'))['aggregate']
print(f'Baseline overall: {b[\"overall_pass_rate\"]:.1%}')
print(f'Pillar 6 overall: {p[\"overall_pass_rate\"]:.1%}')
print(f'Delta: {(p[\"overall_pass_rate\"] - b[\"overall_pass_rate\"])*100:+.1f}pp')
print()
print('By category:')
for cat in sorted(set(b['by_category']) | set(p['by_category'])):
    bp = b['by_category'].get(cat, {})
    pp = p['by_category'].get(cat, {})
    print(f'  {cat:30s}  baseline {bp.get(\"pass\",0)}  pillar6 {pp.get(\"pass\",0)}')
"
```

### Step 1B.9 — Decision

- **Delta ≥ +5%**: Pillar 6 ships. Move to Day 2 (Pillar 7).
- **Delta 0 to +5%**: Marginal. Keep enabled but tune prompt or move to Day 2 cautiously.
- **Delta < 0**: Disable Pillar 6 immediately (`HG_PILLAR_6=false` + restart). Investigate. Do not proceed.

**Day 1 complete.** First production hallucination defense live.

---

## Day 2 — Ship Pillar 7 (citation-forcing) — [stub; see PILLAR_7_INTEGRATION.md when ready]

Day 2 ships citation-forcing structured output. **Higher risk** than Pillar 6 (touches synthesizer output shape). Pre-conditions:
- Day 1 Pillar 6 stable for ≥ 24 hours
- No regressions reported on real traffic
- Eval pass-rate improved

Detailed steps will be in `PILLAR_7_INTEGRATION.md` (TBD by next deliverable batch).

---

## Day 3 — Ship Pillars 1 + 4 + 11 — [stubs]

In order:
- Pillar 1 (anaphora resolver) — low risk, additive
- Pillar 11 (sycophancy + injection detector) — low risk, pre-filter
- Pillar 4 (per-claim verifier) — medium risk, adds latency

Detailed steps to follow.

---

## Day 4 — Full Integration Eval

Master flag + all 5 pillar sub-flags ON. Re-run adversarial suite. Compare to baseline.

**Target:** -50% overall hallucination (≥ 95% adversarial pass for Pillars 6+7+11 categories).

Write decision file `decisions/2026-XX-XX_hallucination-guard-v2-shipped.md` with:
- baseline grades
- per-pillar grades
- final grades
- regression observations (if any)
- next steps (e.g., when to enable Pillar 8 ensemble — out of this patch's scope)

---

## Rollback (any time)

```bash
# Option 1 — disable everything via flag (30 seconds)
sudo sed -i 's/^HALLUCINATION_GUARD_ENABLED=.*/HALLUCINATION_GUARD_ENABLED=false/' \
    /home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent-v31/.env
cd /home/ubuntu/staging/hallucination_guard/scripts
./preflight_check.sh    # VCSAI safety
sudo systemctl restart rag-agent.service

# Option 2 — full code rollback to baseline snapshot
./rollback.sh <baseline_tag>
# When prompted: type YES-ROLLBACK
```

## Open Questions / Risks to Track

- **R1:** Does the existing 8001 synthesizer assemble system prompt in one obvious place? If not, integration may require touching multiple call sites. Inspect synth files before Day 1B Step 1B.3.
- **R2:** Persona compiler (per memory `persona_phase3_state`) may also prepend instructions. Confirm Pillar 6 doesn't double-prepend or conflict with persona instructions.
- **R3:** /query/stream endpoint streams tokens. Confirm system prompt is set BEFORE first token; otherwise streaming bypass.
- **R4:** Baseline pass-rate may be lower than expected due to Pillar 7 (citation-forcing) not yet active. Don't over-interpret Pillar 6 marginal improvement until full patch lands.

## Sign-off Checklist (per HARD RULE compliance)

- [ ] User approval for Tactical Patch v2 scope: **YES** (this conversation, 2026-06-03)
- [ ] VCSAI safety check before every service bounce
- [ ] Baseline snapshot created (tag: __________)
- [ ] Rollback procedure dry-run validated
- [ ] PROD key (`ai_assistant.pem`) NOT used from staging workspace
- [ ] `drawings_v2`/`specifications_v2` collections not touched (verified via code search)
- [ ] All env flags default OFF in code; explicit-enable only via `.env`
- [ ] Decision file written post-deployment (one per pillar)
