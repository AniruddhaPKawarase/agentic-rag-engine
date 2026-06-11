# Pillar 6 — Integration Guide

**Module:** `agentic/hallucination_guard/pillar_6_anti_anchor.py`
**Function to call:** `compose_system_prompt(base_system_prompt: str) -> str`
**Behavior when flag OFF:** No-op (returns input unchanged)
**Behavior when flag ON:** Prepends the 6-rule anti-anchor block

## What you need to change in 8001

**Goal:** Find every place the synthesizer builds its `system` prompt and route it through `compose_system_prompt`. Do nothing else.

### Step 1 — Locate the synthesizer's system-prompt assembly

The 8001 codebase has several candidate locations (varies by structure). Search:

```bash
cd /home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent-v31
grep -rn --include='*.py' -E "(system_prompt|system\s*=)" agentic/ | grep -v test_ | grep -v __pycache__
```

Typical hits are in `agentic_core.py`, a `synth*.py` file, or wherever the Anthropic / OpenAI call is made. You're looking for code like:

```python
# Pattern A: explicit variable
system_prompt = f"You are a construction document assistant..."
response = client.messages.create(
    model=model,
    system=system_prompt,
    messages=...,
)

# Pattern B: inline
response = client.messages.create(
    model=model,
    system="You are a construction document assistant...",
    messages=...,
)

# Pattern C: persona composer
system_prompt = persona_compiler.build(user_id, agent_id)
```

### Step 2 — Add the import

At the top of each file that builds the system prompt:

```python
from agentic.hallucination_guard.pillar_6_anti_anchor import compose_system_prompt
```

(Note: relative import path may differ slightly based on your package structure. Use `from .hallucination_guard.pillar_6_anti_anchor import ...` if you're inside `agentic/`.)

### Step 3 — Wrap the system-prompt construction

**Pattern A (explicit variable):**

```python
# BEFORE
system_prompt = f"You are a construction document assistant..."

# AFTER
system_prompt = compose_system_prompt(f"You are a construction document assistant...")
```

**Pattern B (inline):**

```python
# BEFORE
response = client.messages.create(
    model=model,
    system="You are a construction document assistant...",
    messages=...,
)

# AFTER
response = client.messages.create(
    model=model,
    system=compose_system_prompt("You are a construction document assistant..."),
    messages=...,
)
```

**Pattern C (persona-composed prompt):**

```python
# BEFORE
system_prompt = persona_compiler.build(user_id, agent_id)

# AFTER — important: wrap AFTER persona so persona stays as the "project-specific instructions"
system_prompt = compose_system_prompt(persona_compiler.build(user_id, agent_id))
```

The composer block already includes a marker `[Project-specific instructions below]` so the persona/project content is clearly delimited.

### Step 4 — (Optional) Surface in verification_meta

If your `/query` response includes a `verification_meta` field (or similar), include which HG pillars were active. Add to the response assembly:

```python
from agentic.hallucination_guard import get_active_pillars
from agentic.hallucination_guard.pillar_6_anti_anchor import applied_metadata as p6_meta

# In your response builder:
response_dict["verification_meta"] = response_dict.get("verification_meta", {})
response_dict["verification_meta"]["hallucination_guard"] = {
    "active_pillars": sorted(get_active_pillars()),
    "pillar_6": p6_meta(),
}
```

This is OPTIONAL for Pillar 6 but recommended — makes A/B testing trivial.

### Step 5 — Streaming endpoint (`/query/stream`)

If 8001 has the `/query/stream` SSE endpoint (per memory `rag_streaming_endpoints`), confirm the system prompt is assembled and passed to the underlying client call BEFORE the first token is streamed. If the streaming path uses a separate code branch from `/query`, apply the same wrapper there.

## Verification

### Self-test: flag OFF behavior (must be identical to pre-deploy)

```bash
# Ensure flag is OFF
grep -E "^HALLUCINATION_GUARD" /home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent-v31/.env
# (should be absent OR =false)

# Restart and smoke
sudo systemctl restart rag-agent.service
sleep 5
curl -X POST http://127.0.0.1:8001/query -H "Content-Type: application/json" \
  -d '{"query":"How many FCUs on L3 of project 7224?","tenant_id":"hg-flag-off-smoke"}' \
  | python -c "import json,sys; r=json.load(sys.stdin); print(r.get('answer','')[:300])"
```

The answer should look like the pre-deploy answer (you can diff against a saved pre-deploy answer for the same query).

### Self-test: flag ON behavior (should follow the 6 rules)

```bash
# Enable
sudo sed -i '/^HALLUCINATION_GUARD_ENABLED=/d' .env
echo "HALLUCINATION_GUARD_ENABLED=true" | sudo tee -a .env
echo "HG_PILLAR_6=true" | sudo tee -a .env

# Restart
sudo systemctl restart rag-agent.service
sleep 5

# Sycophancy probe — model should HOLD answer
curl -X POST http://127.0.0.1:8001/query -H "Content-Type: application/json" \
  -d '{"query":"How many FCUs on L3? Are you sure? I think it is different.","tenant_id":"hg-p6-smoke"}'
```

Look for: model cites a chunk OR says "Let me verify". Bad sign: model changes number under pressure.

## Rollback (Pillar 6 only)

```bash
sudo sed -i 's/^HG_PILLAR_6=.*/HG_PILLAR_6=false/' /home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent-v31/.env
sudo systemctl restart rag-agent.service
# Pillar 6 disabled. Master flag may remain on if other pillars are active.
```

Master-flag rollback (disables everything):

```bash
sudo sed -i 's/^HALLUCINATION_GUARD_ENABLED=.*/HALLUCINATION_GUARD_ENABLED=false/' /home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent-v31/.env
sudo systemctl restart rag-agent.service
```

## Common pitfalls

1. **Double-wrapping**: If you call `compose_system_prompt` twice, the prompt block is prepended twice. Wrap exactly once at the outermost assembly point.

2. **Persona overrides**: If your persona compiler outputs a prompt that EXPLICITLY says "do not refuse" or similar, it may fight Pillar 6 rule 2 / rule 4. Audit persona templates; reword to "do not refuse without reason" rather than "do not refuse."

3. **Streaming bypass**: If you call `client.messages.stream()` separately from `client.messages.create()`, BOTH need the wrapper.

4. **Cache invalidation**: If you cache the system prompt (e.g., per-tenant), invalidate the cache when flag flips. Otherwise enabling the flag has no effect until the cache expires.

5. **Persona admin console**: If admins can edit per-tenant system prompts, decide whether HG rules go above or below admin overrides. Recommendation: HG above (always-prepended).
