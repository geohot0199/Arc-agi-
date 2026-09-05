# ATLAS v2 — Complete System Manifest

Every tool, component, feature, mechanism, and the evidence behind each. This is the full inventory of what the system uses, post-critique-adjudication (`03-atlas-v2-critique-review.md`). No code yet — Phase 0 starts on approval.

---

## A. External stack (what it runs ON)

| # | Tool / Asset | Role | Why this one |
|---|---|---|---|
| A1 | **Kaggle Notebook** | Submission format (hard constraint) | Notebook-only competition; Phase A validation run + Phase B hidden-set rerun; auto-generated submission file |
| A2 | **RTX 6000 (48 GB VRAM, ~48 vCPU)** — GCP `g4-standard-48` | Primary accelerator profile | Largest legal brain offline: 27–32B FP8 with full KV headroom; alternates `t4`/`p100`/`cpu` profiles |
| A3 | **`arc-agi` package (PyPI)** | Game engine + API client | Ships the **same engine the Kaggle gateway runs**; `OperationMode.OFFLINE` = seconds-fast local dev; `COMPETITION` = eval mode |
| A4 | **ARC-AGI-3-Agents framework** | Agent skeleton | `Agent.is_done()` / `Agent.choose_action()` contract + `Swarm` orchestration; our agent subclasses this |
| A5 | **vLLM** | Model server | (1) continuous batching — all concurrent game sessions share one GPU; (2) **automatic prefix caching** — stable prompt prefix = physically reused KV = our offline "retained reasoning"; (3) FP8 quantization |
| A6 | **Qwen3.6-VL 27B FP8** | Primary brain | Multimodal (grid-as-image + ASCII), strong coding; the exact class Milestone-1 winner (The Duck) proved fits 48 GB and the harness style |
| A7 | **Alternates: GPT-OSS-120B (MXFP4), Gemma-4-31B, Qwen3.6-VL-8B / GPT-OSS-20B** | Model bake-off candidates + small-hardware profiles | Decided empirically in Phase 0 (A6 v2); official-template base (GPT-OSS) and Reki's proven pick (Gemma) included |
| A8 | **25 public game files** (`environment_files/`) | Primary dev/test set | Same engine, local, infinite free iterations |
| A9 | **~200+ community games** (arc-interactive) | Generalization testing | Private set is unseen; breadth of unseen games is the only honest proxy |
| A10 | **5-game holdout** (public games, never tuned against) | Overfit guard | 3rd-place "forge" proved local public scores mislead LB; holdout keeps us honest |
| A11 | **Pre-trained action-value net** (small CNN/transformer) | Fallback stage 4 | Trained *before* submission on public + community replays; attached as a **public Kaggle dataset** — explicitly legal ("freely & publicly available external data, including pre-trained models") |
| A12 | **Kaggle CLI + ARC-AGI-3-Kaggle-Starter** | Dev→submit pipeline | Edit locally, `make play-local` (seconds), push notebook, one daily submission treated as a sacred experiment |

---

## B. Layer 0 — Orchestrator (the manager)

| Feature | What it does | Reasoning |
|---|---|---|
| **Concurrency pool (8–16 sessions)** | Plays up to 16 games simultaneously | Games are independent + turn-based; keeps the vLLM batch full; 110 games ÷ ~515 min demands it |
| **Global watchdog** | Owns the 9 h clock minus ~15 min boot / ~10 min save reserve; per-game soft deadlines; kills stalls | Overrun = no submission = 0. Non-negotiable |
| **Marginal-value scheduler** | Routes time to highest expected marginal score; *Invest* (learning games) vs *Harvest* (certified games with heavy later levels) | Level weights rise with index and completion caps the game score ⇒ finishing A's level 5 > polishing B's level 1 |
| **Per-level stop-loss** | Tracks action count vs estimated human baseline; push / retry / abandon at ~5× | Beyond ~5× human actions a level earns ≈ 0; grinding is worthless |
| **Checkpointing** | Persists every game's memory artifacts continuously | Hour-7 crash must not lose the run |
| **Hardware profile auto-select** | Boots the right model+concurrency for `rtx6000` / `t4` / `p100` / `cpu` | Same notebook degrades gracefully; never OOM on a smaller machine |
| **Rerun-hard-games pass** (Phase 3) | Games scoring below threshold get a second run with a stronger config; keep best | Schema's fixed fallback rule (98.98% used exactly this) |

---

## C. Layer 1 — Perception (the eyes — 100% deterministic Python, zero tokens)

| Feature | What it does | Reasoning |
|---|---|---|
| **Grid-stack normalizer** | Handles multi-layer frames; detects animation quiescence | Frames are stacks of grids; acting mid-animation wastes scored actions |
| **Frame differencer** | Sparse cell-level changed/appeared/vanished/moved sets | Cheapest, most reliable signal of what an action *did* |
| **Segmentation** | Connected components by color; shapes/rects; HUD & border detection (regions static across frames) | Separates furniture from play objects; shrinks the LLM's visual field |
| **Entity tracker** | Identifies the candidate player object (moves under ACTION1–5); persistent IDs; movement vectors | "Which blob am I?" answered in code saves probe actions and tokens |
| **Change classifier** | Labels every step: `NOOP / MOVED / OBJECT_CHANGED / NEW_OBJECT / DEATH / LEVEL_UP / WIN / ANIMATION` | Free supervision signal for memory + world model certification |
| **Salient-target extractor** | ACTION6 click candidates: object centroids, player-adjacent cells, recently-changed cells, rare-color objects | Collapses 4096 clicks → ~5–15 plausible (Blind Squirrel's valid-action insight) |
| **Renderers ×3** | (a) ASCII with coordinate rulers, (b) upscaled color PNG for the VLM, (c) cropped zoom tool | The Duck: let the model pick the representation; multimodal + text > either alone |
| **Response-signature stats** (v2) | Aggregates probe outcomes into the archetype signature | Feeds the Archetype Router (E1) |

---

## D. Layer 2 — Memory (the agent's "weights" — the OpenAI lesson lives here)

| Artifact / Mechanism | What it does | Reasoning |
|---|---|---|
| **Timeline** | Append-only `(state_hash, action, outcome, next_state)` ground truth | What certification backtests against; replayable reality (Schema) |
| **game_notes.md** | Action-semantics table ("ACTION3 = left"), object glossary, goal hypothesis, hazards, failed hypotheses | Externalized retained reasoning — always in-context, never evicted |
| **world_model.py** | Versioned executable `step(state, action)` + `is_goal(state)` | Interpretable, verifiable, searchable (Schema's program-not-vector thesis) |
| **Level briefs** | Distilled per-level packets: what still holds + what's new | Blind Squirrel cross-level transfer; mechanics persist across levels by design |
| **Compactor** | On context pressure, transcript → one-line structured conclusions (hypothesis→experiment→result→still-believed?); protected core never evicted | OpenAI: compaction tripled score where rolling truncation lost knowledge |
| **Stable-prefix prompt layout** | System + notes + world model first; transcript appended after | Makes vLLM prefix caching actually hit — the offline `previous_response_id` |
| **Cross-level transfer protocol** | On LEVEL_UP: freeze brief, diff mechanics, keep semantics, switch plan-first; re-probe only what surprise invalidates | Discovery paid once on the tutorial, amortized over heavier levels |

---

## E. Layer 3 — Deliberation core (the brain — one local model, six modes + router)

| Mode / Feature | What it does | Reasoning |
|---|---|---|
| **E1. Archetype Router** (v2) | ~5–7 probe actions → response signature → `MOVEMENT / CLICK_PUZZLE / COLOR_LOGIC / SELECTION / NO_RESPONSE`; routes explore strategy, DSL bias, fallback arming | Kills the "probing for movement in a click-puzzle" failure; cheap, deterministic, evidence-based |
| **E2. EXPLORE** | Minimal systematic probes (disambiguate ACTION1–5 in ≤5 trials; sample salient clicks); every probe logged as hypothesis-test | Can't reason untouched, but probes are scored ⇒ minimal + never repeated (no-op ledger) |
| **E3. HYPOTHESIZE** | Propose/revise mechanic hypotheses + the *goal predicate* from structured diffs + notes | Even "what counts as winning" is hidden (`is_goal` must be inferred) |
| **E4. CODE_WORLD_MODEL** | Write/edit the program from the **6-family DSL** via Python-REPL tool | Small models are best at games *through code* (The Duck); DSL families: (1) movement/physics incl. gravity & per-color channels, (2) color-logic (recolor/gates/rotators/toggles), (3) symmetry & pattern (completion, tiling, periodicity), (4) topology (components, interior/exterior fill, contact), (5) arithmetic & resources (counters, timers, meters), (6) object algebra (union/XOR, pairing, size-rank) |
| **E5. CERTIFY** (v2 upgraded) | **Best-of-n tournament**: sample 4–8 candidates (temp ~0.8), backtest ALL against the full Timeline, keep the **simplest** passing 100% (Occam); anti-vacuity rules (≥N transitions, unexercised actions = unknown); repair loop seeded with best-failing counterexamples | Wrong model caught here costs zero actions; wrong model acted on costs quadratically (RHAE). Best-of-n = affordable test-time scaling for a weak brain (critique's best point) |
| **E6. PLAN** | BFS/A*/greedy search *inside* the certified model → action queue + predicted observations | **The core exploit**: search is free; only `env.step()` is metered (Schema beat humans 42-vs-500 on one level this way) |
| **E7. WATCH** | Execute queue; per-step prediction check vs change classifier; any surprise **voids the queue** → counterexample may indict the *representation*, not just the rule | Schema's deepest insight: persistent failure means the state representation is wrong |
| **E8. Agent tools** | Python REPL, Timeline query, zoom/segmentation, action-history stats | Free compute the model can invoke between scored actions |
| **E9. Reflex fallback** | No-LLM policy: legal-action filter + no-op ledger + death avoidance + salient clicks + movement heuristics | Guarantees forward motion when the model is unsure/looping/slow |

---

## F. Layer 4 — Execution guard (the hands)

| Feature | What it does | Reasoning |
|---|---|---|
| **Protocol validator** | Every action checked vs `available_actions` + ACTION6 bounds before leaving the agent | Invalid action = wasted turn or crash (Prime Agent's guard pattern) |
| **Death-state graph** | Hashed observed states; blocks actions that previously led to GAME_OVER from matching states | Blind Squirrel's state graph; deterministic games make it reliable |
| **No-op ledger** | Per-action change statistics; historically useless actions suppressed | Never pay twice for the same information |
| **Prediction-check voiding** | Queue execution aborts on first surprise | No plan survives contact with a wrong model — stop paying for it |
| **Stagnation detector** | Repeated states / no new info ⇒ escalate strategy ⇒ deliberate RESET (knowledge retained) | RESET with a certified model is cheap; looping is death by stop-loss |

---

## G. v2 fallback chain (per game, budget-sliced, in order)

1. **Certified world model + plan** (primary path)
2. **Archetype policy template** — deterministic per-archetype heuristics (CLICK_PUZZLE: click cells completing detected symmetry; MOVEMENT: greedy goal-directed pathing with death avoidance)
3. **Level-analogy policy** — match new level to previous level's template, apply learned delta
4. **Pre-trained action-value net** (A11) — Blind-Squirrel-style NN proposing, guard disposing
5. **Reflex floor** (E9) — never zero, never stalls

---

## H. Evidence map (every load-bearing feature → its proof)

| Feature | Proven by |
|---|---|
| Retained memory + compaction (D) | OpenAI GPT-5.6: 13.3% → **38.3%**, 6× fewer tokens |
| Certified world model + offline planning (E4–E6) | Schema: **98.98%** public; 42-vs-500 human actions on M0R0 L4 |
| Harness structure ≫ brain size | GPT-5.5 0.4% (plain harness) vs GPT-5.6 38.3% (structured) |
| REPL/code-first agent (E4, E8) | The Duck — Kaggle Milestone-1 **winner**, Qwen 27B + Python tool |
| VLM-as-policy + reflection + ablation flags (C-renderers, A6/A7) | Reki — 2nd place, Gemma-4-31B |
| Valid-action collapse (C-targets) | Blind Squirrel — 4102 actions → handful, 2nd place preview comp |
| Cross-level transfer (D-briefs, G3) | Blind Squirrel action-value transfer + game design (semantics persist) |
| Death-state avoidance (F) | Blind Squirrel state graph |
| Best-of-n + hard filter (E5) | o3's 50-program sampling, scaled to our budget (critique, translated) |
| Free-compute-for-actions (E6, whole thesis) | RHAE math: internal ops unmetered; ratio squared (technical report) |
| Rerun-hard-games (B) | Schema's fixed fallback rule |
| Holdout + community games (A9, A10) | forge's warning: local score ≠ LB score |

---

## I. Budget model (9-hour math)

```
540 min total
 − ~15 min boot (wheels, model load, game discovery)   → 525 min
 − ~10 min final (artifact write, checkpoint flush)    → ~515 min play
Concurrency 8–16 ⇒ effective ~70–100 min/game-equivalent across 110 games
RTX 6000 FP8 ≈ 2–4k output tok/s aggregate under batching
 ⇒ ~30–60M tokens ⇒ ~15–30k deliberations ⇒ ~140–270 per game
Contexts compaction-capped (~24–48k) so prefix caching stays effective
Per-turn wall-clock = gateway latency + inference; batching hides most inference
```

---

## J. Deliberately NOT used

| Excluded | Why |
|---|---|
| External APIs (GPT/Claude/Gemini) at eval | Internet disabled — rules; we borrow findings, not models |
| Static-task solvers (TRANSFORM/ANALOGY/RECURSIVE paths) | ARC-AGI-3 has no grid-to-grid tasks — that's the ARC-AGI-2 competition; dead code here |
| Game-specific hardcoding | Overfits the 25 public games; spirit-of-competition risk; private set punishes it |
| Online RL training inside the notebook | 9 h is too little; per-game training doesn't transfer to 110 unseen games |
| Frontier-model score expectations (65%+) | Not replicable offline on 48 GB; verified Kaggle top = 1.21% |

---

## K. Dev & evaluation tooling (outside the submission)

| Tool | Purpose |
|---|---|
| **Local RHAE scorer** | Replicates official formula (1.15 cap, squared ratio, level weights, completion caps, 5× budget) — identical scoring to Kaggle |
| **Feature flags on every component** | Ablation discipline (Reki); n≥20 runs/game for significance (The Duck variance ±0.45%) |
| **Trace viewer** | JSONL recordings → replays with predicted-vs-actual overlays (where did the world model lie?) |
| **Model bake-off harness** (Phase 0) | World-model coding microevals across DSL families + 5-game end-to-end, per hardware profile |
| **Submission gate** | No Kaggle push without local gate + holdout check; 1 submission/day = 1 experiment/day |
