# ATLAS — System Architecture for ARC Prize 2026 (ARC-AGI-3)

> **v2.1** — calibration layer added (archetype posterior with abstention, CERTIFY complexity/coverage metrics, EV stop-loss with hard backstop, latency governor, asymmetric death matching, cross-game priors); see [`05-atlas-v2p1-calibration-review.md`](05-atlas-v2p1-calibration-review.md). **v2** changes from the first external critique are in [`03-atlas-v2-critique-review.md`](03-atlas-v2-critique-review.md) (adopted: archetype router, 6-family DSL, best-of-n certification, diversified fallback chain; rejected: static-task solvers — ARC-AGI-3 has no grid-to-grid tasks; that's the ARC-AGI-2 competition).

**ATLAS** = *Adaptive Task-Learning Agent System*. One sentence: **a local VLM/LLM running inside a structured harness that (1) never forgets what it learned, (2) builds a certified executable world model of each game, and (3) plans inside that model so real actions are spent only on the optimal path.**

It is the synthesis of the three proven playbooks from research (`01-research.md`):
- **OpenAI's GPT-5.6 experiment** → memory retention + compaction (≈3× score)
- **Schema harness (98.98%)** → world-model induction + certification + free offline planning
- **Blind Squirrel / The Duck / Reki** → valid-action model, cross-level transfer, REPL tools, ablation discipline

All of it runs offline in one Kaggle notebook, ≤ 9 h, no internet, on one RTX 6000 (48 GB).

---

## 1. Design principles (in priority order)

1. **Spend thoughts, not actions.** RHAE punishes actions quadratically; internal compute is free. Any decision that can be made by code or search must be.
2. **Never re-learn, never silently forget.** All knowledge is persisted in durable artifacts; context overflow ⇒ structured summarization (compaction), never drop-oldest truncation.
3. **Certify before you trust; verify as you execute.** A world model only drives actions after it backtests perfectly against recorded history; every executed action is checked against its prediction; surprise voids the plan.
4. **Amortize discovery.** Explore deliberately on level 1 (tutorial), transfer semantics to all later levels, plan-first everywhere after.
5. **Degrade gracefully.** Every LLM-dependent component has a deterministic fallback (reflex layer, safe heuristics). The notebook must never crash, stall, or burn the clock — a global watchdog owns the budget.
6. **Stay generic.** Priors at the level of *game-mechanic primitives* (movement, push, key-door, hazards, counters) — never game-specific hardcoding. Private-set generalization is the whole game, and hardcoding risks both.

---

## 2. System overview

```
                                KAGGLE NOTEBOOK (offline, ≤9h, RTX 6000 48GB)
┌─────────────────────────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR (Swarm/Scheduler)                                                 │
│  110 games ▸ concurrency pool ▸ wall-clock & action budgets ▸ checkpointing     │
│        │ spawns one GameSession per game (8–16 in flight)                       │
│        ▼                                                                        │
│  ┌────────────────────────── GameSession (per game) ─────────────────────────┐  │
│  │                                                                           │  │
│  │  PERCEPTION LAYER (token-free, deterministic)                             │  │
│  │  grid stack ▸ frame diff ▸ segmentation/objects ▸ entity tracker ▸        │  │
│  │  change classifier (noop/move/death/levelup) ▸ salient-target extractor   │  │
│  │        │ structured observations (ASCII + rendered image + diffs)         │  │
│  │        ▼                                                                  │  │
│  │  MEMORY LAYER  ───────────── the agent's "weights"                       │  │
│  │  · Timeline (append-only (state, action, next_state) ground truth)        │  │
│  │  · game_notes.md   (rules/action semantics/object glossary/goals)         │  │
│  │  · world_model.py  (executable step() + is_goal(), versioned)             │  │
│  │  · level briefs    (cross-level transfer packets)                         │  │
│  │  · Compactor (structured summarizer, replaces truncation)                 │  │
│  │        │                                                                  │  │
│  │        ▼                                                                  │  │
│  │  DELIBERATION CORE (local VLM/LLM via vLLM, prefix-cached prompts)        │  │
│  │  modes: EXPLORE ▸ HYPOTHESIZE ▸ CODE_WORLD_MODEL ▸ CERTIFY ▸ PLAN ▸ WATCH │  │
│  │        │ commits action queues                                            │  │
│  │        ▼                                                                  │  │
│  │  EXECUTION GUARD                                                          │  │
│  │  queue runner ▸ per-step prediction check ▸ surprise⇒void ▸ legal-action   │  │
│  │  filter ▸ death-state blocker (state graph) ▸ reflex fallback             │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│        │ env.step()                                                              │
│        ▼                                                                          │
│  ARC-AGI GATEWAY (arc-agi pkg, COMPETITION mode — hidden games, submission auto) │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component specifications

### 3.0 Orchestrator (Swarm + Scheduler)
- Discovers games from the gateway, runs **N concurrent GameSessions** (N ≈ 8–16; games are independent and turn-based, so concurrency multiplies effective throughput and keeps the GPU batch full).
- **Global watchdog**: owns the 9 h clock (minus ~15 min boot/save reserve). Per-game soft deadlines; kills/flags stalled sessions; guarantees the notebook completes and writes output.
- **Marginal-value scheduling**: score is level-weighted with completion caps ⇒ time goes where expected marginal score is highest. Two regimes per game: *Invest* (explore/learn on early levels) vs *Harvest* (plan-first execution on later levels, where the weights are). If a game is hopeless (no model after budget), divert time to unfinished promising games.
- **Stop-loss per level**: RHAE dies at ~5× human baseline. Estimator keeps a running human-baseline proxy (from level index, mechanic complexity, observed human-scale action counts) and decides: push / retry / abandon-level.
- Checkpoints all game state (memory artifacts) so a crash never loses the run.

### 3.1 Perception layer (100% deterministic Python — zero tokens)
Every frame passes through:
1. **Grid stack normalization** — handle multi-grid frames (layers/animation), detect quiescence (don't act mid-animation).
2. **Frame differencing** — exactly which cells changed, appeared, vanished, moved (cheap optical-flow on color indices: `Δ = f(t+1) − f(t)` as sparse sets).
3. **Segmentation / objects** — connected components by color; rectangles, clusters, symmetries; HUD/border detection (static regions across frames = non-game furniture).
4. **Entity tracking** — candidate "player" objects (the thing that moves when you press ACTION1–5); persistent IDs across frames; movement vectors.
5. **Change classifier** — label each step's outcome: `NOOP / MOVED / OBJECT_CHANGED / NEW_OBJECT / DEATH / LEVEL_UP / WIN / ANIMATION`. This is the supervision signal for memory and the world model — computed by code, not tokens.
6. **Salient-target extractor** — click candidates for ACTION6: object centroids, cells adjacent to the tracked player, recently-changed cells, rare-colored objects. Collapses 4096 clicks → ~5–15 plausible ones.
7. **Renderers** — (a) compact ASCII with coordinate rulers, (b) PNG image (upscaled, color-coded) for the VLM, (c) cropped zooms on demand (a tool, like The Duck's segmentation zoom).

### 3.2 Memory layer — "the agent's weights" (implements the OpenAI lesson)
Four artifacts per game, all persisted and always reconstructable:

| Artifact | Role | Analog |
|---|---|---|
| **Timeline** | Append-only `(state_hash, action, outcome, next_state)` — the ground truth the world model must replay | Schema's Timeline |
| **game_notes.md** | Living knowledge: action-semantics table ("ACTION3 = left"), object glossary ("cyan blob = player"), goal hypothesis, hazards, mechanics, failed hypotheses | OpenAI's retained reasoning, externalized |
| **world_model.py** | Executable `step(state, action)` + `is_goal(state)`, versioned; edits must pass certification | Schema's induced program |
| **Level briefs** | Distilled packets carried across levels: "what still holds" + per-level deltas | Blind Squirrel's cross-level value model |

**Compactor** (not truncation): when a session's context crosses the budget (~24–48k tokens), the oldest transcript segments are *summarized into structured notes* — each probe becomes one line: `hypothesis → experiment → result → conclusion → still-believed?`. The world model, action table, and glossary are **never** compacted away. Prompts are laid out **stable-prefix-first** (system + notes + world model … then append-only transcript) so **vLLM automatic prefix caching** gives near-free "retained reasoning" across steps.

**Cross-level transfer protocol**: on `LEVEL_UP` → freeze a level brief, diff mechanics vs. previous level, keep action semantics (they persist by game design), switch session to *plan-first* mode; only re-probe what the new level invalidated (detected by prediction surprise).

### 3.3 Reasoning core (local model, vLLM)
- **Model** (Phase-0 benchmark decides; all offline via Kaggle Models):
  - Primary: **Qwen3.6-VL 27B/32B FP8** — multimodal (image+ASCII grids), strong coding — matches The Duck's proven-fit class.
  - Alternates: GPT-OSS-120B (MXFP4, partial CPU offload on 48 GB), Gemma-4-31B (Reki's pick), Qwen3.6-VL-8B / GPT-OSS-20B (T4 fallback profile).
- Served by a **single vLLM process with continuous batching** — all concurrent GameSessions share the GPU; perception text is prepared by the harness so prompts are small.
- **Six deliberation modes** (state machine per session; each is a prompt template + tool set):
  1. **EXPLORE** — systematic, minimal probes: disambiguate ACTION1–5 (probably U/D/L/R/interact) with ≤5 cheap trials on level 1; sample salient clicks; every probe logged as hypothesis-test.
  2. **HYPOTHESIZE** — given structured diffs + notes, propose/revise mechanic hypotheses and the *goal predicate*.
  3. **CODE_WORLD_MODEL** — write/edit `world_model.py` over a **primitive library** (grid, objects, movement, push, collision, counter, key-door, color-logic, spawn/kill) — generic DSL blocks, composable, never game-specific.
  4. **CERTIFY** — backtest the program against the *entire* Timeline: every recorded `(s, a)` must predict `s′` exactly (or within tolerated abstraction). Fail ⇒ pointed bug report ⇒ repair loop. This gate is what makes planning safe.
  5. **PLAN** — search inside the certified model (BFS/A*/greedy over action macros) ⇒ emit an **action queue** with predicted observations. Zero environment cost.
  6. **WATCH** — execute the queue; per-step prediction check (from the change classifier): green = continue; **surprise = void remaining queue** → back to HYPOTHESIZE/CODE with the counterexample (which may indict the *representation*, not just the rule — Schema's key insight).
- **Reflex fallback** (no LLM): if the model is low-confidence, times out, or loops: legal-action filter + no-repeat-(state,action) + death-state graph avoidance + movement heuristics. Guarantees forward motion.

### 3.4 Execution guard
- Validates every action against `available_actions` and ACTION6 bounds (protocol guard, à la Prime Agent).
- **State-graph death avoidance**: hashed observed states; never voluntarily take an action that previously led to `GAME_OVER` from a matching state.
- **No-op ledger**: per-action statistics; actions that historically change nothing get suppressed.
- Stagnation detector (repeated states / no new information) ⇒ escalate: change strategy → deliberate RESET (fresh eyes, level knowledge retained) if budget allows.

### 3.5 Budget model (the 9-hour math)
```
540 min total
 − ~15 min boot: model load, wheel install, game discovery        → 525 min play
 − ~10 min final: submission artifact, checkpoint flush            → ~515 min
Concurrency 8–16 sessions ⇒ effective 70–100 min/game-equivalent
GPU: RTX 6000 Ada FP8 ≈ 2–4k output tok/s aggregate under batching
      ⇒ ~30–60M tokens total; at ~1.5–3k tokens/deliberation ⇒ ~15–30k deliberations
      ⇒ ~140–270 deliberations per game ⇒ feasible with compaction-capped contexts
Wall-clock per turn: gateway latency + inference; batching hides most of it
```
Profiles: `rtx6000` (27B FP8), `t4/p100` (8B/20B quantized), `cpu` (reflex-only) — auto-selected at boot.

---

## 4. How the GPT-5.6/Astra findings map onto ATLAS

| OpenAI finding | In their system | In ATLAS (offline/local) |
|---|---|---|
| Reasoning discarded after each action → model re-derives the game every step | Fixed by `previous_response_id` (retained reasoning) | Reasoning is **externalized into durable artifacts** (notes, world model, action table) that are always in-context; plus prefix caching so the KV of prior context is physically reused |
| Rolling truncation → loses old observations; fuller context impairs perf | Fixed by **compaction** (summarize + continue) | **Structured compactor**: transcripts → one-line conclusions appended to notes; world model/glossary never evicted |
| 3× score, 6× fewer output tokens | GPT-5.6 Sol: 13.3% → 38.3% | We expect the same *mechanisms* (less per-step re-derivation, coherent accumulated strategy) to be our cheapest wins |
| Their conclusion: "evals measure the harness bundle, not the model alone" | Provider-adapter harness: GPT-6 Astra → 99.9% (ARC-verified) | We can't call Astra offline — so we substitute **compute-for-actions**: certified world model + free offline planning converts invisible compute into RHAE efficiency (Schema's 98.98% proves the ceiling of this substitution) |

---

## 5. Development loop & evaluation

- **Local-first**: `arc-agi` offline engine + 25 public games + community games (200+, arc-interactive) for generalization testing.
- **Local RHAE scorer** replicating the official formula (incl. 1.15 cap, level weights, completion caps, 5× budget) — score every experiment identically to Kaggle.
- **5-game holdout** (never tuned against) to fight the "local score ≠ LB" trap that forge documented.
- **Feature flags for every component** (Reki's discipline): perception, memory, compaction, world model, transfer, reflex — each ablatable via env var; nightly ablation grid on 20-runs-per-game (The Duck's variance: ±0.45% means n≥20 for significance).
- **Trace viewer**: JSONL recordings rendered as replays with predicted-vs-actual overlays (where did the world model lie?).
- Submission cadence: local gate before every Kaggle run; daily submission budget treated as sacred experiments.

---

## 6. Roadmap (today = Sept 4, 2026; Milestone 2 = Sept 30; Final = Nov 2)

| Phase | Window | Deliverable | Exit criterion |
|---|---|---|---|
| **0 — Infra** | Week 1 (Sept 4–11) | Starter-kit integration, local engine + RHAE scorer, vLLM serving offline on RTX 6000 profile, random+reflex baseline, **first valid Kaggle submission** | Submission scores > 0; eval loop < 10 min |
| **1 — Memory & Perception** | Sept 11–20 | Perception layer, notes/memory + compaction, valid-action proposer, EXPLORE/HYPOTHESIZE/WATCH modes, watchdog + budgets | Public-local ≥ 2% (beats The Duck's 1.6 local / 1.21 LB) |
| **2 — World models** | Sept 20–28 | CODE_WORLD_MODEL + CERTIFY + PLAN, primitive DSL, surprise loop, cross-level transfer | Public-local ≥ 4–6%; **Milestone-2 submission Sept 28** |
| **3 — Optimization** | Oct | Scheduler marginal-value tuning, stop-loss, model swap-in benchmarking, two-profile fallback (Schema's rerun-hard-games pattern), robustness | Private-ready; local holdout within ~1 pt of public |
| **4 — Final & open source** | Late Oct | MIT-0/CC0 release, write-up (also eligible for the $450k Paper Track), final submission Nov 2 | Prizes require open source — non-negotiable |

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| 27B model too weak to induce world models | Certification gate means partial/wrong models are *caught*, not fatal; DSL primitives make coding tractable; reflex layer always underneath; Schema-style approach degrades to "hypothesis notes + cautious probing" |
| Time overrun (9 h) | Watchdog owns clock; per-game soft deadlines; checkpointing; concurrency tuned in Phase 0 |
| Overfitting to public games | 5-game holdout; generic-priors rule; community games for generalization checks |
| Gateway latency/instability | Retries w/ backoff, protocol guard, recordings for post-mortem |
| RESET/quiescence semantics unknown | Phase-0 empirical tests (research doc §5) |
| Non-determinism breaks state graph | Hash on observed frames; treat matches as probabilistic evidence, not proof |

## 8. Explicit non-goals / rules-compliance
- No game-specific hardcoding of solutions or mechanics keyed to specific game IDs (spirit-of-competition violation + overfitting).
- No external APIs at eval time (internet is disabled anyway).
- Everything open-sourced CC0/MIT-0 before private scoring (prize requirement).
