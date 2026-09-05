# ARC Prize 2026 — Research Notes (as of 2026-09-04)

Everything below informed the architecture in `02-architecture.md`. Sources linked inline.

---

## 1. The benchmark mechanics (ground truth from docs/toolkit)

**Environment interface** (from [arc-agi-3 PyPI](https://pypi.org/project/arc-agi-3/) and [ARC-AGI Toolkit docs](https://docs.arcprize.org/toolkit/overview)):

- Agent implements two methods: `is_done(frames, latest_frame)` and `choose_action(frames, latest_frame)`. A `Swarm` orchestrates parallel play across games.
- `FrameData` contains:
  - `game_id`
  - `frame` — **3D list of grids** (a stack of 2D grids, not just one), each up to 64×64, cells 0–15 (16 colors), (0,0) top-left
  - `state` — `NOT_PLAYED | NOT_FINISHED | WIN | GAME_OVER`
  - `score` — current level (0–254); level-ups are observable
  - `available_actions` — which actions are legal right now
- Actions: `RESET`, `ACTION1–5` (simple; typically Up/Down/Left/Right/Interact), `ACTION6` (complex: click at (x, y) — 4096 variants), `ACTION7`. `action.reasoning` is attachable; `ACTION6` uses `set_data({"x":…, "y":…})`.
- Total action space per turn ≈ **4102 actions** (5 basic + 4096 click cells + reset) — per [Blind Squirrel (2nd place, preview comp)](https://github.com/wd13ca/ARC-AGI-3-Agents).
- Games are **turn-based and deterministic** (there is a `seed` param on `Arcade.make()`; ARC's own validation ran deterministic qualification regimes — [ARC-AGI-3 technical background](https://aiwiki.ai/wiki/arc-agi_3)).
- Toolkit: `arc_agi.Arcade` with `OperationMode.OFFLINE` (local `environment_files/`, no API — the dev loop), `ONLINE` (API + scorecards), and `COMPETITION` (what the Kaggle gateway uses). Recordings are JSONL.

**Scoring — RHAE (Relative Human Action Efficiency)** ([DataCamp explainer](https://www.datacamp.com/blog/arc-agi-3), [TechTimes](https://www.techtimes.com/articles/321661/20260727/claude-opus-5-took-arc-agi-3-record-equation-no-ai-had-written-before.htm)):

```
level_score  = min(1.15, human_actions / ai_actions)^2
game_score   = weighted avg of level scores, weight = level index (1-indexed)
total_score  = avg of game scores, capped by completion (no 100% unless every level cleared)
```

- Human baseline = **upper-median best first-run** of ~10 first-time testers; a game is only retained if ≥2 humans fully solve it.
- Squaring ⇒ waste is punished **quadratically** (2× actions = 25% score, 10× ≈ 1%).
- Operational cap ≈ **5× human action budget per level** (beyond that a level earns ~0).
- Average human tester ≈ **48% RHAE** (OpenAI's estimate from official logs) — even humans don't play at the cap.

**Key consequences for design:**
1. Every environment action is expensive; **internal compute (thinking, search, simulation) is free**. The meter only counts `env.step()` calls.
2. **Later levels weigh more** (weights 1..n) and completion gates the score ⇒ finishing games matters more than acing level 1.
3. Exploration must be **front-loaded and amortized**: pay discovery cost on the tutorial level, then play efficiently.

---

## 2. Competition & infrastructure constraints

From the [Kaggle competition page](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3) + [ARC-AGI-3-Kaggle-Starter docs](https://docs.arcprize.org/arc-prize-2026):

- **Notebook-only** submission; Phase A "Save & Run All" validates; **Phase B rerun** plays the hidden set. Submission file auto-generated once any action is taken.
- **Internet disabled** during eval ⇒ no GPT/Claude/Gemini APIs. Open-weight local models only (Kaggle Models / attached datasets allowed). This is deliberate — ARC wants open, locally-executable systems ([aiwiki](https://aiwiki.ai/wiki/arc-agi_3)).
- **≤ 9 hours runtime**; accelerators: `cpu`, `t4`, `p100`, `rtx6000`. RTX 6000 machine = GCP `g4-standard-48` ⇒ **48 GB VRAM**, ~48 vCPU, lots of system RAM. That fits a 27–32B FP8 model comfortably, with headroom for a 70B-class 4-bit model.
- Eval set: **110 private games** — 55 for Public LB, 55 for Private LB.
- Prize eligibility requires **open-sourcing (CC0 / MIT-0)** before private scores are awarded.
- Local dev loop: `arc-agi` PyPI package ships the **same game engine** the Kaggle gateway runs; `make play-local` runs the 25 public games in seconds ([starter docs](https://docs.arcprize.org/arc-prize-2026)).
- Caution from 3rd-place Milestone-1 ("forge"): **local public-game scores are not a reliable LB proxy** — the private set differs; keep holdout games and don't overfit.
- Community games exist for extra training/generalization testing ([arc-interactive](https://github.com/theredbluepill/arc-interactive): 200+ community games).

---

## 3. Where the field stands (Sept 2026)

### 3a. Kaggle competition (offline, local models — our actual battlefield)

| Rank (Milestone 1, June 30 2026) | Solution | Score | Approach |
|---|---|---|---|
| 1st | Tufa Labs — **"The Duck"** | **1.21%** (1.30% retracted) | Qwen 3.6 27B FP8 via vLLM + Python-REPL harness; game state as Python variables; context eviction for indefinite play ([write-up](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133)) |
| 2nd | **Reki** | ~1% | Gemma-4-31B **VLM-as-policy**: renders frames as labeled images → one JSON action/step; reflection memory every ~10 steps; plan queue of 1–4 actions; every feature behind env-var flags for ablation ([ARC blog](https://arcprize.org/blog/arc-prize-2026-milestone-1)) |
| 3rd | **forge** | <1% | GPT-OSS-120B official template wrapped in configurable framework; candidate-generator + arbiter + confidence-based safe moves (top run disabled the extra machinery) |

The Duck's own eval: **1.60 ± 0.45%** over 20 runs on the 25 public games. Public set is *harder* than semi-private (ARC: expect semi-private ≥ public by up to 15 pts).

Also notable: a discussion post reports that attempting **only L0 of every game** (quick wins, breadth-first) yielded 0.02–0.05 — pure breadth doesn't work; you need real level-clearing depth ([discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/703990)).

### 3b. Preview competition (2025, 3 public + 3 private games)

| Solution | Score | Approach |
|---|---|---|
| **Stochastic Goose** (Tufa Labs) | **12.58%** | RL with CNNs trained per-game |
| **Blind Squirrel** (Will Dick) | 2nd | No LLM: **state graph** (avoid known-death states), **rule-based valid-actions model** (shrink 4102 actions, down-weight historically-no-op actions), **NN action-value model trained on previous levels** (cross-level transfer!) ([repo](https://github.com/wd13ca/ARC-AGI-3-Agents)) |

### 3c. Frontier-model harness results (API, not available to us — but the playbook is)

| System | Score | Set | Key |
|---|---|---|---|
| GPT-5.5 High (official harness) | 0.4% | semi-private | amnesia per step |
| GPT-5.6 Sol max (official harness) | 7.78% semi-private / 13.33% public | | per-step re-derivation |
| **GPT-5.6 Sol + retained reasoning + compaction (OpenAI Responses API harness)** | **38.3%** | public | **~3× score, 6× fewer output tokens** — [OpenAI post](https://openai.com/blog/arc-agi-3-harness) |
| **[schema] Opus 4.8 + Fable 5** | **98.98% (self-reported)** | public | executable world-model induction + certification + free offline planning ([schema-harness.github.io](https://schema-harness.github.io/)) |
| **GPT-6 Astra max** | **62.7% standard / 99.9% provider-adapter (ARC-verified, Sept 3 2026)** | semi-private | provider harness with reasoning retention + compaction; used fewer actions than human baseline on 96% of completed levels |

Human reference: ~48% average first-time tester; 100% = upper-median-best human.

---

## 4. The five lessons that define our architecture

### L1 — Memory is the multiplier (OpenAI GPT-5.6 experiment)
Official harness **discarded all private reasoning after every action** and used **rolling truncation** on history. Turning on (a) **retained reasoning** and (b) **compaction** (summarize-then-continue instead of drop-oldest) tripled the score (13.3% → 38.3%) and cut output tokens 6×. Two mechanisms: the model stops re-deriving the game each step (less thinking per action), and it accumulates coherent strategy over time.
→ **Our translation:** per-game persistent memory (`game_notes.md` + world model + action-semantics table) that is always in-context; transcript compaction by *structured summarization*, never silent truncation. With local models we can't keep hidden chain-of-thought tokens, but we CAN (1) externalize reasoning as durable artifacts, and (2) exploit **vLLM prefix caching** with a stable-prompt-prefix design so the growing context is physically cached in KV — our local analog of `previous_response_id`.

### L2 — Actions are quadratic, thinking is free (RHAE math + Schema)
RHAE squares the action ratio; internal operations (tool calls, reasoning, search, retries) **are not counted**. Schema's ~99% runs: induce an executable `step(state, action)` + `is_goal(state)` program, **certify** it by backtesting against the entire recorded timeline, then **plan inside the program** (BFS/search costs zero environment actions) and commit only the optimized action queue, with per-step prediction checks (surprise ⇒ void the queue, re-deliberate). Pay discovery once; recompute plans for free.
→ **Our translation:** world-model induction + certification + offline planning is our core efficiency engine. A 27B model won't induce a perfect world model — but even a *partially* correct model shrinks action counts massively, and the certification gate prevents acting on wrong theories.

### L3 — Action semantics persist across levels (Blind Squirrel + game design)
Within a game, ACTION1–7 keep their meaning across levels; levels escalate by combining mechanics. Blind Squirrel's action-value model trained on level *k* predictions for level *k+1*. Humans read the tutorial once. Level 1 is usually a tutorial — cheap to clear, and the exploration spent there is the cheapest education you'll get.
→ **Our translation:** explicit **cross-level transfer** — on level-up, distill "what still holds" (action semantics, object glossary, hazards, goal hypothesis) into the persistent notes; never re-learn from scratch; each new level starts with plan-first, probe-only-if-surprised.

### L4 — Small local models need structure, not scale (Kaggle results)
A 27B FP8 model with a good REPL harness scores ~1.2–1.6%. Same-class models as JSON-action-policies score ~1%. Frontier models with the *same* simple harness score 13%. The delta between 13% and 99% is **harness structure** (memory + world models + planning). Since we can't buy model scale offline, we must buy structure: token-free perception, curated prompts, certified world models, heuristic fallbacks.
→ Also: model choice still matters — multimodality was called out as a main improvement driver in The Duck's development; grids rendered as *images* + ASCII both, with zoom/segmentation tools.

### L5 — The action space must be collapsed (Blind Squirrel)
4102 actions/turn can't be searched or sampled naively. Blind Squirrel's valid-actions model cut it to a plausible handful and down-weighted historically-useless actions.
→ **Our translation:** a deterministic, token-free **action proposer**: only click *salient* targets (object centroids, cells adjacent to the player-object, changed cells, colored-key objects), never repeat a (state, action) that was a no-op, maintain per-action no-op statistics; keep ACTION1–5 ordering hypotheses (probably movement) and disambiguate them with 4 cheap probes on level 1.

---

## 5. Open questions we'll resolve empirically (Phase 0/1)

1. Does `RESET` count toward the level's action total? (Test locally; affects retry policy.)
2. How much per-turn latency does the Kaggle gateway add, and what concurrency does it tolerate? (Calibrates scheduler.)
3. Grid stack semantics: when do frames contain multiple 2D grids, and do layers differ semantically (e.g., background/animation)?
4. Animation handling: some actions trigger multi-frame animations — how many no-op steps until quiescence, and do those count as actions? (Blind Squirrel noted grid changes arrive "sometimes as a series of grids that together produce an animation".)
5. Model choice on RTX 6000 48 GB: Qwen3.6-VL-27B/32B FP8 vs GPT-OSS-120B (needs offload) vs Gemma-4-31B — benchmark on 25 public games + community games.
6. Determinism in practice: is the private gateway seeded identically per RESET? (If yes, state-graph replay is a reliable death-avoidance tool.)
