# Critique Review & ATLAS v2 — Adjudication of the "Brutal Assessment"

A reviewer model assessed ATLAS against ARC-AGI-1/2 (static grid-to-grid puzzles). This competition is **ARC-AGI-3** (interactive games). That one confusion invalidates the critique's central claim — but three of its four "flaws" still contain real improvements once translated to the actual benchmark. This document records the verdict, the adoptions, the rejections, and the resulting **ATLAS v2** deltas.

---

## 1. The category error: there are no static tasks in ARC-AGI-3

The critique's core claim:

> "Critical Flaw #1: ~30% of ARC tasks are NOT simulatable games … `input_grid → cognitive_leap → output_grid` (atemporal, single-step)" — visual analogies, grid algebra, fractal recursion, etc.

**That describes ARC-AGI-1/2. It is false for ARC-AGI-3.** Evidence, from the competition's own materials and every independent source in `01-research.md`:

1. **The competition brief itself** (Kaggle page): "Competition evaluation uses a separate, private set of **110 games**." Tasks "take place in hidden, **interactive environments** that require exploration and multi-step reasoning." The submission is auto-generated **from gameplay** ("As long as the agent takes action on any of the games…"). There is no output-grid to produce, anywhere.
2. **The toolkit API has no task channel**: `FrameData = {game_id, frame (grid stack), state, score (level), available_actions}`. There is no `train_examples` field, no input/output pair, no single-step submit. The only interface is `RESET / ACTION1–7` in a turn-based loop.
3. **All 25 public games are interactive environments** (playable at arcprize.org/tasks; shipped as engine files in `environment_files/`, not as task JSONs).
4. **Every verified result on this benchmark is an agent playing games**: The Duck, Reki, forge, Stochastic Goose, Blind Squirrel, GPT-5.6 Sol, GPT-6 Astra. Nobody reports "transformation tasks" because none exist here.
5. The o3/"75.7%"/"10,000 tokens per task"/"generate 50 programs fitting all examples" numbers the critique cites are from the **static ARC lineage**. In the same era, on ARC-AGI-3, frontier models scored 0.4%–13.3% (official harness). Different benchmark, different physics.
6. The user's own brief states it plainly: *"there are three active competitions for ARC in 2026: this competition [ARC-AGI-3], ARC-AGI-2, and a paper track."* The tasks the critique describes live in the **ARC-AGI-2 competition** — a separate contest we are not entering.

**Consequence:** the proposed "Layer 0 Task Classifier" (`GAME / TRANSFORM / ANALOGY / RECURSIVE`) would return `GAME` 110 times out of 110. Its recommended time split hands **40% of the 9-hour budget to solvers that can never execute** (`TRANSFORM` 30% + `ANALOGY` 5% + `RECURSIVE` 5%). On this competition that is not a safety net — it is self-sabotage. Rejected.

**Where the instinct is still right (adopted):** puzzle-*like cognitive content* can appear **inside** the interactive shell. A game can be "click the cells that complete the symmetry" or "rotate colors until the pattern matches" — the frame is still a game loop, but the winning skill is pattern/symmetry/color reasoning, not pathfinding. Real ARC-AGI-3 games include mechanisms like **color rotators** and **refuel rings** (per the Schema write-up's own examples), not just movement. So: **mechanic breadth is a legitimate concern** → adoption A2 below.

---

## 2. Point-by-point verdict

| Critique claim | Verdict | Rationale |
|---|---|---|
| "#1: ~30–40% of tasks are atemporal grid transformations" | **FALSE** | Category error — describes ARC-AGI-1/2. ARC-AGI-3 is 110 interactive games; the API has no output-grid channel. |
| "#1 sub-point: DSL biased to movement games, missing color/symmetry/topology/arithmetic mechanics" | **RIGHT (translated)** | Game mechanics genuinely span more than move/push/collide (color rotators are real ARC-AGI-3 mechanics). DSL extended → A2. |
| "#1 sub-point: EXPLORE will burn 20+ actions probing movement that doesn't exist" | **OVERSTATED / MITIGATED** | v1 already capped movement probes at ≤5 with a no-op ledger. But the concern is real for click-puzzle games → new cheap **archetype probe** makes the pivot explicit after ~5–7 actions → A1. |
| "#1 sub-point: CERTIFY can pass vacuously (empty timeline)" | **RIGHT (edge case)** | Add coverage rules: no certification below a minimum of recorded transitions; unexercised actions stay "unknown," never "safe" → A4. |
| "#2: Qwen 27B is not a human-level reasoner; harness ≠ brain" | **TRUE BUT MISFRAMED** | We said this ourselves in v1 ("a 27B model won't reach Schema's 99%"). It proves too much: GPT-5.5 — a frontier brain — scored **0.4%** with the plain harness, while the same-class GPT-5.6 with a structured harness hit **38.3%**. On today's Kaggle meta (top = 1.21% with a 27B), structure is precisely the lever that's under-exploited. Mitigations adopted → A3, A5. |
| "#2: o3 used 50-program sampling; you can afford 5–10" | **RIGHT & ADOPTED** | Best-of-n sampling + hard certification filter is the affordable version of test-time scaling for a weak model → A3. |
| "#3: DSL too narrow (gravity, topology, arithmetic, symmetry, pattern completion, object algebra)" | **PARTLY RIGHT** | Wrong as "missing ARC task types," right as "missing game-mechanic families." Six DSL families now → A2. |
| "#4: no fallback when symbolic modeling fails → 0 points" | **PARTLY RIGHT** | v1 already had a reflex layer + stop-loss (not "random moves"), but the fallback *chain* lacked diversity. Adopted: archetype policy templates, level-analogy policy, pre-trained action-value net → A5. |
| "Ceiling: 65–75% with flawless execution" | **NOT CREDIBLE** | Hosted frontier models with the official harness: single digits. Only self-reported mega-harnesses (Schema 98.98% public) and one provider-adapter run (Astra 99.9%, hosted) exceed that — neither is replicable offline on 48 GB. Realistic flawless-execution band for an offline 27B system: mid-single-digits to low-double-digits — which is **milestone-podium class** given the 1.21% top score. |
| "You will not score 100%" | **AGREED** | See §4. Neither will anyone else this year: 100% requires Astra-class hosted models, which the no-internet rule excludes. The $700k bonus is safe from the field. |

---

## 3. ATLAS v2 — adopted changes

### A1. Game Archetype Router (new, replaces the critique's "task classifier")
A deterministic **response-signature probe** at game start (≈5–7 actions, all logged as hypothesis tests):
1. Take ACTION1–5 once each (if legal) from the start state; observe the change classifier's verdict per action.
2. Take one salient ACTION6 click; observe.
3. Signature → archetype:
   - an entity translates → **MOVEMENT** (player, paths, physics)
   - cells toggle/recolor locally at clicks → **CLICK_PUZZLE** (pattern/symmetry reasoning)
   - global recolor/rotation/spawn events → **COLOR_LOGIC / STATE_MACHINE**
   - score/level changes with tiny local edits → **SELECTION/META**
   - nothing responds → probe ACTION6 variety + ACTION7, then RESET-probe
The archetype routes (a) which EXPLORE strategy runs, (b) which DSL family the world-model coder biases toward, (c) which fallback policy template is armed. Kills the "20 wasted probes" failure mode the critique feared.

### A2. Six-family world-model DSL (extended from v1's movement bias)
1. **Movement/physics** — move, push/pull, collide, gravity (incl. per-color-channel), springs, wraps
2. **Color-logic** — recolor rules, color gates/keys, rotators, on/off toggles
3. **Symmetry & pattern** — symmetry detection/completion, tiling/periodicity, pattern extension
4. **Topology** — connected components, interior/exterior fill, boundary/contact relations
5. **Arithmetic & resources** — counters, timers, meters, cell-wise algebra
6. **Object algebra** — union/intersection/XOR, matching/pairing, size/rank ordering

Families 2–6 exist because real games use them (color rotator, refuel ring), **not** because static tasks exist. Bloat control: the router (A1) biases generation so the model composes from 1–2 families first; the full library only engages when certification keeps failing.

### A3. Best-of-n world-model tournament (CERTIFY upgrade)
Instead of one induced program: sample **n = 4–8** candidate `world_model.py` variants (temperature ~0.8), backtest **all** against the full Timeline, select the **simplest program that passes 100%** (Occam). If none pass → repair loop seeded with the best-failing candidate's counterexamples. This converts "weak model writes wrong code" from a fatal fault into a discarded candidate — the affordable analog of the 50-program sampling the critique correctly identified.

### A4. Anti-vacuity & coverage rules for certification
- Certification requires ≥ N recorded transitions (N ≈ 10) covering every action type the model will be allowed to drive.
- Actions never exercised in the Timeline are labeled **unknown**, never predicted-safe; PLAN may only use certified action semantics.
- Vacuous or trivially-true models (e.g., "always NOOP" on a game that moved) are rejected by construction.

### A5. Diversified fallback chain (per game, in order, each with a budget slice)
1. **Certified world model + plan** (primary)
2. **Archetype policy template** — deterministic heuristics per archetype (e.g., CLICK_PUZZLE: "click cells that complete detected symmetry"; MOVEMENT: "greedy path toward goal-colored object with death avoidance")
3. **Level-analogy policy** — new level matched to previous level's template; apply learned delta (mechanics persist across levels by design)
4. **Pre-trained action-value net** — small CNN/transformer trained *before* submission on replays from public + community games (attached as a public Kaggle dataset — explicitly allowed: "freely & publicly available external data … including pre-trained models"); Blind-Squirrel-style, it proposes, perception/guard disposes
5. **Reflex floor** — legal-action filter, no-op ledger, death-state graph avoidance, salient clicks (v1 unchanged)

### A6. Model-swap benchmark (Phase 0, explicit)
Qwen3.6-VL-27B-FP8 vs GPT-OSS-120B (MXFP4) vs Gemma-4-31B on (a) world-model coding microevals from the DSL families, (b) 5 public games end-to-end. The critique is right that the brain is the binding constraint — so measure it before committing, per hardware profile.

### Unchanged from v1 (the critique never actually touched these)
Orchestrator/marginal-value scheduler, watchdog & 9-hour math, memory layer + structured compaction (the OpenAI retained-reasoning/compaction lesson), Timeline, execution guard, level briefs & cross-level transfer, holdout & ablation discipline, open-source requirement.

---

## 4. So will this score 100%? (the honest answer)

No — and the critique's own 65–75% "ceiling" is equally untethered. The verified landscape:

| System | Brain | Harness | Score |
|---|---|---|---|
| Kaggle Milestone-1 winner (The Duck) | Qwen 3.6 27B FP8, offline | REPL harness | **1.21%** (LB) |
| GPT-5.5 High | frontier, hosted | official | 0.4% |
| GPT-5.6 Sol max | frontier, hosted | official | 7.8–13.3% |
| GPT-5.6 Sol | frontier, hosted | retained reasoning + compaction | 38.3% (public) |
| GPT-6 Astra max | frontier, hosted | provider adapter | 99.9% (verified) |
| Schema | frontier, hosted | world-model induction | 98.98% (public, self-reported) |
| **ATLAS v2** | ≤32B open-weight, **offline** | all of the above techniques | **target: 2% → 4–6% → stretch low-double-digits** |

The no-internet rule excludes every 60%+ result in that table. Our realistic goal: **beat 1.21% decisively and take a Milestone-2 / final-LB podium**, with upside if the world-model tournament over-delivers. Anything more is marketing, not forecasting.
