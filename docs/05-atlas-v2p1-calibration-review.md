# ATLAS v2.1 — Calibration Layer Review (second external critique, adopted)

Adjudication of the six cautionary points + the "missing calibration layer" claim.
Source: reviewer feedback received 2026-09-05, after Phase 0 build started.
Result: **5.5/6 adopted, calibration layer adopted, three pushbacks registered.**

---

## 1. Verdicts

| # | Point | Verdict | Key change |
|---|---|---|---|
| 1 | Router must not hard-classify | **Adopt** | Bayesian posterior + entropy abstention + info-gain probes (capped) |
| 2 | Smarter ACTION6 funnel | **Adopt** (extension of existing salience design) | player-proximity, click-success ledger, info-gain scoring |
| 3 | DSL must not be a cage | **Adopt** | world-model language = arbitrary Python; DSL = stdlib *prior*, never an acceptance gate |
| 4 | Stop-loss as EV, not rule | **Adopt + pushback** | EV(continue) vs EV(divert) with learning-value term, inside a **hard backstop ceiling** |
| 5 | 27B bottleneck; aggressive replay tests | **Adopt** (half existed as CERTIFY; additions real) | explicit complexity tiebreak; coverage-weighted certification; posterior updates on WATCH surprise |
| 6 | Latency governor | **Adopt** | telemetry, cheap paths (plan execution = 0 LLM calls), adaptive consult rate |
| ★ | Calibration layer across Router/CERTIFY/scheduler | **Adopt — best item** | numeric confidence + thresholds + logging everywhere |
| ★ | Death-graph hash brittleness | **Adopt + asymmetry rule** | multi-key + near-miss; exact = hard block, near-miss = soft warn only |
| ★ | Cross-game meta-learning for scheduler | **Adopt + guards** | tractability priors from archetype/response signatures, shrinkage, per-game budget floor |

## 2. Pushbacks (where the fix could create new failure modes)

1. **EV stop-loss needs a hard spine.** A 27B-estimated EV is systematically optimistic, and "learning value" can rationalize infinite grinding on level 1. The EV computation decides *within* an absolute action/time ceiling that never moves. Soft brain, hard spine.
2. **Fuzzy death-matching must be asymmetric.** Over-blocking valid actions is a silent failure just as bad as under-blocking. Exact multi-key match ⇒ hard block; Hamming near-miss ⇒ warning + caution bias only, never a block. Hidden state (timers/counters not present in the frame) means near-miss evidence is *weak* evidence.
3. **Calibration thresholds can themselves overfit.** Keep thresholds few and coarse; tune only against the dev split, never the 5-game holdout; log threshold-sensitive decisions so ablations can audit them.

## 3. v2.1 specification deltas

### 3.1 Archetype Posterior (replaces hard router)
- State: `P(archetype)` over {MOVEMENT, CLICK_PUZZLE, COLOR_LOGIC, SELECTION, OTHER}; init uniform-ish with tiny mass on OTHER.
- Update: every classified transition is evidence (movement of a tracked entity ⇒ MOVEMENT; local toggle at click ⇒ CLICK_PUZZLE; global recolor ⇒ COLOR_LOGIC; small edit + score change ⇒ SELECTION). Watch-surprise also updates: repeated movement-plan failures downweight MOVEMENT.
- Abstention: if entropy > threshold after base probes ⇒ do NOT commit; run up to +2 info-gain probes — the action maximizing expected entropy reduction between the top-2 hypotheses — and only if their plans diverge.
- Usage: explore strategy = argmax; DSL bias = top-2 weighted; fallback templates armed by mass, not by winner-take-all.

### 3.2 World-model language
- Acceptance test for CERTIFY = "replays recorded history within abstraction tolerance" — **no DSL-membership requirement**. The model may write `discovered_mechanic(state, action)` free-form Python. DSL primitives exist as a curated stdlib to *speed convergence*, nothing more.

### 3.3 CERTIFY v2.1
- Complexity metric (deterministic Occam tiebreak): `complexity = AST_nodes + 2 × branch_count + 3 × free_parameters`.
- Coverage score: fraction of timeline transitions that *discriminate* between the top candidate models (a transition discriminates if candidates disagree on its predicted next state). Best-of-n winner must (a) pass 100% replay, (b) max coverage, (c) min complexity — in that lexicographic order.
- Certified-with-gaps: if winner's coverage < threshold, plans are restricted to action subsequences whose semantics are gap-free (no free parameters exercised in the gap region).
- Every rejection emits a counterexample used to seed repair.

### 3.4 Scheduler v2.1
- Stop-loss: `EV(continue) = P(clear) × level_value + P(learn) × transfer_value_to_later_levels` vs `EV(divert) = expected marginal score of next-best game × diverted minutes`. Decisions inside hard ceilings: per-level absolute action cap, per-game absolute time cap.
- Cross-game priors: after each game, record (archetype posterior, response signature, time-to-first-level-clear, final score). Scheduler predicts tractability of unseen games via k-NN over signatures with shrinkage toward the mean until ≥10 games are recorded; every game keeps a minimum budget floor regardless of prior.

### 3.5 Calibration bus (the unifying layer)
- One structured log channel. Every judgment emits `{component, judgment, value, threshold, decision, t}`.
- Trace viewer renders calibration timelines — silent failures (router misclassification, coverage-gap certification, near-miss deaths) become visible.
- Components on the bus: archetype posterior/entropy, certification pass/coverage/complexity, plan confidence, stop-loss EV terms, death near-miss distance, latency-governor decisions.

### 3.6 Latency governor
- Telemetry: tokens/action, deliberations/level, actions-saved/deliberation, score/min, gateway RTT.
- Cheap paths: executing a certified plan queue = zero LLM calls (pure Python + prediction check); reflex handles low-stakes turns; LLM consulted only at genuine decision points; consult rate adapts to measured score/min.
- Per-deliberation token ceilings; world-model tournaments budgeted in tokens, not just candidate count.

### 3.7 Death graph v2.1
- Multi-key: exact grid hash, object-multiset hash, player-relative layout hash.
- Near-miss: Hamming distance below threshold on the primary grid.
- Asymmetry: exact ⇒ block; near-miss ⇒ warn (bias scores, never hard block).

## 4. Phase 0 code alignment (agent/my_agent.py already in repo)

Phase 0 ships a fixed probe script and evidence-weighted reflex scoring — deliberately below v2.1 ambition. Upgrade path, in order:
1. `noop_ledger`/`action_progress` already implement point-2-style evidence weighting at action granularity (per-cell-region click ledger comes with the perception upgrade).
2. Add archetype posterior + abstention before the world-model phase (small, self-contained).
3. CERTIFY/complexity/coverage land with the world-model phase; calibration bus lands with the trace viewer.
4. Latency governor lands when the LLM advisor moves from optional to default (vLLM integration phase).
