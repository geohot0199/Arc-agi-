# Arc-agi- — ARC Prize 2026 (ARC-AGI-3) Agent Workspace

Working branch for the Kaggle competition **ARC Prize 2026 — ARC-AGI-3** (notebook-only, offline, 9h, open-weights models).

## Documentation

- [`docs/01-research.md`](docs/01-research.md) — benchmark mechanics, RHAE scoring math, field survey (Kaggle milestone winners, preview competition, frontier harness results), and the five lessons that drive the design.
- [`docs/02-architecture.md`](docs/02-architecture.md) — **ATLAS** system architecture: orchestrator, perception, memory (retained-reasoning + compaction), world-model induction & certification, offline planning, execution guard, budgets, roadmap to Milestone 2 (Sept 30) and Final (Nov 2).
- [`docs/03-atlas-v2-critique-review.md`](docs/03-atlas-v2-critique-review.md) — point-by-point adjudication of an external "brutal assessment" (which conflated ARC-AGI-3 with static ARC-AGI-1/2), and the ATLAS v2 revisions it motivated.
- [`docs/05-atlas-v2p1-calibration-review.md`](docs/05-atlas-v2p1-calibration-review.md) — adjudication of the second critique (6 cautions + calibration layer) and the v2.1 spec deltas: archetype posterior with abstention, deterministic Occam tiebreak, coverage-weighted certification, EV stop-loss with hard backstop, latency governor, asymmetric death matching, cross-game tractability priors.

## Status

- [x] Research & architecture design (docs 01–05: research, ATLAS v2/v2.1, two critique adjudications, full manifest)
- [x] Phase 0 — agent `agent/my_agent.py` (perception v0, archetype posterior with abstention, click funnel v2, multi-key death graph, calibration bus, stop-loss hard spine, latency governor, guarded LLM advisor) + tests (`python3 -m unittest discover -s tests`) + notebook builder (`scripts/build_notebook.py`) + Kaggle guide (docs/06)
- [ ] Phase 0 — first valid Kaggle submission (user: accept rules, attach model, Save & Run All, Submit — see docs/06)
- [ ] Phase 1 — memory & perception upgrade (structured notes, compaction, level briefs)
- [ ] Phase 2 — world models (CODE/CERTIFY/PLAN/WATCH, DSL-as-prior, complexity+coverage metrics)
- [ ] Phase 3 — optimization & robustness (scheduler EV, cross-game priors)
- [ ] Phase 4 — open-source release (CC0/MIT-0, prize requirement) & final submission
