#!/usr/bin/env python3
"""Run ATLAS against the local benchmark fixtures on the REAL ARC-AGI engine.

This is the end-to-end check for the harness.  It uses:

  * ``arc_agi.Arcade`` in OFFLINE mode          — the shipped engine, unmodified
  * ``agents.agent.Agent`` from a checkout of ARC-AGI-3-Agents — the real loop
  * ``arc_agi.scorecard``                       — the real RHAE calculation

so the numbers it prints come from the same code paths the Kaggle gateway uses,
not from a re-implementation.  Only the games are ours (the 110 scored games are
never downloadable).

Usage
-----
    harness/setup.sh                                   # once: engine + framework
    python harness/run_local.py                        # all fixtures, 4 threads
    python harness/run_local.py --games mv01,hz01 -v   # subset, verbose agent
    python harness/run_local.py --agent random         # baseline for comparison
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import baselines  # noqa: E402

DEFAULT_AGENTS_DIR = os.environ.get(
    "ATLAS_AGENTS_DIR", os.path.join(ROOT, "third_party", "ARC-AGI-3-Agents"))


# ---------------------------------------------------------------------------
# module loading
# ---------------------------------------------------------------------------
def load_agents_base(agents_dir: str):
    """Import the real ``agents.agent.Agent`` without the framework's extras.

    ``agents/__init__.py`` imports every template (langchain, langgraph,
    smolagents, ...).  We only need the base class, so we register a stub
    package that points at the real directory and load ``agents/agent.py``
    directly; its relative imports (``.recorder``, ``.tracing``) resolve
    through the stub's ``__path__`` and have no heavy dependencies.
    """
    if not os.path.isdir(os.path.join(agents_dir, "agents")):
        raise SystemExit(
            f"ARC-AGI-3-Agents not found at {agents_dir}\n"
            "Run harness/setup.sh, or set ATLAS_AGENTS_DIR to a checkout.")
    package = types.ModuleType("agents")
    package.__path__ = [os.path.join(agents_dir, "agents")]
    sys.modules["agents"] = package
    path = os.path.join(agents_dir, "agents", "agent.py")
    spec = importlib.util.spec_from_file_location("agents.agent", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["agents.agent"] = module
    spec.loader.exec_module(module)
    return module.Agent


def load_agent(agent_file: str):
    spec = importlib.util.spec_from_file_location("atlas_agent", agent_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules["atlas_agent"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# one game
# ---------------------------------------------------------------------------
def play_game(agent_cls, arc, game_id: str, card_id: str, quiet: bool) -> dict:
    env = arc.make(game_id, scorecard_id=card_id)
    if env is None:
        return {"game_id": game_id, "error": "environment failed to load"}
    agent = agent_cls(
        card_id=card_id, game_id=env.info.game_id, agent_name="atlas-local",
        ROOT_URL="http://localhost:8001", record=False, arc_env=env, tags=[],
    )
    started = time.time()
    error = None
    try:
        agent.main()
    except Exception as exc:                     # a crash must be visible
        error = f"{type(exc).__name__}: {exc}"
    return {
        "game_id": game_id,
        "actions": agent.action_counter,
        "levels": agent.levels_completed,
        "seconds": round(time.time() - started, 1),
        "telemetry": agent.telemetry() if hasattr(agent, "telemetry") else {},
        "error": error,
    }


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", default="",
                        help="comma-separated fixture ids (default: all)")
    parser.add_argument("--jobs", type=int, default=4,
                        help="games played concurrently (the Swarm is threaded)")
    parser.add_argument("--max-actions", type=int, default=0)
    parser.add_argument("--deadline", type=float, default=0.0)
    parser.add_argument("--agents-dir", default=DEFAULT_AGENTS_DIR)
    parser.add_argument("--agent-file",
                        default=os.path.join(ROOT, "agent", "my_agent.py"))
    parser.add_argument("--json", default=os.path.join(ROOT, "working",
                                                       "local_results.json"))
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="keep the agent's per-transition logging")
    parser.add_argument("--llm", default="0",
                        help="ATLAS_LLM value (0 = deterministic only)")
    args = parser.parse_args()

    os.environ["ATLAS_LLM"] = args.llm
    os.environ["ATLAS_LOG"] = "1" if args.verbose else "0"
    os.environ.setdefault("ATLAS_TRACE_DIR", os.path.join(ROOT, "working", "traces"))
    if args.max_actions:
        os.environ["ATLAS_MAX_ACTIONS"] = str(args.max_actions)
    if args.deadline:
        os.environ["ATLAS_DEADLINE_S"] = str(args.deadline)

    logging.disable(logging.WARNING)

    baselines.write_metadata()
    agent_base = load_agents_base(args.agents_dir)          # noqa: F841
    agent_module = load_agent(args.agent_file)
    agent_cls = agent_module.MyAgent

    from arc_agi import Arcade, OperationMode

    games_dir = os.path.join(HERE, "games")
    wanted = [g.strip() for g in args.games.split(",") if g.strip()]
    game_ids = [gid for gid, _c, _k in baselines.GAMES
                if not wanted or gid in wanted]
    if not game_ids:
        raise SystemExit(f"no fixtures matched {wanted}")

    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=games_dir,
        recordings_dir=os.path.join(ROOT, "working", "recordings"),
    )
    available = {e.game_id.split("-")[0] for e in arc.available_environments}
    missing = [g for g in game_ids if g not in available]
    if missing:
        raise SystemExit(f"fixtures not found by the engine: {missing}")

    card_id = arc.open_scorecard(tags=["atlas-local"])
    print(f"engine: arc_agi OFFLINE | games={len(game_ids)} | "
          f"jobs={args.jobs} | scorecard={card_id[:8]}", flush=True)

    started = time.time()
    results = []
    if args.jobs <= 1:
        for gid in game_ids:
            results.append(play_game(agent_cls, arc, gid, card_id, args.verbose))
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(play_game, agent_cls, arc, gid, card_id,
                                   args.verbose): gid for gid in game_ids}
            for future in futures:
                results.append(future.result())

    scorecard = arc.close_scorecard(card_id)
    elapsed = time.time() - started

    # ---- report -----------------------------------------------------------
    optimal = baselines.optimal_table()
    rows = {}
    for environment in scorecard.environments:
        run = max(environment.runs, key=lambda r: r.score)
        rows[environment.id.split("-")[0]] = {
            "rhae": round(run.score, 2),
            "levels": run.levels_completed,
            "total_levels": len(run.level_scores or []),
            "actions": run.actions,
            "level_actions": run.level_actions,
            "level_scores": [round(s, 1) for s in (run.level_scores or [])],
        }

    print("\n" + "=" * 78)
    print(f"{'game':8} {'RHAE':>7} {'levels':>8} {'actions':>8} "
          f"{'optimal':>8}  per-level actions (ours / baseline)")
    print("-" * 78)
    total = 0.0
    for gid in game_ids:
        row = rows.get(gid)
        result = next(r for r in results if r["game_id"] == gid)
        if row is None:
            print(f"{gid:8} {'--':>7}  NO SCORE  error={result.get('error')}")
            continue
        total += row["rhae"]
        base = baselines.baseline_table()[gid]
        detail = " ".join(f"{a}/{b}" for a, b in
                          zip(row["level_actions"], base))
        print(f"{gid:8} {row['rhae']:7.2f} "
              f"{row['levels']:>3}/{row['total_levels']:<4} "
              f"{row['actions']:>8} {sum(optimal[gid]):>8}  {detail}")
        if result.get("error"):
            print(f"{'':8} !! {result['error']}")
    mean = total / len(game_ids) if game_ids else 0.0
    print("-" * 78)
    print(f"TOTAL RHAE (mean over {len(game_ids)} games): {mean:.2f}%   "
          f"wall clock {elapsed:.1f}s")
    won = sum(1 for r in results
              if rows.get(r["game_id"], {}).get("levels", 0)
              >= rows.get(r["game_id"], {}).get("total_levels", 1))
    print(f"games fully cleared: {won}/{len(game_ids)}   "
          f"total actions: {sum(r['actions'] for r in results)}")
    crashes = [r["game_id"] for r in results if r.get("error")]
    if crashes:
        print(f"CRASHED: {crashes}")
    print("=" * 78)

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump({"total_rhae": round(mean, 3), "games": rows,
                       "runs": results, "seconds": round(elapsed, 1)},
                      fh, indent=2, default=str)
        print(f"wrote {args.json}")
    return 1 if crashes else 0


if __name__ == "__main__":
    sys.exit(main())
