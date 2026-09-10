#!/usr/bin/env python3
"""Assemble the whole submission into ONE pasteable Python file.

The notebook build (scripts/build_notebook.py) splits the submission across
cells because that is what `kaggle kernels push` wants.  This builder produces
the equivalent as a single script for people who would rather paste one blob
into one Kaggle cell.

Layout matters and is not arbitrary:

  1. prelude  - install the offline wheels, stub the framework package, and set
                every ATLAS_* flag.  All of this must happen FIRST, because the
                agent reads its configuration exactly once, at import time.
  2. agent    - agent/my_agent.py verbatim.  Its base-class resolution tries
                `from .agent import Agent`, then `from agents.agent import
                Agent`; the prelude has already put the latter in sys.modules.
  3. driver   - the offline evaluation loop, adapted to use the MyAgent class
                defined above instead of re-importing it from /tmp.

Usage:  python scripts/build_single_file.py  ->  notebooks/kaggle_run.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "agent" / "my_agent.py"
OUT = ROOT / "notebooks" / "kaggle_run.py"

PRELUDE = '''\
# =====================================================================
# ATLAS -- ARC Prize 2026 (ARC-AGI-3)  |  single-file Kaggle runner
# =====================================================================
# Paste this whole file into ONE Kaggle code cell and run it.
#
# It plays the public games Kaggle ships in `environment_files` and prints
# RHAE per game, scored by the competition's own EnvironmentScorecard.
#
# Requirements: the competition input must be attached, so these exist:
#   /kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files
#   /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels
#   /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents
#
# Tunables are in the CONFIG block just below.
# ---------------------------------------------------------------------
import importlib.util
import json
import logging
import os
import subprocess
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
COMP = '/kaggle/input/competitions/arc-prize-2026-arc-agi-3'
WHEELS_DIR = COMP + '/arc_agi_3_wheels'
MAX_GAMES = 25            # Kaggle currently ships 25 public games
MAX_WORKERS = 6           # games played concurrently
WALL_CLOCK_MINUTES = 150  # hard stop for the whole evaluation
WORK_DIR = os.environ.get('ATLAS_WORK_DIR', '/kaggle/working')

# ---------------------------------------------------------------------
# 1a. install the toolkit from the offline wheelhouse (no internet)
# ---------------------------------------------------------------------
try:
    import arc_agi  # noqa: F401
except ImportError:
    print('installing arc-agi from', WHEELS_DIR)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index',
                    '--find-links', WHEELS_DIR, 'arc-agi', 'python-dotenv'],
                   check=False)

# ---------------------------------------------------------------------
# 1b. make `agents.agent.Agent` importable without the framework's extras
#
# The agent inherits its action loop (Agent.main) from ARC-AGI-3-Agents, but
# that package's __init__ eagerly imports langchain / langgraph / smolagents /
# openai, none of which are installed here.  Only the base class is needed, so
# a stub package pointing at the real directory is registered and agent.py is
# loaded by path; its own relative imports resolve through the stub's __path__.
# ---------------------------------------------------------------------
def _find_framework_dir():
    override = os.environ.get('ATLAS_AGENTS_DIR')
    if override and os.path.isdir(os.path.join(override, 'agents')):
        return override
    for path in (COMP + '/ARC-AGI-3-Agents',
                 '/kaggle/input/ARC-AGI-3-Agents',
                 WORK_DIR + '/ARC-AGI-3-Agents'):
        if os.path.isdir(os.path.join(path, 'agents')):
            return path
    import glob
    hits = sorted(glob.glob('/kaggle/input/**/ARC-AGI-3-Agents', recursive=True))
    return hits[0] if hits else None


def load_framework():
    if 'agents.agent' in sys.modules:
        return sys.modules['agents.agent'].Agent
    framework = _find_framework_dir()
    if framework is None:
        return None
    package = types.ModuleType('agents')
    package.__path__ = [os.path.join(framework, 'agents')]
    sys.modules['agents'] = package
    path = os.path.join(framework, 'agents', 'agent.py')
    spec = importlib.util.spec_from_file_location('agents.agent', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['agents.agent'] = module
    spec.loader.exec_module(module)
    return framework


FRAMEWORK_DIR = load_framework()

# ---------------------------------------------------------------------
# 1c. ATLAS flags -- must be set BEFORE the agent code below is executed,
#     because the agent reads the environment once, at import time.
# ---------------------------------------------------------------------
os.environ.setdefault('ATLAS_MAX_ACTIONS', '360')
os.environ.setdefault('ATLAS_LEVEL_BUDGET', '110')
os.environ.setdefault('ATLAS_LEVEL_HARD_CAP', '220')
os.environ.setdefault('ATLAS_LOG', '0')
os.environ.setdefault('ATLAS_LLM', '0')       # deterministic; no model needed
os.environ.setdefault('ATLAS_TRACE_DIR', os.path.join(WORK_DIR, 'atlas_traces'))
os.environ.setdefault('ATLAS_CAL_FILE',
                      os.path.join(WORK_DIR, 'atlas_calibration.jsonl'))
logging.disable(logging.WARNING)


# =====================================================================
# 2. THE AGENT  (agent/my_agent.py, verbatim)
# =====================================================================
'''

DRIVER = '''

# =====================================================================
# 3. DRIVER -- play the games and report RHAE
# =====================================================================
def find_games_dir():
    override = os.environ.get('ATLAS_GAMES_DIR')
    if override and os.path.isdir(override):
        return override
    for path in (COMP + '/environment_files',
                 '/kaggle/input/arc-prize-2026-arc-agi-3/environment_files',
                 '/kaggle/input/environment_files'):
        if os.path.isdir(path):
            return path
    import glob
    hits = sorted(glob.glob('/kaggle/input/**/environment_files', recursive=True))
    return hits[0] if hits else None


def play(agent_cls, arc, game_id, card_id):
    env = arc.make(game_id, scorecard_id=card_id)
    if env is None:
        return {'game_id': game_id, 'actions': 0, 'levels': 0,
                'error': 'environment failed to load'}
    agent = agent_cls(
        card_id=card_id, game_id=env.info.game_id, agent_name='atlas',
        ROOT_URL='http://localhost:8001', record=False, arc_env=env, tags=[],
    )
    started = time.time()
    error = None
    try:
        agent.main()
    except Exception as exc:                       # a crash must stay visible
        error = '%s: %s' % (type(exc).__name__, exc)
    return {
        'game_id': game_id,
        'actions': agent.action_counter,
        'levels': agent.levels_completed,
        'seconds': round(time.time() - started, 1),
        'telemetry': agent.telemetry() if hasattr(agent, 'telemetry') else {},
        'error': error,
    }


def offline_eval():
    if FRAMEWORK_DIR is None:
        print('ARC-AGI-3-Agents not found. The agent inherits its action loop')
        print('(Agent.main) from that framework, so it cannot play without it.')
        print('Attach the competition input, or set ATLAS_AGENTS_DIR.')
        return
    games_dir = find_games_dir()
    if games_dir is None:
        print('environment_files not found. Attach the competition input:')
        print('  ' + COMP + '/environment_files')
        return

    from arc_agi import Arcade, OperationMode

    print('framework   :', FRAMEWORK_DIR)
    print('games dir   :', games_dir)
    print('agent       :', MyAgent.__name__,
          '| MAX_ACTIONS =', MyAgent.MAX_ACTIONS,
          '| ATLAS_LLM =', os.environ.get('ATLAS_LLM', '0'))

    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=games_dir,
        recordings_dir=os.path.join(WORK_DIR, 'recordings'),
    )
    available = list(arc.available_environments)
    if not available:
        print('the engine found no playable environments in', games_dir)
        return

    game_ids = [e.game_id for e in available][:MAX_GAMES]
    print('environments:', len(available), '| playing:', len(game_ids),
          '| workers:', MAX_WORKERS)

    card_id = arc.open_scorecard(tags=['offline-eval'])
    started = time.time()
    deadline = started + WALL_CLOCK_MINUTES * 60

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = []
        for gid in game_ids:
            if time.time() > deadline:
                print('wall-clock budget reached; not starting', gid)
                break
            futures.append((gid, pool.submit(play, MyAgent, arc, gid, card_id)))
        for gid, future in futures:
            remaining = deadline - time.time()
            try:
                results.append(future.result(timeout=max(60.0, remaining)))
            except Exception as exc:
                future.cancel()
                results.append({'game_id': gid, 'actions': 0, 'levels': 0,
                                'error': '%s: %s' % (type(exc).__name__, exc)})

    scorecard = arc.close_scorecard(card_id)
    elapsed = time.time() - started

    rows = {}
    for environment in scorecard.environments:
        run = max(environment.runs, key=lambda r: r.score)
        rows[environment.id] = {
            'rhae': round(run.score, 2),
            'levels': run.levels_completed,
            'total_levels': len(run.level_scores or []),
            'actions': run.actions,
            'level_actions': run.level_actions,
            'level_baselines': run.level_baseline_actions,
            'level_scores': [round(s, 1) for s in (run.level_scores or [])],
        }

    print()
    print('=' * 78)
    print('%-16s %7s %8s %8s  per-level actions (ours / human baseline)'
          % ('game', 'RHAE', 'levels', 'actions'))
    print('-' * 78)
    total = 0.0
    for gid in game_ids:
        row = rows.get(gid)
        outcome = next((r for r in results if r['game_id'] == gid), {})
        if row is None:
            print('%-16s %7s  NO SCORE  error=%s'
                  % (gid[:16], '--', outcome.get('error')))
            continue
        total += row['rhae']
        detail = ' '.join('%s/%s' % (a, b) for a, b in
                          zip(row['level_actions'], row['level_baselines']))
        print('%-16s %7.2f %4s/%-4s %8s  %s'
              % (gid[:16], row['rhae'], row['levels'], row['total_levels'],
                 row['actions'], detail))
        if outcome.get('error'):
            print('%-16s !! %s' % ('', outcome['error']))

    mean = total / len(game_ids) if game_ids else 0.0
    print('-' * 78)
    print('MEAN RHAE over %d games: %.2f%%   wall clock %.1fs'
          % (len(game_ids), mean, elapsed))
    cleared = sum(1 for gid in game_ids
                  if rows.get(gid)
                  and rows[gid]['levels'] >= rows[gid]['total_levels'])
    print('games fully cleared: %d/%d   total actions: %d'
          % (cleared, len(game_ids), sum(r['actions'] for r in results)))
    crashed = [r['game_id'] for r in results if r.get('error')]
    if crashed:
        print('CRASHED:', crashed)
    print('=' * 78)

    payload = {'mean_rhae': round(mean, 3), 'games_played': len(game_ids),
               'games_cleared': cleared, 'seconds': round(elapsed, 1),
               'games': rows, 'runs': results}
    try:
        os.makedirs(WORK_DIR, exist_ok=True)
        out = os.path.join(WORK_DIR, 'offline_results.json')
        with open(out, 'w') as fh:
            json.dump(payload, fh, indent=2, default=str)
        print('wrote', out)
    except Exception as exc:
        print('could not write offline_results.json:', exc)


if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    print('competition rerun detected: use the notebook build instead')
    print('(scripts/build_notebook.py -> notebooks/submission.ipynb), which')
    print('drives the gateway and emits submission.parquet.')
else:
    offline_eval()
'''


def split_future(text: str) -> tuple[str, str]:
    """Pull `from __future__` imports out of the agent body.

    They are only legal as the first statement in a file, and the agent is no
    longer at the top once the prelude is prepended -- so they move to line 1
    of the assembled script.  `ast.parse` does NOT catch this; only the
    compiler does, which is why the assembled file has to be executed, not
    merely parsed, before it is called working.
    """
    kept, rest = [], []
    for line in text.splitlines():
        if line.startswith("from __future__"):
            kept.append(line)
        else:
            rest.append(line)
    return "\n".join(kept), "\n".join(rest)


def main() -> None:
    if not AGENT_SRC.exists():
        raise SystemExit(f"Could not find {AGENT_SRC}")
    future, agent = split_future(AGENT_SRC.read_text())
    agent = agent.strip("\n") + "\n"
    header = (future + "\n" if future else "") + PRELUDE
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(header + agent + DRIVER)
    print(f"[build_single_file] wrote {OUT.relative_to(ROOT)} "
          f"({len(OUT.read_text().splitlines())} lines)")


if __name__ == "__main__":
    main()
