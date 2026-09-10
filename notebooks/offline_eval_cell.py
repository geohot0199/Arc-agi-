# =====================================================================
# OFFLINE EVALUATION -- play the public games Kaggle ships
# =====================================================================
# An interactive Kaggle session has no gateway sidecar, so this cell drives
# the engine straight from `environment_files`.  It goes through the same
# arc_agi.Arcade + EnvironmentScorecard code path that the competition rerun
# uses, so the RHAE numbers printed below are computed by the competition's
# own scorer rather than by a re-implementation.
#
# Nothing here can block a submission: on a real rerun this cell is a no-op,
# and every failure path prints a reason and carries on.

import importlib.util
import json
import logging
import os
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor

MAX_GAMES = 25          # Kaggle currently ships 25 public games
MAX_WORKERS = 6         # games played concurrently
WALL_CLOCK_MINUTES = 150  # hard stop for the whole evaluation

# Both are overridable so the cell can be pointed at any games directory and
# any output directory (which is also what makes it testable off-Kaggle).
WORK_DIR = os.environ.get('ATLAS_WORK_DIR', '/kaggle/working')


def find_games_dir():
    override = os.environ.get('ATLAS_GAMES_DIR')
    if override and os.path.isdir(override):
        return override
    candidates = [
        '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files',
        '/kaggle/input/arc-prize-2026-arc-agi-3/environment_files',
        '/kaggle/input/environment_files',
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    import glob
    hits = sorted(glob.glob('/kaggle/input/**/environment_files', recursive=True))
    return hits[0] if hits else None


def find_framework_dir():
    override = os.environ.get('ATLAS_AGENTS_DIR')
    if override and os.path.isdir(os.path.join(override, 'agents')):
        return override
    candidates = [
        '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents',
        '/kaggle/input/ARC-AGI-3-Agents',
        '/kaggle/working/ARC-AGI-3-Agents',
    ]
    for path in candidates:
        if os.path.isdir(os.path.join(path, 'agents')):
            return path
    import glob
    hits = sorted(glob.glob('/kaggle/input/**/ARC-AGI-3-Agents', recursive=True))
    return hits[0] if hits else None


def load_framework():
    """Make `agents.agent.Agent` importable without the framework's extras.

    `agents/__init__.py` eagerly imports every bundled template (langchain,
    langgraph, smolagents, openai, ...), none of which are installed here.
    Only the base class is needed, so a stub package pointing at the real
    directory is registered and `agents/agent.py` is loaded by path; its own
    relative imports resolve through the stub's `__path__`.
    """
    if 'agents.agent' in sys.modules:
        return sys.modules['agents.agent'].Agent
    framework = find_framework_dir()
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
    print('framework   :', framework)
    return module.Agent


def load_agent(path='/tmp/my_agent.py'):
    """Import the agent written by the previous cell.

    The framework's own `agents/__init__.py` eagerly imports optional
    templates (langchain, smolagents, openai, ...) that are not installed
    here, so the module is loaded by path instead of by package import.
    """
    spec = importlib.util.spec_from_file_location('atlas_agent', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['atlas_agent'] = module
    spec.loader.exec_module(module)
    return module.MyAgent


def play(agent_cls, arc, game_id, card_id):
    env = arc.make(game_id, scorecard_id=card_id)
    if env is None:
        return {'game_id': game_id, 'actions': 0, 'levels': 0,
                'error': 'environment failed to load'}
    agent = agent_cls(
        card_id=card_id, game_id=env.info.game_id, agent_name='atlas-offline',
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
    games_dir = find_games_dir()
    if games_dir is None:
        print('environment_files not found. Attach the competition input:')
        print('  /kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files')
        return

    # The agent reads its configuration once, at import time, so every flag
    # has to be in the environment BEFORE the module is loaded.  These mirror
    # the .env the rerun cell writes, so offline and scored runs behave alike.
    os.environ.setdefault('ATLAS_MAX_ACTIONS', '360')
    os.environ.setdefault('ATLAS_LEVEL_BUDGET', '110')
    os.environ.setdefault('ATLAS_LEVEL_HARD_CAP', '220')
    os.environ.setdefault('ATLAS_LOG', '0')
    os.environ.setdefault('ATLAS_TRACE_DIR', os.path.join(WORK_DIR, 'atlas_traces'))
    os.environ.setdefault('ATLAS_CAL_FILE', os.path.join(WORK_DIR, 'atlas_calibration.jsonl'))
    logging.disable(logging.WARNING)

    base = None
    try:
        base = load_framework()
    except Exception as exc:
        print('WARNING: could not load the ARC-AGI-3-Agents base class:', exc)
    if base is None:
        print('ARC-AGI-3-Agents not found. The agent inherits its action loop')
        print('(Agent.main) from that framework, so it cannot play without it.')
        print('Attach the competition input, or set ATLAS_AGENTS_DIR.')
        return

    try:
        agent_cls = load_agent()
    except Exception as exc:
        print('could not import /tmp/my_agent.py:', exc)
        print('run the "%%writefile /tmp/my_agent.py" cell first.')
        return
    if not issubclass(agent_cls, base):
        print('WARNING: the agent did not bind to the framework base class;')
        print('it will have no main() loop. Check ATLAS_AGENTS_DIR.')
        return

    from arc_agi import Arcade, OperationMode

    print('games dir   :', games_dir)
    print('agent       :', agent_cls.__name__,
          '| MAX_ACTIONS =', agent_cls.MAX_ACTIONS,
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
            futures.append((gid, pool.submit(play, agent_cls, arc, gid, card_id)))
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

    # ---- report --------------------------------------------------------
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
    scored = 0
    for gid in game_ids:
        row = rows.get(gid)
        outcome = next((r for r in results if r['game_id'] == gid), {})
        if row is None:
            print('%-16s %7s  NO SCORE  error=%s' % (gid[:16], '--',
                                                     outcome.get('error')))
            continue
        total += row['rhae']
        scored += 1
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
                  if rows.get(gid) and rows[gid]['levels'] >= rows[gid]['total_levels'])
    print('games fully cleared: %d/%d   total actions: %d'
          % (cleared, len(game_ids), sum(r['actions'] for r in results)))
    crashed = [r['game_id'] for r in results if r.get('error')]
    if crashed:
        print('CRASHED:', crashed)
    print('=' * 78)

    payload = {
        'mean_rhae': round(mean, 3),
        'games_played': len(game_ids),
        'games_cleared': cleared,
        'seconds': round(elapsed, 1),
        'games': rows,
        'runs': results,
    }
    try:
        with open(os.path.join(WORK_DIR, 'offline_results.json'), 'w') as fh:
            json.dump(payload, fh, indent=2, default=str)
        print('wrote', os.path.join(WORK_DIR, 'offline_results.json'))
    except Exception as exc:
        print('could not write offline_results.json:', exc)


if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    print('competition rerun: the gateway plays the games -- skipping offline eval')
else:
    offline_eval()
