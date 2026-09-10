from __future__ import annotations
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
"""ATLAS v3 — Adaptive Task-Learning Agent System for ARC Prize 2026 (ARC-AGI-3).

A single-file, stdlib-only agent. The Kaggle notebook splices this file into the
ARC-AGI-3-Agents framework as ``agents/templates/my_agent.py`` and registers the
class ``MyAgent`` under the name ``myagent``.

WHY THIS DESIGN (verified facts, not folklore)
---------------------------------------------
The scoring function was read out of the shipped wheel ``arc_agi/scorecard.py``:

    level_score_i = min(115, 100 * (baseline_actions_i / actions_taken_i) ** 2)   # only if completed
    raw_game      = sum(level_score_i * i) / sum(i)                              # weight = level index
    completion_cap= 100 * sum(i for completed) / sum(i for all)
    RHAE(game)    = min(raw_game, completion_cap)
    RHAE(total)   = mean over games

Consequences that shape every line below:

1. Actions are squared; internal compute is free.  Search, simulation, retries,
   re-induction and LLM calls cost nothing.  Only ``env.step()`` is metered.
2. ``Card.inc_reset_count`` increments ``actions`` -> RESET COSTS ONE METERED
   ACTION.  It is a recovery tool, not a free undo.
3. ``ARCBaseGame.handle_reset`` performs a *level* reset mid-game (full reset
   only when the engine action count is 0 or the state is WIN), so RESET replays
   the current level from its start and its cost is charged to that level.
4. Later levels carry more weight (weight = level index) and the completion cap
   gates everything -> finishing games beats acing level 1.
5. ``EnvironmentScoreList.score = max(run.score for run in runs)`` -> the best
   play of a game is what counts, so a clean late run can rescue a messy start.

The architecture follows the only published harness that reaches ~99% RHAE on
the public set ([schema], 98.98% with Opus 4.8 + Fable 5): an append-only
Timeline of ground-truth transitions, an *executable* world model that is
CERTIFIED by replaying that history, planning inside the certified model (free),
a single commit channel from thinking to action, and execution under per-step
prediction checks so a surprise voids the queue instead of costing more actions.

GPT-6 Astra's verified ARC runs (62.7% standard vs 99.9% provider-adapter, with
the adapter staying above 96% even at reasoning effort "none") show that the
memory policy dominates model reasoning depth.  We cannot retain a local
model's hidden reasoning, so we externalise it: durable compact symbolic notes
(Astra was observed inventing shorthand for objects, coordinates, rules and
unfinished plans) plus a stable prompt prefix so vLLM's KV cache does the work
that ``previous_response_id`` does for a hosted model.

LAYER MAP
---------
 0  config, calibration bus, logging
 1  engine adapters            (action ids, frame access, protocol)
 2  perception                 (token-free: diff, segment, entity, events)
 3  symbolic state             (compact, hashable, diffable)
 4  Notes                      (durable memory = the agent's "weights")
 5  Timeline                   (append-only ground truth)
 6  WorldModel + induction     (deterministic, no LLM)
 7  certify                    (backtest + Occam + anti-vacuity)
 8  plan                       (BFS inside the certified model: free actions)
 9  guard                      (protocol validity, death graph, ledgers)
10  scheduler                  (EV stop-loss with an immovable hard spine)
11  advisor                    (optional local LLM, sandboxed, governed)
12  MyAgent                    (the 10-step policy loop + fallback chain)
"""


import ast
import json
import math
import os
import random
import time
import urllib.request
import zlib
from collections import Counter, defaultdict, deque

# ---------------------------------------------------------------------------
# Framework imports.  Three contexts must work:
#   (a) installed as agents/templates/my_agent.py  -> relative import
#   (b) ARC-AGI-3-Agents on sys.path               -> absolute import
#   (c) standalone (unit tests / local harness)    -> object base class
# ---------------------------------------------------------------------------
try:  # (a)
    from .agent import Agent  # type: ignore
except ImportError:  # pragma: no cover
    try:  # (b)
        from agents.agent import Agent  # type: ignore
    except ImportError:  # (c)
        Agent = object  # type: ignore

try:
    from arcengine import GameAction  # type: ignore
except ImportError:  # pragma: no cover
    GameAction = None  # type: ignore


# ===========================================================================
# 0. CONFIG / CALIBRATION BUS / LOGGING
# ===========================================================================
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


CFG = {
    # ---- action budget (metered resource) --------------------------------
    "MAX_ACTIONS": _env_int("ATLAS_MAX_ACTIONS", 1400),
    "LEVEL_BUDGET": _env_int("ATLAS_LEVEL_BUDGET", 90),       # soft: exploit
    "LEVEL_HARD_CAP": _env_int("ATLAS_LEVEL_HARD_CAP", 240),  # hard: conserve
    "DEADLINE_S": _env_float("ATLAS_DEADLINE_S", 1500.0),     # per-game wall clock
    # ---- deliberation ----------------------------------------------------
    "PLAN_DEPTH": _env_int("ATLAS_PLAN_DEPTH", 14),
    "PLAN_NODES": _env_int("ATLAS_PLAN_NODES", 6000),
    "CERTIFY_ACCURACY": _env_float("ATLAS_CERTIFY_ACCURACY", 0.85),
    "CERTIFY_ACCURACY_LOOSE": _env_float("ATLAS_CERTIFY_ACCURACY_LOOSE", 0.70),
    "CERTIFY_COVERAGE": _env_float("ATLAS_CERTIFY_COVERAGE", 0.30),
    "DELIBERATE_EVERY": _env_int("ATLAS_DELIBERATE_EVERY", 4),
    "DELTA_TRIALS": _env_int("ATLAS_DELTA_TRIALS", 24),
    # ---- perception ------------------------------------------------------
    "SALIENT_CAP": _env_int("ATLAS_SALIENT_CAP", 12),
    "ABSTAIN_ENTROPY": _env_float("ATLAS_ABSTAIN_ENTROPY", 1.3),
    "MAX_INFO_PROBES": _env_int("ATLAS_MAX_INFO_PROBES", 2),
    "NEARMISS_FRAC": _env_float("ATLAS_NEARMISS_FRAC", 0.06),
    "ANIM_MIN_FRAMES": _env_int("ATLAS_ANIM_MIN_FRAMES", 2),
    # ---- recovery --------------------------------------------------------
    "STAGNATION_RESET": _env_int("ATLAS_STAGNATION_RESET", 7),
    "MAX_RESETS_PER_LEVEL": _env_int("ATLAS_MAX_RESETS_PER_LEVEL", 3),
    # ---- LLM advisor -----------------------------------------------------
    "LLM_MODE": os.getenv("ATLAS_LLM", "auto").strip().lower(),
    "LLM_URL": os.getenv("ATLAS_LLM_URL", "http://127.0.0.1:1234/v1").rstrip("/"),
    "LLM_TIMEOUT": _env_float("ATLAS_LLM_TIMEOUT", 45.0),
    "LLM_EVERY": _env_int("ATLAS_LLM_EVERY", 6),
    "LLM_MAX_FAILS": _env_int("ATLAS_LLM_MAX_FAILS", 4),
    "LLM_MAX_EVERY": _env_int("ATLAS_LLM_MAX_EVERY", 24),
    "LLM_ALLOW_CODE": _env_bool("ATLAS_LLM_ALLOW_CODE", True),
    # ---- telemetry -------------------------------------------------------
    "LOG": _env_bool("ATLAS_LOG", True),
    "CAL_FILE": os.getenv("ATLAS_CAL_FILE", ""),
    "TRACE_DIR": os.getenv("ATLAS_TRACE_DIR", ""),
    "SEED": _env_int("ATLAS_SEED", 1337),
}

AGENT_NAME = "atlas-v3"
_COLOR_CHARS = "0123456789abcdef"
ARCHETYPES = ("MOVEMENT", "CLICK_PUZZLE", "COLOR_LOGIC", "SELECTION", "OTHER")

_CAL_BUF: list = []


def cal_log(component: str, judgment: str, value, threshold, decision: str) -> None:
    """Emit one calibration record.  No judgment in ATLAS is ever silent."""
    entry = {
        "t": round(time.time(), 3),
        "component": component,
        "judgment": judgment,
        "value": round(value, 4) if isinstance(value, float) else value,
        "threshold": threshold,
        "decision": decision,
    }
    _CAL_BUF.append(entry)
    if len(_CAL_BUF) > 4000:
        del _CAL_BUF[:2000]
    if CFG["LOG"]:
        print(f"[cal] {component}.{judgment}={entry['value']} "
              f"(thr={threshold}) -> {decision}", flush=True)
    if CFG["CAL_FILE"]:
        try:
            with open(CFG["CAL_FILE"], "a") as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception:
            pass


def _log(msg: str) -> None:
    if CFG["LOG"]:
        print(f"[atlas] {msg}", flush=True)


# ===========================================================================
# 1. ENGINE ADAPTERS
# ===========================================================================
# The engine ships GameAction as an IntEnum-like Enum whose members are
# process-wide singletons; FrameData.available_actions is a list[int] of ids.
# We keep our own tables so the agent also runs without arcengine installed.
ACTION_IDS = {
    "RESET": 0, "ACTION1": 1, "ACTION2": 2, "ACTION3": 3,
    "ACTION4": 4, "ACTION5": 5, "ACTION6": 6, "ACTION7": 7,
}
ID_TO_NAME = {v: k for k, v in ACTION_IDS.items()}
SIMPLE_ACTIONS = ("RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4",
                  "ACTION5", "ACTION7")
COMPLEX_ACTIONS = ("ACTION6",)          # requires x, y in [0, 63]
PLAY_ACTIONS = tuple(f"ACTION{i}" for i in range(1, 8))
COORD_LIMIT = 63                        # ComplexAction: x/y Field(ge=0, le=63)


def action_name(action) -> str:
    """Normalise any action representation (enum / int / str) to a NAME."""
    if action is None:
        return "RESET"
    name = getattr(action, "name", None)
    if isinstance(name, str):
        return name.upper()
    if isinstance(action, int):
        return ID_TO_NAME.get(action, "RESET")
    try:
        return str(action).upper()
    except Exception:
        return "RESET"


def is_complex(name: str) -> bool:
    return name in COMPLEX_ACTIONS


def available_names(latest_frame) -> set:
    """Legal action NAMES for the current frame.

    ``FrameData.available_actions`` is ``list[int]`` in the shipped engine, but
    be tolerant: older builds exposed enum members, mocks expose strings.
    """
    try:
        raw = list(getattr(latest_frame, "available_actions", None) or [])
    except Exception:
        raw = []
    out = set()
    for item in raw:
        if isinstance(item, int) and not isinstance(item, bool):
            out.add(ID_TO_NAME.get(item, ""))
        else:
            out.add(action_name(item))
    out.discard("")
    if not out:  # no information: assume the documented default set
        out = {"RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4",
               "ACTION5", "ACTION6"}
    return out


def frame_levels(latest_frame) -> int:
    for attr in ("levels_completed", "score"):
        val = getattr(latest_frame, attr, None)
        if isinstance(val, int):
            return val
    return 0


def frame_state(latest_frame) -> str:
    state = getattr(latest_frame, "state", None)
    try:
        return str(getattr(state, "name", state)).upper()
    except Exception:
        return "UNKNOWN"


def grids_of(frame) -> list:
    """Normalise any frame payload to a list of 2D grids (frame stack)."""
    payload = getattr(frame, "frame", frame)
    if payload is None:
        return []
    try:
        payload = list(payload)
    except TypeError:
        return []
    if not payload:
        return []
    first = payload[0]
    if isinstance(first, (int, float)):          # a bare 2D grid
        return [payload]
    try:
        if len(first) and isinstance(first[0], (int, float)):
            return [payload]                     # a bare 2D grid
    except TypeError:
        return []
    return payload


def dominant_grid(frame) -> tuple:
    """The quiescent grid: the LAST layer of the stack.

    ``ARCBaseGame.perform_action`` keeps rendering frames until the action is
    complete, so a multi-layer stack is an animation and its last layer is the
    settled state.
    """
    stack = grids_of(frame)
    if not stack:
        return ()
    try:
        return tuple(tuple(int(c) for c in row) for row in stack[-1])
    except Exception:
        return ()


def grid_hash(grid: tuple) -> int:
    """Process-stable hash (zlib.crc32, not Python's salted ``hash``)."""
    if not grid:
        return 0
    return zlib.crc32(repr(grid).encode("utf-8"))


# ===========================================================================
# 2. PERCEPTION  (pure python, zero tokens)
# ===========================================================================
def diff_cells(g1: tuple, g2: tuple, cap: int = 500) -> list:
    """[(x, y, old, new)] for every changed cell."""
    out = []
    if not g1 or not g2:
        return out
    for y in range(min(len(g1), len(g2))):
        r1, r2 = g1[y], g2[y]
        for x in range(min(len(r1), len(r2))):
            if r1[x] != r2[x]:
                out.append((x, y, int(r1[x]), int(r2[x])))
                if len(out) >= cap:
                    return out
    return out


def color_histogram(grid: tuple) -> tuple:
    hist = [0] * 16
    for row in grid:
        for cell in row:
            hist[cell & 0xF] += 1
    return tuple(hist)


def _cells_of_color(grid: tuple, colour: int, cap: int = 64) -> list:
    """Coordinates of every cell of ``colour``, row-major, capped."""
    out = []
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell == colour:
                out.append((x, y))
                if len(out) >= cap:
                    return out
    return out


def segment(grid: tuple, min_size: int = 1, max_objects: int = 64) -> list:
    """4-connected monochrome components with size / bbox / centroid."""
    if not grid:
        return []
    h, w = len(grid), len(grid[0])
    seen = [[False] * w for _ in range(h)]
    objs = []
    for y in range(h):
        for x in range(w):
            colour = grid[y][x]
            if seen[y][x] or colour == 0:
                continue
            stack = [(x, y)]
            seen[y][x] = True
            cells = []
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy),
                               (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] \
                            and grid[ny][nx] == colour:
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            if len(cells) < min_size:
                continue
            xs = [p[0] for p in cells]
            ys = [p[1] for p in cells]
            objs.append({
                "color": int(colour),
                "size": len(cells),
                "cells": cells,
                "bbox": (min(xs), min(ys), max(xs), max(ys)),
                "centroid": (sum(xs) / len(xs), sum(ys) / len(ys)),
            })
            if len(objs) >= max_objects:
                return objs
    objs.sort(key=lambda o: (-o["size"], o["color"]))
    return objs


def object_multiset(grid: tuple) -> frozenset:
    return frozenset((o["color"], o["size"]) for o in segment(grid, 1, 80))


def downsample(grid: tuple, side: int = 16) -> tuple:
    if not grid:
        return ()
    h, w = len(grid), len(grid[0])
    sy, sx = max(1, (h + side - 1) // side), max(1, (w + side - 1) // side)
    return tuple(tuple(grid[y][x] for x in range(0, w, sx))
                 for y in range(0, h, sy))


def hamming_frac(a: tuple, b: tuple) -> float:
    if not a or not b:
        return 1.0
    total = mismatch = 0
    for ra, rb in zip(a, b):
        for ca, cb in zip(ra, rb):
            total += 1
            if ca != cb:
                mismatch += 1
    return (mismatch / total) if total else 1.0


def classify_event(prev_levels: int, cur_levels: int, state: str,
                   diff_n: int) -> str:
    if state == "WIN":
        return "WIN"
    if state == "GAME_OVER":
        return "DEATH"
    if cur_levels > prev_levels:
        return "LEVEL_UP"
    if diff_n == 0:
        return "NOOP"
    if diff_n <= 3:
        return "TINY"
    if diff_n > 80:
        return "BIG"
    return "CHANGE"


def event_is_progress(event: str) -> bool:
    return event in ("LEVEL_UP", "WIN")


def find_mover(diff: list):
    """If the diff is one entity translating, return (old, new, colour).

    Scale-agnostic on purpose: a "one cell" move in the agent's 64x64 frame is
    a whole sprite's worth of pixels (5x5, 4x4, ...), so the test is on shape
    and colour agreement, never on a raw cell count.
    """
    vanished = [(x, y, o) for x, y, o, n in diff if o != 0 and n == 0]
    appeared = [(x, y, n) for x, y, o, n in diff if o == 0 and n != 0]
    recolored = [(x, y, o, n) for x, y, o, n in diff if o != 0 and n != 0]
    if not vanished or not appeared or recolored:
        return None
    if len(vanished) != len(appeared) or len(vanished) > 400:
        return None
    if len({c for _x, _y, c in vanished}) != 1:
        return None
    if len({c for _x, _y, c in appeared}) != 1:
        return None
    colour = vanished[0][2]
    if appeared[0][2] != colour:
        return None
    vxs = [p[0] for p in vanished]
    vys = [p[1] for p in vanished]
    axs = [p[0] for p in appeared]
    ays = [p[1] for p in appeared]
    if (max(vxs) - min(vxs), max(vys) - min(vys)) != \
            (max(axs) - min(axs), max(ays) - min(ays)):
        return None
    old = (int(round(sum(vxs) / len(vxs))), int(round(sum(vys) / len(vys))))
    new = (int(round(sum(axs) / len(axs))), int(round(sum(ays) / len(ays))))
    return old, new, colour


def render_ascii(grid: tuple, max_side: int = 44) -> str:
    """Compact ASCII rendering for the advisor prompt."""
    if not grid:
        return "<empty>"
    h, w = len(grid), len(grid[0])
    sy = max(1, (h + max_side - 1) // max_side)
    sx = max(1, (w + max_side - 1) // max_side)
    lines = []
    if sx > 1 or sy > 1:
        lines.append(f"(downsampled {sy}x{sx})")
    for y in range(0, h, sy):
        lines.append("".join(_COLOR_CHARS[grid[y][x] & 0xF]
                             for x in range(0, w, sx)))
    return "\n".join(lines)


# ===========================================================================
# 3. SYMBOLIC STATE
# ===========================================================================
class StateSig:
    """Compact symbolic description of a frame — Astra-style shorthand.

    Deliberately small and hashable: it is the state the world model reasons
    about, the key of the death graph, and the vocabulary of the notes.
    """

    __slots__ = ("player", "hist", "nobj", "rare", "ghash", "frames")

    def __init__(self, grid: tuple, player=None, frames: int = 1):
        self.player = player
        self.hist = color_histogram(grid)
        objs = segment(grid, 1, 48)
        self.nobj = len(objs)
        self.rare = tuple(sorted(c for c, n in enumerate(self.hist)
                                 if c and 1 <= n <= 4))
        self.ghash = grid_hash(grid)
        self.frames = frames

    # -- keys -------------------------------------------------------------
    def exact_key(self) -> tuple:
        return (self.ghash,)

    def object_key(self) -> tuple:
        return (self.nobj, self.hist)

    def relative_key(self) -> tuple:
        """Position-invariant: same objects, any player position."""
        return (self.nobj, self.hist)

    def brief(self) -> str:
        p = f"({self.player[0]},{self.player[1]})" if self.player else "-"
        nz = {c: n for c, n in enumerate(self.hist) if n}
        top = sorted(nz.items(), key=lambda kv: -kv[1])[:5]
        return (f"p={p} obj={self.nobj} colors=" +
                ",".join(f"{c}:{n}" for c, n in top))


# ===========================================================================
# 4. NOTES — durable memory ("the agent's weights")
# ===========================================================================
class Notes:
    """Append-mostly structured memory that survives compaction and levels.

    Rendered as compact symbolic text (never prose paragraphs) so it stays
    cheap in tokens and stable in the prompt prefix, which is what lets vLLM's
    prefix cache serve as our stand-in for retained reasoning.
    """

    def __init__(self) -> None:
        self.actions: dict = {}        # name -> "semantics" shorthand
        self.glossary: dict = {}       # colour -> role shorthand
        self.goal_hypotheses: list = []
        self.hazards: list = []
        self.failed: list = []
        self.level_briefs: list = []
        self.win_sequences: list = []
        self.pending_plan: str = ""
        self.facts: list = []

    # -- writers ----------------------------------------------------------
    def note_action(self, name: str, text: str) -> None:
        if self.actions.get(name) != text:
            self.actions[name] = text

    def note_color(self, colour: int, role: str) -> None:
        if colour and self.glossary.get(colour) != role:
            self.glossary[colour] = role

    def note_goal(self, text: str) -> None:
        if text not in self.goal_hypotheses:
            self.goal_hypotheses.append(text)
            if len(self.goal_hypotheses) > 6:
                self.goal_hypotheses = self.goal_hypotheses[-6:]

    def note_hazard(self, text: str) -> None:
        if text not in self.hazards:
            self.hazards.append(text)
            if len(self.hazards) > 12:
                self.hazards = self.hazards[-12:]

    def note_failure(self, text: str) -> None:
        if text not in self.failed:
            self.failed.append(text)
            if len(self.failed) > 14:
                self.failed = self.failed[-14:]

    def note_fact(self, text: str) -> None:
        if text not in self.facts:
            self.facts.append(text)
            if len(self.facts) > 24:
                self.facts = self.facts[-24:]

    def brief_level(self, index: int, text: str) -> None:
        self.level_briefs.append((index, text))
        if len(self.level_briefs) > 8:
            self.level_briefs = self.level_briefs[-8:]

    def record_win_sequence(self, index: int, actions: list) -> None:
        effective = [a for a in actions if a]
        if effective:
            self.win_sequences.append((index, effective))
            if len(self.win_sequences) > 6:
                self.win_sequences = self.win_sequences[-6:]

    # -- rendering --------------------------------------------------------
    def render(self, budget_chars: int = 2600) -> str:
        parts = []
        if self.actions:
            parts.append("ACTIONS " + " ".join(
                f"{k}={v}" for k, v in sorted(self.actions.items())))
        if self.glossary:
            parts.append("COLORS " + " ".join(
                f"{c}:{r}" for c, r in sorted(self.glossary.items())))
        if self.goal_hypotheses:
            parts.append("GOAL " + " | ".join(self.goal_hypotheses[-3:]))
        if self.hazards:
            parts.append("HAZARD " + " | ".join(self.hazards[-6:]))
        if self.failed:
            parts.append("TRIED_FAILED " + " | ".join(self.failed[-7:]))
        if self.facts:
            parts.append("FACTS " + " | ".join(self.facts[-10:]))
        if self.level_briefs:
            parts.append("LEVELS " + " | ".join(
                f"L{i}:{t}" for i, t in self.level_briefs[-4:]))
        if self.win_sequences:
            parts.append("WON_BY " + " | ".join(
                f"L{i}:{'>'.join(s[-14:])}" for i, s in self.win_sequences[-3:]))
        if self.pending_plan:
            parts.append("PLAN " + self.pending_plan)
        text = "\n".join(parts)
        return text[:budget_chars]

    # -- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "actions": self.actions, "glossary": self.glossary,
            "goal_hypotheses": self.goal_hypotheses, "hazards": self.hazards,
            "failed": self.failed, "level_briefs": self.level_briefs,
            "win_sequences": self.win_sequences,
            "pending_plan": self.pending_plan, "facts": self.facts,
        }

    def load_dict(self, data: dict) -> None:
        for key, value in (data or {}).items():
            if hasattr(self, key):
                setattr(self, key, value)


def compact_transcript(records: list, keep_recent: int = 12) -> str:
    """Compaction, not truncation: older exchanges become one-line conclusions.

    This is the mechanism OpenAI identified as worth ~3x RHAE (13.3% -> 38.3%)
    and the reason the Astra provider-adapter harness beat the standard one at
    every reasoning tier.
    """
    if len(records) <= keep_recent:
        recent, older = records, []
    else:
        older, recent = records[:-keep_recent], records[-keep_recent:]
    lines = []
    if older:
        counter = Counter(r.get("event", "?") for r in older)
        acts = Counter(r.get("action", "?") for r in older)
        lines.append(
            "SUMMARY_EARLIER n=%d events=%s actions=%s" % (
                len(older),
                ",".join(f"{k}x{v}" for k, v in counter.most_common(6)),
                ",".join(f"{k}x{v}" for k, v in acts.most_common(6)),
            )
        )
    for r in recent:
        lines.append("#%s %s -> %s d=%s %s" % (
            r.get("i", "?"), r.get("action", "?"), r.get("event", "?"),
            r.get("diff", 0), r.get("after", "")))
    return "\n".join(lines)[-3000:]


# ===========================================================================
# 5. TIMELINE — append-only ground truth
# ===========================================================================
class Timeline:
    """Every real transition, replayable.  The world model is certified
    against this and never against the agent's own imagination."""

    def __init__(self, cap: int = 4000) -> None:
        self.records: list = []
        self.cap = cap

    def append(self, record: dict) -> None:
        self.records.append(record)
        if len(self.records) > self.cap:
            del self.records[: self.cap // 4]

    def __len__(self) -> int:
        return len(self.records)

    def tail(self, n: int = 12) -> list:
        return self.records[-n:]

    def exercised(self) -> list:
        """Transitions whose action the current model actually covers."""
        return self.records


# ===========================================================================
# 6. WORLD MODEL + INDUCTION  (deterministic, no LLM)
# ===========================================================================
class WorldModel:
    """An executable model over the symbolic state.

    ``step(player, action, grid)`` predicts ``(new_player, event_class)``.
    Learned entirely from recorded transitions:

      deltas[a]      translation of the tracked entity for simple action ``a``
      walls          cells the entity provably could not enter
      deadly         cells whose entry produced GAME_OVER
      goal_color     colour whose contact produced LEVEL_UP
      clear_color    colour whose disappearance produced LEVEL_UP
      click_rules    colour clicked -> observed event class
      noop_rate[a]   how often ``a`` changes nothing
    """

    def __init__(self, version: int = 0, variant: str = "strict") -> None:
        self.version = version
        self.variant = variant
        self.deltas: dict = {}
        self.walls: set = set()
        self.deadly: set = set()
        self.goal_color: int = 0
        self.clear_color: int = 0
        self.deadly_color: int = 0
        self.clear_action: str = ""
        self.click_rules: dict = {}
        self.noop_rate: dict = defaultdict(lambda: [0, 0])   # a -> [n, noops]
        self.death_actions: set = set()
        self.support: dict = {}
        self.certified = False
        self.accuracy = 0.0
        self.coverage = 0.0

    # -- introspection ----------------------------------------------------
    def known_actions(self) -> set:
        return set(self.deltas)

    def complexity(self) -> int:
        """Deterministic Occam metric: fewer parameters is simpler."""
        return (len(self.deltas) + len(self.walls) // 4 + len(self.deadly) +
                (1 if self.goal_color else 0) + (1 if self.clear_color else 0)
                + (1 if self.deadly_color else 0)
                + len(self.click_rules) + len(self.death_actions))

    def describe(self) -> str:
        bits = []
        if self.deltas:
            bits.append("deltas=" + ",".join(
                f"{k}:{'+' if v[0] >= 0 else ''}{v[0]}{'+' if v[1] >= 0 else ''}{v[1]}"
                for k, v in sorted(self.deltas.items())))
        if self.goal_color:
            bits.append(f"goal_color={self.goal_color}")
        if self.clear_color:
            bits.append(f"clear_color={self.clear_color}"
                        + (f"/{self.clear_action}" if self.clear_action else ""))
        if self.deadly_color:
            bits.append(f"deadly_color={self.deadly_color}")
        if self.walls:
            bits.append(f"walls={len(self.walls)}")
        if self.deadly:
            bits.append("deadly=" + ",".join(
                f"{x},{y}" for x, y in sorted(self.deadly)[:8]))
        if self.death_actions:
            bits.append("death_actions=" + ",".join(sorted(self.death_actions)))
        if self.click_rules:
            bits.append("click=" + ",".join(
                f"c{c}->{e}" for c, e in sorted(self.click_rules.items())[:6]))
        bits.append(f"v{self.version}/{self.variant}")
        bits.append(f"acc={self.accuracy:.2f} cov={self.coverage:.2f}")
        return " ".join(bits)

    # -- prediction -------------------------------------------------------
    def step(self, player, action: str, grid: tuple):
        """Predict (new_player, event_class).  event may be 'UNKNOWN'.

        Lethality is a property of a CELL, never of an action: the same move
        that kills on a hazard is the move that wins everywhere else.  Treating
        an action as globally deadly poisons the model (it then mispredicts
        every safe use of that action), so only ``deadly`` cells veto here.
        """
        delta = self.deltas.get(action)
        if delta is None or player is None:
            return player, "UNKNOWN"
        nxt = (player[0] + delta[0], player[1] + delta[1])
        if nxt in self.deadly or (
                self.deadly_color
                and _color_at(grid, nxt) == self.deadly_color):
            return nxt, "DEATH"
        if nxt in self.walls or not _in_bounds(nxt, grid):
            return player, "NOOP"
        if self.goal_color and _color_at(grid, nxt) == self.goal_color:
            return nxt, "LEVEL_UP"
        return nxt, "CHANGE"

    def is_goal(self, grid: tuple, player) -> bool:
        if player is None:
            return False
        if self.goal_color and _color_at(grid, player) == self.goal_color:
            return True
        if self.clear_color and color_histogram(grid)[self.clear_color] == 0:
            return True
        return False

    def clone(self) -> "WorldModel":
        copy = WorldModel(self.version, self.variant)
        copy.deltas = dict(self.deltas)
        copy.walls = set(self.walls)
        copy.deadly = set(self.deadly)
        copy.deadly_color = self.deadly_color
        copy.clear_action = self.clear_action
        copy.goal_color = self.goal_color
        copy.clear_color = self.clear_color
        copy.click_rules = dict(self.click_rules)
        copy.noop_rate = defaultdict(lambda: [0, 0],
                                     {k: list(v) for k, v in self.noop_rate.items()})
        copy.death_actions = set(self.death_actions)
        copy.support = dict(self.support)
        return copy


def _in_bounds(pos, grid: tuple) -> bool:
    if not grid:
        return False
    x, y = pos
    return 0 <= y < len(grid) and 0 <= x < len(grid[0])


def _color_at(grid: tuple, pos) -> int:
    if not _in_bounds(pos, grid):
        return 0
    return grid[pos[1]][pos[0]]


def induce_world_model(timeline: Timeline, variant: str = "strict") -> WorldModel:
    """Mine a deterministic world model out of recorded transitions.

    Two evidence thresholds give two tournament candidates (strict / loose);
    ``certify`` then picks whichever survives backtesting, Occam-tiebroken.
    """
    min_support = 2 if variant == "strict" else 1
    model = WorldModel(version=1, variant=variant)
    records = timeline.records

    # ---- pass 1: entity translations -> deltas --------------------------
    delta_votes: dict = defaultdict(Counter)
    for rec in records:
        mover = rec.get("mover")
        name = rec.get("action")
        if not mover or not name or name == "RESET":
            continue
        old, new, _colour = mover
        delta_votes[name][(new[0] - old[0], new[1] - old[1])] += 1
    for name, votes in delta_votes.items():
        best, count = votes.most_common(1)[0]
        total = sum(votes.values())
        if count >= min_support and count >= 0.6 * total and best != (0, 0):
            model.deltas[name] = best
            model.support[name] = count

    # ---- pass 2: no-ops -> walls (needs a known delta) ------------------
    for rec in records:
        name = rec.get("action")
        delta = model.deltas.get(name)
        before = rec.get("player_before")
        if not delta or before is None:
            continue
        model.noop_rate[name][0] += 1
        if rec.get("event") == "NOOP":
            model.noop_rate[name][1] += 1
            model.walls.add((before[0] + delta[0], before[1] + delta[1]))

    # ---- pass 3: deaths (positional only) --------------------------------
    lethal = Counter()
    for rec in records:
        if rec.get("event") != "DEATH":
            continue
        name = rec.get("action")
        before = rec.get("player_before")
        delta = model.deltas.get(name)
        if before is not None and delta:
            entered = (before[0] + delta[0], before[1] + delta[1])
            model.deadly.add(entered)
            colour = _color_at(rec.get("grid_before") or (), entered)
            if colour:
                lethal[colour] += 1
    # NB: no `death_actions` set is built.  An action that killed once is not a
    # deadly action; the cell it entered was.  See WorldModel.step.
    #
    # A hazard is a COLOUR, not a coordinate.  Dying once on a colour-8 cell
    # means every colour-8 cell is lethal, and generalising immediately is the
    # largest single action saving available in a dense-hazard game: without it
    # the agent pays a death, a RESET and a full retrace for every hazard on
    # the board.  One observation is enough to form the hypothesis; CERTIFY is
    # what vetoes it if stepping onto that colour later turns out to be safe.
    if lethal:
        model.deadly_color = lethal.most_common(1)[0][0]

    # ---- pass 4: goal predicate -----------------------------------------
    reach = Counter()
    cleared = Counter()
    clicked = Counter()
    for rec in records:
        if rec.get("event") not in ("LEVEL_UP", "WIN"):
            continue
        before = rec.get("player_before")
        after = rec.get("player_after")
        grid_before = rec.get("grid_before")
        if before is not None and grid_before:
            # On a level-up the "after" position belongs to the NEXT level, so
            # looking it up in the pre-move grid reads whatever happens to sit
            # there (usually the player's own start cell).  The cell that
            # actually triggered the transition is the one stepped INTO.
            delta = model.deltas.get(rec.get("action") or "")
            cell = None
            if delta is not None:
                cell = (before[0] + delta[0], before[1] + delta[1])
            elif after is not None:
                cell = after
            touched = _color_at(grid_before, cell) if cell is not None else 0
            if touched:
                reach[touched] += 1
        hist_b = rec.get("hist_before")
        hist_a = rec.get("hist_after")
        if hist_b and hist_a:
            for colour in range(1, 16):
                if hist_b[colour] > 0 and hist_a[colour] == 0:
                    cleared[colour] += 1
        if rec.get("action") == "ACTION6" and rec.get("click_color"):
            clicked[rec["click_color"]] += 1
    if reach and reach.most_common(1)[0][1] >= min_support:
        model.goal_color = reach.most_common(1)[0][0]
    if cleared and cleared.most_common(1)[0][1] >= min_support:
        model.clear_color = cleared.most_common(1)[0][0]

    # ---- pass 5: click rules --------------------------------------------
    click_votes: dict = defaultdict(Counter)
    for rec in records:
        if rec.get("action") == "ACTION6" and rec.get("click_color"):
            click_votes[rec["click_color"]][rec.get("event", "?")] += 1
    for colour, ev in click_votes.items():
        best, count = ev.most_common(1)[0]
        if count >= min_support:
            model.click_rules[colour] = best
    return model


# ===========================================================================
# 7. CERTIFY — never act on an unverified theory
# ===========================================================================
def certify(model: WorldModel, timeline: Timeline,
            accuracy_floor: float, coverage_floor: float):
    """Backtest the model against the whole recorded history.

    Returns ``(ok, accuracy, coverage, failing_indices)``.  The anti-vacuity
    gate is the point: a model that "passes" by explaining nothing is rejected,
    and unexplained history stays UNKNOWN rather than being treated as safe.
    """
    exercised = matched = 0
    failing: list = []
    for i, rec in enumerate(timeline.records):
        name = rec.get("action")
        if not name or name == "RESET" or name not in model.deltas:
            continue
        before = rec.get("player_before")
        observed = rec.get("event")
        if before is None or observed in (None, "LEVEL_UP", "WIN"):
            continue
        if observed == "DEATH":
            # hold the model accountable for predicting lethality *positionally*
            _predicted_pos, predicted_event = model.step(
                before, name, rec.get("grid_before") or ())
            exercised += 1
            if predicted_event == "DEATH":
                matched += 1
            else:
                failing.append(i)
            continue
        predicted_pos, predicted_event = model.step(
            before, name, rec.get("grid_before") or ())
        exercised += 1
        actual_pos = rec.get("player_after")
        pos_ok = (actual_pos is None) or (predicted_pos == actual_pos)
        if observed == "NOOP":
            event_ok = predicted_event in ("NOOP", "UNKNOWN")
        elif observed in ("CHANGE", "TINY", "BIG"):
            event_ok = predicted_event in ("CHANGE", "NOOP", "UNKNOWN")
        else:
            event_ok = True
        if pos_ok and event_ok:
            matched += 1
        else:
            failing.append(i)

    total = max(1, len(timeline.records))
    coverage = exercised / total
    accuracy = (matched / exercised) if exercised else 0.0
    required = max(2, math.ceil(coverage_floor * total))
    ok = (exercised >= required and accuracy >= accuracy_floor
          and coverage >= coverage_floor)
    model.accuracy = accuracy
    model.coverage = coverage
    model.certified = ok
    return ok, accuracy, coverage, failing


def repair_model(model: WorldModel, timeline: Timeline, failing: list,
                 variant: str) -> WorldModel:
    """Drop the rules implicated by the counterexamples and re-induce (v+1)."""
    blamed_actions = Counter()
    blamed_cells = []
    for idx in failing:
        rec = timeline.records[idx]
        if rec.get("action"):
            blamed_actions[rec["action"]] += 1
        before = rec.get("player_before")
        delta = model.deltas.get(rec.get("action"))
        if before and delta:
            blamed_cells.append((before[0] + delta[0], before[1] + delta[1]))
    fresh = induce_world_model(timeline, variant)
    fresh.version = model.version + 1
    for action, count in blamed_actions.items():
        if count >= 2 and action in fresh.deltas:
            del fresh.deltas[action]
    for cell in blamed_cells:
        fresh.walls.discard(cell)
    return fresh


# ===========================================================================
# 8. PLAN — free compute buys metered actions
# ===========================================================================
def plan_bfs(model: WorldModel, grid: tuple, player, depth: int, node_budget: int,
             visited: set = None):
    """Breadth-first search INSIDE the certified model.

    Zero environment actions are spent here — this is the whole efficiency
    engine.  Returns ``(action_queue, predicted_events)`` or ``([], [])``.

    Two objectives, in priority order:

    1. the learned goal (``goal_color`` / ``clear_color``) — pure exploitation;
    2. the nearest cell the agent has never occupied — systematic coverage.

    Objective 2 is what breaks the chicken-and-egg: the goal predicate can only
    be induced from a level-up, and a level-up cannot be planned for until the
    predicate exists.  Frontier search guarantees the goal is eventually walked
    into, after which every later plan goes straight at it.
    """
    if player is None or not model.deltas:
        return [], []
    visited = visited or set()
    goals = set()
    if model.goal_color:
        for y, row in enumerate(grid):
            for x, cell in enumerate(row):
                if cell == model.goal_color:
                    goals.add((x, y))
    start = (player[0], player[1])
    queue = deque([(start, [])])
    seen = {start}
    frontier = None      # nearest never-occupied cell, held as a fallback
    nodes = 0
    moves = sorted(model.deltas.items(),
                   key=lambda kv: (abs(kv[1][0]) + abs(kv[1][1]), kv[0]))
    while queue and nodes < node_budget:
        pos, path = queue.popleft()
        nodes += 1
        if len(path) >= depth:
            continue
        for name, delta in moves:
            nxt = (pos[0] + delta[0], pos[1] + delta[1])
            if nxt in seen or not _in_bounds(nxt, grid):
                continue
            if nxt in model.walls or nxt in model.deadly:
                continue
            seen.add(nxt)
            new_path = path + [name]
            if (goals and nxt in goals) or model.is_goal(grid, nxt):
                cal_log("plan", "bfs_goal", len(new_path), depth,
                        f"goal_color={model.goal_color} nodes={nodes}")
                return new_path, ["CHANGE"] * (len(new_path) - 1) + ["LEVEL_UP"]
            # Remember the nearest unvisited cell but keep searching.  A goal
            # that is currently unreachable (a door before its key, a hazard
            # field not yet mapped) must not switch exploration off, or the
            # agent stalls forever in sight of something it cannot yet take.
            if frontier is None and nxt not in visited:
                frontier = new_path
            queue.append((nxt, new_path))
    if frontier is not None:
        cal_log("plan", "bfs_frontier", len(frontier), depth,
                f"goal unreachable (goal_color={model.goal_color}); "
                f"covering new ground, visited={len(visited)} "
                f"nodes={nodes}")
        return frontier, ["CHANGE"] * len(frontier)
    cal_log("plan", "bfs_exhausted", nodes, node_budget,
            f"no path (goal_color={model.goal_color} walls={len(model.walls)} "
            f"visited={len(visited)} deltas={len(model.deltas)})")
    return [], []


# ===========================================================================
# 9a. ARCHETYPE POSTERIOR — soft router with abstention
# ===========================================================================
class ArchetypePosterior:
    """P(archetype).  Evidence multiplies odds; nothing ever reaches zero, and
    candidate scoring is bias-weighted rather than winner-take-all, so a
    misclassification costs accuracy instead of the whole game."""

    WEIGHTS = {
        #                        MOV   CLICK  COLOR  SELECT OTHER
        "entity_moved":        (2.2, 0.2, 0.1, 0.2, 0.1),
        "simple_actions_move": (1.6, 0.1, 0.0, 0.1, 0.1),
        "click_local_change":  (0.2, 2.2, 0.4, 0.6, 0.2),
        "click_level_change":  (0.1, 0.6, 0.3, 2.2, 0.2),
        "global_recolor":      (0.1, 0.3, 2.4, 0.2, 0.3),
        "simple_noop_only":    (0.3, 1.0, 0.6, 0.4, 0.6),
        "surprise_on_plan":    (0.5, 0.5, 0.5, 0.5, 1.2),
    }

    def __init__(self) -> None:
        self.p = {a: 0.2 for a in ARCHETYPES}
        self._floor = 0.02

    def update(self, evidence: dict) -> None:
        for key, val in evidence.items():
            if not val or key not in self.WEIGHTS:
                continue
            weights = self.WEIGHTS[key]
            for i, arch in enumerate(ARCHETYPES):
                self.p[arch] *= (1.0 + weights[i])
        total = sum(self.p.values()) or 1.0
        self.p = {a: max(v / total, self._floor) for a, v in self.p.items()}
        z = sum(self.p.values())
        self.p = {a: v / z for a, v in self.p.items()}

    def entropy(self) -> float:
        return -sum(v * math.log(v + 1e-9) for v in self.p.values())

    def top(self, n: int = 2):
        return sorted(self.p.items(), key=lambda kv: -kv[1])[:n]

    def abstain(self, threshold: float) -> bool:
        return self.entropy() > threshold

    def summary(self) -> str:
        return " ".join(f"{a[:5]}={self.p[a]:.2f}" for a in ARCHETYPES)


# ===========================================================================
# 9b. GUARD — nothing invalid, nothing fatal, nothing twice
# ===========================================================================
class Guard:
    """Protocol validity plus the death graph plus the useless-action ledgers.

    Asymmetry is deliberate: exact evidence blocks, fuzzy evidence only warns.
    Blocking a legal move on a near-miss is worse than risking it, because a
    blocked move is a wasted metered action with zero information.
    """

    def __init__(self) -> None:
        self.death_exact: Counter = Counter()     # (grid_hash, action_key) -> n
        self.death_object: Counter = Counter()    # (object_key, action) -> n
        self.death_relative: Counter = Counter()  # (relative_key, action) -> n
        self.death_snaps: list = []               # [(ds_grid, action_key)]
        self.noop: dict = defaultdict(lambda: [0, 0])    # a -> [used, noops]
        self.progress: dict = defaultdict(lambda: [0, 0])  # a -> [used, progress]
        self.clicks: dict = defaultdict(lambda: [0, 0])    # (x,y) -> [n, changes]
        self.tried_here: dict = defaultdict(set)           # grid_hash -> {key}

    # -- protocol ---------------------------------------------------------
    @staticmethod
    def validate(name: str, xy, avail: set, dims) -> bool:
        """True iff this action can legally be emitted right now."""
        if name not in ACTION_IDS:
            return False
        if avail and name not in avail:
            return False
        if is_complex(name):
            if not xy:
                return False
            x, y = xy
            if not (0 <= x <= COORD_LIMIT and 0 <= y <= COORD_LIMIT):
                return False
            if dims and dims[1] and not (0 <= x < dims[1] and 0 <= y < dims[0]):
                return False
        return True

    # -- death graph ------------------------------------------------------
    def mark_death(self, sig: StateSig, key: str, grid: tuple) -> None:
        self.death_exact[(sig.ghash, key)] += 1
        self.death_object[(sig.object_key(), key.split("@")[0])] += 1
        self.death_relative[(sig.relative_key(), key.split("@")[0])] += 1
        if len(self.death_snaps) < 40:
            self.death_snaps.append((downsample(grid), key))

    def death_score(self, sig: StateSig, key: str) -> float:
        name = key.split("@")[0]
        exact = self.death_exact.get((sig.ghash, key), 0)
        if exact:
            return 6.0 + exact                     # hard block
        soft = 0.0
        soft += 0.6 * self.death_object.get((sig.object_key(), name), 0)
        soft += 0.3 * self.death_relative.get((sig.relative_key(), name), 0)
        return soft                                # soft warn only

    def near_miss(self, grid: tuple, key: str, frac: float) -> bool:
        ds = downsample(grid)
        for snap_grid, snap_key in self.death_snaps:
            if snap_key == key:
                return True
            if hamming_frac(ds, snap_grid) < frac:
                return True
        return False

    # -- ledgers ----------------------------------------------------------
    def record(self, key: str, event: str) -> None:
        name = key.split("@")[0]
        if name == "RESET":
            return
        self.noop[name][0] += 1
        self.progress[name][0] += 1
        if event == "NOOP":
            self.noop[name][1] += 1
        if event_is_progress(event):
            self.progress[name][1] += 1
        if name == "ACTION6" and "@" in key:
            try:
                xs, ys = key.split("@")[1].split(",")
                cell = (int(xs), int(ys))
                self.clicks[cell][0] += 1
                if event != "NOOP":
                    self.clicks[cell][1] += 1
            except Exception:
                pass

    def noop_rate(self, name: str) -> float:
        used, noops = self.noop[name]
        return (noops / used) if used else 0.0

    def affinity(self, name: str) -> float:
        used, prog = self.progress[name]
        return (prog / used) if used else 0.0


# ===========================================================================
# 10. SCHEDULER — EV stop-loss with an immovable hard spine
# ===========================================================================
class Scheduler:
    """Decides explore / exploit / conserve.

    The soft brain may rationalise; the hard cap may not move.  That separation
    is what keeps a long autonomous run from grinding one level to zero.
    """

    def __init__(self) -> None:
        self.mode = "explore"
        self.level_index = 1
        self.level_weight_total = 1

    def expected_weight(self, level_index: int, total_levels: int) -> float:
        """RHAE weight of this level as a fraction of the game score."""
        total_levels = max(total_levels, level_index, 1)
        denom = total_levels * (total_levels + 1) / 2.0
        return level_index / denom if denom else 0.0

    def update(self, on_level: int, progress_rate: float, learning_left: int,
               game_id: str = "") -> str:
        weight = self.expected_weight(self.level_index, self.level_weight_total)
        ev_exploit = progress_rate * (1.0 + 3.0 * weight)
        ev_explore = 0.35 * learning_left * (0.4 + weight)
        previous = self.mode
        if on_level > CFG["LEVEL_HARD_CAP"]:
            self.mode = "conserve"
        elif on_level > CFG["LEVEL_BUDGET"]:
            self.mode = "exploit"
        else:
            self.mode = "explore" if ev_explore > ev_exploit else "exploit"
        if self.mode != previous:
            cal_log("scheduler", "ev_mode", round(ev_explore - ev_exploit, 3),
                    0.0, f"{previous}->{self.mode} on_level={on_level} "
                         f"weight={weight:.2f} {game_id}")
        return self.mode


# ===========================================================================
# 11. ADVISOR — optional local LLM, sandboxed and latency-governed
# ===========================================================================
_ALLOWED_AST = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call,
    ast.Name, ast.Load, ast.Constant, ast.Tuple, ast.List, ast.Dict, ast.Set,
    ast.Subscript, ast.Index, ast.Slice, ast.IfExp, ast.Attribute,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.And, ast.Or,
    ast.Not, ast.USub, ast.UAdd, ast.Return, ast.FunctionDef, ast.arguments,
    ast.arg, ast.Assign, ast.AugAssign, ast.If, ast.For, ast.While, ast.Break,
    ast.Continue, ast.Pass, ast.Module, ast.comprehension, ast.ListComp,
    ast.GeneratorExp, ast.Starred,
)
_ALLOWED_NAMES = {
    "len", "range", "min", "max", "abs", "sum", "sorted", "enumerate", "zip",
    "int", "float", "bool", "tuple", "list", "dict", "set", "any", "all",
    "True", "False", "None",
}


def safe_compile(source: str):
    """AST-whitelist a model-authored function.  No imports, no dunders, no IO.

    Returns a callable or None.  The resulting code is still only ever used as
    a *prediction*, and it must pass the same backtest as an induced model.
    """
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST + (ast.expr, ast.stmt, ast.operator,
                                                ast.boolop, ast.cmpop,
                                                ast.unaryop, ast.expr_context)):
            return None
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return None
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            return None
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id not in _ALLOWED_NAMES:
                return None
            if isinstance(fn, ast.Attribute):
                return None
    namespace: dict = {}
    try:
        exec(compile(tree, "<atlas-model>", "exec"),  # noqa: S102 - whitelisted
             {"__builtins__": {n: getattr(__builtins__, n, None)
                               for n in _ALLOWED_NAMES}
              if not isinstance(__builtins__, dict) else
              {n: __builtins__.get(n) for n in _ALLOWED_NAMES}},
             namespace)
    except Exception:
        return None
    for value in namespace.values():
        if callable(value) and not getattr(value, "__name__", "").startswith("__"):
            return value
    return None


class AtlasLLM:
    """Dependency-free client for a local OpenAI-compatible server.

    Local-only by construction: the competition runs with internet disabled, so
    an external API is not an option, only a failure mode.  After
    ``LLM_MAX_FAILS`` failures the advisor disables itself and the agent
    continues on its deterministic machinery.
    """

    def __init__(self) -> None:
        self.url = CFG["LLM_URL"]
        self.timeout = CFG["LLM_TIMEOUT"]
        self.fails = 0
        self.disabled = False
        self.model = None
        self.last_latency = 0.0
        self.calls = 0
        self.tokens = 0

    def probe(self) -> bool:
        if self.disabled:
            return False
        try:
            with urllib.request.urlopen(f"{self.url}/models", timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            ids = [d.get("id", "") for d in data.get("data", [])]
            if not ids:
                return False
            self.model = ids[0]
            return True
        except Exception:
            self.fails += 1
            if self.fails >= CFG["LLM_MAX_FAILS"]:
                self.disabled = True
            return False

    def ask(self, messages: list, max_tokens: int = 320,
            temperature: float = 0.2) -> str:
        if self.disabled:
            return ""
        payload = {
            "model": self.model or "atlas",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        started = time.time()
        try:
            req = urllib.request.Request(
                f"{self.url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            self.last_latency = time.time() - started
            self.calls += 1
            usage = data.get("usage") or {}
            self.tokens += int(usage.get("completion_tokens", 0) or 0)
            message = (data.get("choices") or [{}])[0].get("message") or {}
            # --reasoning-parser qwen3 puts chain-of-thought in reasoning_content
            text = message.get("content") or message.get("reasoning_content") or ""
            return text if isinstance(text, str) else ""
        except Exception:
            self.last_latency = time.time() - started
            self.fails += 1
            if self.fails >= CFG["LLM_MAX_FAILS"]:
                self.disabled = True
                _log("advisor disabled after repeated failures; "
                     "deterministic machinery only")
            return ""

    @staticmethod
    def parse_json(text: str):
        text = (text or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            obj = json.loads(text[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None


# ===========================================================================
# 12. THE AGENT
# ===========================================================================
class MyAgent(Agent):
    """ATLAS v3: perceive free, remember durably, plan for free, act rarely."""

    MAX_ACTIONS = CFG["MAX_ACTIONS"]

    # ------------------------------------------------------------------
    def __init__(self, *args, **kwargs) -> None:
        if Agent is not object:
            try:
                super().__init__(*args, **kwargs)
            except Exception:
                self.frames = []
                self.action_counter = 0
                self.game_id = kwargs.get("game_id", "mock")
        else:  # standalone / tests
            self.frames = []
            self.action_counter = 0
            self.game_id = kwargs.get("game_id", "mock")

        self.rng = random.Random(CFG["SEED"] ^ zlib.crc32(
            str(self.game_id).encode("utf-8")))
        self.t_start = time.time()

        # memory ----------------------------------------------------------
        self.timeline = Timeline()
        self.notes = Notes()
        self.guard = Guard()
        self.posterior = ArchetypePosterior()
        self.scheduler = Scheduler()
        self.model = WorldModel()
        self.model_candidates: list = []

        # perception state ------------------------------------------------
        self.prev_grid: tuple = ()
        self.prev_levels = 0
        self.prev_player = None
        self.player = None
        self.player_color = None
        self.player_confirmed = False
        self.visited: set = set()
        self.delta_trials: Counter = Counter()
        self.prev_sig: StateSig = None
        self.sig: StateSig = None
        self.dims = (0, 0)
        self.last_diff: list = []
        self.last_event: str = ""
        self.last_key: str = ""
        self.last_click_color = 0
        self.consec_noops = 0
        self.surprises = 0

        # level / budget state --------------------------------------------
        self.levels_seen = 0
        self.win_levels = 0
        self.level_start_action = 0
        self.resets_this_level = 0
        self.level_action_counts: list = []
        self.level_actions_current: list = []
        self.info_probes_used = 0
        self.click_probes_used = 0
        self.probe_queue: list = []
        self._probes_built = False
        self.probing = True
        self.deadline = self.t_start + CFG["DEADLINE_S"]

        # planning / execution --------------------------------------------
        self.plan: list = []
        self.plan_expected: list = []
        self.plan_pos: list = []      # predicted position per queued step
        self._plan_stamp = None
        self.deliberations = 0
        self.plans_built = 0
        self.plan_actions = 0
        self.wins = 0

        # advisor ----------------------------------------------------------
        self.advisor = None
        self.advisor_ok = False
        self.advisor_every = CFG["LLM_EVERY"]
        self._advisor_asks = 0
        self._advisor_accepted = 0
        self._advisor_rejected = 0
        self.click_usable = True
        self._pending_data: dict = {}
        self._pending_reasoning = None
        self.mode = "explore"

    # ==================================================================
    # FRAMEWORK INTEGRATION
    # ==================================================================
    def do_action_request(self, action):
        """Emit an action WITHOUT reading shared enum state.

        ``GameAction`` members are process-wide singletons and the framework's
        ``Agent.do_action_request`` reads ``action.action_data`` after
        ``choose_action`` returns.  The Swarm runs one agent per game in its own
        thread, so with N-way concurrency another game's ACTION6 coordinates can
        land in this game's request.  We carry our own payload instead.
        """
        data = dict(self._pending_data or {"game_id": self.game_id})
        raw = self.arc_env.step(action, data=data, reasoning=self._pending_reasoning)
        return self._convert_raw_frame_data(raw)

    def is_done(self, frames, latest_frame) -> bool:
        try:
            state = frame_state(latest_frame)
            if state == "WIN":
                self.wins += 1
                self.checkpoint("win")
                return True
            if self.action_counter >= self.MAX_ACTIONS:
                cal_log("budget", "max_actions", self.action_counter,
                        self.MAX_ACTIONS, "stop")
                return True
            if time.time() > self.deadline:
                cal_log("budget", "deadline", round(time.time() - self.t_start, 1),
                        CFG["DEADLINE_S"], "stop")
                return True
            return False
        except Exception:
            return False

    def choose_action(self, frames, latest_frame):
        """Fail-safe wrapper: an internal error must never cost the whole game."""
        try:
            return self._choose(latest_frame)
        except Exception as exc:  # pragma: no cover - defensive
            cal_log("failsafe", "choose_exception", str(exc)[:80], "-",
                    "guarded RESET")
            return self._emit("RESET", None, "failsafe")

    # ==================================================================
    # OBSERVE
    # ==================================================================
    def _observe(self, frame) -> None:
        """Fold one real transition into memory.  Free, and the only source of
        truth the world model is ever allowed to learn from."""
        grid = dominant_grid(frame)
        stack = grids_of(frame)
        levels = frame_levels(frame)
        state = frame_state(frame)
        if grid:
            self.dims = (len(grid), len(grid[0]) if grid[0] else 0)

        diff = diff_cells(self.prev_grid, grid) if (self.prev_grid and grid) else []
        mover = find_mover(diff)
        event = classify_event(self.prev_levels, levels, state, len(diff))

        if self.last_key == "RESET":
            # A reset teleports the entity back to the level start, so the
            # tracked position is stale.  Drop it and re-lock on the next
            # mover rather than poisoning the map with a phantom position.
            self.player = None
            self.player_confirmed = False

        # entity tracking --------------------------------------------------
        # A mover is the strongest evidence there is: it is the entity that
        # moved.  Until one is seen we hold only a provisional guess, so a
        # wrong first guess can never block the real lock-on.
        if mover:
            old, new, colour = mover
            if not self.player_confirmed or self.player_color == colour:
                self.player_color = colour
                self.player = new
                self.player_confirmed = True
            elif self.player is not None and \
                    abs(old[0] - self.player[0]) + abs(old[1] - self.player[1]) <= 8:
                self.player = new
        elif grid and self.player is None:
            self.player = self._guess_player(grid)
        if self.player_confirmed and self.player is not None:
            # the occupancy record: what makes frontier planning possible
            self.visited.add(self.player)

        sig = StateSig(grid, self.player, len(stack)) if grid else None

        # outcome bookkeeping for the action we emitted --------------------
        if self.last_key and sig is not None:
            name = self.last_key.split("@")[0]
            record = {
                "i": len(self.timeline),
                "action": name,
                "key": self.last_key,
                "event": event,
                "diff": len(diff),
                "levels": levels,
                "state": state,
                "player_before": self.prev_player,
                "player_after": self.player,
                "mover": mover,
                "grid_before": self.prev_grid,
                "hist_before": self.prev_sig.hist if self.prev_sig else None,
                "hist_after": sig.hist,
                "ghash_before": self.prev_sig.ghash if self.prev_sig else 0,
                "ghash_after": sig.ghash,
                "click_color": self.last_click_color,
                "after": sig.brief(),
            }
            self.timeline.append(record)
            self.guard.record(self.last_key, event)
            if event == "NOOP":
                self.consec_noops += 1
            else:
                self.consec_noops = 0
            if event == "DEATH":
                self.guard.mark_death(self.prev_sig or sig, self.last_key,
                                      self.prev_grid or grid)
                self.notes.note_hazard(f"{self.last_key}->DEATH")
            self._archetype_evidence(name, event, diff)
            self._watch(name, event, sig)
            if name != "RESET":
                self.level_actions_current.append(name)
            self._log_transition(name, event, len(diff))

        # level lifecycle ---------------------------------------------------
        if levels > self.prev_levels:
            self._on_level_up(levels)
        self.levels_seen = levels
        win = getattr(frame, "win_levels", None)
        if isinstance(win, int) and win:
            self.win_levels = win
            self.scheduler.level_weight_total = max(self.win_levels, 1)

        self.prev_grid = grid
        self.prev_player = self.player
        self.prev_sig = sig
        self.sig = sig
        self.prev_levels = levels
        self.last_diff = diff
        self.last_event = event
        self.last_key = ""
        self.last_click_color = 0

    def _guess_player(self, grid: tuple):
        """Heuristic first guess: a small, rare, movable blob."""
        objs = segment(grid, 1, 48)
        if not objs:
            return None
        counts = Counter(o["color"] for o in objs)
        best = None
        for obj in objs:
            if obj["size"] > 6 or counts[obj["color"]] > 3:
                continue
            cx, cy = obj["centroid"]
            score = -obj["size"] - counts[obj["color"]]
            if best is None or score > best[0]:
                best = (score, (int(round(cx)), int(round(cy))), obj["color"])
        if best:
            self.player_color = best[2]
            return best[1]
        cx, cy = objs[0]["centroid"]
        return (int(round(cx)), int(round(cy)))

    def _archetype_evidence(self, name: str, event: str, diff: list) -> None:
        ev = {k: False for k in ArchetypePosterior.WEIGHTS}
        if event in ("LEVEL_UP", "WIN") and name == "ACTION6":
            ev["click_level_change"] = True
        elif event == "NOOP" and name in PLAY_ACTIONS and name != "ACTION6":
            ev["simple_noop_only"] = True
        elif diff:
            mover = find_mover(diff)
            if mover:
                ev["entity_moved"] = True
                if name != "ACTION6":
                    ev["simple_actions_move"] = True
            if name == "ACTION6" and len(diff) <= 80:
                ev["click_local_change"] = True
            recolored = sum(1 for _x, _y, o, n in diff if o and n)
            if len(diff) > 80 and recolored > 0.6 * len(diff):
                ev["global_recolor"] = True
        if any(ev.values()):
            self.posterior.update(ev)

    def _log_transition(self, name: str, event: str, diff_n: int) -> None:
        if not CFG["LOG"]:
            return
        if event in ("NOOP", "CHANGE", "TINY") and self.action_counter % 20:
            return
        _log(f"{self.game_id} #{self.action_counter} {name} -> {event} "
             f"(d{diff_n}) lvl={self.levels_seen}/{self.win_levels} "
             f"[{self.posterior.summary()}]")

    # ==================================================================
    # LEVEL LIFECYCLE + CROSS-LEVEL TRANSFER
    # ==================================================================
    def _on_level_up(self, levels: int) -> None:
        spent = self.action_counter - self.level_start_action
        self.level_action_counts.append(spent)
        self.notes.brief_level(
            levels - 1,
            f"cleared in {spent} actions; model[{self.model.describe()[:90]}]")
        self.notes.record_win_sequence(levels - 1, self.level_actions_current)
        _log(f"{self.game_id} LEVEL_UP -> {levels} "
             f"({spent} actions on the level)")
        cal_log("progress", "level_cleared", spent, CFG["LEVEL_BUDGET"],
                f"level {levels} in {spent} actions")
        # cross-level transfer: action semantics, hazards and the goal
        # hypothesis persist across levels (that is what a human carries over
        # from the tutorial); only the map is re-earned per level.
        keep_deltas = dict(self.model.deltas)
        keep_death = set(self.model.death_actions)
        keep_goal = self.model.goal_color
        keep_clear = self.model.clear_color
        self.model = WorldModel()
        self.model.deltas = keep_deltas
        self.model.death_actions = keep_death
        self.model.goal_color = keep_goal
        self.model.clear_color = keep_clear
        self.model.version = 1
        self.notes.note_fact(
            f"carried over {len(keep_deltas)} action semantics, "
            f"goal_color={keep_goal}, clear_color={keep_clear}")
        self.plan = []
        self.plan_expected = []
        self.plan_pos = []
        self._plan_stamp = None
        self.visited = set()          # a new level is a new map to cover
        self.player = None            # re-anchor on the new level's first mover
        self.player_confirmed = False
        self.level_start_action = self.action_counter
        self.level_actions_current = []
        self.resets_this_level = 0
        self.scheduler.level_index = max(1, levels + 1)
        self.checkpoint("level_up")

    def checkpoint(self, why: str) -> None:
        if not CFG["TRACE_DIR"]:
            return
        try:
            os.makedirs(CFG["TRACE_DIR"], exist_ok=True)
            path = os.path.join(
                CFG["TRACE_DIR"],
                f"atlas_{str(self.game_id)[:40]}.json")
            payload = {
                "game_id": self.game_id, "why": why,
                "actions": self.action_counter,
                "levels": self.levels_seen,
                "win_levels": self.win_levels,
                "notes": self.notes.to_dict(),
                "model": self.model.describe(),
                "posterior": self.posterior.p,
                "telemetry": self.telemetry(),
                "calibration_tail": _CAL_BUF[-40:],
            }
            with open(path, "w") as fh:
                json.dump(payload, fh, default=str)
        except Exception:
            pass

    def telemetry(self) -> dict:
        advisor = self.advisor if isinstance(self.advisor, AtlasLLM) else None
        return {
            "actions": self.action_counter,
            "levels": self.levels_seen,
            "deliberations": self.deliberations,
            "plans_built": self.plans_built,
            "plan_actions": self.plan_actions,
            "surprises": self.surprises,
            "advisor_calls": advisor.calls if advisor else 0,
            "advisor_tokens": advisor.tokens if advisor else 0,
            "timeline": len(self.timeline),
            "visited": len(self.visited),
            "model": self.model.describe()[:160],
            "mode": self.mode,
        }

    # ==================================================================
    # DELIBERATE: induce -> certify -> plan  (all free)
    # ==================================================================
    def _deliberate(self, force: bool = False) -> None:
        since = self.action_counter - self.level_start_action
        if not force and self.deliberations and \
                since and since % CFG["DELIBERATE_EVERY"]:
            return
        if len(self.timeline) < 2:
            return
        self.deliberations += 1
        candidates = [induce_world_model(self.timeline, "strict"),
                      induce_world_model(self.timeline, "loose")]
        # an advisor-authored model joins the tournament and must pass the same
        # backtest as an induced one
        authored = self._advisor_model()
        if authored is not None:
            candidates.append(authored)

        ranked = []
        for cand in candidates:
            ok, acc, cov, failing = certify(
                cand, self.timeline,
                CFG["CERTIFY_ACCURACY"] if cand.variant == "strict"
                else CFG["CERTIFY_ACCURACY_LOOSE"],
                CFG["CERTIFY_COVERAGE"])
            ranked.append((ok, round(acc, 4), -cand.complexity(), cand, failing))
        ranked.sort(key=lambda r: (not r[0], -r[1], -r[2]))
        best = ranked[0]
        ok, acc, _cov, cand, failing = best
        cal_log("certify", "world_model", acc, CFG["CERTIFY_ACCURACY"],
                f"{'CERTIFIED' if ok else 'rejected'} "
                f"[{cand.describe()[:120]}]")
        if ok:
            self.model = cand
            self._publish_model_notes()
            self._replan()
            return

        # repair loop: drop the implicated rules and try once more
        repaired = repair_model(cand, self.timeline, failing, cand.variant)
        ok2, acc2, _cov2, failing2 = certify(
            repaired, self.timeline, CFG["CERTIFY_ACCURACY_LOOSE"],
            CFG["CERTIFY_COVERAGE"])
        cal_log("certify", "repair", acc2, CFG["CERTIFY_ACCURACY_LOOSE"],
                f"{'CERTIFIED' if ok2 else 'still rejected'} "
                f"v{repaired.version}")
        if ok2:
            self.model = repaired
            self._publish_model_notes()
            self._replan()
        else:
            # keep the most accurate model for prediction, but never plan on it
            self.model = cand
            self.model.certified = False
            self.plan = []
            self.plan_expected = []
            self.plan_pos = []

    def _publish_model_notes(self) -> None:
        model = self.model
        for name, delta in model.deltas.items():
            self.notes.note_action(
                name, f"move({delta[0]:+d},{delta[1]:+d})")
        if model.goal_color:
            self.notes.note_goal(f"reach color {model.goal_color}")
            self.notes.note_color(model.goal_color, "GOAL")
        if model.clear_color:
            self.notes.note_goal(f"clear color {model.clear_color}")
            self.notes.note_color(model.clear_color, "CONSUMABLE")
        for name in model.death_actions:
            self.notes.note_hazard(f"{name} can kill")
        for colour, event in model.click_rules.items():
            self.notes.note_color(colour, f"click->{event}")
            if event_is_progress(event):
                self.notes.note_goal(f"click color {colour}")
        for name, (used, noops) in list(model.noop_rate.items()):
            if used >= 3 and noops / used >= 0.8:
                self.notes.note_action(name, f"mostly noop ({noops}/{used})")

    def _replan(self, force: bool = False) -> None:
        if not self.model.certified or self.player is None:
            return
        if not self.player_confirmed:
            # A guessed position (rarest-colour heuristic, or screen centre
            # when nothing on the board is rare) is a hypothesis, not a fact.
            # Planning from it yields a queue that is plausible on paper and
            # wrong in the world, and every metered action spent walking it is
            # unrecoverable.  Confirm the player first: one exploratory move
            # costs less than one bad plan.
            self.plan = []
            self.plan_expected = []
            self.plan_pos = []
            return
        stamp = (self.model.version, self.model.variant, len(self.model.walls),
                 len(self.model.deadly), self.model.goal_color)
        # planning is free but not instantaneous: do not re-search while a
        # queue is in flight against an unchanged model
        if self.plan and not force and stamp == getattr(self, "_plan_stamp", None):
            return
        queue, expected = plan_bfs(
            self.model, self.prev_grid or (), self.player,
            CFG["PLAN_DEPTH"], CFG["PLAN_NODES"], visited=self.visited)
        if queue:
            self.plan = queue
            self.plan_expected = expected
            # Predicted position per step, so WATCH can check the model
            # belief-by-belief instead of only checking the event class.
            preds = []
            pos = self.player
            grid = self.prev_grid or ()
            for nm in queue:
                if nm in self.model.deltas:
                    pos, _ev = self.model.step(pos, nm, grid)
                    preds.append(pos)
                else:
                    preds.append(None)
            self.plan_pos = preds
            self._plan_stamp = stamp
            self.plans_built += 1
            self.notes.pending_plan = ">".join(queue[:16])
            cal_log("plan", "committed_queue", len(queue), CFG["PLAN_DEPTH"],
                    f"{'|'.join(queue[:10])}")

    # ==================================================================
    # WATCH: execute a plan under prediction check
    # ==================================================================
    def _watch(self, name: str, event: str, sig: StateSig) -> None:
        """Per-step self-check.  A surprise voids the rest of the queue and
        feeds the counterexample back into induction — that is what stops one
        wrong theory from costing a dozen metered actions."""
        if name == "RESET":
            # a reset invalidates every positional prediction in the queue
            if self.plan or self.plan_expected:
                cal_log("watch", "plan_voided_by_reset", len(self.plan), 0,
                        "positions no longer valid")
            self.plan = []
            self.plan_expected = []
            self.plan_pos = []
            return
        if not self.plan_expected:
            return
        expected_event = self.plan_expected.pop(0)
        expected_pos = self.plan_pos.pop(0) if self.plan_pos else None
        if self.plan:
            self.plan.pop(0)
        # Position check.  Matching the event class is not enough: a queue is
        # only safe if the model is right about WHERE the agent ended up, since
        # every later step of the queue is relative to that cell.
        if (expected_pos is not None and self.player_confirmed
                and self.player is not None
                and (self.player[0], self.player[1]) != tuple(expected_pos)):
            self.surprises += 1
            self.plan = []
            self.plan_expected = []
            self.plan_pos = []
            self.posterior.update({"surprise_on_plan": True})
            self.notes.note_failure(
                f"plan step {name} expected to arrive at {tuple(expected_pos)} "
                f"but the agent is at {(self.player[0], self.player[1])}")
            cal_log("watch", "surprise_position", self.surprises, 1,
                    f"{name}: expected {tuple(expected_pos)} got "
                    f"{(self.player[0], self.player[1])}; queue voided")
            return
        if expected_event in ("LEVEL_UP", "WIN") and event_is_progress(event):
            return
        if expected_event == "NOOP" and event == "NOOP":
            return
        if expected_event in ("CHANGE", "TINY", "BIG") and event in (
                "CHANGE", "TINY", "BIG", "NOOP"):
            return
        if event_is_progress(event):
            return
        self.surprises += 1
        self.plan = []
        self.plan_expected = []
        self.plan_pos = []
        self.posterior.update({"surprise_on_plan": True})
        self.notes.note_failure(f"plan step {name} expected {expected_event} "
                                f"got {event}")
        cal_log("watch", "surprise", self.surprises, 1,
                f"{name}: expected {expected_event} got {event}; queue voided")

    # ==================================================================
    # CLICK FUNNEL: 4096 cells -> a dozen scored candidates
    # ==================================================================
    def _salient_targets(self, grid: tuple):
        if not self.click_usable or not grid:
            return []
        h, w = len(grid), len(grid[0])
        objs = segment(grid, 1, 40)
        hist = color_histogram(grid)
        rare = {c for c, n in enumerate(hist) if c and 1 <= n <= 4}
        recent = {(x, y) for x, y, _o, _n in self.last_diff[:60]}
        raw = []

        def add(score, x, y):
            x = max(0, min(w - 1, int(x)))
            y = max(0, min(h - 1, int(y)))
            raw.append((score, x, y))

        for obj in objs:
            cx, cy = obj["centroid"]
            bx0, by0, bx1, by1 = obj["bbox"]
            score = 1.0
            if obj["color"] in rare:
                score += 3.0
            if obj["size"] <= 4:
                score += 2.0
            if self.player:
                dist = math.hypot(cx - self.player[0], cy - self.player[1])
                score += 2.5 * math.exp(-dist / 8.0)
            if (int(cx), int(cy)) in recent:
                score += 1.0
            add(score, cx, cy)
            add(score * 0.8, bx0, by0)
            if obj["size"] >= 6:
                add(score * 0.7, (bx0 + bx1) // 2, (by0 + by1) // 2)
        for x, y in list(recent)[:10]:
            add(4.0, x, y)

        seen, out = {}, []
        for score, x, y in sorted(raw, reverse=True):
            if (x, y) in seen:
                continue
            clicks, changes = self.guard.clicks[(x, y)]
            if clicks and changes == 0:
                score -= 2.5                      # proven useless cell
            elif clicks:
                score += 2.0 * (changes / clicks)  # proven productive cell
            else:
                score += 1.0                       # information gain
            key = f"ACTION6@{x},{y}"
            death = self.guard.death_score(self.sig, key) if self.sig else 0.0
            if death >= 6.0:
                continue                           # exact death: hard block
            score -= death
            if self.guard.near_miss(grid, key, CFG["NEARMISS_FRAC"]):
                score -= 1.0                       # soft warn only
            seen[(x, y)] = True
            out.append((score, x, y))
            if len(out) >= CFG["SALIENT_CAP"]:
                break
        return out

    # ==================================================================
    # ADVISOR (optional, local, governed)
    # ==================================================================
    def _advisor_init(self) -> None:
        if CFG["LLM_MODE"] in ("0", "off", "false", "no"):
            self.advisor = "disabled"
            self.advisor_ok = False
            return
        self.advisor = AtlasLLM()
        self.advisor_ok = self.advisor.probe()
        cal_log("advisor", "available", 1.0 if self.advisor_ok else 0.0, 0.5,
                "ready" if self.advisor_ok else "deterministic-only")

    def _governor(self) -> None:
        """Back off when proposals keep getting rejected: latency is real even
        though actions are the only metered resource."""
        if self._advisor_asks >= 6 and \
                self._advisor_rejected > 2 * max(1, self._advisor_accepted):
            if self.advisor_every < CFG["LLM_MAX_EVERY"]:
                self.advisor_every = min(CFG["LLM_MAX_EVERY"],
                                         self.advisor_every * 2)
                cal_log("advisor", "backoff", self.advisor_every,
                        CFG["LLM_MAX_EVERY"], "consult less often")
                self._advisor_rejected //= 2

    def _advisor_prompt(self, ask: str) -> list:
        """Stable prefix first (notes, model, rules), transcript last.

        The prefix barely changes between turns, which is what lets the local
        server's KV/prefix cache stand in for retained reasoning.
        """
        system = (
            "You are the reasoning core of ATLAS, an agent playing ARC-AGI-3. "
            "The grid is 64x64, 16 colours, origin top-left. Only environment "
            "actions are metered; your thinking is free. Reply with a single "
            "JSON object and nothing else."
        )
        context = (
            "NOTES\n" + (self.notes.render() or "(none)") +
            "\nWORLD_MODEL " + self.model.describe() +
            "\nARCHETYPES " + self.posterior.summary() +
            "\nLEGAL " + ",".join(sorted(self._avail)) +
            "\nSTATE " + (self.sig.brief() if self.sig else "-")
        )
        transcript = compact_transcript(self.timeline.tail(14), 10)
        return [
            {"role": "system", "content": system},
            {"role": "system", "content": context},
            {"role": "user", "content": "HISTORY\n" + transcript + "\n\n" + ask},
        ]

    def _advisor_action(self):
        if not self.advisor_ok or not isinstance(self.advisor, AtlasLLM):
            return None
        self._advisor_asks += 1
        if self._advisor_asks % self.advisor_every:
            return None
        ask = (
            "Propose the single next action. JSON: "
            '{"action":"ACTION1..ACTION7","x":int,"y":int,"why":"<=12 words"} '
            "Omit x/y unless the action needs coordinates."
        )
        text = self.advisor.ask(self._advisor_prompt(ask), 200)
        obj = AtlasLLM.parse_json(text)
        if not obj:
            self._advisor_rejected += 1
            return None
        name = str(obj.get("action", "")).upper()
        if name not in ACTION_IDS or name == "RESET":
            self._advisor_rejected += 1
            return None
        xy = None
        if is_complex(name):
            try:
                xy = (int(obj.get("x", -1)), int(obj.get("y", -1)))
            except Exception:
                xy = None
        if not self.guard.validate(name, xy, self._avail, self.dims):
            self._advisor_rejected += 1
            cal_log("advisor", "invalid_proposal", name, "-", "rejected")
            return None
        self._advisor_accepted += 1
        why = str(obj.get("why", ""))[:60]
        self.notes.note_fact(f"advisor:{name}({why})")
        return name, xy, why

    def _advisor_model(self):
        """Ask the advisor to author world-model rules; they must pass the same
        backtest as induced rules, so a hallucination cannot reach an action."""
        if not (self.advisor_ok and CFG["LLM_ALLOW_CODE"]):
            return None
        if not isinstance(self.advisor, AtlasLLM):
            return None
        if self._advisor_asks % max(3, self.advisor_every):
            return None
        ask = (
            "Author the game's transition rules. JSON: "
            '{"deltas":{"ACTION1":[dx,dy]},"goal_color":int,'
            '"clear_color":int,"death_actions":["ACTION2"],'
            '"click_rules":{"3":"LEVEL_UP"}} '
            "Use only evidence in HISTORY. dx/dy are integer cell offsets."
        )
        text = self.advisor.ask(self._advisor_prompt(ask), 300)
        obj = AtlasLLM.parse_json(text)
        if not obj:
            return None
        cand = WorldModel(version=99, variant="authored")
        deltas = obj.get("deltas") or {}
        if isinstance(deltas, dict):
            for name, value in deltas.items():
                name = str(name).upper()
                if name in PLAY_ACTIONS and isinstance(value, (list, tuple)) \
                        and len(value) == 2:
                    try:
                        cand.deltas[name] = (int(value[0]), int(value[1]))
                    except Exception:
                        continue
        for key, attr in (("goal_color", "goal_color"),
                          ("clear_color", "clear_color")):
            try:
                val = int(obj.get(key, 0) or 0)
                if 0 <= val <= 15:
                    setattr(cand, attr, val)
            except Exception:
                pass
        for name in (obj.get("death_actions") or []):
            name = str(name).upper()
            if name in PLAY_ACTIONS:
                cand.death_actions.add(name)
        rules = obj.get("click_rules") or {}
        if isinstance(rules, dict):
            for colour, event in rules.items():
                try:
                    cand.click_rules[int(colour)] = str(event).upper()
                except Exception:
                    continue
        if not cand.deltas:
            return None
        cal_log("advisor", "authored_model", cand.complexity(), "-",
                cand.describe()[:90])
        return cand

    # ==================================================================
    # EMISSION
    # ==================================================================
    def _emit(self, name: str, xy, reason: str):
        """Build and return the GameAction, arming our private payload."""
        name = name if name in ACTION_IDS else "RESET"
        action = None
        if GameAction is not None:
            try:
                action = GameAction.from_name(name)
            except Exception:
                action = None
        data = {"game_id": str(self.game_id)}
        if is_complex(name) and xy:
            data["x"] = int(xy[0])
            data["y"] = int(xy[1])
            key = f"{name}@{int(xy[0])},{int(xy[1])}"
            self.last_click_color = _color_at(self.prev_grid, (int(xy[0]),
                                                              int(xy[1])))
        else:
            key = name
        self._pending_data = data
        self._pending_reasoning = {"r": reason[:120]}
        # Also set the shared enum payload for framework/recording compatibility,
        # but never read it back (see do_action_request).
        if action is not None:
            try:
                action.set_data(dict(data))
                action.reasoning = self._pending_reasoning
            except Exception:
                pass
        self.last_key = key
        if name in PLAY_ACTIONS and not is_complex(name):
            self.delta_trials[name] += 1
        if self.sig is not None:
            self.guard.tried_here[self.sig.ghash].add(key)
        return action if action is not None else name

    # ==================================================================
    # THE POLICY LOOP
    # ==================================================================
    def _choose(self, latest_frame):
        self._observe(latest_frame)
        self._avail = available_names(latest_frame)
        state = frame_state(latest_frame)

        # 1) lifecycle ------------------------------------------------------
        if state in ("NOT_PLAYED", "GAME_OVER"):
            if self.advisor is None:
                self._advisor_init()
            reason = "start game" if state == "NOT_PLAYED" else "recover from death"
            self.level_actions_current = []
            return self._emit("RESET", None, reason)
        if state == "WIN":
            return self._emit("RESET", None, "won; continue")

        # 2) free deliberation ----------------------------------------------
        self._deliberate(force=(self.surprises > 0 and not self.plan))
        # Planning inside the learned model costs zero metered actions, so it
        # must not wait for the (deliberately throttled) induction cadence:
        # whenever the queue has run dry, search again immediately.  _replan
        # no-ops while a queue is still in flight against an unchanged model.
        self._replan()

        # 3) execute the certified plan under WATCH --------------------------
        while self.plan:
            name = self.plan[0]
            if not self.guard.validate(name, None, self._avail, self.dims):
                self.plan.pop(0)
                if self.plan_expected:
                    self.plan_expected.pop(0)
                if self.plan_pos:
                    self.plan_pos.pop(0)
                continue
            self.plan_actions += 1
            return self._emit(name, None, "plan")

        # 4) budget / mode ---------------------------------------------------
        on_level = self.action_counter - self.level_start_action
        self.mode = self.scheduler.update(
            on_level, self.guard.affinity("ACTION1") +
            sum(self.guard.progress[a][1] for a in PLAY_ACTIONS) * 0.01,
            max(0, 8 - len(self.model.deltas)), str(self.game_id))

        # 5) stagnation recovery (RESET costs one metered action — spend it
        #    only when the alternative is flailing) ---------------------------
        if self.consec_noops >= CFG["STAGNATION_RESET"] and \
                self.resets_this_level < CFG["MAX_RESETS_PER_LEVEL"]:
            self.resets_this_level += 1
            self.consec_noops = 0
            cal_log("recovery", "stagnation_reset", on_level,
                    CFG["STAGNATION_RESET"], "deliberate reset, memory kept")
            return self._emit("RESET", None, "stagnation")

        # 6) advisor proposal -------------------------------------------------
        proposal = self._advisor_action()
        if proposal is not None:
            name, xy, why = proposal
            return self._emit(name, xy, f"advisor:{why}")
        self._governor()

        # 7) probe phase: identify the action space cheaply, with abstention ---
        probe = self._probe_step()
        if probe is not None:
            return probe

        # 8) deliberate policies, then the fallback chain -----------------------
        return self._policy_action(on_level)

    # ------------------------------------------------------------------
    def _probe_step(self):
        if not self.probing:
            return None
        if not self._probes_built and not self.probe_queue:
            self.probe_queue = [a for a in PLAY_ACTIONS
                                if a in self._avail and a != "ACTION6"]
            self._probes_built = True
        while self.probe_queue:
            name = self.probe_queue.pop(0)
            key = name
            if self.guard.death_score(self.sig, key) >= 6.0 if self.sig else False:
                continue
            if self.sig is not None and key in self.guard.tried_here[self.sig.ghash]:
                continue
            return self._emit(name, None, "probe action semantics")

        ambiguous = self.posterior.abstain(CFG["ABSTAIN_ENTROPY"])
        if ambiguous and self.info_probes_used < CFG["MAX_INFO_PROBES"]:
            (top_a, _pa), (top_b, _pb) = self.posterior.top(2)
            self.info_probes_used += 1
            want_click = ("CLICK_PUZZLE" in (top_a, top_b) or
                          "SELECTION" in (top_a, top_b)) and \
                "ACTION6" in self._avail
            grid = self.prev_grid
            if want_click:
                for _score, x, y in self._salient_targets(grid)[:1]:
                    key = f"ACTION6@{x},{y}"
                    if self.sig is not None and \
                            key in self.guard.tried_here[self.sig.ghash]:
                        continue
                    cal_log("router", "info_probe", self.posterior.entropy(),
                            CFG["ABSTAIN_ENTROPY"],
                            f"click ({x},{y}) splits {top_a}/{top_b}")
                    return self._emit("ACTION6", (x, y), "info-gain probe")
            for name in PLAY_ACTIONS:
                if name == "ACTION6" or name not in self._avail:
                    continue
                if self.sig is not None and \
                        name in self.guard.tried_here[self.sig.ghash]:
                    continue
                cal_log("router", "info_probe", self.posterior.entropy(),
                        CFG["ABSTAIN_ENTROPY"],
                        f"move {name} splits {top_a}/{top_b}")
                return self._emit(name, None, "info-gain probe")

        if self.click_probes_used < 3 and not ambiguous and \
                "ACTION6" in self._avail:
            for _score, x, y in self._salient_targets(self.prev_grid)[:2]:
                key = f"ACTION6@{x},{y}"
                if self.sig is not None and \
                        key in self.guard.tried_here[self.sig.ghash]:
                    continue
                self.click_probes_used += 1
                return self._emit("ACTION6", (x, y), "probe click target")

        self.probing = False
        cal_log("router", "committed", self.posterior.entropy(),
                CFG["ABSTAIN_ENTROPY"],
                f"probes done [{self.posterior.summary()}]")
        _log(f"{self.game_id} probes done at #{self.action_counter}: "
             f"{self.posterior.summary()}")
        return None

    # ------------------------------------------------------------------
    def _policy_action(self, on_level: int):
        """Five-layer fallback chain; no game ever stalls at zero."""
        sig = self.sig
        tried = self.guard.tried_here.get(sig.ghash, set()) if sig else set()
        grid = self.prev_grid

        # layer 2: archetype policy templates --------------------------------
        template = self._template_action(tried)
        if template is not None:
            return template

        # layer 3: level analogy — replay a similar level's effective sequence
        analogy = self._analogy_action(tried)
        if analogy is not None:
            return analogy

        # layer 4/5: learned value + reflex floor ------------------------------
        move_bias = self.posterior.p["MOVEMENT"]
        click_bias = max(self.posterior.p["CLICK_PUZZLE"],
                         self.posterior.p["SELECTION"],
                         self.posterior.p["COLOR_LOGIC"])
        candidates = []
        for name in sorted(self._avail - {"RESET", "ACTION6"}):
            score = (0.9 * self.guard.affinity(name)
                     + 0.5 * self.guard.progress[name][1]
                     - 1.6 * self.guard.noop_rate(name))
            score += 1.2 * move_bias
            if self.guard.noop[name][0] >= 2 and \
                    self.guard.noop_rate(name) <= 0.5:
                score += 2.0 * move_bias          # proven locomotion: chain it
            if self.guard.noop[name][0] == 0:
                score += 1.0                      # untried: information value
            # delta completion: an action whose effect we cannot yet predict is
            # worth more than any exploitation, because every later plan
            # depends on knowing it.  A wall may have hidden it on every
            # earlier attempt, so the budget is generous and the reward decays
            # slowly — but it only applies where the action is still untried,
            # which is what stops it from becoming a loop.
            if name in PLAY_ACTIONS and name not in self.model.deltas and \
                    self.delta_trials[name] < CFG["DELTA_TRIALS"] and \
                    name not in tried:
                score += 3.0 - 0.1 * self.delta_trials[name]
            # one-step frontier lookahead: entering a cell we have never
            # occupied beats re-walking ground the map already knows
            delta = self.model.deltas.get(name)
            if delta and self.player:
                nxt = (self.player[0] + delta[0], self.player[1] + delta[1])
                if nxt in self.model.deadly:
                    score -= 6.0
                elif nxt in self.model.walls:
                    score -= 2.5
                elif _in_bounds(nxt, grid) and nxt not in self.visited:
                    score += 2.5
                else:
                    score -= 0.8
            if name in tried:
                score -= 1.2
                if move_bias > 0.4 and self.guard.noop_rate(name) <= 0.5:
                    score += 0.9                  # re-moving is normal
            if sig is not None:
                score -= self.guard.death_score(sig, name)
            if self.mode == "conserve" and self.guard.progress[name][1] == 0:
                score -= 2.0                      # only proven actions
            if self.mode == "exploit" and self.guard.affinity(name) <= 0:
                score -= 1.0
            candidates.append((score, name, None))

        for score, x, y in self._salient_targets(grid):
            key = f"ACTION6@{x},{y}"
            if key in tried:
                continue
            candidates.append((0.4 + 0.25 * score + 1.2 * click_bias,
                               "ACTION6", (x, y)))

        if candidates:
            self.rng.shuffle(candidates)
            candidates.sort(key=lambda c: -c[0])
            if os.getenv("ATLAS_DEBUG_POLICY"):
                _log("cands " + " | ".join(
                    f"{c[1]}{('@%d,%d' % c[2]) if c[2] else ''}:{c[0]:.2f}"
                    for c in candidates[:4]))
            for score, name, xy in candidates:
                if not self.guard.validate(name, xy, self._avail, self.dims):
                    continue
                return self._emit(name, xy,
                                  f"policy {score:.2f} mode={self.mode} "
                                  f"lvl_act={on_level}")

        # layer 5: reflex floor — always legal, never stalls
        for name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5",
                     "ACTION7"):
            if name in self._avail:
                return self._emit(name, None, "reflex floor")
        return self._emit("RESET", None, "no legal action")

    def _template_action(self, tried: set):
        """Deterministic archetype policies that need no learned model."""
        grid = self.prev_grid
        if not grid or self.sig is None:
            return None
        top = self.posterior.top(1)[0][0]
        if top in ("CLICK_PUZZLE", "COLOR_LOGIC", "SELECTION"):
            # rarest untried colour is the highest-information click
            hist = color_histogram(grid)
            rare = sorted((n, c) for c, n in enumerate(hist) if c and n)
            for _n, colour in rare[:3]:
                for x, y in _cells_of_color(grid, colour):
                    key = f"ACTION6@{x},{y}"
                    if key in tried:
                        continue
                    if self.guard.death_score(self.sig, key) >= 6.0:
                        continue
                    if not self.guard.validate("ACTION6", (x, y),
                                               self._avail, self.dims):
                        continue
                    return self._emit("ACTION6", (x, y),
                                      f"template rarest color {colour}")
        if top == "MOVEMENT" and self.model.deltas and self.player:
            # symmetry completion: head for the least-visited quadrant
            h, w = len(grid), len(grid[0])
            targets = [(w // 4, h // 4), (3 * w // 4, h // 4),
                       (w // 4, 3 * h // 4), (3 * w // 4, 3 * h // 4)]
            targets.sort(key=lambda t: math.hypot(t[0] - self.player[0],
                                                  t[1] - self.player[1]))
            for name, delta in self.model.deltas.items():
                step = (self.player[0] + delta[0], self.player[1] + delta[1])
                if step in self.model.walls or step in self.model.deadly:
                    continue
                best = targets[0]
                if (best[0] - self.player[0]) * delta[0] + \
                        (best[1] - self.player[1]) * delta[1] > 0:
                    if self.guard.validate(name, None, self._avail, self.dims):
                        return self._emit(name, None, "template quadrant seek")
        return None

    def _analogy_action(self, tried: set):
        """Cross-level transfer: replay an effective sequence from a level we
        already cleared.  Guarded against repeating a known no-op loop."""
        if not self.notes.win_sequences:
            return None
        if self.consec_noops >= 3:
            return None
        for _level, sequence in reversed(self.notes.win_sequences):
            index = len(self.level_actions_current)
            if index >= len(sequence):
                continue
            name = sequence[index]
            if name not in self._avail or name == "ACTION6":
                continue
            if self.guard.noop[name][0] >= 3 and \
                    self.guard.noop_rate(name) >= 0.9:
                continue
            if self.guard.validate(name, None, self._avail, self.dims):
                return self._emit(name, None, f"analogy L{_level} step {index}")
        return None


AGENT_NAME = "atlas-v3"


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
