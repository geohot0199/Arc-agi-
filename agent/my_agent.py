"""ATLAS Phase 0 agent — ARC Prize 2026 (ARC-AGI-3). v2.1

Adaptive Task-Learning Agent System. Single-file agent (stdlib only) that the
Kaggle notebook splices into the ARC-AGI-3-Agents framework as
``agents/templates/my_agent.py`` and registers as ``myagent``.

v2.1 upgrades implemented here (see docs/05-atlas-v2p1-calibration-review.md):
  * ArchetypePosterior — soft probabilities over game archetypes with entropy
    abstention (+ up to 2 info-gain probes) instead of a hard router. Updates
    on every classified transition. Candidate scoring is bias-weighted by the
    posterior, never winner-take-all.
  * Click funnel v2 — ACTION6 candidates scored by rarity/size/recency/
    player-proximity/click-success-ledger/info-gain, with exact-death cells
    excluded and near-miss-death regions soft-warned. 4096 -> <=12 candidates.
  * Death graph v2 — multi-key (exact grid hash + object-multiset hash) hard
    blocks; downsampled Hamming near-miss gives a soft warn only (asymmetry:
    never block on fuzzy evidence).
  * Calibration bus — every judgment emits {component, judgment, value,
    threshold, decision} to stdout and, when present, a jsonl file, so silent
    failures are visible in traces.
  * Stop-loss with a hard spine — soft budget switches to exploit mode;
    absolute hard cap stops exploration entirely. (Full EV(continue) vs
    EV(divert) needs swarm-level control: Phase 1.)
  * Latency governor — LLM advisor consults adaptively (backs off on
    rejection), wall-time telemetry logged.
  * (DSL-as-prior-not-cage and CERTIFY complexity/coverage metrics land with
    the world-model phase; the calibration bus they will emit is here now.)

Phase 0 core (unchanged): perception v0 (hash/diff/segment/salience), memory
ledgers, probe script, reflex policy, guarded LLM advisor, never-crash guards.

Also fixed vs the first draft: LLM-proposed actions now set last_action_key so
their outcomes are recorded in the ledgers (previously silently unattributed).
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.request
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Imports: framework template context first, fallbacks for local development.
# ---------------------------------------------------------------------------
try:  # normal placement: ARC-AGI-3-Agents/agents/templates/my_agent.py
    from .agent import Agent  # type: ignore
except ImportError:  # pragma: no cover - local dev / test shims
    try:
        from agents.agent import Agent  # type: ignore
    except ImportError:
        Agent = object  # type: ignore  (standalone import for tooling)

try:
    from arcengine import GameAction  # type: ignore
except ImportError:  # pragma: no cover
    GameAction = None  # type: ignore

try:
    from arcengine import GameState  # type: ignore
except ImportError:  # pragma: no cover
    GameState = None  # type: ignore


# ---------------------------------------------------------------------------
# Configuration (env-overridable; the notebook writes these into .env)
# ---------------------------------------------------------------------------
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


CFG = {
    "MAX_ACTIONS": _env_int("ATLAS_MAX_ACTIONS", 360),
    "LEVEL_BUDGET": _env_int("ATLAS_LEVEL_BUDGET", 110),  # soft: exploit mode
    "LEVEL_HARD_CAP": _env_int("ATLAS_LEVEL_HARD_CAP", 220),  # hard spine
    "LLM_MODE": (os.getenv("ATLAS_LLM", "auto").strip().lower()),
    "LLM_URL": os.getenv("ATLAS_LLM_URL", "http://127.0.0.1:8000/v1").rstrip("/"),
    "LLM_TIMEOUT": _env_int("ATLAS_LLM_TIMEOUT", 45),
    "LLM_EVERY": _env_int("ATLAS_LLM_EVERY", 3),
    "LLM_MAX_FAILS": _env_int("ATLAS_LLM_MAX_FAILS", 5),
    "LLM_MAX_EVERY": 9,  # latency governor backoff ceiling
    "STAGNATION_RESET": _env_int("ATLAS_STAGNATION_RESET", 8),
    "MAX_RESETS_PER_LEVEL": _env_int("ATLAS_MAX_RESETS_PER_LEVEL", 4),
    "SALIENT_CAP": _env_int("ATLAS_SALIENT_CAP", 12),
    "ABSTAIN_ENTROPY": float(os.getenv("ATLAS_ABSTAIN_ENTROPY", "1.3")),
    "MAX_INFO_PROBES": _env_int("ATLAS_MAX_INFO_PROBES", 2),
    "NEARMISS_FRAC": float(os.getenv("ATLAS_NEARMISS_FRAC", "0.06")),  # of cells
    "CAL_FILE": os.getenv("ATLAS_CAL_FILE", ""),  # e.g. /kaggle/working/atlas_calibration.jsonl
    "LOG": os.getenv("ATLAS_LOG", "1").strip() not in ("0", "false", "False"),
    "SEED": _env_int("ATLAS_SEED", 1337),
}

_COLOR_CHARS = "0123456789abcdef"
ARCHETYPES = ("MOVEMENT", "CLICK_PUZZLE", "COLOR_LOGIC", "SELECTION", "OTHER")


def _log(msg: str) -> None:
    if CFG["LOG"]:
        print(f"[atlas] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Calibration bus (v2.1): every judgment is a logged number with a threshold.
# ---------------------------------------------------------------------------
_CAL_BUF: list = []


def cal_log(component: str, judgment: str, value, threshold, decision: str) -> None:
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
    path = CFG["CAL_FILE"]
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Small structural helpers (defensive against wheel-version differences)
# ---------------------------------------------------------------------------
def norm_state(state) -> str:
    try:
        return str(getattr(state, "name", state)).upper()
    except Exception:
        return "UNKNOWN"


def action_name(action) -> str:
    try:
        return str(getattr(action, "name", action)).upper()
    except Exception:
        return "UNKNOWN"


def available_names(latest_frame) -> set:
    try:
        raw = list(getattr(latest_frame, "available_actions", None) or [])
    except Exception:
        raw = []
    return {action_name(a) for a in raw}


def grids_of(frame) -> list:
    try:
        f = getattr(frame, "frame", None)
        if f is None:
            return []
        f = list(f)
    except Exception:
        return []
    if not f:
        return []
    first = f[0]
    if isinstance(first, (int, float)):
        return [f]
    if first and isinstance(first[0], (int, float)):
        return [f]
    return f


def _tuple_grid(grid) -> tuple:
    try:
        return tuple(tuple(int(c) for c in row) for row in grid)
    except Exception:
        return ()


def frame_hash(latest_frame) -> str:
    return "|".join(str(hash(_tuple_grid(g))) for g in grids_of(latest_frame))


def object_multiset_hash(grid: tuple) -> int:
    """Second death-graph key: invariant to object positions."""
    objs = segment(grid, min_size=1, max_objects=80)
    return hash(frozenset((o["color"], o["size"]) for o in objs))


def grid_dims(latest_frame):
    grids = grids_of(latest_frame)
    if not grids:
        return (0, 0)
    g = grids[0]
    return (len(g), len(g[0]) if g else 0)


def downsample(grid: tuple, side: int = 16) -> tuple:
    """Stride-downsample a grid to <= side x side for cheap near-miss tests."""
    if not grid:
        return ()
    h, w = len(grid), len(grid[0])
    sy, sx = max(1, (h + side - 1) // side), max(1, (w + side - 1) // side)
    return tuple(
        tuple(grid[y][x] for x in range(0, w, sx)) for y in range(0, h, sy)
    )


def hamming_frac(a: tuple, b: tuple) -> float:
    if not a or not b:
        return 1.0
    n = mism = 0
    for ra, rb in zip(a, b):
        for ca, cb in zip(ra, rb):
            n += 1
            if ca != cb:
                mism += 1
    return (mism / n) if n else 1.0


def diff_cells(g1, g2, cap: int = 400):
    out = []
    h = min(len(g1), len(g2))
    for y in range(h):
        r1, r2 = g1[y], g2[y]
        w = min(len(r1), len(r2))
        for x in range(w):
            if r1[x] != r2[x]:
                out.append((x, y, int(r1[x]), int(r2[x])))
                if len(out) >= cap:
                    return out
    return out


def dominant_grid(latest_frame) -> tuple:
    grids = grids_of(latest_frame)
    return _tuple_grid(grids[0]) if grids else ()


# ---------------------------------------------------------------------------
# Perception: segmentation (4-connected components, pure python)
# ---------------------------------------------------------------------------
def segment(grid: tuple, min_size: int = 1, max_objects: int = 60):
    if not grid:
        return []
    h, w = len(grid), len(grid[0])
    seen = [[False] * w for _ in range(h)]
    objs = []
    for y in range(h):
        for x in range(w):
            c = grid[y][x]
            if seen[y][x] or c == 0:
                continue
            stack = [(x, y)]
            seen[y][x] = True
            cells = []
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] and grid[ny][nx] == c:
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            if len(cells) < min_size:
                continue
            xs = [p[0] for p in cells]
            ys = [p[1] for p in cells]
            objs.append({
                "color": int(c),
                "size": len(cells),
                "cells": cells,
                "bbox": (min(xs), min(ys), max(xs), max(ys)),
                "centroid": (sum(xs) / len(xs), sum(ys) / len(ys)),
            })
            if len(objs) >= max_objects:
                break
    objs.sort(key=lambda o: -o["size"])
    return objs


def render_ascii(latest_frame, max_side: int = 48) -> str:
    grids = grids_of(latest_frame)
    if not grids:
        return "<empty frame>"
    g = grids[0]
    h, w = len(g), len(g[0]) if g else 0
    if h == 0 or w == 0:
        return "<empty grid>"
    sy = max(1, (h + max_side - 1) // max_side)
    sx = max(1, (w + max_side - 1) // max_side)
    lines = [f"(downsampled {sy}x{sx}; origin top-left)"] if (sx > 1 or sy > 1) else []
    lines.append("    " + "".join(
        (str((x // sx) % 10) if ((x // sx) % 8 == 0) else " ") for x in range(0, w, sx)))
    for y in range(0, h, sy):
        row = "".join(_COLOR_CHARS[int(g[y][x]) & 0xF] for x in range(0, w, sx))
        lines.append(f"{y:3d} {row}")
    return "\n".join(lines)


def classify_event(prev_levels, cur_levels, cur_state, diff) -> str:
    s = norm_state(cur_state)
    if s == "WIN":
        return "WIN"
    if s == "GAME_OVER":
        return "DEATH"
    if cur_levels > prev_levels:
        return "LEVEL_UP"
    n = len(diff)
    if n == 0:
        return "NOOP"
    if n <= 3:
        return "TINY_CHANGE"
    if n > 80:
        return "BIG_CHANGE"
    return "CHANGE"


# ---------------------------------------------------------------------------
# Archetype posterior (v2.1): soft router with abstention + info-gain probes
# ---------------------------------------------------------------------------
class ArchetypePosterior:
    """P(archetype) over {MOVEMENT, CLICK_PUZZLE, COLOR_LOGIC, SELECTION, OTHER}.

    Evidence is a dict of booleans; each archetype scores evidence via weights;
    posterior odds multiply and normalize (with a floor so nothing hits 0).
    """

    WEIGHTS = {
        # evidence key:        MOV   CLICK  COLOR  SELECT  OTHER
        "entity_moved":       (2.2, 0.2, 0.1, 0.2, 0.1),
        "simple_actions_move": (1.6, 0.1, 0.0, 0.1, 0.1),
        "click_local_change":  (0.2, 2.2, 0.4, 0.6, 0.2),
        "click_level_change":  (0.1, 0.6, 0.3, 2.2, 0.2),
        "global_recolor":      (0.1, 0.3, 2.4, 0.2, 0.3),
        "simple_noop_only":    (0.3, 1.0, 0.6, 0.4, 0.6),
        "surprise_on_plan":    (0.5, 0.5, 0.5, 0.5, 1.2),  # plan failed: OTHER up
    }

    def __init__(self) -> None:
        self.p = {a: 0.2 for a in ARCHETYPES}
        self._floor = 0.02

    def update(self, evidence: dict) -> None:
        for key, val in evidence.items():
            if not val or key not in self.WEIGHTS:
                continue
            w = self.WEIGHTS[key]
            for i, a in enumerate(ARCHETYPES):
                self.p[a] *= (1.0 + w[i])
        lo = self._floor
        total = sum(self.p.values()) or 1.0
        self.p = {a: max(v / total, lo) for a, v in self.p.items()}
        z = sum(self.p.values())
        self.p = {a: v / z for a, v in self.p.items()}

    def entropy(self) -> float:
        import math
        return -sum(v * math.log(v + 1e-9) for v in self.p.values())

    def top(self, n=2):
        return sorted(self.p.items(), key=lambda kv: -kv[1])[:n]

    def abstain(self, threshold: float) -> bool:
        h = self.entropy()
        return h > threshold

    def summary(self) -> str:
        return " ".join(f"{a[:6]}={self.p[a]:.2f}" for a in ARCHETYPES)


# ---------------------------------------------------------------------------
# Optional LLM advisor (OpenAI-compatible local endpoint, urllib only)
# ---------------------------------------------------------------------------
class AtlasLLM:
    """Dependency-free client for a local vLLM/OpenAI-compatible server.

    Returns a parsed JSON proposal dict or None (None => reflex). After
    CFG['LLM_MAX_FAILS'] failures it disables itself for the rest of the game.
    """

    def __init__(self) -> None:
        self.url = CFG["LLM_URL"]
        self.timeout = CFG["LLM_TIMEOUT"]
        self.fails = 0
        self.disabled = False
        self.model = None
        self.last_latency = 0.0

    def probe(self) -> bool:
        if self.disabled:
            return False
        try:
            with urllib.request.urlopen(f"{self.url}/models", timeout=8) as r:
                data = json.loads(r.read().decode("utf-8"))
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

    def propose(self, system: str, user: str) -> dict | None:
        if self.disabled:
            return None
        payload = {
            "model": self.model or "atlas",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": 300,
        }
        t0 = time.time()
        try:
            req = urllib.request.Request(
                f"{self.url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            self.last_latency = time.time() - t0
            text = data["choices"][0]["message"]["content"] or ""
            return self._parse_json(text)
        except Exception:
            self.last_latency = time.time() - t0
            self.fails += 1
            if self.fails >= CFG["LLM_MAX_FAILS"]:
                self.disabled = True
                _log("LLM disabled after repeated failures; reflex only.")
            return None

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        text = text.strip()
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


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
class MyAgent(Agent):
    """ATLAS Phase-0 agent: probe -> perceive -> believe (softly) -> act."""

    MAX_ACTIONS = CFG["MAX_ACTIONS"]

    def __init__(self, *args, **kwargs) -> None:
        if Agent is not object:
            try:
                super().__init__(*args, **kwargs)
            except Exception:
                pass  # never die in construction
        else:  # standalone/test context
            self.frames = []
            self.action_counter = 0
            self.game_id = kwargs.get("game_id", "mock")
        import zlib
        self.rng = random.Random(CFG["SEED"] ^ zlib.crc32(self.game_id.encode()))
        self._t0 = time.time()

        # memory ------------------------------------------------------------
        self.timeline = []
        self.noop_ledger = defaultdict(lambda: [0, 0])     # action name -> [used, noops]
        self.progress_ledger = defaultdict(lambda: [0, 0])  # action name -> [used, progress]
        self.click_ledger = defaultdict(lambda: [0, 0])     # (x,y) -> [clicks, changes]
        self.death_exact = Counter()                        # (state_hash, key) -> n
        self.death_mset = Counter()                         # (mset_hash, name) -> n
        self.death_snaps = []                               # [(ds_grid, key, x, y)]
        self.tried_here = defaultdict(set)                  # state_hash -> {key}
        self.state_visits = Counter()
        self.posterior = ArchetypePosterior()
        self.player_pos = None                              # (x, y) tracked entity
        self.levels_completed = 0
        self.win_levels = None
        self.prev_grid = None
        self.prev_state = None
        self.prev_levels = 0
        self.last_action_key = None
        self.last_state_hash = None
        self.last_diff = []
        self.last_event = None
        self.consec_noops = 0
        self.level_start_actions = 0
        self.resets_this_level = 0
        self.info_probes_used = 0
        self.click_probes_used = 0
        self._probes_built = False
        self._cur_hash = None
        self._mset_now = None
        self.probe_queue = []
        self.probing = True
        # LLM / latency governor
        self.click_usable = True  # circuit breaker if engine lacks set_data
        self.llm = None
        self.llm_ok = False
        self.llm_every = CFG["LLM_EVERY"]
        self._llm_asks = 0
        self._llm_accepted = 0
        self._llm_rejected = 0
        self.last_ascii = ""
        self.mode = "explore"  # explore -> exploit (soft) -> conserve (hard spine)

    # ------------------------------------------------------------------
    # outcome bookkeeping: the free supervision signal
    # ------------------------------------------------------------------
    def _mark_outcome(self, latest_frame) -> None:
        if self.last_action_key is None:
            return
        key = self.last_action_key
        name = key.split("@")[0]
        grid = dominant_grid(latest_frame)
        diff = diff_cells(self.prev_grid, grid) if (self.prev_grid and grid) else []
        event = classify_event(
            self.prev_levels,
            getattr(latest_frame, "levels_completed", 0) or 0,
            getattr(latest_frame, "state", None),
            diff,
        )
        s_hash = frame_hash(latest_frame)
        self.state_visits[s_hash] += 1
        self.last_diff = diff
        self.last_event = event
        if event == "NOOP":
            self.consec_noops += 1
        else:
            self.consec_noops = 0

        if name != "RESET":  # RESET is bookkeeping, not evidence
            self.timeline.append({
                "action": name, "event": event, "diff": len(diff),
                "levels": getattr(latest_frame, "levels_completed", 0) or 0,
                "state": norm_state(getattr(latest_frame, "state", None)),
            })
            if len(self.timeline) > 2000:
                self.timeline = self.timeline[-1000:]

            self.noop_ledger[name][0] += 1
            self.progress_ledger[name][0] += 1
            if event in ("LEVEL_UP", "WIN"):
                self.progress_ledger[name][1] += 1
            if event == "NOOP":
                self.noop_ledger[name][1] += 1
            if name == "ACTION6" and key.startswith("ACTION6@"):
                try:
                    x, y = key.split("@")[1].split(",")
                    cell = (int(x), int(y))
                    self.click_ledger[cell][0] += 1
                    if event != "NOOP":
                        self.click_ledger[cell][1] += 1
                except Exception:
                    pass
            if event == "DEATH":
                self.death_exact[(self.last_state_hash or s_hash, key)] += 1
                if grid:
                    self.death_mset[(object_multiset_hash(grid), name)] += 1
                    if len(self.death_snaps) < 30:
                        self.death_snaps.append((downsample(grid), key, None, None))

        # entity tracking + archetype evidence -----------------------------
        ev = self._evidence(name, event, diff, grid)
        if any(ev.values()):
            self.posterior.update(ev)

        self._log_event(event, name, len(diff))
        # advance
        self.prev_grid = grid
        self.prev_state = getattr(latest_frame, "state", None)
        self.prev_levels = getattr(latest_frame, "levels_completed", 0) or 0
        self.last_action_key = None
        self.last_state_hash = None

    def _evidence(self, name: str, event: str, diff, grid: tuple) -> dict:
        """Turn one transition into archetype evidence (+ player tracking)."""
        ev = {
            "entity_moved": False, "simple_actions_move": False,
            "click_local_change": False, "click_level_change": False,
            "global_recolor": False, "simple_noop_only": False,
        }
        if event in ("LEVEL_UP", "WIN") and name == "ACTION6":
            ev["click_level_change"] = True
            return ev
        if event == "DEATH" or not grid:
            return ev
        if not diff:  # no-op: cheap evidence about which archetype we're NOT in
            if name in (f"ACTION{i}" for i in range(1, 6)):
                ev["simple_noop_only"] = True
            return ev
        vanished = [(x, y) for x, y, o, n in diff if o != 0 and n == 0]
        appeared = [(x, y) for x, y, o, n in diff if o == 0 and n != 0]
        recolored = [(x, y, o, n) for x, y, o, n in diff if o != 0 and n != 0]
        moved = (
            0 < len(vanished) <= 6 and 0 < len(appeared) <= 6
            and len(recolored) == 0
        )
        if moved:
            cx = sum(p[0] for p in appeared) / len(appeared)
            cy = sum(p[1] for p in appeared) / len(appeared)
            old = (sum(p[0] for p in vanished) / len(vanished),
                   sum(p[1] for p in vanished) / len(vanished))
            if (cx - old[0]) ** 2 + (cy - old[1]) ** 2 <= 25:
                ev["entity_moved"] = True
                if name in (f"ACTION{i}" for i in range(1, 6)):
                    ev["simple_actions_move"] = True
                self.player_pos = (cx, cy)
        if name == "ACTION6":
            near = False
            try:
                x, y = self.last_action_key.split("@")[1].split(",")
                ax, ay = float(x), float(y)
                near = any((ax - px) ** 2 + (ay - py) ** 2 <= 9 for px, py, _o, _n2 in diff)
            except Exception:
                pass
            if near and len(diff) <= 80:
                ev["click_local_change"] = True
        if len(diff) > 80 and len(recolored) > 0.6 * len(diff):
            ev["global_recolor"] = True
        if (not moved) and name in (f"ACTION{i}" for i in range(1, 6)):
            # simple actions doing nothing (so far) hints at click/colour games
            ev["simple_noop_only"] = True
        return ev

    def _log_event(self, event: str, name: str, diffn: int) -> None:
        if CFG["LOG"] and (event not in ("NOOP", "CHANGE", "TINY_CHANGE")
                           or self.action_counter % 25 == 0):
            _log(f"{self.game_id} #{self.action_counter} {name} -> {event} "
                 f"(Δ{diffn}) lvl={self.prev_levels} [{self.posterior.summary()}]")

    def _track_level(self, latest_frame) -> None:
        lv = getattr(latest_frame, "levels_completed", 0) or 0
        wl = getattr(latest_frame, "win_levels", None)
        if wl:
            self.win_levels = wl
        if lv > self.prev_levels:
            _log(f"{self.game_id} LEVEL_UP -> {lv} "
                 f"(actions on level: {self.action_counter - self.level_start_actions})")
            self.level_start_actions = self.action_counter
            self.resets_this_level = 0
            cal_log("scheduler", "level_clear_actions",
                    self.action_counter - self.level_start_actions,
                    CFG["LEVEL_BUDGET"], "level cleared")
        self.levels_completed = lv

    # ------------------------------------------------------------------
    # click funnel v2: 4096 -> <=SALIENT_CAP scored candidates
    # ------------------------------------------------------------------
    def _salient_targets(self, latest_frame, h, w):
        if not self.click_usable:
            return []
        grid = dominant_grid(latest_frame)
        if not grid:
            return []
        objs = segment(grid, min_size=1, max_objects=40)
        counts = Counter(cell for row in grid for cell in row)
        rare = {c for c, n in counts.items() if c != 0 and 1 <= n <= 4}
        recent = {(x, y) for x, y, _o, _n in self.last_diff[:60]}
        targets = []

        def _add(score, x, y):
            x = max(0, min(w - 1, int(x)))
            y = max(0, min(h - 1, int(y)))
            targets.append((score, x, y))

        for o in objs:
            cx, cy = o["centroid"]
            bx0, by0, bx1, by1 = o["bbox"]
            s = 1.0
            if o["color"] in rare:
                s += 3.0
            if o["size"] <= 4:
                s += 2.0
            if self.player_pos:
                d = ((cx - self.player_pos[0]) ** 2 + (cy - self.player_pos[1]) ** 2) ** 0.5
                s += 2.5 * pow(2.718281828, -d / 8.0)  # proximity decay
            if (int(cx), int(cy)) in recent:
                s += 1.0
            _add(s, cx, cy)                                   # interior/centroid
            _add(s * 0.8, bx0, by0)                           # boundary corner
            if o["size"] >= 6:
                _add(s * 0.7, (bx0 + bx1) // 2, (by0 + by1) // 2)  # mid cell
        for (x, y) in list(recent)[:10]:                       # changed cells
            _add(4.0, x, y)
        # click-success ledger and suppression
        ds_grid = downsample(grid)
        nm_cap = CFG["NEARMISS_FRAC"]
        seen = {}
        out = []
        for sc, x, y in sorted(targets, reverse=True):
            if (x, y) in seen:
                continue
            clicks, changes = self.click_ledger[(x, y)]
            if clicks and changes == 0:
                sc -= 2.5               # known no-op cell
            elif clicks:
                sc += 2.0 * (changes / clicks)  # previously successful
            else:
                sc += 1.0               # information gain: untried cell
            # exact death cell (multi-key): exclude
            if self.death_exact.get((self._cur_hash, f"ACTION6@{x},{y}"), 0):
                continue
            # near-miss death region: soft warn only (asymmetry)
            for dgrid, dkey, _dx, _dy in self.death_snaps:
                if dkey and dkey.startswith("ACTION6@"):
                    try:
                        kx, ky = dkey.split("@")[1].split(",")
                        if (int(kx) - x) ** 2 + (int(ky) - y) ** 2 <= 9:
                            sc -= 1.5
                            break
                    except Exception:
                        pass
                elif hamming_frac(ds_grid, dgrid) < nm_cap:
                    sc -= 0.5
                    break
            seen[(x, y)] = True
            out.append((sc, x, y))
            if len(out) >= CFG["SALIENT_CAP"]:
                break
        return out

    # ------------------------------------------------------------------
    # LLM advisory (optional) with latency governor
    # ------------------------------------------------------------------
    def _llm_init(self) -> None:
        if CFG["LLM_MODE"] in ("0", "off", "false"):
            self.llm_ok = False
            self.llm = "skipped"  # sentinel: don't re-init
            return
        self.llm = AtlasLLM()
        self.llm_ok = self.llm.probe()
        cal_log("llm", "available", 1.0 if self.llm_ok else 0.0, 0.5,
                "ready" if self.llm_ok else "reflex-only")
        _log(f"{self.game_id} LLM advisor "
             f"{'ready' if self.llm_ok else 'unavailable -> reflex only'}")

    def _llm_governor(self) -> None:
        """Back off consult frequency when proposals keep being rejected."""
        if self._llm_asks >= 6 and self._llm_rejected > 2 * max(1, self._llm_accepted):
            if self.llm_every < CFG["LLM_MAX_EVERY"]:
                self.llm_every = min(CFG["LLM_MAX_EVERY"], self.llm_every * 2)
                cal_log("latency", "llm_backoff", self.llm_every,
                        CFG["LLM_MAX_EVERY"], f"consult every {self.llm_every}")
                self._llm_rejected = self._llm_rejected // 2

    def _llm_action(self, latest_frame, avail, h, w):
        if not self.llm_ok or not isinstance(self.llm, AtlasLLM):
            return None
        self._llm_asks += 1
        if self._llm_asks % self.llm_every != 1:
            return None
        grid = dominant_grid(latest_frame)
        stats = {
            n: {"used": v[0], "noops": v[1],
                "progress": self.progress_ledger[n][1]}
            for n, v in list(self.noop_ledger.items())[:8]
        }
        user = json.dumps({
            "level": self.levels_completed,
            "win_levels": self.win_levels,
            "available_actions": sorted(avail),
            "archetype_posterior": {k: round(v, 2) for k, v in self.posterior.p.items()},
            "action_stats": stats,
            "recent_events": [f"{t['action']}:{t['event']}" for t in self.timeline[-12:]],
            "actions_taken": self.action_counter,
            "player_pos": self.player_pos,
            "grid_ascii": self.last_ascii or render_ascii(latest_frame),
            "objects": [
                {"color": o["color"], "size": o["size"],
                 "at": [int(o["centroid"][0]), int(o["centroid"][1])]}
                for o in (segment(grid, 1, 12) if grid else [])
            ],
            "salient_clicks": [[x, y] for _s, x, y
                               in self._salient_targets(latest_frame, h, w)[:6]],
            "task": "Choose ONE next action to make progress. Reply ONLY JSON: "
                    '{"action":"ACTION3","x":0,"y":0,"reason":"..."} '
                    "(x,y only for ACTION6, must be one of salient_clicks).",
        })
        obj = self.llm.propose(
            "You are ATLAS, an agent playing an unknown ARC-AGI-3 grid game. "
            "Grid chars are hex colors 0-f, (0,0) top-left, y is the row. "
            "Prefer actions that historically caused progress; avoid no-ops and "
            "deaths. Output ONLY the JSON object.", user)
        if obj is None:
            self._llm_rejected += 1
            self._llm_governor()
            return None
        name = str(obj.get("action", "")).upper()
        if name not in avail or name == "RESET" or (name == "ACTION6" and not self.click_usable):
            self._llm_rejected += 1
            self._llm_governor()
            return None
        act = self._make_action(name)
        x = y = None
        if name == "ACTION6":
            try:
                x, y = int(obj.get("x", -1)), int(obj.get("y", -1))
            except (TypeError, ValueError):
                self._llm_rejected += 1
                return None
            if not (0 <= x < w and 0 <= y < h):
                self._llm_rejected += 1
                return None
            sal = self._salient_targets(latest_frame, h, w)
            if sal:
                bx, by = min(((sx, sy) for _s, sx, sy in sal),
                             key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
                if (bx - x) ** 2 + (by - y) ** 2 <= 16:
                    x, y = bx, by
            if not self._set_click(act, x, y):
                self._llm_rejected += 1
                return None
        self._llm_accepted += 1
        key = f"ACTION6@{x},{y}" if (name == "ACTION6" and x is not None) else name
        try:
            act.reasoning = f"llm: {str(obj.get('reason',''))[:200]}"
        except Exception:
            pass
        return act, key

    # ------------------------------------------------------------------
    # action construction (defensive against GameAction API differences)
    # ------------------------------------------------------------------
    def _make_action(self, name: str):
        if GameAction is not None:
            try:
                return GameAction[name]
            except Exception:
                pass
        return name  # test shim

    def _set_click(self, act, x: int, y: int) -> bool:
        try:
            if hasattr(act, "set_data"):
                act.set_data({"x": int(x), "y": int(y)})
                return True
            if hasattr(act, "action_data"):
                act.action_data.x = int(x)
                act.action_data.y = int(y)
                return True
        except Exception:
            pass
        return False

    def _finalize(self, act, name: str, key: str, s_hash, reason: str):
        try:
            if hasattr(act, "reasoning"):
                act.reasoning = reason
        except Exception:
            pass
        self.last_action_key = key
        self.last_state_hash = s_hash
        return act

    # ------------------------------------------------------------------
    # framework interface
    # ------------------------------------------------------------------
    def is_done(self, frames, latest_frame) -> bool:
        try:
            s = norm_state(getattr(latest_frame, "state", None))
            lv = getattr(latest_frame, "levels_completed", 0) or 0
            wl = getattr(latest_frame, "win_levels", None) or self.win_levels
            if s == "WIN" and wl and lv >= int(wl):
                _log(f"{self.game_id} DONE: WIN at level {lv}/{wl}")
                return True
            if s == "WIN" and not wl:
                done = self.action_counter > max(8, 2 * max(1, lv) * 10)
                if done:
                    _log(f"{self.game_id} DONE: WIN with unknown win_levels")
                return done
        except Exception:
            pass
        return False

    def choose_action(self, frames, latest_frame):
        try:
            return self._choose(frames, latest_frame)
        except Exception as e:
            _log(f"{self.game_id} choose_action fallback after error: {e!r}")
            try:
                if GameAction is not None:
                    return GameAction.RESET
            except Exception:
                pass
            return "RESET"

    # -- the real policy -------------------------------------------------
    def _choose(self, frames, latest_frame):
        avail = available_names(latest_frame)
        if not avail:
            avail = {f"ACTION{i}" for i in range(1, 8)} | {"RESET"}
        s = norm_state(getattr(latest_frame, "state", None))
        h, w = grid_dims(latest_frame)

        # 1) record previous action's outcome
        if s != "NOT_PLAYED":
            self._mark_outcome(latest_frame)
        self._track_level(latest_frame)
        self.last_ascii = render_ascii(latest_frame)
        s_hash = frame_hash(latest_frame)
        self._cur_hash = s_hash
        grid_now = dominant_grid(latest_frame)
        self._mset_now = object_multiset_hash(grid_now) if grid_now else None

        # 2) lifecycle states
        if s in ("NOT_PLAYED", "GAME_OVER"):
            act = self._make_action("RESET")
            self.resets_this_level += 1
            return self._finalize(act, "RESET", "RESET", s_hash,
                                  f"restart level (state={s})")
        if s == "WIN":
            act = self._make_action("RESET")
            return self._finalize(act, "RESET", "RESET", s_hash,
                                  "won a level; continue to next")

        # 3) one-time setup (probe queue is built exactly once per game)
        if self.llm is None:
            self._llm_init()
        if self.probing and not self.probe_queue and not self._probes_built:
            self.probe_queue = [f"ACTION{i}" for i in range(1, 6)
                                if f"ACTION{i}" in avail]
            if "ACTION7" in avail:
                self.probe_queue.append("ACTION7")
            self._probes_built = True

        # 4) optional LLM advice (validated + attributed)
        if self.llm_ok:
            got = self._llm_action(latest_frame, avail, h, w)
            if got is not None:
                act, key = got
                self.tried_here[s_hash].add(key)
                return self._finalize(act, key.split("@")[0], key, s_hash,
                                      "llm proposal accepted")

        # 5) probe phase + abstention (v2.1: entropy gate before commitment)
        if self.probing and self.probe_queue:
            while self.probe_queue:
                name = self.probe_queue.pop(0)
                if name == "RESET" or (s_hash, name) in self.death_exact:
                    continue
                self.tried_here[s_hash].add(name)
                act = self._make_action(name)
                return self._finalize(act, name, name, s_hash,
                                      "probe: identify action")
        if self.probing:
            if (self.posterior.abstain(CFG["ABSTAIN_ENTROPY"])
                    and self.info_probes_used < CFG["MAX_INFO_PROBES"]):
                # info-gain probe: run the experiment that best separates the
                # top-2 hypotheses (movement probe vs click probe).
                self.info_probes_used += 1
                (top_a, _pa), (top_b, _pb) = self.posterior.top(2)
                want_click = {top_a, top_b} >= {"CLICK_PUZZLE"} or top_a == "CLICK_PUZZLE"
                sal = self._salient_targets(latest_frame, h, w)
                if want_click and sal:
                    _sc, x, y = sal[0]
                    key = f"ACTION6@{x},{y}"
                    if key not in self.tried_here[s_hash]:
                        self.tried_here[s_hash].add(key)
                        act = self._make_action("ACTION6")
                        if self._set_click(act, x, y):
                            cal_log("router", "info_probe", self.posterior.entropy(),
                                    CFG["ABSTAIN_ENTROPY"],
                                    f"click probe ({x},{y}) to split "
                                    f"{top_a}/{top_b}")
                            return self._finalize(act, "ACTION6", key, s_hash,
                                                  "info-gain click probe")
                else:
                    untried = [f"ACTION{i}" for i in range(1, 6)
                               if f"ACTION{i}" in avail
                               and f"ACTION{i}" not in self.tried_here[s_hash]]
                    if untried:
                        name = untried[0]
                        self.tried_here[s_hash].add(name)
                        act = self._make_action(name)
                        cal_log("router", "info_probe", self.posterior.entropy(),
                                CFG["ABSTAIN_ENTROPY"],
                                f"move probe {name} to split {top_a}/{top_b}")
                        return self._finalize(act, name, name, s_hash,
                                              "info-gain move probe")
            elif not self.posterior.abstain(CFG["ABSTAIN_ENTROPY"]) or \
                    self.info_probes_used >= CFG["MAX_INFO_PROBES"]:
                if self.click_probes_used < 3:
                    sal = self._salient_targets(latest_frame, h, w)
                    for _sc, x, y in sal[:2]:
                        key = f"ACTION6@{x},{y}"
                        if key in self.tried_here[s_hash] or \
                                (s_hash, key) in self.death_exact:
                            continue
                        act = self._make_action("ACTION6")
                        if not self._set_click(act, x, y):
                            continue
                        self.click_probes_used += 1
                        self.tried_here[s_hash].add(key)
                        return self._finalize(act, "ACTION6", key, s_hash,
                                              f"probe click at ({x},{y})")
                self.probing = False
                cal_log("router", "committed", self.posterior.entropy(),
                        CFG["ABSTAIN_ENTROPY"],
                        f"probes done; posterior {self.posterior.summary()}")
                _log(f"{self.game_id} probes done at #{self.action_counter}; "
                     f"policy mode [{self.posterior.summary()}]")

        # 6) mode selection: soft budget -> exploit; hard cap -> conserve
        on_level = self.action_counter - self.level_start_actions
        if on_level > CFG["LEVEL_HARD_CAP"]:
            if self.mode != "conserve":
                self.mode = "conserve"
                cal_log("scheduler", "hard_spine", on_level,
                        CFG["LEVEL_HARD_CAP"],
                        "conserve: exploit known progress only")
        elif on_level > CFG["LEVEL_BUDGET"]:
            if self.mode != "exploit":
                self.mode = "exploit"
                cal_log("scheduler", "soft_budget", on_level,
                        CFG["LEVEL_BUDGET"], "exploit mode")

        # 7) stagnation control: consecutive no-ops = no new information.
        if self.consec_noops >= CFG["STAGNATION_RESET"] and \
                self.resets_this_level < CFG["MAX_RESETS_PER_LEVEL"] and \
                self.mode != "conserve":
            self.resets_this_level += 1
            self.consec_noops = 0
            act = self._make_action("RESET")
            return self._finalize(act, "RESET", "RESET", s_hash,
                                  "stagnation: deliberate reset (memory kept)")

        # 8) reflex policy, bias-weighted by the posterior (never winner-take-all)
        p = self.posterior.p
        move_bias = p["MOVEMENT"]
        click_bias = max(p["CLICK_PUZZLE"], p["SELECTION"], p["COLOR_LOGIC"])
        cands = []
        # ACTION6 candidates come ONLY from the scored funnel (with xy);
        # bare ACTION6 without coordinates is not a valid action.
        for name in sorted(avail - {"RESET", "ACTION6"}):
            key = name
            used, noops = self.noop_ledger[name]
            prog = self.progress_ledger[name]
            affinity = (prog[1] / prog[0]) if prog[0] else 0.0
            noop_rate = (noops / used) if used else 0.0
            change_rate = 1.0 - noop_rate
            score = 0.6 * affinity + 0.5 * prog[1] - 1.5 * noop_rate
            if name.startswith("ACTION") and name != "ACTION6":
                score += 1.2 * move_bias          # posterior bias, soft
                # reliability: a proven state-changer is a locomotion action;
                # chain it instead of cycling untried no-ops.
                if used >= 2 and change_rate >= 0.5:
                    score += 2.0 * move_bias * change_rate
            if used == 0:
                score += 1.0
            if key in self.tried_here[s_hash]:
                score -= 1.2
                if used >= 2 and change_rate >= 0.5 and move_bias > 0.4:
                    score += 0.9  # re-moving from a state is normal in movement games
            score -= 3.0 * self.death_exact.get((s_hash, key), 0)
            if self._mset_now:
                score -= 0.5 * self.death_mset.get((self._mset_now, name), 0)
            if self.mode == "exploit" and affinity <= 0:
                score -= 1.0
            if self.mode == "conserve" and prog[1] == 0:
                score -= 2.0                      # only proven actions
            cands.append((score, key, name, None))
        for sc, x, y in self._salient_targets(latest_frame, h, w):
            key = f"ACTION6@{x},{y}"
            if key in self.tried_here[s_hash] or (s_hash, key) in self.death_exact:
                continue
            score = 0.4 + 0.25 * sc + 1.2 * click_bias
            cands.append((score, key, "ACTION6", (x, y)))
        if not cands:
            act = self._make_action("RESET")
            return self._finalize(act, "RESET", "RESET", s_hash,
                                  "all candidates suppressed")

        self.rng.shuffle(cands)
        cands.sort(key=lambda c: -c[0])
        if os.getenv("ATLAS_DEBUG_POLICY"):
            _log("cands: " + " | ".join(
                f"{c[2]}{('@%d,%d' % c[3]) if c[3] else ''}:{c[0]:.2f}"
                for c in cands[:4]))
        for score, key, name, xy in cands:
            act = self._make_action(name)
            if name == "ACTION6":
                if xy is None or not self._set_click(act, xy[0], xy[1]):
                    # engine can't carry coordinates: disable clicks, next cand
                    if self.click_usable:
                        self.click_usable = False
                        cal_log("guard", "click_unusable", 1.0, 0.5,
                                "ACTION6 disabled; simple actions only")
                    continue
            self.tried_here[s_hash].add(key)
            return self._finalize(act, name, key, s_hash,
                                  f"policy score={score:.2f} mode={self.mode} "
                                  f"budget_left={self.MAX_ACTIONS - self.action_counter}")
        act = self._make_action("RESET")
        return self._finalize(act, "RESET", "RESET", s_hash,
                              "all candidates suppressed")


AGENT_NAME = "atlas-p0"
