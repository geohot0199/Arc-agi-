"""Optimal-play analysis for the benchmark fixtures -> RHAE baselines.

The competition's ``baseline_actions`` are human first-run measurements and ship
inside each game's metadata.  Our fixtures need the same field for the real
scorecard to produce a number, so we compute the optimal (shortest surviving)
solution per level and inflate it by BASELINE_MULT, which stands in for the
exploration a first-time player does.  The multiplier is a fixture convention,
not a claim about humans; it only has to be consistent so that agent variants
are comparable.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from gamekit import (  # noqa: E402
    DEADLY, DOOR, GOAL, KEY, LIGHT, WALL, parse_map,
)

BASELINE_MULT = float(os.getenv("ATLAS_BASELINE_MULT", "2.2"))

# (directory / game_id, class name, archetype key used for the baseline maths)
GAMES = [
    ("mv01", "Mv01", "mover"),
    ("cl01", "Cl01", "clicks"),
    ("lt01", "Lt01", "lights"),
    ("kd01", "Kd01", "mover"),
    ("sq01", "Sq01", "sequence"),
    ("hz01", "Hz01", "mover"),
]

_STEP = ((0, -1), (0, 1), (-1, 0), (1, 0))


def _load_game_module(game_id: str):
    path = os.path.join(HERE, "games", game_id, f"{game_id}.py")
    spec = importlib.util.spec_from_file_location(f"fixture_{game_id}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _passable(grid, pos) -> bool:
    x, y = pos
    if not (0 <= y < len(grid) and 0 <= x < len(grid[0])):
        return False
    return grid[y][x] not in (WALL, DEADLY)


def optimal_mover(rows) -> int | None:
    """Shortest surviving path, honouring the key/door dependency."""
    grid, feat = parse_map(rows)
    start = feat["player"]
    goal, key, door = feat["goal"], feat["key"], feat["door"]
    queue = deque([(start, False, 0)])
    seen = {(start, False)}
    while queue:
        pos, has_key, dist = queue.popleft()
        for dx, dy in _STEP:
            nxt = (pos[0] + dx, pos[1] + dy)
            if not _passable(grid, nxt):
                continue
            if door and nxt == door and not has_key:
                continue                      # locked: impassable
            held = has_key or (key is not None and nxt == key)
            if nxt == goal or (door and nxt == door):
                return dist + 1
            if (nxt, held) in seen:
                continue
            seen.add((nxt, held))
            queue.append((nxt, held, dist + 1))
    return None


def _distances(grid, src):
    out = {src: 0}
    queue = deque([src])
    while queue:
        pos = queue.popleft()
        for dx, dy in _STEP:
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in out or not _passable(grid, nxt):
                continue
            out[nxt] = out[pos] + 1
            queue.append(nxt)
    return out


def optimal_lights(rows) -> int | None:
    """Greedy nearest-light tour: an upper bound, and an honest one."""
    grid, feat = parse_map(rows)
    remaining = [tuple(p) for p in feat["lights"]]
    current = tuple(feat["player"])
    total = 0
    while remaining:
        dist = _distances(grid, current)
        reachable = [(dist[p], p) for p in remaining if p in dist]
        if not reachable:
            return None
        cost, target = min(reachable)
        total += cost + 1                     # walk there, then press
        remaining.remove(target)
        current = target
    return total


def optimal_clicks(rows) -> int:
    _grid, feat = parse_map(rows)
    return len(feat["targets"])


def optimal_for(game_id: str, kind: str) -> list:
    module = _load_game_module(game_id)
    if kind == "sequence":
        return [len(seq) for seq in module.SEQS]
    out = []
    for rows in module.MAPS:
        if kind == "mover":
            value = optimal_mover(rows)
        elif kind == "lights":
            value = optimal_lights(rows)
        elif kind == "clicks":
            value = optimal_clicks(rows)
        else:  # pragma: no cover - table is fixed
            raise ValueError(f"unknown kind {kind}")
        if value is None:
            raise ValueError(f"{game_id}: level is not solvable")
        out.append(int(value))
    return out


def optimal_table() -> dict:
    return {gid: optimal_for(gid, kind) for gid, _cls, kind in GAMES}


def baseline_table() -> dict:
    return {
        gid: [max(1, int(round(v * BASELINE_MULT))) for v in values]
        for gid, values in optimal_table().items()
    }


def write_metadata(force: bool = False) -> dict:
    """Write metadata.json for every fixture so the real engine can load them."""
    baselines = baseline_table()
    written = {}
    for game_id, class_name, _kind in GAMES:
        directory = os.path.join(HERE, "games", game_id)
        path = os.path.join(directory, "metadata.json")
        payload = {
            "game_id": f"{game_id}-atlas01",
            "title": f"ATLAS benchmark fixture {game_id}",
            "class_name": class_name,
            "baseline_actions": baselines[game_id],
            "tags": ["atlas-bench", game_id],
        }
        existing = None
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    existing = json.load(fh)
            except Exception:
                existing = None
        if force or existing != payload:
            with open(path, "w") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.write("\n")
        written[game_id] = payload
    return written


if __name__ == "__main__":
    table = write_metadata(force="--force" in sys.argv)
    print(f"baseline multiplier: {BASELINE_MULT}")
    for game_id, payload in table.items():
        print(f"  {game_id}: optimal={optimal_for(game_id, dict((g, k) for g, _c, k in GAMES)[game_id])} "
              f"baseline={payload['baseline_actions']}")
