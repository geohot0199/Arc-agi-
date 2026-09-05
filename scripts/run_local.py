#!/usr/bin/env python3
"""Run the ATLAS agent locally — no Kaggle, no GPU, no internet needed.

Modes:
  --mock          Demo the agent on built-in mock games (pure stdlib, always
                  works). Probes -> posterior -> policy -> WIN.
  --list          List games found in the local engine (if arc-agi installed).
  --games a,b     Play specific games on the real offline engine
                  (requires: pip install arc-agi, Python 3.12, and game files
                  in ./environment_files or $ENVIRONMENTS_DIR).

Examples:
  python3 scripts/run_local.py --mock
  python3 scripts/run_local.py --games ls20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "tests"))

os.environ.setdefault("ATLAS_LLM", "0")  # never probe a server in local runs
os.environ.setdefault("ATLAS_LOG", "1")

import my_agent as ma  # noqa: E402


# ---------------------------------------------------------------------------
# Built-in mock games (deterministic, stdlib-only)
# ---------------------------------------------------------------------------
class MockMovementGame:
    """ACTION4 moves the player right to a goal; ACTION2 kills; else no-op."""

    title = "mock-movement (ACTION4=right, ACTION2=death)"
    win_levels = 3

    def __init__(self) -> None:
        self.player, self.levels, self.state = (2, 2), 0, "NOT_PLAYED"

    def grid(self):
        g = [[0] * 8 for _ in range(8)]
        g[self.player[1]][self.player[0]] = 5
        g[2][6] = 9
        return g

    def frame(self):
        return _Frame(self.grid(), self.state, self.levels, self.win_levels)

    def step(self, action: str, xy=None):
        if action == "RESET":
            self.state, self.player = "NOT_FINISHED", (2, 2)
        elif self.state == "NOT_FINISHED":
            if action == "ACTION2":
                self.state = "GAME_OVER"
            elif action == "ACTION4":
                x, y = self.player
                if x + 1 == 6:
                    self.levels += 1
                    self.player = (2, 2)
                    if self.levels >= self.win_levels:
                        self.state = "WIN"
                elif x + 1 < 8:
                    self.player = (x + 1, y)
        return self.frame()


class MockClickGame:
    """Only clicking the rare-colored cell (color 3) clears the level."""

    title = "mock-click (ACTION6 on the rare color-3 cell)"
    win_levels = 2
    spots = [(2, 2), (5, 5), (3, 6), (6, 1)]  # where the target appears next

    def __init__(self) -> None:
        self.levels, self.state, self.clicks = 0, "NOT_PLAYED", 0

    def grid(self):
        g = [[0] * 8 for _ in range(8)]
        tx, ty = self.spots[self.levels % len(self.spots)]
        g[ty][tx] = 3                      # rare target
        g[1][1] = g[1][6] = g[6][3] = 7    # decoys
        return g

    def frame(self):
        return _Frame(self.grid(), self.state, self.levels, self.win_levels)

    def step(self, action: str, xy=None):
        if action == "RESET":
            self.state = "NOT_FINISHED"
        elif self.state == "NOT_FINISHED":
            if action == "ACTION6" and xy is not None:
                self.clicks += 1
                tx, ty = self.spots[self.levels % len(self.spots)]
                if (int(xy[0]), int(xy[1])) == (tx, ty):
                    self.levels += 1
                    if self.levels >= self.win_levels:
                        self.state = "WIN"
        return self.frame()


class _Frame:
    def __init__(self, grid, state, levels, win_levels):
        self.frame = [grid]
        self.state = state
        self.levels_completed = levels
        self.win_levels = win_levels
        self.available_actions = ["RESET"] + [f"ACTION{i}" for i in range(1, 8)]


def _click_of(act, fallback_game):
    """Extract (x, y) from a GameAction-like object (mocks use tuples)."""
    if isinstance(act, tuple) and len(act) == 2:
        return act
    data = getattr(act, "action_data", None)
    if data is not None:
        try:
            return (int(getattr(data, "x", -1)), int(getattr(data, "y", -1)))
        except Exception:
            return None
    return None


def run_mock(max_steps: int = 200) -> bool:
    class _MockAct:
        """String-like action object that can carry (x, y) for ACTION6."""

        def __init__(self, name: str):
            self.name = name
            self.x = self.y = -1
            self.reasoning = ""

    ok_all = True
    for Game in (MockMovementGame, MockClickGame):
        game = Game()
        agent = ma.MyAgent(game_id=Game.__name__)
        # mock-mode plumbing: real action objects + attribute-based clicks
        agent._make_action = lambda name: _MockAct(name)
        agent._set_click = lambda act, x, y: (
            setattr(act, "x", int(x)), setattr(act, "y", int(y)), True)[2]
        frames = [game.frame()]
        history = []
        for _ in range(max_steps):
            f = frames[-1]
            if agent.is_done(frames, f):
                break
            act = agent.choose_action(frames, f)
            name = act if isinstance(act, str) else ma.action_name(act)
            xy = None
            if not isinstance(act, str) and getattr(act, "x", -1) >= 0:
                xy = (act.x, act.y)
            history.append((name, xy))
            agent.action_counter += 1  # mirror the framework's main loop
            frames.append(game.step(name, xy))
        won = game.state == "WIN"
        ok_all &= won
        print(f"\n=== {Game.title} ===")
        print(f"result: {'WIN' if won else 'INCOMPLETE'} "
              f"| levels {game.levels}/{Game.win_levels} "
              f"| actions {len(history)}")
        print(f"posterior: {agent.posterior.summary()}")
        print(f"mode={agent.mode} clicks_used={sum(1 for n, _ in history if n == 'ACTION6')}")
    return ok_all


# ---------------------------------------------------------------------------
# Real offline engine (arc-agi toolkit). Best-effort; API per docs.arcprize.org
# ---------------------------------------------------------------------------
def run_engine(game_ids):
    try:
        from arc_agi import Arcade, OperationMode  # type: ignore
        from arcengine import GameAction  # type: ignore
    except ImportError as e:
        sys.exit(f"arc-agi not installed ({e}). Install Python 3.12 + "
                 "`pip install arc-agi` and place game files in "
                 f"{ROOT / 'environment_files'} (or set ENVIRONMENTS_DIR).")

    env_dir = os.getenv("ENVIRONMENTS_DIR") or str(ROOT / "environment_files")
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=env_dir)
    known = {e.game_id for e in arc.get_environments()}
    for gid in game_ids:
        if gid not in known:
            print(f"!! {gid}: not found (have: {sorted(known)})")
            continue
        agent = ma.MyAgent(game_id=gid)
        env = arc.make(gid)
        obs = getattr(env, "observation_space", None)
        frames = []
        step_fn = getattr(env, "step", None)
        print(f"== {gid} ==")
        try:
            for i in range(agent.MAX_ACTIONS):
                f = obs if obs is not None else (frames[-1] if frames else None)
                if f is None:
                    break
                frames.append(f)
                if agent.is_done(frames, f):
                    break
                act = agent.choose_action(frames, f)
                out = step_fn(act)
                obs = getattr(out, "observation_space", out)
                if i % 25 == 0:
                    print(f"  #{i} {agent.posterior.summary()}")
        except Exception as e:  # engine surface differs across versions
            print(f"  engine loop stopped: {e!r} — see docs and adapt; "
                  "mock mode always works.")
            break
        print(f"  done: levels={agent.levels_completed} "
              f"actions={agent.action_counter} {agent.posterior.summary()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="run built-in mock games")
    ap.add_argument("--games", default="", help="comma-separated game ids (real engine)")
    ap.add_argument("--list", action="store_true", help="list local engine games")
    args = ap.parse_args()

    if args.list:
        try:
            from arc_agi import Arcade, OperationMode  # type: ignore
            arc = Arcade(operation_mode=OperationMode.OFFLINE)
            for e in arc.get_environments():
                print(f"{e.game_id}: {getattr(e, 'title', '')}")
        except ImportError:
            sys.exit("arc-agi not installed — use --mock for the dependency-free demo")
        return
    if args.games:
        run_engine([g.strip() for g in args.games.split(",") if g.strip()])
        return
    ok = run_mock()
    print("\nmock suite:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
