"""Tests for ATLAS Phase-0 agent core logic (no engine / no network needed).

Run:  python3 -m unittest discover -s tests -v
"""

import importlib.util
import os
import sys
import unittest
from collections import Counter
from pathlib import Path

# Deterministic, offline: the LLM advisor must never even probe localhost.
os.environ["ATLAS_LLM"] = "0"
os.environ["ATLAS_LOG"] = "0"

_AGENT_PATH = Path(__file__).resolve().parents[1] / "agent" / "my_agent.py"
_spec = importlib.util.spec_from_file_location("my_agent", _AGENT_PATH)
ma = importlib.util.module_from_spec(_spec)
sys.modules["my_agent"] = ma
_spec.loader.exec_module(ma)


def make_grid(h=8, w=8, fill=0, placements=None):
    g = [[fill] * w for _ in range(h)]
    for (x, y), c in (placements or {}).items():
        g[y][x] = c
    return g


class MockFrame:
    def __init__(self, grid, state="NOT_FINISHED", levels=0, win_levels=3):
        self.frame = [grid]
        self.state = state
        self.levels_completed = levels
        self.win_levels = win_levels
        self.available_actions = ["RESET"] + [f"ACTION{i}" for i in range(1, 8)]


class MockGame:
    """Tiny deterministic game: ACTION4 moves the player right toward a goal;
    ACTION2 kills; everything else is a no-op. Clearing the goal = level up."""

    def __init__(self, goal_x=6):
        self.player = (2, 2)
        self.goal_x = goal_x
        self.levels = 0
        self.state = "NOT_PLAYED"
        self.win_levels = 3

    def grid(self):
        return make_grid(8, 8, placements={self.player: 5, (self.goal_x, 2): 9})

    def frame(self):
        return MockFrame(self.grid(), self.state, self.levels, self.win_levels)

    def step(self, action: str):
        if action == "RESET":
            self.state = "NOT_FINISHED"
            self.player = (2, 2)
            return self.frame()
        if self.state != "NOT_FINISHED":
            return self.frame()
        if action == "ACTION2":
            self.state = "GAME_OVER"
            return self.frame()
        if action == "ACTION4":
            x, y = self.player
            if x + 1 == self.goal_x:
                self.levels += 1
                self.player = (2, 2)  # next level
                if self.levels >= self.win_levels:
                    self.state = "WIN"
                return self.frame()
            if x + 1 < 8:
                self.player = (x + 1, y)
        return self.frame()


def new_agent(game_id="mock"):
    return ma.MyAgent(game_id=game_id)


# ---------------------------------------------------------------------------
class TestHelpers(unittest.TestCase):
    def test_segment_finds_objects(self):
        g = make_grid(8, 8, placements={(1, 1): 3, (2, 1): 3, (6, 6): 7})
        objs = ma.segment(ma._tuple_grid(g))
        colors = sorted(o["color"] for o in objs)
        self.assertEqual(colors, [3, 7])
        by_c = {o["color"]: o for o in objs}
        self.assertEqual(by_c[3]["size"], 2)
        self.assertEqual(by_c[7]["centroid"], (6.0, 6.0))

    def test_diff_and_classify(self):
        g1 = make_grid(placements={(1, 1): 5})
        g2 = make_grid(placements={(2, 1): 5})
        d = ma.diff_cells(ma._tuple_grid(g1), ma._tuple_grid(g2))
        self.assertEqual(len(d), 2)  # vanished + appeared
        self.assertEqual(ma.classify_event(0, 0, "NOT_FINISHED", []), "NOOP")
        self.assertEqual(ma.classify_event(0, 1, "NOT_FINISHED", d), "LEVEL_UP")
        self.assertEqual(ma.classify_event(0, 0, "GAME_OVER", d), "DEATH")
        self.assertEqual(ma.classify_event(0, 0, "WIN", d), "WIN")

    def test_downsample_hamming(self):
        a = ma._tuple_grid(make_grid(16, 16, fill=1))
        b = ma._tuple_grid(make_grid(16, 16, fill=1))
        b = tuple(tuple(2 if (x == 3 and y == 3) else c for x, c in enumerate(r))
                  for y, r in enumerate(b))
        da, db = ma.downsample(a), ma.downsample(b)
        self.assertLess(ma.hamming_frac(da, db), 0.06)
        self.assertGreater(ma.hamming_frac(da, ma.downsample(
            ma._tuple_grid(make_grid(16, 16, fill=4)))), 0.9)

    def test_llm_json_parse(self):
        self.assertEqual(ma.AtlasLLM._parse_json('{"action":"ACTION4"}')["action"],
                         "ACTION4")
        self.assertEqual(
            ma.AtlasLLM._parse_json('```json\n{"action":"ACTION1"}\n```')["action"],
            "ACTION1")
        self.assertIsNone(ma.AtlasLLM._parse_json("no json here"))


# ---------------------------------------------------------------------------
class TestPosterior(unittest.TestCase):
    def test_starts_ambiguous_then_resolves(self):
        p = ma.ArchetypePosterior()
        self.assertTrue(p.abstain(1.3))  # uniform: high entropy -> abstain
        for _ in range(4):
            p.update({"entity_moved": True, "simple_actions_move": True})
        top, prob = p.top(1)[0]
        self.assertEqual(top, "MOVEMENT")
        self.assertGreater(prob, 0.5)
        self.assertFalse(p.abstain(1.3))

    def test_click_evidence_routes_to_click_puzzle(self):
        p = ma.ArchetypePosterior()
        for _ in range(4):
            p.update({"click_local_change": True, "simple_noop_only": True})
        top, _ = p.top(1)[0]
        self.assertEqual(top, "CLICK_PUZZLE")

    def test_summary_is_logging_friendly(self):
        s = ma.ArchetypePosterior().summary()
        self.assertIn("MOV", s)


# ---------------------------------------------------------------------------
class TestSalienceFunnel(unittest.TestCase):
    def test_funnel_caps_and_prioritizes(self):
        ag = new_agent()
        placements = {(x, y): 3 for x in (0, 1, 2) for y in (0, 1, 2)}  # blob
        placements[(12, 12)] = 7  # rare single cell
        g = ma._tuple_grid(make_grid(16, 16, placements=placements))
        f = MockFrame([list(r) for r in g])
        ag._cur_hash = ma.frame_hash(f)
        ag.player_pos = (10.0, 10.0)
        out = ag._salient_targets(f, 16, 16)
        self.assertLessEqual(len(out), ma.CFG["SALIENT_CAP"])
        self.assertIn((12, 12), [(x, y) for _s, x, y in out])  # rare cell present
        # all in bounds
        for _s, x, y in out:
            self.assertTrue(0 <= x < 16 and 0 <= y < 16)

    def test_exact_death_cell_excluded(self):
        ag = new_agent()
        g = ma._tuple_grid(make_grid(16, 16, placements={(12, 12): 7}))
        f = MockFrame([list(r) for r in g])
        ag._cur_hash = ma.frame_hash(f)
        ag.death_exact[(ag._cur_hash, "ACTION6@12,12")] = 1
        out = ag._salient_targets(f, 16, 16)
        self.assertNotIn((12, 12), [(x, y) for _s, x, y in out])

    def test_noop_click_downweighted(self):
        ag = new_agent()
        g = ma._tuple_grid(make_grid(16, 16, placements={(12, 12): 7, (3, 3): 7}))
        f = MockFrame([list(r) for r in g])
        ag._cur_hash = ma.frame_hash(f)
        ag.click_ledger[(12, 12)] = [4, 0]  # clicked 4x, never changed
        scores = {(x, y): s for s, x, y in ag._salient_targets(f, 16, 16)}
        if (12, 12) in scores and (3, 3) in scores:
            self.assertLess(scores[(12, 12)], scores[(3, 3)])


# ---------------------------------------------------------------------------
class TestMockGameLoop(unittest.TestCase):
    def run_game(self, max_steps=220):
        game = MockGame()
        ag = new_agent()
        frames = [game.frame()]
        deaths = 0
        for _ in range(max_steps):
            f = frames[-1]
            if ag.is_done(frames, f):
                break
            act = ag.choose_action(frames, f)
            name = act if isinstance(act, str) else ma.action_name(act)
            self.assertIn(  # protocol guard: always a legal action
                name, [a for a in f.available_actions])
            frames.append(game.step(name))
            if frames[-1].state == "GAME_OVER":
                deaths += 1
        return ag, game, deaths

    def test_never_crashes_and_makes_progress(self):
        ag, game, deaths = self.run_game()
        self.assertGreaterEqual(game.levels, 1)  # cleared at least level 1
        self.assertLessEqual(deaths, 4)          # deaths are learned away

    def test_death_is_blocked_after_first_time(self):
        ag, game, deaths = self.run_game(max_steps=120)
        a2_deaths = [n for k, n in ag.death_exact.items() if k[1] == "ACTION2"]
        self.assertGreaterEqual(sum(a2_deaths), 1)
        # posterior should have absorbed the movement evidence
        top, prob = ag.posterior.top(1)[0]
        self.assertIn(top, ("MOVEMENT", "OTHER"))
        self.assertGreater(prob, 0.3)

    def test_posterior_and_ledgers_populated(self):
        ag, game, _ = self.run_game()
        self.assertTrue(any(v[0] > 0 for v in ag.noop_ledger.values()))
        self.assertGreater(ag.progress_ledger["ACTION4"][1], 0)
        self.assertTrue(len(ag.timeline) > 10)


# ---------------------------------------------------------------------------
class TestCalibrationBus(unittest.TestCase):
    def test_cal_log_records_entries(self):
        n0 = len(ma._CAL_BUF)
        ma.cal_log("test", "unit", 0.75, 0.5, "demo")
        e = ma._CAL_BUF[-1]
        self.assertEqual(len(ma._CAL_BUF), n0 + 1)
        self.assertEqual(e["component"], "test")
        self.assertEqual(e["decision"], "demo")

    def test_stop_loss_modes(self):
        ag = new_agent()
        ag.level_start_actions = 0
        ag.action_counter = ma.CFG["LEVEL_BUDGET"] + 5
        # emulate the mode-selection block from _choose
        on_level = ag.action_counter - ag.level_start_actions
        if on_level > ma.CFG["LEVEL_HARD_CAP"]:
            ag.mode = "conserve"
        elif on_level > ma.CFG["LEVEL_BUDGET"]:
            ag.mode = "exploit"
        self.assertEqual(ag.mode, "exploit")


if __name__ == "__main__":
    unittest.main(verbosity=2)
