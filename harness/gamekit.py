"""Shared base for the ATLAS local benchmark games.

These are *test fixtures*, not competition content: the 110 scored games are
served by the Kaggle gateway and are never downloadable.  What the fixtures do
provide is a way to run the real ``arc_agi`` engine, the real
``agents.agent.Agent`` loop and the real RHAE scorecard against known,
controllable games — so the harness is verified end to end instead of by
inspection.

Each game is written the way a real ARC-AGI-3 game is: an ``ARCBaseGame``
subclass whose ``step()`` implements the mechanics, rendered through the engine
camera into a 64x64 frame of 16 colours.  None of this is visible to the agent.
"""

from __future__ import annotations

import numpy as np
from arcengine import (
    ARCBaseGame,
    BlockingMode,
    Camera,
    GameAction,
    Level,
    Sprite,
)

PALETTE = {
    ".": 0,   # floor
    "#": 1,   # wall
    "G": 2,   # goal
    "K": 3,   # key / consumable
    "D": 4,   # door
    "P": 5,   # player
    "L": 6,   # light (toggle target)
    "T": 7,   # target blob (clickable)
    "X": 8,   # deadly
    "B": 9,   # decoy blob
}
FLOOR, WALL, GOAL, KEY, DOOR, PLAYER, LIGHT, TARGET, DEADLY, DECOY = (
    PALETTE["."], PALETTE["#"], PALETTE["G"], PALETTE["K"], PALETTE["D"],
    PALETTE["P"], PALETTE["L"], PALETTE["T"], PALETTE["X"], PALETTE["B"],
)


def parse_map(rows) -> tuple:
    """ASCII map -> (grid of colour ints, features dict)."""
    grid = []
    features: dict = {"player": (1, 1), "goal": None, "key": None, "door": None,
                      "deadly": [], "lights": [], "targets": [], "decoys": []}
    for y, row in enumerate(rows):
        line = []
        for x, char in enumerate(row):
            colour = PALETTE.get(char, 0)
            if char == "P":
                features["player"] = (x, y)
                line.append(FLOOR)
                continue
            line.append(colour)
            if char == "G":
                features["goal"] = (x, y)
            elif char == "K":
                features["key"] = (x, y)
            elif char == "D":
                features["door"] = (x, y)
            elif char == "X":
                features["deadly"].append((x, y))
            elif char == "L":
                features["lights"].append((x, y))
            elif char == "T":
                features["targets"].append((x, y))
            elif char == "B":
                features["decoys"].append((x, y))
        grid.append(line)
    return grid, features


class GridGame(ARCBaseGame):
    """Board-sprite game base: a single sprite holds the whole grid.

    Subclasses implement ``act(action_id, x, y)``; this base handles rendering,
    level switching and reset.  ``self.grid`` is the authoritative state.
    """

    def __init__(self, game_id: str, maps: list, win_score: int,
                 available_actions: list, seed: int = 0) -> None:
        parsed = [parse_map(rows) for rows in maps]
        # _clean must exist before super().__init__, which calls set_level(0)
        # and therefore on_set_level().
        self._clean = [(grid, features) for grid, features in parsed]
        levels = []
        for grid, _features in parsed:
            board = Sprite(
                np.array(grid, dtype=np.int8),
                name="board", x=0, y=0, layer=0,
                blocking=BlockingMode.NOT_BLOCKED, collidable=False,
            )
            levels.append(Level(sprites=[board],
                                grid_size=(len(grid[0]), len(grid))))
        super().__init__(
            game_id, levels,
            Camera(width=len(parsed[0][0][0]), height=len(parsed[0][0]),
                   background=0, letter_box=0),
            win_score=win_score, available_actions=available_actions, seed=seed,
        )
        self.board = None
        self.grid: list = []
        self.features: dict = {}
        self.has_key = False
        self.on_set_level(self.current_level)

    # -- engine hooks ------------------------------------------------------
    def on_set_level(self, level: Level) -> None:
        sprites = level.get_sprites()
        self.board = sprites[0] if sprites else None
        grid, features = self._clean[self._current_level_index]
        self.grid = [row[:] for row in grid]
        self.features = {
            key: (list(value) if isinstance(value, list) else value)
            for key, value in features.items()
        }
        self.has_key = False
        self._on_level_ready()
        self.sync()

    def _on_level_ready(self) -> None:
        """Subclass hook, runs after self.grid/self.features are loaded."""

    def sync(self) -> None:
        if self.board is not None:
            self.board.pixels = np.array(self.grid, dtype=np.int8)

    def step(self) -> None:
        action = self.action
        x = y = -1
        if action.id.is_complex():
            try:
                x = int(action.data.get("x", -1))
                y = int(action.data.get("y", -1))
            except Exception:
                x = y = -1
        self.act(action.id, x, y)
        self.sync()
        self.complete_action()

    # -- helpers -----------------------------------------------------------
    def inside(self, pos) -> bool:
        x, y = pos
        return 0 <= y < len(self.grid) and 0 <= x < len(self.grid[0])

    def at(self, pos) -> int:
        if pos is None or not self.inside(pos):
            return FLOOR
        return self.grid[pos[1]][pos[0]]

    def set_cell(self, pos, colour: int) -> None:
        if self.inside(pos):
            self.grid[pos[1]][pos[0]] = colour

    def count_color(self, colour: int) -> int:
        return sum(row.count(colour) for row in self.grid)

    def cell_from_pixel(self, x: int, y: int):
        """Map a 64x64 frame coordinate back to a board cell.

        Mirrors ``Camera._calculate_scale_and_offset`` exactly: integer scale
        plus centred letterbox offsets.
        """
        if not self.grid:
            return None
        h, w = len(self.grid), len(self.grid[0])
        if not w or not h:
            return None
        scale = max(1, min(64 // w, 64 // h))
        off_x = (64 - w * scale) // 2
        off_y = (64 - h * scale) // 2
        cx, cy = (x - off_x) // scale, (y - off_y) // scale
        if 0 <= cx < w and 0 <= cy < h:
            return (cx, cy)
        return None

    def act(self, action_id: GameAction, x: int, y: int) -> None:
        raise NotImplementedError


class MoverGame(GridGame):
    """Grid game with a tracked player entity and 4-way movement.

    ACTION1 up, ACTION2 down, ACTION3 left, ACTION4 right.  Walls block, ``X``
    kills, ``G`` completes the level, ``K`` is consumed, ``D`` needs the key.
    """

    DELTAS = {
        GameAction.ACTION1: (0, -1),
        GameAction.ACTION2: (0, 1),
        GameAction.ACTION3: (-1, 0),
        GameAction.ACTION4: (1, 0),
    }

    def _on_level_ready(self) -> None:
        self.player = tuple(self.features.get("player") or (1, 1))
        self.under = FLOOR
        self.paint()

    def paint(self) -> None:
        self.set_cell(self.player, PLAYER)

    def unpaint(self) -> None:
        self.set_cell(self.player, self.under)

    def _relocate(self, pos, under: int) -> None:
        self.unpaint()
        self.under = under
        self.player = pos
        self.paint()

    def act(self, action_id: GameAction, x: int, y: int) -> None:
        delta = self.DELTAS.get(action_id)
        if delta is not None:
            self.move(delta[0], delta[1])

    def move(self, dx: int, dy: int) -> None:
        target_pos = (self.player[0] + dx, self.player[1] + dy)
        if not self.inside(target_pos):
            return
        colour = self.at(target_pos)
        if colour == WALL:
            return
        if colour == DEADLY:
            self._relocate(target_pos, colour)
            self.lose()
            return
        if colour == DOOR and not self.has_key:
            return                       # locked: a true no-op
        was_goal = colour == GOAL
        was_door = colour == DOOR
        if colour == KEY:
            self.has_key = True
            colour = FLOOR               # consumed
        self._relocate(target_pos, colour)
        if was_goal or was_door:
            self.next_level()
