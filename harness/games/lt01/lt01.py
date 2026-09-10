"""lt01 — switch off every light.  MOVEMENT + INTERACT (COLOR_LOGIC goal).

ACTION1..ACTION4 move the player; ACTION5 switches off the light (colour 6) in
the cell the player is facing.  The level ends when no light is left, so the
goal predicate is "clear colour 6", not "reach a cell".
"""

from arcengine import GameAction

from gamekit import FLOOR, LIGHT, MoverGame

MAPS = [
    [
        "############",
        "#P.L.......#",
        "#..........#",
        "#.......L..#",
        "#..........#",
        "############",
    ],
    [
        "############",
        "#P...L.....#",
        "#.########.#",
        "#.L......L.#",
        "#.########.#",
        "#.....L....#",
        "############",
    ],
    [
        "############",
        "#P.L...L...#",
        "#.########.#",
        "#....L.....#",
        "#.########.#",
        "#.L.....L..#",
        "############",
    ],
]


class Lt01(MoverGame):
    def __init__(self, seed: int = 0) -> None:
        super().__init__("lt01", MAPS, win_score=len(MAPS),
                         available_actions=[1, 2, 3, 4, 5], seed=seed)

    def _on_level_ready(self) -> None:
        super()._on_level_ready()
        self.facing = (1, 0)

    def act(self, action_id: GameAction, x: int, y: int) -> None:
        delta = self.DELTAS.get(action_id)
        if delta is not None:
            self.facing = delta
            super().act(action_id, x, y)
            return
        if action_id is GameAction.ACTION5:
            front = (self.player[0] + self.facing[0],
                     self.player[1] + self.facing[1])
            if self.at(front) == LIGHT:
                self.set_cell(front, FLOOR)
                if self.count_color(LIGHT) == 0:
                    self.next_level()
