"""sq01 — press the hidden sequence.  SELECTION archetype.

Only ACTION1..ACTION3 exist and none of them moves anything.  Each level wants
a specific order; a correct press advances a visible bar, a wrong press clears
it.  Nothing in the frame reveals the order, so the only route is to remember
what worked — this is the fixture that exercises the notes/win-sequence memory
and the level-analogy fallback.
"""

from arcengine import GameAction

from gamekit import GOAL, WALL, GridGame

SEQS = [[1, 2, 3], [2, 3, 1], [3, 1, 2, 1]]

MAPS = [
    [
        "########",
        "#......#",
        "#......#",
        "#......#",
        "########",
    ],
    [
        "########",
        "#......#",
        "#......#",
        "#......#",
        "########",
    ],
    [
        "########",
        "#......#",
        "#......#",
        "#......#",
        "########",
    ],
]


class Sq01(GridGame):
    def __init__(self, seed: int = 0) -> None:
        super().__init__("sq01", MAPS, win_score=len(MAPS),
                         available_actions=[1, 2, 3], seed=seed)

    def _on_level_ready(self) -> None:
        self.target = SEQS[self._current_level_index % len(SEQS)]
        self.progress = 0
        self.render_bar()

    def render_bar(self) -> None:
        for index in range(len(self.target)):
            self.set_cell((1 + index, 2),
                          GOAL if index < self.progress else WALL)

    def act(self, action_id: GameAction, x: int, y: int) -> None:
        pressed = action_id.value
        if self.progress < len(self.target) and pressed == self.target[self.progress]:
            self.progress += 1
            if self.progress >= len(self.target):
                self.next_level()
                return
        else:
            self.progress = 0
        self.render_bar()
