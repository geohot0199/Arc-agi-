"""cl01 — click the rare target.  CLICK_PUZZLE archetype.

ACTION1..ACTION5 are inert; only ACTION6 (a click at x, y) does anything.
Clicking a target blob (colour 7) consumes it, and the level ends when none
remain.  Decoy blobs (colour 9) are no-ops.  The target is always the rarest
colour on the board, which is the signal a competent agent should find.
"""

from arcengine import GameAction

from gamekit import FLOOR, TARGET, GridGame

MAPS = [
    [
        "################",
        "#..............#",
        "#..B....T...B..#",
        "#..............#",
        "#....B.....B...#",
        "#..............#",
        "################",
    ],
    [
        "################",
        "#..B........B..#",
        "#.....T........#",
        "#.B.........B..#",
        "#........T.....#",
        "#..B........B..#",
        "################",
    ],
    [
        "################",
        "#.B....T....B..#",
        "#......B.......#",
        "#..T........T..#",
        "#.B....B....B..#",
        "#......B.......#",
        "################",
    ],
]


class Cl01(GridGame):
    def __init__(self, seed: int = 0) -> None:
        super().__init__("cl01", MAPS, win_score=len(MAPS),
                         available_actions=[1, 2, 3, 4, 5, 6], seed=seed)

    def act(self, action_id: GameAction, x: int, y: int) -> None:
        if action_id is not GameAction.ACTION6:
            return
        cell = self.cell_from_pixel(x, y)
        if cell is None or self.at(cell) != TARGET:
            return
        self.set_cell(cell, FLOOR)
        if self.count_color(TARGET) == 0:
            self.next_level()
