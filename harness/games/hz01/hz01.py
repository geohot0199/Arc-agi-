"""hz01 — hazard field.  MOVEMENT with dense lethality.

The goal sits across an open field scattered with deadly cells (colour 8).  A
safe corridor exists down column 1 and along the bottom row, but nothing marks
it, so the agent has to build a death graph and then plan around it.  This is
the fixture that measures whether the guard actually prevents repeat deaths.
"""

from gamekit import MoverGame

MAPS = [
    [
        "############",
        "#P..X..X...#",
        "#..X...X.X.#",
        "#.X...X....#",
        "#...X...X..#",
        "#.X..X..X..#",
        "#....X...X.#",
        "#.X...X....#",
        "#.........G#",
        "############",
    ],
    [
        "############",
        "#P.X...X..X#",
        "#...X.X....#",
        "#.X..X..X..#",
        "#..X...X.X.#",
        "#.X.X..X...#",
        "#...X...X..#",
        "#.X..X..X.G#",
        "############",
    ],
    [
        "############",
        "#P.X.X.X.X.#",
        "#..........#",
        "#X.X.X.X.X.#",
        "#..........#",
        "#.X.X.X.X.X#",
        "#..........#",
        "#X.X.X.X.XG#",
        "############",
    ],
]


class Hz01(MoverGame):
    def __init__(self, seed: int = 0) -> None:
        super().__init__("hz01", MAPS, win_score=len(MAPS),
                         available_actions=[1, 2, 3, 4], seed=seed)
