"""kd01 — key then door.  MOVEMENT with a two-stage goal.

The door (colour 4) is impassable until the key (colour 3) has been collected,
so an agent that only ever heads for the door makes no progress at all.  This
is the fixture that punishes a single fixed goal hypothesis.
"""

from gamekit import MoverGame

MAPS = [
    [
        "############",
        "#P....#...D#",
        "#.###.#.##.#",
        "#...#....#.#",
        "#.#.######.#",
        "#K.........#",
        "############",
    ],
    [
        "############",
        "#P.#......D#",
        "#..#.####.##",
        "#....#..#..#",
        "####.#..#.##",
        "#K...#.....#",
        "############",
    ],
    [
        "############",
        "#P...#.....#",
        "#.##.#.###.#",
        "#..#...#K#.#",
        "#.######.#.#",
        "#.........D#",
        "############",
    ],
]


class Kd01(MoverGame):
    def __init__(self, seed: int = 0) -> None:
        super().__init__("kd01", MAPS, win_score=len(MAPS),
                         available_actions=[1, 2, 3, 4], seed=seed)
