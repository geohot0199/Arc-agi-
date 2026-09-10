"""mv01 — serpentine corridors.  MOVEMENT archetype.

ACTION1..ACTION4 move up/down/left/right; reach the goal cell (colour 2).
Level 2 onward puts a deadly cell (colour 8) on the direct route, so the only
way through is to learn the hazard and route around it.
"""

from arcengine import GameAction

from gamekit import MoverGame

MAPS = [
    [
        "############",
        "#P.........#",
        "#.########.#",
        "#..........#",
        "#.########.#",
        "#.........G#",
        "############",
    ],
    [
        "############",
        "#P...X....G#",
        "#.########.#",
        "#..........#",
        "#.########.#",
        "#..........#",
        "############",
    ],
    [
        "############",
        "#P...X.....#",
        "#.########.#",
        "#....X.....#",
        "#.########.#",
        "#.........G#",
        "############",
    ],
]


class Mv01(MoverGame):
    def __init__(self, seed: int = 0) -> None:
        super().__init__("mv01", MAPS, win_score=len(MAPS),
                         available_actions=[1, 2, 3, 4], seed=seed)
