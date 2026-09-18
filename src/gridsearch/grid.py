"""ASCII grid maps and the search problem defined over them."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from gridsearch.errors import MapError

WALL = "#"
START = "T"
GOAL = "P"

#: Move name -> (dx, dy). Four-connected: no diagonals, which is what makes
#: Manhattan an admissible heuristic here.
MOVES: Mapping[str, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

#: Terrain: the cost of *entering* a cell. Everything not listed costs 1.
#: Cost belongs to the cell, not to the direction of travel. Direction-dependent
#: cost is the obvious alternative and it cannot produce the experiment this
#: repository needs: on a 4-connected grid the net displacement fixes how many
#: left, right, up and down moves any path must contain, so every minimum-step
#: route costs the same and breadth-first search is *always* cost-optimal.
#: Terrain breaks that — a longer route over cheap cells can be cheaper.
TERRAIN_COSTS: Mapping[str, float] = {str(d): float(d) for d in range(1, 10)}
DEFAULT_COST = 1.0


@dataclass(frozen=True, slots=True)
class Grid:
    """A rectangular ASCII map with exactly one start and one goal."""

    rows: tuple[str, ...]
    start: tuple[int, int]
    goal: tuple[int, int]

    @property
    def width(self) -> int:
        return len(self.rows[0])

    @property
    def height(self) -> int:
        return len(self.rows)

    def is_wall(self, position: tuple[int, int]) -> bool:
        """Out of bounds counts as a wall, so no caller needs a bounds check."""
        x, y = position
        if not (0 <= y < self.height and 0 <= x < self.width):
            return True
        return self.rows[y][x] == WALL

    def render(self, path: list[tuple[int, int]] | None = None) -> str:
        """Draw the map, marking a path with `.` but never over T, P or a wall."""
        cells = [list(row) for row in self.rows]
        for x, y in path or []:
            if cells[y][x] not in (START, GOAL, WALL):
                cells[y][x] = "."
        return "\n".join("".join(row) for row in cells)


def parse(text: str) -> Grid:
    """Parse an ASCII map, rejecting everything a search cannot be run on.

    Blank lines are dropped and short rows are right-padded with walls, so a
    map written in a docstring does not depend on trailing whitespace
    surviving an editor. Anything else — no start, two goals, an empty map —
    raises here rather than producing a search that silently walks off the
    board or never terminates.
    """

    rows = [line for line in text.splitlines() if line.strip()]
    if not rows:
        raise MapError("map is empty")

    width = max(len(row) for row in rows)
    rows = [row.ljust(width, WALL) for row in rows]

    starts = list(_find(rows, START))
    goals = list(_find(rows, GOAL))
    for name, found in (("start (T)", starts), ("goal (P)", goals)):
        if len(found) != 1:
            raise MapError(f"map must contain exactly one {name}; found {len(found)}")

    return Grid(rows=tuple(rows), start=starts[0], goal=goals[0])


def _find(rows: list[str], marker: str) -> Iterator[tuple[int, int]]:
    for y, row in enumerate(rows):
        for x, cell in enumerate(row):
            if cell.upper() == marker:
                yield (x, y)


@dataclass(frozen=True, slots=True)
class GridProblem:
    """The search problem: successors, goal test, and step cost.

    The cost of a step is the cost of the cell being entered. A digit in the
    map is that cell's cost; every other walkable cell costs `default_cost`.
    This is what separates "fewest steps" from "cheapest path", and therefore
    what makes breadth-first search visibly wrong on the terrain map.
    """

    grid: Grid
    terrain: Mapping[str, float] = None  # type: ignore[assignment]
    default_cost: float = DEFAULT_COST

    def __post_init__(self) -> None:
        terrain = dict(TERRAIN_COSTS if self.terrain is None else self.terrain)
        if self.default_cost <= 0 or any(v <= 0 for v in terrain.values()):
            # A zero or negative step cost breaks uniform cost and A* outright:
            # the frontier can be reopened forever and "cheapest" stops being
            # well defined. Rejecting it here beats a hang later.
            raise MapError("every cell cost must be positive")
        object.__setattr__(self, "terrain", terrain)

    def cost_of_entering(self, position: tuple[int, int]) -> float:
        x, y = position
        return self.terrain.get(self.grid.rows[y][x], self.default_cost)

    @property
    def min_cell_cost(self) -> float:
        """The cheapest step on this map.

        Manhattan distance is admissible only when no step costs less than 1 —
        it counts steps, so a map with a cell costing 0.5 makes the textbook
        heuristic an overestimate. `heuristics.check_admissible` verifies this
        rather than assuming it.
        """
        seen = {
            self.cost_of_entering((x, y))
            for y in range(self.grid.height)
            for x in range(self.grid.width)
            if not self.grid.is_wall((x, y))
        }
        return min(seen) if seen else self.default_cost

    @property
    def initial(self) -> tuple[int, int]:
        return self.grid.start

    def is_goal(self, state: tuple[int, int]) -> bool:
        return state == self.grid.goal

    def successors(self, state: tuple[int, int]):
        """Legal moves from `state`, in a fixed order.

        The order is fixed (`MOVES` is an ordinary dict, insertion-ordered) so
        that two runs of the same algorithm on the same map return the same
        path. Iterating a set here is the usual reason a search result changes
        between runs for no visible reason.
        """
        x, y = state
        for action, (dx, dy) in MOVES.items():
            nxt = (x + dx, y + dy)
            if not self.grid.is_wall(nxt):
                yield action, nxt, self.cost_of_entering(nxt)
