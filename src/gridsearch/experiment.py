"""The four experiments, and the table that makes an inadmissible run visible."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from gridsearch.grid import GridProblem, parse
from gridsearch.heuristics import chebyshev, check_admissible, manhattan, weighted, zero
from gridsearch.search import (
    SearchResult,
    astar,
    breadth_first,
    depth_first,
    uniform_cost,
)

# The map for experiments 1-3: open enough that several routes are plausible.
OPEN_MAP = """
##########
#   P    #
# ## #  ##
# #  #  ##
#T   ##  #
#        #
##########
"""

# Built so that scaling Manhattan returns a genuinely cheaper-looking but more
# expensive route. On most maps an inadmissible heuristic gets away with it;
# the point of having this one is that "usually fine" is not a guarantee.
TRAP_MAP = """
##########
#T       #
# ##### ##
# #   # ##
# # # # ##
#   #  P##
##########
"""

# Terrain map: digits are the cost of entering that cell. The direct corridor
# from T to P runs through a wall of 9s; the long way round is all 1s. Fewest
# steps and cheapest path are now different paths.
TERRAIN_MAP = """
##########
#T 999   #
#  999   #
#  999   #
#  999  P#
#        #
##########
"""


@dataclass(frozen=True, slots=True)
class Comparison:
    """A set of runs on one map, judged against the cheapest cost found."""

    results: tuple[SearchResult, ...]

    @property
    def optimal_cost(self) -> float:
        costs = [r.cost for r in self.results if r.found]
        return min(costs) if costs else float("inf")

    def excess(self, result: SearchResult) -> float:
        """How much more than optimal this run's path costs."""
        if not result.found:
            return float("inf")
        return result.cost - self.optimal_cost

    def is_optimal(self, result: SearchResult) -> bool:
        # Float comparison with a tolerance: costs are sums of floats, and an
        # exact `==` would mark a genuinely optimal path suboptimal after
        # enough additions.
        return result.found and abs(self.excess(result)) < 1e-9

    def table(self) -> str:
        header = (
            f"{'algorithm':<16}{'heuristic':<16}{'cost':>8}{'steps':>7}"
            f"{'visited':>9}{'frontier':>10}{'optimal':>9}{'excess':>9}"
        )
        lines = [header, "-" * len(header)]
        for r in self.results:
            if not r.found:
                lines.append(f"{r.algorithm:<16}{r.heuristic:<16}{'NO PATH':>8}")
                continue
            lines.append(
                f"{r.algorithm:<16}{r.heuristic:<16}{r.cost:>8.1f}{r.steps:>7}"
                f"{r.visited:>9}{r.max_frontier:>10}"
                f"{('yes' if self.is_optimal(r) else 'NO'):>9}"
                f"{self.excess(r):>9.1f}"
            )
        return "\n".join(lines)


def uninformed(map_text: str = OPEN_MAP) -> Comparison:
    """Experiment 1 — do the uninformed strategies agree, and at what effort?"""
    problem = GridProblem(parse(map_text))
    return Comparison(
        (breadth_first(problem), depth_first(problem), uniform_cost(problem))
    )


def terrain(map_text: str = TERRAIN_MAP) -> Comparison:
    """Experiment 2 — when cells cost different amounts, BFS optimises the wrong thing.

    BFS minimises the *number of steps*, which stops being the same question as
    soon as one cell costs more than another. It returns the short route
    straight through the expensive terrain, looking entirely healthy, at a cost
    no one asked for.
    """
    grid = parse(map_text)
    problem = GridProblem(grid)
    return Comparison(
        (
            breadth_first(problem),
            uniform_cost(problem),
            astar(problem, manhattan(grid.goal), heuristic_name="manhattan"),
        )
    )


def admissible_pair(map_text: str = TERRAIN_MAP) -> Comparison:
    """Experiment 3 — three admissible heuristics; only the effort differs."""
    grid = parse(map_text)
    problem = GridProblem(grid)
    return Comparison(
        (
            astar(problem, manhattan(grid.goal), heuristic_name="manhattan"),
            astar(problem, chebyshev(grid.goal), heuristic_name="chebyshev"),
            astar(problem, zero(grid.goal), heuristic_name="zero (= UCS)"),
        )
    )


def inadmissible(
    map_text: str = TRAP_MAP, weights: Sequence[float] = (2.0, 3.0, 5.0)
) -> Comparison:
    """Experiment 4 — where weighting Manhattan returns a worse path, silently."""
    grid = parse(map_text)
    problem = GridProblem(grid)
    runs = [
        uniform_cost(problem),
        astar(problem, manhattan(grid.goal), heuristic_name="manhattan"),
    ]
    runs += [
        astar(problem, weighted(grid.goal, w), heuristic_name=f"manhattan x{w:g}")
        for w in weights
    ]
    return Comparison(tuple(runs))


def bound_check(map_text: str = TRAP_MAP, weight: float = 3.0) -> str:
    """Confirm the w-bound: a weighted run costs at most w times optimal.

    This is the sentence that turns weighting from an accident into a decision.
    The bound holds here; the point is that it has to be *stated*, because the
    returned path gives no indication that a cheaper one existed.
    """
    grid = parse(map_text)
    problem = GridProblem(grid)
    optimal = uniform_cost(problem)
    weighted_run = astar(
        problem, weighted(grid.goal, weight), heuristic_name=f"x{weight:g}"
    )
    ratio = weighted_run.cost / optimal.cost if optimal.cost else float("inf")
    check = check_admissible(problem, weighted(grid.goal, weight))
    return (
        f"optimal cost {optimal.cost:.1f}   "
        f"weighted (x{weight:g}) cost {weighted_run.cost:.1f}\n"
        f"   ratio {ratio:.2f}   bound {weight:g}   "
        f"within bound: {ratio <= weight + 1e-9}\n"
        f"{check}"
    )
