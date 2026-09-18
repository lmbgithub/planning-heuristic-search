"""Heuristics, and the machinery to check whether they are actually admissible.

A heuristic is admissible when it never overestimates the true remaining cost.
That is the whole guarantee behind A*'s optimality, and it is a property of the
heuristic *and the cost model together* — Manhattan is admissible on unit costs
and stops being so the moment a step can cost less than 1.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from gridsearch.errors import GridSearchError
from gridsearch.grid import GridProblem

Heuristic = Callable[[tuple[int, int]], float]


def manhattan(goal: tuple[int, int]) -> Heuristic:
    """|dx| + |dy|. Admissible on a 4-connected grid with unit costs."""
    gx, gy = goal

    def h(state: tuple[int, int]) -> float:
        x, y = state
        return float(abs(x - gx) + abs(y - gy))

    return h


def chebyshev(goal: tuple[int, int]) -> Heuristic:
    """max(|dx|, |dy|). Admissible and strictly dominated by Manhattan here.

    Dominated means never larger, both being lower bounds — so it is the weaker
    of the two and A* expands at least as many nodes with it. Same answer, more
    work: that is what a weaker heuristic costs, and it is the point of the
    experiment that compares them.
    """
    gx, gy = goal

    def h(state: tuple[int, int]) -> float:
        x, y = state
        return float(max(abs(x - gx), abs(y - gy)))

    return h


def zero(goal: tuple[int, int]) -> Heuristic:
    """h = 0. Admissible, uninformative; A* with it is uniform cost."""

    def h(state: tuple[int, int]) -> float:
        return 0.0

    return h


def weighted(goal: tuple[int, int], weight: float) -> Heuristic:
    """Manhattan x weight. Inadmissible as soon as weight > 1 on unit costs.

    Weighting is a legitimate technique — with weight *w* the returned path is
    at most *w* times the optimal cost — but the bound only exists as a stated
    decision. Scaling a heuristic to make search faster and continuing to call
    the result "shortest" is the thing this package exists to show is false.
    """
    if weight <= 0:
        raise GridSearchError(f"weight must be positive; got {weight}")
    base = manhattan(goal)

    def h(state: tuple[int, int]) -> float:
        return weight * base(state)

    return h


HEURISTICS: Mapping[str, Callable[..., Heuristic]] = {
    "manhattan": manhattan,
    "chebyshev": chebyshev,
    "zero": zero,
    "weighted": weighted,
}


@dataclass(frozen=True, slots=True)
class AdmissibilityCheck:
    admissible: bool
    worst_state: tuple[int, int] | None
    worst_overestimate: float
    #: The map's cheapest step, when it is below 1.0. A step-counting heuristic
    #: stops being a lower bound there, so the number is worth naming in the
    #: message rather than leaving the reader to wonder why.
    cheapest_step: float | None = None

    def __str__(self) -> str:
        if self.admissible:
            return "admissible: h never exceeds the true remaining cost"
        reason = ""
        if self.cheapest_step is not None:
            reason = (
                f" (the cheapest step on this map costs {self.cheapest_step:g}, "
                f"so a step-counting heuristic is not a lower bound)"
            )
        return (
            f"INADMISSIBLE: overestimates by {self.worst_overestimate:.2f} "
            f"at {self.worst_state}{reason}"
        )


def check_admissible(problem: GridProblem, h: Heuristic) -> AdmissibilityCheck:
    """Verify admissibility exhaustively against true costs.

    The true remaining cost from every reachable cell is computed once by a
    backwards uniform-cost sweep from the goal, then compared against `h`. This
    is only tractable because the maps are small — but on maps this small there
    is no reason to *assume* a textbook property when it can be checked, and the
    check is what turns "Manhattan is admissible" from a claim into a test.
    """

    from gridsearch.search import true_costs_to_goal

    # Manhattan counts steps, so it is a lower bound only while no step costs
    # less than one. Checking the map's cheapest cell first turns a silent
    # wrong answer into a named precondition.
    cheapest = problem.min_cell_cost if problem.min_cell_cost < 1.0 else None

    true = true_costs_to_goal(problem)
    worst_state: tuple[int, int] | None = None
    worst_gap = 0.0
    for state, cost in true.items():
        gap = h(state) - cost
        if gap > worst_gap + 1e-12:
            worst_gap, worst_state = gap, state

    return AdmissibilityCheck(
        admissible=worst_state is None,
        worst_state=worst_state,
        worst_overestimate=worst_gap,
        cheapest_step=cheapest,
    )
