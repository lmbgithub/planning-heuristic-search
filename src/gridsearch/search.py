"""The four search strategies, written out and instrumented.

They are implemented here rather than imported from a library for two reasons.
The experiment is *about* what the frontier does — the node counts are the
result, and a library that does not expose them cannot produce it — and the
whole repository then needs no third-party dependency at all.

Every strategy returns the same `SearchResult`, so the comparison table is a
list of identical records rather than four different shapes of output.
"""

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import pairwise

from gridsearch.errors import GridSearchError
from gridsearch.grid import GridProblem
from gridsearch.heuristics import Heuristic

State = tuple[int, int]


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One run. Both numbers matter and they answer different questions.

    `cost` says whether the answer is right; `visited` says what it took to get
    there. Reporting only the first is exactly how an inadmissible heuristic
    goes unnoticed — the path looks fine, because it is a real path.
    """

    algorithm: str
    heuristic: str
    path: tuple[State, ...]
    cost: float
    visited: int
    max_frontier: int
    found: bool = True

    @property
    def steps(self) -> int:
        """Number of moves, which is not the same as cost once moves differ."""
        return max(len(self.path) - 1, 0)

    def summary(self) -> str:
        if not self.found:
            return (
                f"{self.algorithm:<16}{self.heuristic:<16}NO PATH   "
                f"visited {self.visited}"
            )
        return (
            f"{self.algorithm:<16}{self.heuristic:<16}"
            f"cost {self.cost:>7.1f}  steps {self.steps:>4}  "
            f"visited {self.visited:>5}  max frontier {self.max_frontier:>4}"
        )


def _no_path(
    algorithm: str, heuristic: str, visited: int, max_frontier: int
) -> SearchResult:
    """An unreachable goal is a result, not an exception.

    Returning an empty path with `found=False` rather than raising keeps the
    comparison table complete: "this strategy found nothing after N expansions"
    is a row worth having next to the strategies that did.
    """
    return SearchResult(
        algorithm=algorithm,
        heuristic=heuristic,
        path=(),
        cost=float("inf"),
        visited=visited,
        max_frontier=max_frontier,
        found=False,
    )


def _reconstruct(
    parents: Mapping[State, State | None], state: State
) -> tuple[State, ...]:
    path = [state]
    while (parent := parents[path[-1]]) is not None:
        path.append(parent)
    return tuple(reversed(path))


def breadth_first(problem: GridProblem) -> SearchResult:
    """Fewest *steps*. Optimal on cost only while every move costs the same."""
    start = problem.initial
    parents: dict[State, State | None] = {start: None}
    frontier: deque[State] = deque([start])
    visited = 0
    max_frontier = 1

    while frontier:
        state = frontier.popleft()
        visited += 1
        if problem.is_goal(state):
            path = _reconstruct(parents, state)
            return SearchResult(
                "breadth first",
                "-",
                path,
                path_cost(problem, path),
                visited,
                max_frontier,
            )
        for _, nxt, _ in problem.successors(state):
            if nxt not in parents:
                parents[nxt] = state
                frontier.append(nxt)
        max_frontier = max(max_frontier, len(frontier))

    return _no_path("breadth first", "-", visited, max_frontier)


def depth_first(problem: GridProblem) -> SearchResult:
    """Finds *a* path. There is no reason for it to be a good one."""
    start = problem.initial
    parents: dict[State, State | None] = {start: None}
    stack: list[State] = [start]
    seen: set[State] = set()
    visited = 0
    max_frontier = 1

    while stack:
        state = stack.pop()
        if state in seen:
            continue
        seen.add(state)
        visited += 1
        if problem.is_goal(state):
            path = _reconstruct(parents, state)
            return SearchResult(
                "depth first", "-", path, path_cost(problem, path), visited, max_frontier
            )
        for _, nxt, _ in problem.successors(state):
            if nxt not in seen:
                parents.setdefault(nxt, state)
                # A node already on the stack is re-pushed; the `seen` guard
                # above makes the duplicate harmless. Repairing the parent
                # pointer instead would quietly turn this into a best-first
                # search, which is not what is being demonstrated.
                stack.append(nxt)
        max_frontier = max(max_frontier, len(stack))

    return _no_path("depth first", "-", visited, max_frontier)


def best_first(
    problem: GridProblem,
    h: Heuristic | None = None,
    *,
    algorithm: str = "A*",
    heuristic_name: str = "-",
) -> SearchResult:
    """Uniform cost (h = 0) and A* are the same loop with a different priority.

    Writing them as one function is the honest framing: A* is not a different
    algorithm, it is uniform cost with an estimate of the remaining distance
    added to the priority. If the estimate is 0 the two are identical, and the
    test suite asserts exactly that.
    """

    h = h or (lambda state: 0.0)
    start = problem.initial
    best_cost: dict[State, float] = {start: 0.0}
    parents: dict[State, State | None] = {start: None}
    # The counter breaks priority ties in insertion order, making the result
    # deterministic. Without it, two nodes with equal f compare by their
    # coordinate tuples, and the returned path depends on grid geometry.
    counter = 0
    frontier: list[tuple[float, int, State]] = [(h(start), counter, start)]
    closed: set[State] = set()
    visited = 0
    max_frontier = 1

    while frontier:
        _, _, state = heapq.heappop(frontier)
        if state in closed:
            continue
        closed.add(state)
        visited += 1

        if problem.is_goal(state):
            path = _reconstruct(parents, state)
            return SearchResult(
                algorithm,
                heuristic_name,
                path,
                path_cost(problem, path),
                visited,
                max_frontier,
            )

        for _, nxt, step in problem.successors(state):
            candidate = best_cost[state] + step
            if candidate < best_cost.get(nxt, float("inf")):
                best_cost[nxt] = candidate
                parents[nxt] = state
                counter += 1
                heapq.heappush(frontier, (candidate + h(nxt), counter, nxt))
        max_frontier = max(max_frontier, len(frontier))

    return _no_path(algorithm, heuristic_name, visited, max_frontier)


def uniform_cost(problem: GridProblem) -> SearchResult:
    """Cheapest path, no heuristic. The ground truth every other row is judged against."""
    return best_first(problem, None, algorithm="uniform cost", heuristic_name="-")


def astar(
    problem: GridProblem, h: Heuristic, *, heuristic_name: str = "custom"
) -> SearchResult:
    return best_first(problem, h, algorithm="A*", heuristic_name=heuristic_name)


def path_cost(problem: GridProblem, path: tuple[State, ...]) -> float:
    """Sum the real step costs along a path, rejecting impossible moves.

    Scoring the path independently of the search that produced it is deliberate:
    a bug in a frontier that reports its own accumulated cost would be invisible.
    Here the number comes from replaying the path against the cost model.
    """
    total = 0.0
    for current, nxt in pairwise(path):
        for _, candidate, step in problem.successors(current):
            if candidate == nxt:
                total += step
                break
        else:
            raise GridSearchError(f"path contains an illegal move {current} -> {nxt}")
    return total


def true_costs_to_goal(problem: GridProblem) -> dict[State, float]:
    """Exact remaining cost from every cell, by a backwards sweep from the goal.

    Used to check admissibility. The sweep is genuinely backwards: the cost of
    the step from a predecessor into `state` is the cost of entering *state*,
    not of entering the predecessor. Getting that the wrong way round produces
    a plausible cost map that is wrong by one cell everywhere, and the
    admissibility check then verifies the wrong thing.
    """
    from gridsearch.grid import MOVES

    costs: dict[State, float] = {problem.grid.goal: 0.0}
    frontier: list[tuple[float, State]] = [(0.0, problem.grid.goal)]

    while frontier:
        cost, state = heapq.heappop(frontier)
        if cost > costs.get(state, float("inf")):
            continue
        x, y = state
        step = problem.cost_of_entering(state)
        for dx, dy in MOVES.values():
            predecessor = (x - dx, y - dy)
            if problem.grid.is_wall(predecessor):
                continue
            candidate = cost + step
            if candidate < costs.get(predecessor, float("inf")):
                costs[predecessor] = candidate
                heapq.heappush(frontier, (candidate, predecessor))

    return costs


ALGORITHMS: Mapping[str, Callable[[GridProblem], SearchResult]] = {
    "breadth-first": breadth_first,
    "depth-first": depth_first,
    "uniform-cost": uniform_cost,
}
