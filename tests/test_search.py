"""Search strategies, judged against hand-verifiable maps."""

import pytest

from gridsearch.grid import Grid, GridProblem, parse
from gridsearch.heuristics import chebyshev, manhattan, weighted, zero
from gridsearch.search import (
    astar,
    breadth_first,
    depth_first,
    path_cost,
    true_costs_to_goal,
    uniform_cost,
)

CORRIDOR = """
#######
#T   P#
#######
"""

# The direct route is four steps through cost-9 cells; going round is six
# steps of cost 1. Fewest steps and cheapest path are different answers.
TERRAIN = """
########
#T99  P#
#      #
########
"""

UNREACHABLE = """
#######
#T ## P#
#######
"""


def problem(text):
    return GridProblem(parse(text))


@pytest.mark.parametrize("algorithm", [breadth_first, depth_first, uniform_cost])
def test_every_strategy_finds_the_corridor(algorithm):
    result = algorithm(problem(CORRIDOR))
    assert result.found
    assert result.path[0] == (1, 1)
    assert result.path[-1] == (5, 1)
    assert result.cost == 4.0


def test_steps_and_cost_are_different_questions():
    result = breadth_first(problem(TERRAIN))
    assert result.steps == 5
    assert result.cost > result.steps


def test_breadth_first_is_cost_optimal_only_on_uniform_terrain():
    # This is the experiment: BFS returns the short expensive route.
    bfs = breadth_first(problem(TERRAIN))
    ucs = uniform_cost(problem(TERRAIN))
    assert bfs.steps <= ucs.steps
    assert bfs.cost > ucs.cost


def test_astar_with_a_zero_heuristic_is_uniform_cost():
    grid = parse(TERRAIN)
    p = GridProblem(grid)
    a = astar(p, zero(grid.goal))
    u = uniform_cost(p)
    assert a.cost == u.cost
    assert a.visited == u.visited


@pytest.mark.parametrize("heuristic", [manhattan, chebyshev])
def test_admissible_heuristics_return_the_optimal_cost(heuristic):
    grid = parse(TERRAIN)
    p = GridProblem(grid)
    assert astar(p, heuristic(grid.goal)).cost == uniform_cost(p).cost


def test_a_stronger_admissible_heuristic_expands_no_more_nodes():
    # Dominance guarantees "no more", not "fewer" — on small maps the two
    # frequently tie, and claiming an improvement that is not there is exactly
    # the sort of thing this suite exists to catch.
    grid = parse(TERRAIN)
    p = GridProblem(grid)
    assert (
        astar(p, manhattan(grid.goal)).visited <= astar(p, chebyshev(grid.goal)).visited
    )


def test_an_unreachable_goal_is_a_result_not_an_exception():
    result = uniform_cost(problem(UNREACHABLE))
    assert not result.found
    assert result.path == ()
    assert result.cost == float("inf")
    assert result.visited > 0


@pytest.mark.parametrize("algorithm", [breadth_first, depth_first, uniform_cost])
def test_every_strategy_reports_an_unreachable_goal_the_same_way(algorithm):
    assert not algorithm(problem(UNREACHABLE)).found


@pytest.mark.parametrize("algorithm", [breadth_first, depth_first, uniform_cost])
def test_start_equal_to_goal_costs_nothing(algorithm):
    grid = parse(CORRIDOR)
    degenerate = GridProblem(Grid(rows=grid.rows, start=grid.start, goal=grid.start))
    result = algorithm(degenerate)
    assert result.found
    assert result.cost == 0.0
    assert result.steps == 0
    assert result.visited == 1


def test_path_cost_is_recomputed_from_the_map_not_trusted():
    # Scoring the path independently is what would catch a frontier that
    # accumulates its own cost incorrectly.
    p = problem(TERRAIN)
    result = uniform_cost(p)
    assert path_cost(p, result.path) == result.cost


def test_path_cost_rejects_an_impossible_move():
    p = problem(CORRIDOR)
    with pytest.raises(ValueError, match="illegal move"):
        path_cost(p, ((1, 1), (5, 1)))


def test_path_cost_of_a_single_cell_is_zero():
    assert path_cost(problem(CORRIDOR), ((1, 1),)) == 0.0


def test_results_are_deterministic_across_runs():
    grid = parse(TERRAIN)
    p = GridProblem(grid)
    first = astar(p, manhattan(grid.goal))
    second = astar(p, manhattan(grid.goal))
    assert first.path == second.path
    assert first.visited == second.visited


def test_true_costs_measure_the_cell_being_entered():
    # Getting this backwards yields a plausible cost map that is wrong by one
    # cell everywhere, and the admissibility check then verifies the wrong thing.
    costs = true_costs_to_goal(problem(TERRAIN))
    grid = parse(TERRAIN)
    assert costs[grid.goal] == 0.0
    assert costs[grid.start] == uniform_cost(problem(TERRAIN)).cost


def test_true_costs_omit_unreachable_cells():
    costs = true_costs_to_goal(problem(UNREACHABLE))
    assert parse(UNREACHABLE).start not in costs


def test_an_inadmissible_heuristic_can_return_a_worse_path():
    from gridsearch.experiment import TRAP_MAP

    grid = parse(TRAP_MAP)
    p = GridProblem(grid)
    optimal = uniform_cost(p).cost
    greedy = astar(p, weighted(grid.goal, 3.0))
    assert greedy.found  # it returns a complete, valid path
    assert greedy.cost > optimal  # and it is not the cheapest one
    assert greedy.visited < astar(p, manhattan(grid.goal)).visited  # it was faster


def test_the_weight_bound_holds():
    from gridsearch.experiment import TRAP_MAP

    grid = parse(TRAP_MAP)
    p = GridProblem(grid)
    optimal = uniform_cost(p).cost
    for weight in (1.5, 2.0, 3.0, 5.0):
        assert astar(p, weighted(grid.goal, weight)).cost <= weight * optimal + 1e-9
