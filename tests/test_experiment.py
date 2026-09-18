"""The four experiments must keep producing the results the README claims."""

import pytest

from gridsearch import experiment
from gridsearch.grid import GridProblem, parse
from gridsearch.search import SearchResult, uniform_cost


def test_uninformed_agrees_except_for_depth_first():
    comparison = experiment.uninformed()
    by_name = {r.algorithm: r for r in comparison.results}
    assert comparison.is_optimal(by_name["breadth first"])
    assert comparison.is_optimal(by_name["uniform cost"])
    assert not comparison.is_optimal(by_name["depth first"])


def test_terrain_shows_breadth_first_optimising_the_wrong_quantity():
    comparison = experiment.terrain()
    bfs = next(r for r in comparison.results if r.algorithm == "breadth first")
    assert not comparison.is_optimal(bfs)
    assert comparison.excess(bfs) > 0


def test_every_admissible_heuristic_returns_the_same_cost():
    comparison = experiment.admissible_pair()
    assert len({r.cost for r in comparison.results}) == 1
    assert all(comparison.is_optimal(r) for r in comparison.results)


def test_a_heuristic_only_changes_the_effort():
    comparison = experiment.admissible_pair()
    by_name = {r.heuristic: r for r in comparison.results}
    assert by_name["manhattan"].visited <= by_name["zero (= UCS)"].visited


def test_the_trap_map_actually_traps():
    comparison = experiment.inadmissible()
    bad = [r for r in comparison.results if not comparison.is_optimal(r)]
    assert bad, "the trap map no longer produces a suboptimal weighted run"
    assert all(r.found for r in bad), "a suboptimal path is still a complete path"


def test_the_trap_run_is_faster_as_well_as_worse():
    # If it were only worse, nobody would ever weight a heuristic.
    comparison = experiment.inadmissible()
    optimal = next(r for r in comparison.results if r.heuristic == "manhattan")
    worst = max(comparison.results, key=comparison.excess)
    assert worst.visited < optimal.visited


def test_optimal_cost_of_an_empty_comparison_is_infinite():
    assert experiment.Comparison(()).optimal_cost == float("inf")


def test_a_comparison_of_failures_reports_no_path():
    failed = SearchResult("x", "-", (), float("inf"), 3, 1, found=False)
    comparison = experiment.Comparison((failed,))
    assert "NO PATH" in comparison.table()
    assert not comparison.is_optimal(failed)
    assert comparison.excess(failed) == float("inf")


def test_the_bound_check_confirms_the_guarantee():
    assert "within bound: True" in experiment.bound_check(weight=3.0)


def test_the_bound_check_names_the_heuristic_as_inadmissible():
    assert "INADMISSIBLE" in experiment.bound_check(weight=3.0)


def test_weight_one_is_admissible_and_optimal():
    text = experiment.bound_check(weight=1.0)
    assert "admissible" in text
    assert "INADMISSIBLE" not in text


@pytest.mark.parametrize(
    "map_text", [experiment.OPEN_MAP, experiment.TERRAIN_MAP, experiment.TRAP_MAP]
)
def test_every_built_in_map_is_solvable(map_text):
    assert uniform_cost(GridProblem(parse(map_text))).found
