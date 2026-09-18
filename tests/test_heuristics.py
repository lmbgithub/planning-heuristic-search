"""Heuristics, and admissibility verified rather than assumed."""

import pytest

from gridsearch.grid import GridProblem, parse
from gridsearch.heuristics import chebyshev, check_admissible, manhattan, weighted, zero

TERRAIN = """
########
#T99  P#
#      #
########
"""


def test_manhattan_is_zero_at_the_goal():
    assert manhattan((3, 4))((3, 4)) == 0.0


def test_manhattan_counts_both_axes():
    assert manhattan((0, 0))((3, 4)) == 7.0


def test_chebyshev_takes_the_larger_axis():
    assert chebyshev((0, 0))((3, 4)) == 4.0


def test_manhattan_dominates_chebyshev_everywhere():
    m, c = manhattan((0, 0)), chebyshev((0, 0))
    for x in range(6):
        for y in range(6):
            assert m((x, y)) >= c((x, y))


def test_zero_is_zero_everywhere():
    assert zero((3, 4))((0, 0)) == 0.0


def test_weighted_scales_manhattan():
    assert weighted((0, 0), 3.0)((2, 1)) == 9.0


def test_weight_one_is_plain_manhattan():
    assert weighted((0, 0), 1.0)((2, 1)) == manhattan((0, 0))((2, 1))


@pytest.mark.parametrize("weight", [0.0, -1.0])
def test_non_positive_weights_are_rejected(weight):
    with pytest.raises(ValueError, match="positive"):
        weighted((0, 0), weight)


def test_manhattan_is_admissible_on_unit_terrain():
    problem = GridProblem(parse("#####\n#T P#\n#####"))
    assert check_admissible(problem, manhattan(problem.grid.goal)).admissible


def test_manhattan_stays_admissible_when_terrain_is_expensive():
    # Expensive cells raise the true cost, so a step-counting heuristic only
    # under-estimates further.
    problem = GridProblem(parse(TERRAIN))
    assert check_admissible(problem, manhattan(problem.grid.goal)).admissible


def test_the_message_names_the_cheapest_step_when_that_is_the_cause():
    # An inadmissibility caused by sub-unit step costs reads very differently
    # from one caused by a weighted heuristic, and the message says which.
    problem = GridProblem(parse("#######\n#T   P#\n#######"), default_cost=0.5)
    assert "cheapest step" in str(check_admissible(problem, manhattan(problem.grid.goal)))


def test_a_weighted_heuristic_is_not_blamed_on_the_terrain():
    problem = GridProblem(parse("#######\n#T   P#\n#######"))
    assert "cheapest step" not in str(
        check_admissible(problem, weighted(problem.grid.goal, 2.0))
    )


def test_manhattan_is_not_admissible_when_a_step_can_cost_less_than_one():
    # The textbook claim is about unit costs. Halve them and Manhattan — which
    # counts steps — starts overestimating.
    problem = GridProblem(parse("#######\n#T   P#\n#######"), default_cost=0.5)
    check = check_admissible(problem, manhattan(problem.grid.goal))
    assert not check.admissible
    assert check.worst_overestimate == pytest.approx(2.0)


def test_chebyshev_is_admissible():
    problem = GridProblem(parse(TERRAIN))
    assert check_admissible(problem, chebyshev(problem.grid.goal)).admissible


def test_a_weighted_heuristic_is_flagged_inadmissible():
    problem = GridProblem(parse("#######\n#T   P#\n#######"))
    check = check_admissible(problem, weighted(problem.grid.goal, 2.0))
    assert not check.admissible
    assert check.worst_state is not None
    assert "INADMISSIBLE" in str(check)


def test_the_admissibility_report_reads_as_a_sentence():
    problem = GridProblem(parse("#######\n#T   P#\n#######"))
    assert str(check_admissible(problem, manhattan(problem.grid.goal))).startswith(
        "admissible"
    )


def test_weight_just_above_one_is_still_inadmissible():
    problem = GridProblem(parse("#######\n#T   P#\n#######"))
    assert not check_admissible(problem, weighted(problem.grid.goal, 1.01)).admissible
