"""Map parsing rejects everything a search cannot be run on."""

import pytest

from gridsearch.grid import DEFAULT_COST, GridProblem, MapError, parse

SIMPLE = """
#####
#T P#
#####
"""


def test_start_and_goal_are_located():
    grid = parse(SIMPLE)
    assert grid.start == (1, 1)
    assert grid.goal == (3, 1)


def test_dimensions():
    grid = parse(SIMPLE)
    assert (grid.width, grid.height) == (5, 3)


def test_blank_lines_are_ignored():
    assert parse("\n\n#####\n#T P#\n#####\n\n").height == 3


def test_short_rows_are_padded_with_walls():
    # A map written in a docstring must not depend on trailing spaces surviving.
    grid = parse("#####\n#T P\n#####")
    assert grid.rows[1] == "#T P#"


def test_lowercase_markers_are_accepted():
    grid = parse("#####\n#t p#\n#####")
    assert grid.start == (1, 1) and grid.goal == (3, 1)


def test_out_of_bounds_counts_as_a_wall():
    grid = parse(SIMPLE)
    assert grid.is_wall((-1, 0))
    assert grid.is_wall((99, 99))


@pytest.mark.parametrize(
    "text",
    [
        "#####\n#  P#\n#####",  # no start
        "#####\n#T  #\n#####",  # no goal
        "#####\n#TTP#\n#####",  # two starts
        "#####\n#TPP#\n#####",  # two goals
    ],
)
def test_maps_without_exactly_one_start_and_goal_are_rejected(text):
    with pytest.raises(MapError, match="exactly one"):
        parse(text)


def test_an_empty_map_is_rejected():
    with pytest.raises(MapError, match="empty"):
        parse("\n  \n")


def test_render_marks_a_path_without_overwriting_markers():
    grid = parse(SIMPLE)
    drawn = grid.render([(1, 1), (2, 1), (3, 1)])
    assert drawn.splitlines()[1] == "#T.P#"


def test_render_without_a_path_is_the_original_map():
    grid = parse(SIMPLE)
    assert grid.render() == "\n".join(grid.rows)


def test_default_cell_cost():
    problem = GridProblem(parse(SIMPLE))
    assert problem.cost_of_entering((2, 1)) == DEFAULT_COST


def test_digits_are_terrain_costs():
    problem = GridProblem(parse("#####\n#T9P#\n#####"))
    assert problem.cost_of_entering((2, 1)) == 9.0


def test_min_cell_cost_ignores_walls():
    problem = GridProblem(parse("#####\n#T9P#\n#####"))
    assert problem.min_cell_cost == 1.0


@pytest.mark.parametrize("terrain", [{"9": 0.0}, {"9": -1.0}])
def test_non_positive_costs_are_rejected(terrain):
    # A zero-cost cell lets uniform cost reopen the frontier forever.
    with pytest.raises(MapError, match="positive"):
        GridProblem(parse(SIMPLE), terrain)


def test_zero_default_cost_is_rejected():
    with pytest.raises(MapError, match="positive"):
        GridProblem(parse(SIMPLE), default_cost=0.0)


def test_successors_never_enter_a_wall():
    problem = GridProblem(parse(SIMPLE))
    assert [nxt for _, nxt, _ in problem.successors((1, 1))] == [(2, 1)]


def test_successor_order_is_fixed():
    # Iterating a set here is the usual reason a search result changes between
    # runs for no visible reason.
    problem = GridProblem(parse("#####\n#   #\n# T #\n#  P#\n#####"))
    twice = [[action for action, _, _ in problem.successors((2, 2))] for _ in range(2)]
    assert twice[0] == twice[1] == ["up", "down", "left", "right"]


def test_successor_cost_is_the_cost_of_the_cell_entered():
    problem = GridProblem(parse("#####\n#T9P#\n#####"))
    assert [cost for _, _, cost in problem.successors((1, 1))] == [9.0]
