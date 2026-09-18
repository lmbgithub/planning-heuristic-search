import pytest

from gridsearch.cli import build_parser, main


def test_default_runs_everything():
    assert build_parser().parse_args([]).experiment == "all"


def test_unknown_experiment_is_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["astar-plus"])


def test_all_experiments_run(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    for name in ("uninformed", "terrain", "admissible", "inadmissible"):
        assert f"=== {name}" in out
    assert "within bound: True" in out


def test_a_single_experiment_runs_alone(capsys):
    assert main(["terrain"]) == 0
    assert "=== uninformed" not in capsys.readouterr().out


def test_show_paths_draws_the_maps(capsys):
    assert main(["inadmissible", "--show-paths"]) == 0
    assert "optimal:" in capsys.readouterr().out


def test_a_missing_map_file_is_a_usage_error(tmp_path, capsys):
    assert main(["uninformed", "--map", str(tmp_path / "absent.txt")]) == 2
    assert "not found" in capsys.readouterr().out


def test_a_malformed_map_file_is_a_usage_error(tmp_path, capsys):
    path = tmp_path / "bad.txt"
    path.write_text("####\n#  #\n####\n")
    assert main(["uninformed", "--map", str(path)]) == 2
    assert "bad map" in capsys.readouterr().out


def test_a_custom_map_file_is_used(tmp_path, capsys):
    path = tmp_path / "corridor.txt"
    path.write_text("#######\n#T   P#\n#######\n")
    assert main(["uninformed", "--map", str(path)]) == 0
    assert "breadth first" in capsys.readouterr().out
