"""Run the four experiments and print the tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from gridsearch import experiment
from gridsearch.grid import GridProblem, MapError, parse
from gridsearch.heuristics import check_admissible, manhattan, weighted
from gridsearch.search import astar, uniform_cost

EXPERIMENTS = {
    "uninformed": experiment.uninformed,
    "terrain": experiment.terrain,
    "admissible": experiment.admissible_pair,
    "inadmissible": experiment.inadmissible,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gridsearch",
        description=(
            "What an inadmissible heuristic costs, on maps small enough to verify by eye."
        ),
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        default="all",
        choices=[*EXPERIMENTS, "all"],
    )
    parser.add_argument(
        "--map",
        dest="map_file",
        help="read the map from a file instead of using the built-in one",
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=3.0,
        help="Manhattan multiplier for the bound check (>1 is inadmissible)",
    )
    parser.add_argument("--show-paths", action="store_true", help="draw the maps")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    override = None
    if args.map_file:
        try:
            override = Path(args.map_file).read_text(encoding="utf-8")
            parse(override)
        except FileNotFoundError:
            print(f"map file not found: {args.map_file}")
            return 2
        except MapError as exc:
            print(f"bad map: {exc}")
            return 2

    names = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    for name in names:
        print(f"=== {name}")
        run = EXPERIMENTS[name]
        comparison = run(override) if override else run()
        print(comparison.table())
        print()

    if args.experiment in ("all", "inadmissible"):
        text = override or experiment.TRAP_MAP
        print(experiment.bound_check(text, weight=args.weight))

        if args.show_paths:
            grid = parse(text)
            problem = GridProblem(grid)
            print("\noptimal:")
            print(grid.render(list(uniform_cost(problem).path)))
            print(f"\nManhattan x{args.weight:g}:")
            print(
                grid.render(list(astar(problem, weighted(grid.goal, args.weight)).path))
            )
            print(f"\n{check_admissible(problem, manhattan(grid.goal))}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
