"""Heuristic search on grid maps: what an inadmissible heuristic actually costs.

Zero dependencies. The search strategies are implemented here rather than
imported, because the experiment is about what the frontier does — the node
counts are the result, and a library that does not expose them cannot produce
it.
"""

from __future__ import annotations

from gridsearch.errors import GridSearchError, MapError
from gridsearch.grid import Grid, GridProblem, parse
from gridsearch.heuristics import chebyshev, check_admissible, manhattan, weighted, zero
from gridsearch.search import (
    SearchResult,
    astar,
    breadth_first,
    depth_first,
    uniform_cost,
)

__all__ = [
    "Grid",
    "GridProblem",
    "GridSearchError",
    "MapError",
    "SearchResult",
    "astar",
    "breadth_first",
    "chebyshev",
    "check_admissible",
    "depth_first",
    "manhattan",
    "parse",
    "uniform_cost",
    "weighted",
    "zero",
]
