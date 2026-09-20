# planning-heuristic-search

A study of what an inadmissible heuristic costs you in A* search.

A* returns an optimal path only when its heuristic never overestimates the remaining cost. In practice that guarantee is often traded away — a heuristic is scaled up to make search faster, and the result is still described as the shortest path. This project measures both sides of that trade, on maps small enough to check by eye: how much search time you save, and how much path quality you give up.

**Zero dependencies. 86 tests.**

## Skills demonstrated

**Search algorithms** — breadth-first, depth-first, uniform cost and A* written from scratch with instrumented frontiers; uniform cost and A* expressed as one function differing only in the priority, with a test asserting `h = 0` makes them identical

**Heuristic theory** — admissibility verified exhaustively against true costs by a backwards uniform-cost sweep rather than assumed; dominance between Manhattan and Chebyshev; the *w*-bound on weighted A* stated and tested

**Measurement** — path cost recomputed from the map instead of trusted from the frontier, so a mis-accumulating search cannot report consistent wrong numbers; node counts and peak frontier reported beside every answer

**Determinism** — insertion-counter tie-breaking in the priority queue and fixed successor order, because iterating a set is why a route changes between runs

**Software engineering** — zero dependencies, 86 tests covering unreachable goals, degenerate start-equals-goal, cyclic cost models and sub-unit step costs

**Tooling** — ruff, pre-commit, CI matrix on 3.10/3.11/3.12 with a lint job

## What the run looks like

```
$ python examples/run_all.py
=== uninformed
algorithm       heuristic           cost  steps  visited  frontier  optimal   excess
------------------------------------------------------------------------------------
breadth first   -                    6.0      6       18         4      yes      0.0
depth first     -                   14.0     14       20        12       NO      8.0
uniform cost    -                    6.0      6       18         4      yes      0.0

=== terrain
algorithm       heuristic           cost  steps  visited  frontier  optimal   excess
------------------------------------------------------------------------------------
breadth first   -                   34.0     10       39         5       NO     22.0
uniform cost    -                   12.0     12       25        12      yes      0.0
A*              manhattan           12.0     12       19        10      yes      0.0

=== admissible
algorithm       heuristic           cost  steps  visited  frontier  optimal   excess
------------------------------------------------------------------------------------
A*              manhattan           12.0     12       19        10      yes      0.0
A*              chebyshev           12.0     12       19        10      yes      0.0
A*              zero (= UCS)        12.0     12       25        12      yes      0.0

=== inadmissible
algorithm       heuristic           cost  steps  visited  frontier  optimal   excess
------------------------------------------------------------------------------------
uniform cost    -                   10.0     10       22         3      yes      0.0
A*              manhattan           10.0     10       17         3      yes      0.0
A*              manhattan x2        10.0     10       18         3      yes      0.0
A*              manhattan x3        14.0     14       15         2       NO      4.0
A*              manhattan x5        14.0     14       15         2       NO      4.0

optimal cost 10.0   weighted (x3) cost 14.0
   ratio 1.40   bound 3   within bound: True
INADMISSIBLE: overestimates by 20.00 at (1, 1)
```


Read the last table twice. The `manhattan x3` row is a **complete, valid, entirely plausible path** that costs 40% more than necessary — and it was found by expanding *fewer* nodes than the correct answer. Both halves of that sentence matter: if weighting were only worse, nobody would do it.

## The six decisions worth discussing

**1. Two numbers per run, always.** `cost` answers "is the answer right"; `visited` answers "what did it take". Reporting only the first is exactly how an inadmissible heuristic survives review — the path looks fine, because it *is* a path. Reporting only the second is how a fast wrong answer gets shipped.

**2. Cost belongs to the cell, not to the direction of travel.** The obvious way to make steps cost different amounts is per-direction costs (left costs 3, right costs 1). It cannot produce this experiment: on a 4-connected grid the net displacement fixes how many left, right, up and down moves *any* path must contain, so every minimum-step route costs the same and breadth-first search is always cost-optimal. Terrain costs break that — and the `terrain` table above is BFS returning a 10-step route costing 34 where a 12-step route costs 12.

**3. Admissibility is checked, not asserted.** `check_admissible` computes the true remaining cost from every reachable cell by a backwards uniform-cost sweep and compares it against the heuristic. On maps this small there is no reason to *assume* a textbook property. It also catches the case the textbook statement hides: Manhattan is admissible **on unit costs**, and a map with a cell costing 0.5 makes the standard heuristic an overestimate. There is a test for that.

**4. Path cost is recomputed from the map, never taken from the search.** The number in the table comes from replaying the returned path against the cost model, not from the accumulator inside the frontier. A search that mis-adds its own costs would otherwise report a consistent, wrong pair of numbers with nothing to contradict them.

**5. Ties are broken deterministically.** The priority queue carries an insertion counter, so equal-`f` nodes pop in insertion order. Without it, ties break by comparing coordinate tuples and the returned path depends on grid geometry — the classic "why did the route change" bug. Successor order is fixed for the same reason: iterating a set is why a search result changes between runs with no visible cause.

**6. An unreachable goal is a result, not an exception.** Every strategy returns `found=False` with an infinite cost and its real node count, so "found nothing after 39 expansions" is a row in the comparison table next to the strategies that did find something.

### The practical reading

Weighting a heuristic is legitimate when you have *decided* that bounded suboptimality is worth the speed: with weight *w*, the returned path costs at most *w* times optimal, and the test suite checks that bound holds. What is not defensible is scaling the heuristic to make search faster and continuing to describe the result as shortest. The difference is a stated decision, and the `--weight` flag exists so the bound is something you set rather than something you inherit.

## Design

### Why the searches are written out

Four strategies, about 150 lines, no dependency. The experiment is *about* what the frontier does — the node counts are the result — and a library that does not expose them cannot produce this table. Uniform cost and A* are the same function with a different priority, which is the honest framing: A* is not a different algorithm, it is uniform cost plus an estimate of what remains. The test suite asserts that `astar` with `h = 0` matches `uniform_cost` exactly, on both cost and node count.


```
src/gridsearch/
  grid.py         ASCII map parsing, terrain costs, successor generation
  heuristics.py   Manhattan, Chebyshev, zero, weighted + the admissibility check
  search.py       BFS, DFS, uniform cost, A*, independent path scoring
  experiment.py   the four experiments and the comparison table
  cli.py          argument parsing
```

## Usage

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

```bash
pytest -q
python examples/run_all.py
python -m gridsearch inadmissible --show-paths
python -m gridsearch uninformed --map my_map.txt
python -m gridsearch --weight 5
```

As a library:

```python
from gridsearch import GridProblem, astar, check_admissible, manhattan, parse, uniform_cost

grid = parse("""
#######
#T 9 P#
#######
""")
problem = GridProblem(grid)
print(uniform_cost(problem).summary())
print(astar(problem, manhattan(grid.goal)).summary())
print(check_admissible(problem, manhattan(grid.goal)))
```

Map syntax: `T` start, `P` goal, `#` wall, a digit `1`-`9` the cost of entering that cell, anything else costs 1.

`main.ipynb` runs the same four experiments as a narrative, importing the package rather than redefining it.

## Dataset

None. Every map is defined in `gridsearch.experiment`, or passed to the CLI with `--map`. The maps are deliberately small enough to check the answers by hand, which is the point: a search result you cannot verify is not evidence of anything.

## Scope

- **No bounded-suboptimal search family.** Weighted A* is here to demonstrate the failure, not as an implementation of ARA* or anchor search.
- **No large benchmark maps.** Timing on a 100k-node map would measure this Python implementation, not the algorithms.
- **No diagonal movement or jump-point search.** Both would invalidate the Manhattan/Chebyshev dominance argument, which is the subject of one of the four experiments.
- **No consistency (monotonicity) analysis.** Admissibility is the property the optimality guarantee needs here; consistency matters for closed-set re-expansion, and this implementation re-opens nodes when a cheaper path is found, so it does not need it.

## License

MIT — see [LICENSE](LICENSE).
