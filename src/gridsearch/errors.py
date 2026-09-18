"""The package's error hierarchy.

One base class, so a caller catches everything this package raises with a
single `except` and can still discriminate. `GridSearchError` derives from
`ValueError` so handlers written for bad input keep working.
"""

from __future__ import annotations


class GridSearchError(ValueError):
    """Anything this package refuses to do."""


class MapError(GridSearchError):
    """The map is not a grid a search can be defined on."""
