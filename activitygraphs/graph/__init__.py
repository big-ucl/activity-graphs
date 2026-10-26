"""[Stage 3: survey -> graph] This package transforms a survey into a (shared) spatial graph.

The spatial graph is shared for all users. The nodes are zones, with location-level features. The edges are contiguity,
with extra edges if specified for that survey.
"""

from activitygraphs.graph.build import load_graph

__all__ = ["load_graph"]
