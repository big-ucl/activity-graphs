"""[Stage 4: survey + graph -> tensors] This package transforms surveys and graphs to PyG and PyTorch tensors.

It exposes ``SurveyTensors``, a bundle of Tensors used by the ML pipeline, i.e.:
  - the PyG form of the graph (``pyg.data.Data``)
  - the spatial demographics (per-user node features, labels, demographics, and distances).
  - the individual demographics (indiv. demographics and household group IDs)
"""

from activitygraphs.tensors.build import SPATIAL_FEATURE_NAMES, SurveyTensors, build_tensors

__all__ = ["SPATIAL_FEATURE_NAMES", "SurveyTensors", "build_tensors"]
