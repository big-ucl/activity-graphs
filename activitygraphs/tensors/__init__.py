"""[Stage 4: survey + graph -> tensors] This package transforms surveys and graphs to PyG and PyTorch tensors.

It exposes ``SurveyTensors``, a bundle of Tensors used by the ML pipeline, i.e.:
  - the PyG form of the graph (``pyg.data.Data``)
  - the per-user node tensors (node features, visit labels, and distances).
  - the per-user tensors (user attributes and household group IDs)
"""

from activitygraphs.tensors.build import USER_NODE_SCHEMA, SurveyTensors, build_tensors

__all__ = ["USER_NODE_SCHEMA", "SurveyTensors", "build_tensors"]
