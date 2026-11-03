"""[Stage 4: survey + graph -> tensors] This package transforms surveys and graphs to PyTorch tensors, and caches them.

It is split into the following modules:
  - ``build``: ``build_tensors`` and ``load_tensors``, the tensors of a survey: the location graph and the users.
  - ``bundles``: ``LocationTensors`` (the shared graph and the node distances) and ``UserTensors`` (the home nodes,
    visit labels, user attributes and household group IDs), and the tensor bundle validation.
  - ``cache``: ``save_tensors`` and ``read_tensors``, the tensor cache and fingerprinting.
"""

from activitygraphs.tensors.build import USER_NODE_SCHEMA, build_tensors, load_tensors
from activitygraphs.tensors.bundles import LocationTensors, SurveyTensors, UserTensors, check_tensors
from activitygraphs.tensors.cache import (
    Fingerprint,
    read_tensors,
    read_tensors_fingerprint,
    save_tensors,
    tensors_fingerprint,
    tensors_path,
)

__all__ = [
    "USER_NODE_SCHEMA",
    "Fingerprint",
    "LocationTensors",
    "SurveyTensors",
    "UserTensors",
    "build_tensors",
    "check_tensors",
    "load_tensors",
    "read_tensors",
    "read_tensors_fingerprint",
    "save_tensors",
    "tensors_fingerprint",
    "tensors_path",
]
