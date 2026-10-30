"""[Stages 1-2: raw files -> tables -> survey] This package loads the raw files into survey specific ``*Tables``, then processes them into a common ``SurveyData`` interface.

It is split into the following modules:
  - ``registry``: ``load_survey``, the entry point for the rest of the pipeline, which picks the ``SurveyLoader``
    named by the data config's ``survey`` key from ``REGISTRY``.
  - ``base``: the ``SurveyData`` interface, the ``SurveyLoader`` bundle and the ``DataFrameStore`` cache.
  - ``tpg``, ``thats``, ``cmap``: one module per survey, each with: ``read_tables`` (raw files -> ``*Tables``);
  ``build_*_survey`` (``*Tables`` -> ``*Survey``); the census statistics for the graph nodes; and a ``LOADER``.
  - ``demographics``: the demographic categories shared by the surveys.
  - ``gtfs``, ``overture``, ``statistics``, ``routing``: other readers and helpers.
"""
