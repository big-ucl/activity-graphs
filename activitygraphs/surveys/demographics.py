"""Demographic columns shared by the surveys, so that a category gets the same code in every dataset.

Each survey maps its own codes onto the categories declared here and returns ``DEMOGRAPHICS_SCHEMA``.
"""

from collections.abc import Sequence

import polars as pl

from activitygraphs.schema import ColumnKind, ColumnSpec, TensorSchema

AGE_BANDS = ("0-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+")
EMPLOYMENT_STATUSES = ("full_time", "part_time", "self_employed", "not_employed")
STUDENT_STATUSES = ("full_time", "part_time", "not_student")
BOOLEAN_CATEGORIES = ("false", "true")
INCOME_QUARTILES = ("q1", "q2", "q3", "q4")

DEMOGRAPHICS_SCHEMA = TensorSchema(
    "user_attributes",
    (
        ColumnSpec("age_band", ColumnKind.CATEGORICAL, AGE_BANDS),
        ColumnSpec("employment_status", ColumnKind.CATEGORICAL, EMPLOYMENT_STATUSES),
        ColumnSpec("student_status", ColumnKind.CATEGORICAL, STUDENT_STATUSES),
        ColumnSpec("can_drive", ColumnKind.CATEGORICAL, BOOLEAN_CATEGORIES),
        ColumnSpec("has_transit_pass", ColumnKind.CATEGORICAL, BOOLEAN_CATEGORIES),
        ColumnSpec("hh_num_adults", ColumnKind.CONTINUOUS),
        ColumnSpec("hh_num_children", ColumnKind.CONTINUOUS),
        ColumnSpec("hh_num_vehicles", ColumnKind.CONTINUOUS),
        ColumnSpec("income_quartile", ColumnKind.CATEGORICAL, INCOME_QUARTILES),
    ),
)


def quantile_bands(codes: pl.Series, ordered_codes: Sequence[int], quantile_labels: Sequence[str]) -> dict[int, str]:
    """Map each band of an ordinal banded variable to the quantile band that contains the middle row of the band.

    Args:
        codes: Band code of each row, e.g. the income band of each household of a survey. Rows with codes that are not
          in ``ordered_codes``, such as refusals/missing, are left out of the quantiles.
        ordered_codes: Valid band codes, in increasing order of the banded variable.
        quantile_labels: Quantile band labels, in increasing order. The number of labels determines the
          quantiles (4 for quartiles).

    Returns:
        Dict from each code of ``ordered_codes`` to one of ``quantile_labels``.

    Raises:
        ValueError: No row of ``codes`` has a code in ``ordered_codes``.
    """
    valid_codes = codes.filter(codes.is_in(ordered_codes))
    if valid_codes.is_empty():
        raise ValueError(f"No row of `{codes.name}` has a code in {list(ordered_codes)}.")
    code_counts_df = valid_codes.value_counts()
    band_codes, band_counts = code_counts_df[codes.name].to_list(), code_counts_df["count"].to_list()

    counts_per_code = dict(zip(band_codes, band_counts, strict=True))

    band_to_quantile = {}
    counts_below = 0
    for code in ordered_codes:
        count_in_band = counts_per_code.get(code, 0)
        middle_share = (counts_below + count_in_band / 2) / valid_codes.len()

        label_idx = min(int(middle_share * len(quantile_labels)), len(quantile_labels) - 1)
        band_to_quantile[code] = quantile_labels[label_idx]

        counts_below += count_in_band

    return band_to_quantile
