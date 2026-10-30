"""Unit tests for the demographic categories shared by the surveys."""

import polars as pl
import pytest

from activitygraphs.surveys.demographics import INCOME_QUARTILES, quantile_bands

TTS_2016_INCOME_HOUSEHOLDS = {1: 6451, 2: 23811, 3: 23093, 4: 33805, 5: 15938, 6: 28306, 7: 31304}
TTS_2016_INCOME_BANDS = [1, 2, 3, 4, 5, 6]


def repeat_codes(households_by_code: dict[int, int]) -> pl.Series:
    codes = []
    for code, n_households in households_by_code.items():
        codes.extend([code] * n_households)

    return pl.Series("income", codes)


class TestQuantileBands:
    def test_a_band_takes_the_quantile_of_its_middle_row(self):
        codes = pl.Series("income", [1, 2, 2, 3, 3])

        assert quantile_bands(codes, [1, 2, 3], INCOME_QUARTILES) == {1: "q1", 2: "q2", 3: "q4"}

    def test_codes_outside_the_bands_are_left_out_of_the_shares(self):
        codes = pl.Series("income", [1, 2, 2, 3, 3, 999, 999, 999])

        assert quantile_bands(codes, [1, 2, 3], INCOME_QUARTILES) == {1: "q1", 2: "q2", 3: "q4"}

    def test_an_empty_band_still_gets_a_quantile(self):
        codes = pl.Series("income", [1, 3, 3, 3])

        assert quantile_bands(codes, [1, 2, 3], INCOME_QUARTILES) == {1: "q1", 2: "q2", 3: "q3"}

    def test_codes_without_a_valid_band_raise(self):
        codes = pl.Series("income", [999, 999])

        with pytest.raises(ValueError, match="No row of `income` has a code in"):
            quantile_bands(codes, [1, 2, 3], INCOME_QUARTILES)

    def test_tts_2016_bands_spread_over_every_quartile(self):
        codes = repeat_codes(TTS_2016_INCOME_HOUSEHOLDS)

        bands = quantile_bands(codes, TTS_2016_INCOME_BANDS, INCOME_QUARTILES)

        assert bands == {1: "q1", 2: "q1", 3: "q2", 4: "q3", 5: "q3", 6: "q4"}
