"""Regression tests for extreme sample extraction."""

import numpy as np
import pandas as pd
import pytest

from magica.core.data_processor import DataProcessor


@pytest.mark.parametrize("peak", [0.0, 10.0])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"min_separation": 3},
        {"min_separation": "3D"},
        {"min_separation": pd.Timedelta(days=3)},
        {"event_wise": True},
    ],
)
def test_pot_without_exceedances_returns_empty_arrays(peak, options):
    """Values below or equal to the threshold never count as exceedances."""
    series = pd.Series(0.0, index=pd.date_range("2024-01-01", "2024-01-31"))
    series.loc["2024-01-14":"2024-01-18"] = peak
    analyzer = DataProcessor(series).get_extremes_analyzer()

    values, times = analyzer.peaks_over_threshold(threshold=10, **options)

    assert isinstance(values, np.ndarray)
    assert isinstance(times, pd.DatetimeIndex)
    assert values.shape == (0,)
    assert len(times) == 0


def test_empty_time_based_pot_preserves_timezone():
    series = pd.Series(
        [0.0, 10.0],
        index=pd.date_range("2024-01-01", periods=2, tz="America/Sao_Paulo"),
    )
    analyzer = DataProcessor(series).get_extremes_analyzer()

    values, times = analyzer.peaks_over_threshold(10, min_separation=3)

    assert values.size == 0
    pd.testing.assert_index_equal(times, series.index[:0])


@pytest.mark.parametrize("two_events", [False, True])
@pytest.mark.parametrize(
    "options, offsets",
    [
        ({}, [0, 1, 2, 3, 4]),
        ({"min_separation": 3}, [0, 3]),
        ({"event_wise": True}, [0]),
    ],
)
def test_pot_selects_plateau_dates(two_events, options, offsets):
    """Event-wise ties select the first day; time-based keeps days 1 and 4."""
    series = pd.Series(0.0, index=pd.date_range("2024-01-01", "2024-01-31"))
    starts = [pd.Timestamp("2024-01-07")]
    if two_events:
        starts.append(pd.Timestamp("2024-01-21"))
    for start in starts:
        series.loc[pd.date_range(start, periods=5)] = 11.0
    analyzer = DataProcessor(series).get_extremes_analyzer()

    values, times = analyzer.peaks_over_threshold(10, **options)

    expected_dates = [
        start + pd.Timedelta(days=day) for start in starts for day in offsets
    ]
    expected = pd.DatetimeIndex(expected_dates)
    np.testing.assert_array_equal(values, np.full(len(expected), 11.0))
    pd.testing.assert_index_equal(times, expected)
