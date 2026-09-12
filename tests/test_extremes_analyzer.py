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


@pytest.mark.parametrize(
    "selection, expected_days, expected_values",
    [
        ("max", [8, 11], [20.0, 15.0]),
        ("first", [7, 10], [11.0, 13.0]),
        ("last", [9, 11], [12.0, 15.0]),
    ],
)
@pytest.mark.parametrize("window", [3, "3D", pd.Timedelta(days=3)])
def test_time_window_selection(selection, expected_days, expected_values, window):
    series = pd.Series(
        [11.0, 20.0, 12.0, 13.0, 15.0],
        index=pd.date_range("2024-01-07", periods=5),
    )
    analyzer = DataProcessor(series).get_extremes_analyzer()

    values, times = analyzer.peaks_over_threshold(
        10, min_separation=window, peak_selection=selection
    )

    np.testing.assert_array_equal(values, expected_values)
    assert times.day.tolist() == expected_days


def test_time_window_defaults_to_maximum():
    series = pd.Series([11.0, 20.0, 12.0], index=pd.date_range("2024-01-07", periods=3))
    values, times = (
        DataProcessor(series)
        .get_extremes_analyzer()
        .peaks_over_threshold(10, min_separation=3)
    )
    np.testing.assert_array_equal(values, [20.0])
    assert times.tolist() == [pd.Timestamp("2024-01-08")]


def test_windows_do_not_move_to_selected_peak():
    """The right boundary starts a new window, even next to the prior maximum."""
    series = pd.Series(
        [11.0, 12.0, 30.0, 40.0, 13.0, 14.0],
        index=pd.date_range("2024-01-07", periods=6),
    )
    values, times = (
        DataProcessor(series)
        .get_extremes_analyzer()
        .peaks_over_threshold(10, min_separation=3)
    )
    np.testing.assert_array_equal(values, [30.0, 40.0])
    assert times.day.tolist() == [9, 10]


def test_maximum_ties_keep_first_observation_with_duplicate_times():
    dates = pd.to_datetime(["2024-01-09", "2024-01-07", "2024-01-07"])
    series = pd.Series([20.0, 11.0, 20.0], index=dates).tz_localize("UTC")
    values, times = (
        DataProcessor(series)
        .get_extremes_analyzer()
        .peaks_over_threshold(10, min_separation=3)
    )
    np.testing.assert_array_equal(values, [20.0])
    assert times.tolist() == [pd.Timestamp("2024-01-07", tz="UTC")]


@pytest.mark.parametrize("selection", ["max", "first", "last"])
def test_window_selection_handles_empty_and_single_samples(selection):
    series = pd.Series([0.0, 11.0], index=pd.date_range("2024-01-01", periods=2))
    analyzer = DataProcessor(series).get_extremes_analyzer()
    values, times = analyzer.peaks_over_threshold(
        20, min_separation=3, peak_selection=selection
    )
    assert values.size == len(times) == 0
    values, times = analyzer.peaks_over_threshold(
        10, min_separation=3, peak_selection=selection
    )
    np.testing.assert_array_equal(values, [11.0])
    assert times.tolist() == [series.index[1]]


def test_invalid_peak_selection_is_rejected():
    analyzer = DataProcessor(
        pd.Series([0.0], index=pd.date_range("2024-01-01", periods=1))
    ).get_extremes_analyzer()
    with pytest.raises(ValueError, match="peak_selection"):
        analyzer.peaks_over_threshold(10, min_separation=3, peak_selection="median")


def test_windows_restart_after_gaps():
    series = pd.Series(
        [11.0, 20.0, 12.0, 30.0],
        index=pd.to_datetime(["2024-01-07", "2024-01-09", "2024-01-21", "2024-01-23"]),
    )
    values, times = (
        DataProcessor(series)
        .get_extremes_analyzer()
        .peaks_over_threshold(10, min_separation=3)
    )
    np.testing.assert_array_equal(values, [20.0, 30.0])
    assert times.day.tolist() == [9, 23]


@pytest.mark.parametrize("selection", ["max", "first", "last"])
@pytest.mark.parametrize("vary_first", ["percentile", "separation"])
def test_threshold_search_uses_window_selection(selection, vary_first):
    series = pd.Series(
        [0.0, 11.0, 30.0, 12.0, 0.0, 0.0, 11.0, 20.0, 12.0],
        index=pd.date_range("2024-01-01", periods=9),
    )
    analyzer = DataProcessor(series).get_extremes_analyzer()
    result = analyzer.find_optimal_pot_threshold(
        min_samples=1,
        percentile_min=10,
        percentile_max=20,
        min_separation_hours=72,
        max_separation_hours=96,
        vary_first=vary_first,
        peak_selection=selection,
    )
    expected_days = {"max": [3, 8], "first": [2, 7], "last": [4, 9]}
    assert result["success"]
    assert result["exceedance_times"].day.tolist() == expected_days[selection]
    np.testing.assert_array_equal(
        result["exceedances"], series.loc[result["exceedance_times"]].values
    )


@pytest.mark.parametrize("selection", ["max", "first", "last"])
def test_pure_and_event_wise_ignore_window_selection(selection):
    series = pd.Series([11.0, 30.0, 12.0], index=pd.date_range("2024-01-01", periods=3))
    analyzer = DataProcessor(series).get_extremes_analyzer()
    values, times = analyzer.peaks_over_threshold(10, peak_selection=selection)
    np.testing.assert_array_equal(values, series.values)
    pd.testing.assert_index_equal(times, series.index)
    values, times = analyzer.peaks_over_threshold(
        10, min_separation=3, event_wise=True, peak_selection=selection
    )
    np.testing.assert_array_equal(values, [30.0])
    assert times.day.tolist() == [2]


@pytest.mark.parametrize("duration", [-1, "-1D", pd.NaT])
def test_invalid_window_duration(duration):
    analyzer = DataProcessor(
        pd.Series([11.0], index=pd.date_range("2024-01-01", periods=1))
    ).get_extremes_analyzer()
    with pytest.raises(ValueError, match="min_separation"):
        analyzer.peaks_over_threshold(10, min_separation=duration)


def test_zero_window_keeps_all_exceedances():
    series = pd.Series([11.0, 30.0, 12.0], index=pd.date_range("2024-01-01", periods=3))
    values, times = (
        DataProcessor(series)
        .get_extremes_analyzer()
        .peaks_over_threshold(10, min_separation=0)
    )
    np.testing.assert_array_equal(values, series.values)
    pd.testing.assert_index_equal(times, series.index)
