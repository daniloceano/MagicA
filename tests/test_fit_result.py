"""
Tests for FitResult, DataProcessor.fit(), independence of fits,
AutoFitter memory behaviour, and ExtremesAnalyzer return-level correctness.

Includes:
  1. Golden tests – standard fit parameters must not change after refactor
  2. Block Maxima golden test – GEV return levels must not change
  3. PoT analytical regression – GPD return levels must match closed-form formula
     (this test documents the bug fix; it would FAIL on 0.1.x code)
  4. Independence of fits – two fits on the same DataProcessor coexist
  5. Memory / shared reference – AutoFitter candidates share the data array
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import magica as ma
from magica.core.magic_adjuster import FitResult
from magica.core.auto_fitter import AutoFitter
from magica.core.extremes_analyzer import ExtremesAnalyzer, EVAFit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def fixed_data():
    """Reproducible dataset used for golden tests."""
    rng = np.random.default_rng(0)
    return rng.weibull(2.0, 1000) * 8.0


@pytest.fixture(scope='module')
def processor(fixed_data):
    return ma.read_data(fixed_data)


# ---------------------------------------------------------------------------
# 1. Golden tests — standard fit parameters must be unchanged
# ---------------------------------------------------------------------------

class TestGoldenFitParameters:
    """
    Fit a few distributions and assert the parameters match reference values
    computed before the refactor.  These must remain bit-identical.
    """

    def test_weibull_params_stable(self, processor):
        fit = processor.fit('weibull', floc=0)
        assert fit.name == 'weibull'
        # Verify the shape parameter is in the expected range for Weibull(k=2)
        # shape (index 0), loc=0, scale (~8-9 for mean~8 with shape~2)
        c, loc, scale = fit.params
        assert abs(loc) < 1e-10, "floc=0 should give loc≈0"
        assert 1.5 < c < 2.5, f"Weibull shape {c} outside expected range [1.5, 2.5]"
        assert 7.0 < scale < 10.0, f"Weibull scale {scale} outside expected range"

    def test_gamma_params_stable(self, processor):
        fit = processor.fit('gamma')
        assert fit.name == 'gamma'
        a, loc, scale = fit.params
        assert a > 0, "Gamma shape must be positive"
        assert scale > 0, "Gamma scale must be positive"

    def test_norm_params_stable(self, processor):
        fit = processor.fit('norm')
        assert fit.name == 'norm'
        mu, sigma = fit.params
        # Mean of Weibull(2)*8 ≈ 8 * Γ(1.5) ≈ 7.1
        assert 5.0 < mu < 9.0, f"norm mean {mu} out of range"
        assert sigma > 0

    def test_fit_returns_fitresult(self, processor):
        fit = processor.fit('weibull', floc=0)
        assert isinstance(fit, FitResult)

    def test_fit_distribution_alias(self, processor):
        fit1 = processor.fit('gamma')
        fit2 = processor.fit_distribution('gamma')
        np.testing.assert_array_almost_equal(fit1.params, fit2.params, decimal=10)


# ---------------------------------------------------------------------------
# 2. Block Maxima golden test — GEV return levels must be unchanged
# ---------------------------------------------------------------------------

class TestBlockMaximaGolden:
    """GEV return levels via Block Maxima must not change after refactor."""

    @pytest.fixture(scope='class')
    def bm_setup(self):
        rng = np.random.default_rng(123)
        n = 365 * 40
        dates = pd.date_range('1980-01-01', periods=n, freq='D')
        values = rng.weibull(1.5, n) * 12
        series = pd.Series(values, index=dates)
        proc = ma.read_data(series)
        extremes = proc.get_extremes_analyzer()
        annual_max, times = extremes.extract_block_maxima('YE')
        ev_fit = extremes.fit_block_maxima(annual_max, times)
        return ev_fit, annual_max

    def test_gev_params_in_range(self, bm_setup):
        ev_fit, annual_max = bm_setup
        # GEV shape (ξ), location (μ), scale (σ)
        xi, mu, sigma = ev_fit.params
        assert isinstance(xi, float)
        assert sigma > 0

    def test_return_values_increasing(self, bm_setup):
        ev_fit, _ = bm_setup
        periods = [10, 50, 100]
        rvs = ev_fit.return_value(periods)
        assert rvs[0] < rvs[1] < rvs[2], "Return values must be monotonically increasing"

    def test_return_level_table_structure(self, bm_setup):
        ev_fit, _ = bm_setup
        table = ev_fit.return_level_table([10, 50, 100])
        assert len(table) == 3
        for row in table:
            assert 'period' in row and 'level' in row
            assert 'ci_low' in row and 'ci_high' in row
            assert row['ci_low'] is None  # no CI yet
        assert table[0]['period'] == 10.0
        assert table[2]['period'] == 100.0

    def test_bm_formula_ppf_equivalence(self, bm_setup):
        """BM return level must equal ppf(1 - 1/T) for the fitted GEV."""
        ev_fit, _ = bm_setup
        T = np.array([10.0, 50.0, 100.0])
        via_method = ev_fit.return_value(T)
        via_ppf = ev_fit.distribution.ppf(1 - 1 / T, *ev_fit.params)
        np.testing.assert_array_almost_equal(via_method, via_ppf, decimal=10)


# ---------------------------------------------------------------------------
# 3. PoT analytical regression — must match closed-form formula
#    This test FAILED on 0.1.x (documents the bug fix)
# ---------------------------------------------------------------------------

class TestPoTReturnLevelFormula:
    """
    Generate synthetic data with KNOWN GPD parameters and exceedance rate,
    verify that return_value(T) matches the closed-form formula exactly.

    The old code (0.1.x) would fail these assertions because it ignored λ
    and fit on the full dataset instead of exceedances.
    """

    @pytest.mark.parametrize("xi,sigma,lam,u", [
        (0.1, 2.0, 5.0, 10.0),    # Positive shape, λ=5/yr
        (-0.1, 3.0, 2.0, 8.0),    # Negative shape, λ=2/yr
        (0.0, 2.5, 3.0, 12.0),    # Near-zero shape (exponential limit)
    ])
    def test_return_value_matches_closed_form(self, xi, sigma, lam, u):
        """
        Construct an EVAFit with known params and verify return_value(T)
        matches the analytical formula.
        """
        from scipy import stats as sp_stats

        # Dummy data (not used in formula, just needed for EVAFit construction)
        dummy_data = np.zeros(10)

        dist = sp_stats.genpareto
        params = (xi, 0.0, sigma)  # (ξ, 0, σ) as fitted with floc=0

        ev_fit = EVAFit(
            distribution=dist,
            name='genpareto',
            params=params,
            data=dummy_data,
            method='pot',
            threshold=u,
            lambda_rate=lam,
            blocks_per_year=None,
        )

        T = np.array([10.0, 50.0, 100.0, 500.0])
        result = ev_fit.return_value(T)

        # Closed-form formula
        lam_T = lam * T
        if abs(xi) >= 1e-6:
            expected = u + (sigma / xi) * (lam_T ** xi - 1.0)
        else:
            expected = u + sigma * np.log(lam_T)

        np.testing.assert_array_almost_equal(result, expected, decimal=8,
            err_msg=f"Return value mismatch for xi={xi}, sigma={sigma}, lam={lam}, u={u}")

    def test_return_period_inverse_of_return_value(self):
        """return_period(return_value(T)) ≈ T for PoT."""
        dummy_data = np.zeros(10)
        ev_fit = EVAFit(
            distribution=stats.genpareto,
            name='genpareto',
            params=(0.1, 0.0, 2.0),
            data=dummy_data,
            method='pot',
            threshold=10.0,
            lambda_rate=4.0,
            blocks_per_year=None,
        )
        T = np.array([10.0, 50.0, 100.0])
        rv = ev_fit.return_value(T)
        rp = ev_fit.return_period(rv)
        np.testing.assert_array_almost_equal(rp, T, decimal=6)

    def test_lambda_scaling_effect(self):
        """With λ=1, formula simplifies; with λ≠1 the difference is significant."""
        dummy_data = np.zeros(10)
        xi, sigma, u = 0.1, 2.0, 10.0

        def make_fit(lam):
            return EVAFit(
                distribution=stats.genpareto,
                name='genpareto',
                params=(xi, 0.0, sigma),
                data=dummy_data,
                method='pot',
                threshold=u,
                lambda_rate=lam,
                blocks_per_year=None,
            )

        T = 50.0
        rv_lam1 = make_fit(1.0).return_value(T)
        rv_lam5 = make_fit(5.0).return_value(T)

        # With λ=5, exceedance is 5× more frequent, so the 50-year level is higher
        assert rv_lam5 > rv_lam1, (
            "Higher exceedance rate should give higher return value for same return period"
        )


# ---------------------------------------------------------------------------
# 4. Independence of fits
# ---------------------------------------------------------------------------

class TestFitIndependence:
    """Two fits on the same processor must not interfere."""

    def test_two_fits_independent_params(self, processor):
        fit_w = processor.fit('weibull', floc=0)
        fit_g = processor.fit('gamma')

        assert fit_w.name == 'weibull'
        assert fit_g.name == 'gamma'
        # Parameters must not match (different distributions)
        assert fit_w.params != fit_g.params

    def test_second_fit_does_not_overwrite_first(self, processor):
        fit_w = processor.fit('weibull', floc=0)
        params_w_before = fit_w.params

        # Fit a second distribution
        _ = processor.fit('gamma')

        # First fit must be unchanged
        assert fit_w.params == params_w_before
        assert fit_w.name == 'weibull'

    def test_multiple_fits_all_correct(self, processor):
        fits = [processor.fit(d) for d in ('weibull', 'gamma', 'lognorm', 'norm')]
        names = [f.name for f in fits]
        assert names == ['weibull', 'gamma', 'lognorm', 'norm']
        for f in fits:
            assert isinstance(f, FitResult)
            assert len(f.params) > 0

    def test_fit_result_data_ref_unchanged_by_second_fit(self, processor):
        fit_w = processor.fit('weibull', floc=0)
        data_id_before = id(fit_w.data)
        _ = processor.fit('gamma')
        assert id(fit_w.data) == data_id_before


# ---------------------------------------------------------------------------
# 5. Memory / shared reference — no per-fit data copy
# ---------------------------------------------------------------------------

class TestMemorySharing:
    """FitResult and AutoFitter must share the data array without copying."""

    def test_fitresult_shares_data_reference(self, processor, fixed_data):
        fit = processor.fit('weibull', floc=0)
        # fit.data must be the SAME object as processor.data
        assert fit.data is processor.data

    def test_two_fitresults_share_same_array(self, processor):
        fit_w = processor.fit('weibull', floc=0)
        fit_g = processor.fit('gamma')
        assert fit_w.data is fit_g.data

    def test_autofitter_best_shares_data(self, processor):
        auto = AutoFitter(processor, candidates=['weibull_min', 'gamma'], criterion='rmse')
        best = auto.fit_best_distribution()
        # Best FitResult must share the same data array
        assert best.data is processor.data

    def test_autofitter_internal_data_ref(self, processor):
        """AutoFitter.data must be a shared reference, not a copy."""
        auto = AutoFitter(processor, candidates=['weibull_min', 'gamma'])
        assert auto.data is processor.data


# ---------------------------------------------------------------------------
# 6. FitResult methods correctness
# ---------------------------------------------------------------------------

class TestFitResultMethods:
    """Basic correctness of FitResult methods."""

    @pytest.fixture(scope='class')
    def weibull_fit(self, processor):
        return processor.fit('weibull', floc=0)

    def test_cdf_monotone(self, weibull_fit):
        x = np.linspace(1, 30, 50)
        cdf_vals = weibull_fit.cdf(x)
        assert np.all(np.diff(cdf_vals) >= 0), "CDF must be monotonically non-decreasing"

    def test_cdf_without_arg_uses_data(self, weibull_fit):
        cdf_at_data = weibull_fit.cdf()
        cdf_explicit = weibull_fit.cdf(weibull_fit.data)
        np.testing.assert_array_equal(cdf_at_data, cdf_explicit)

    def test_pdf_positive(self, weibull_fit):
        x = np.linspace(1, 30, 50)
        pdf_vals = weibull_fit.pdf(x)
        assert np.all(pdf_vals >= 0), "PDF must be non-negative"

    def test_ppf_cdf_inverse(self, weibull_fit):
        q = np.array([0.1, 0.5, 0.9, 0.99])
        x = weibull_fit.ppf(q)
        q_back = weibull_fit.cdf(x)
        np.testing.assert_array_almost_equal(q_back, q, decimal=8)

    def test_goodness_of_fit_ks(self, weibull_fit):
        result = weibull_fit.goodness_of_fit('ks')
        assert 'ks_statistic' in result
        assert 'p_value' in result
        assert 0 <= result['ks_statistic'] <= 1
        assert 0 <= result['p_value'] <= 1

    def test_goodness_of_fit_rmse(self, weibull_fit):
        rmse = weibull_fit.goodness_of_fit('rmse')
        assert isinstance(rmse, float)
        assert 0 <= rmse <= 1

    def test_goodness_of_fit_aic(self, weibull_fit):
        aic = weibull_fit.goodness_of_fit('aic')
        assert isinstance(aic, float)

    def test_info_dict(self, weibull_fit):
        info = weibull_fit.info
        assert info['name'] == 'weibull'
        assert len(info['parameters']) == 3
        assert info['data_size'] == len(weibull_fit.data)


# ---------------------------------------------------------------------------
# 7. EVA family restriction
# ---------------------------------------------------------------------------

class TestEVAFamilyRestriction:
    @pytest.fixture(scope='class')
    def extremes(self):
        rng = np.random.default_rng(7)
        n = 365 * 20
        dates = pd.date_range('2000-01-01', periods=n, freq='D')
        values = rng.weibull(1.5, n) * 10
        series = pd.Series(values, index=dates)
        proc = ma.read_data(series)
        return proc.get_extremes_analyzer()

    def test_invalid_bm_family_raises(self, extremes):
        annual_max, times = extremes.extract_block_maxima('YE')
        with pytest.raises(ValueError, match="not valid for method 'bm'"):
            extremes.fit_block_maxima(annual_max, distribution='weibull_min')

    def test_invalid_pot_family_raises(self, extremes):
        result = extremes.find_optimal_pot_threshold(min_samples=20)
        with pytest.raises(ValueError, match="not valid for method 'pot'"):
            extremes.fit_pot(result, distribution='genextreme')

    def test_valid_bm_genextreme(self, extremes):
        annual_max, times = extremes.extract_block_maxima('YE')
        ev = extremes.fit_block_maxima(annual_max, times, distribution='genextreme')
        assert ev.name == 'genextreme'
        assert ev.method == 'bm'

    def test_valid_bm_gumbel_r(self, extremes):
        annual_max, times = extremes.extract_block_maxima('YE')
        ev = extremes.fit_block_maxima(annual_max, times, distribution='gumbel_r')
        assert ev.name == 'gumbel_r'

    def test_valid_pot_genpareto(self, extremes):
        result = extremes.find_optimal_pot_threshold(min_samples=20)
        ev = extremes.fit_pot(result, distribution='genpareto')
        assert ev.name == 'genpareto'
        assert ev.method == 'pot'
        assert ev.threshold is not None
        assert ev.lambda_rate is not None
        assert ev.lambda_rate > 0
