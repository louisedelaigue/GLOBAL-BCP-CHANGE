"""Linear trends with Monte Carlo propagation of measurement uncertainty."""
import numpy as np

import config


def ols_slope(y, t):
    """OLS slope along the last axis of `y`, ignoring NaNs (same as xarray polyfit skipna=True)."""
    w = np.isfinite(y)
    n = w.sum(-1)
    tt = np.where(w, t, 0.0)
    yy = np.where(w, y, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        tm = tt.sum(-1) / n
        ym = yy.sum(-1) / n
        num = (w * (tt - tm[..., None]) * (yy - ym[..., None])).sum(-1)
        den = (w * (tt - tm[..., None]) ** 2).sum(-1)
        s = num / den
    s[n < 3] = np.nan
    return s


def monte_carlo_slopes(ds, variables, rng, n_iter=config.N_MONTE_CARLO):
    """Trend (per year) of each variable in each Monte Carlo iteration.

    Gaussian noise with the SPOTS measurement uncertainty is added independently to every
    interpolated value, as in the old XX_02_pres_calculate_change.py scripts.
    Returns {variable: array (n_iter, CTDPRS)}.
    """
    t = ds["datenum"].values.astype(float)
    out = {}
    for v in variables:
        y = ds[v].values
        sigma = config.MEASUREMENT_UNCERTAINTY[v]
        out[v] = np.array([ols_slope(y + rng.normal(0, sigma, y.shape), t)
                           for _ in range(n_iter)]) * 365.25
    return out


def trend_ar1(y, t):
    """OLS trend along axis 0 with standard error adjusted for lag-1 autocorrelation.

    n_eff = n (1 - r1) / (1 + r1). Same as trend_ar1() in GLOBAL-BCP-CHANGE-2.3
    01A1_O2_product_structural_uncertainty.py. Returns (slope, se, n_eff) in units of y per unit t.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    m = np.isfinite(y)
    y, t = y[m], t[m]
    n = y.size
    if n < 3:
        return np.nan, np.nan, np.nan
    dt = t - t.mean()
    sxx = np.sum(dt ** 2)
    slope = np.sum(dt * (y - y.mean())) / sxx
    resid = y - y.mean() - slope * dt
    r1 = np.clip(np.sum(resid[1:] * resid[:-1]) / np.sum(resid ** 2), 0, 0.99)
    neff = max(n * (1 - r1) / (1 + r1), 3.0)
    se = np.sqrt(np.sum(resid ** 2) / (n - 2) / sxx * (n - 2) / (neff - 2))
    return slope, se, neff
