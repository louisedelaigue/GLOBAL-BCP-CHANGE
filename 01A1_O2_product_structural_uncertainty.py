"""
01C - Structural uncertainty of the global-mean AOU trend from the difference
between two oxygen products (used by 01A and 01B)

GOBAI-O2 v2.3 and ML4O2 (Ito et al., 2024) are compared over their common
period (2004-2020), 0-1000 m:
  - both products on the GOBAI horizontal grid and ML4O2 depth levels
  - common footprint (valid in both products, Mediterranean excluded)
  - area-weighted global-mean time series at each depth
  - ML4O2 provides O2 only, so its AOU uses GOBAI O2sat (same T/S); the AOU
    trend difference therefore equals minus the O2 trend difference
  - trends: deseasonalised OLS, standard errors adjusted for lag-1
    autocorrelation (n_eff = n(1-r1)/(1+r1))

Structural uncertainty (1 sigma) of the AOU trend:
    sigma_struct_AOU = |trend_GOBAI - trend_ML4O2| / sqrt(2)
(the difference of two estimates with equal error sigma has spread sqrt(2) sigma)

Output: data/processing_steps/01A1_O2_product_structural_uncertainty.nc
"""

import warnings
import numpy as np
import pandas as pd
import xarray as xr
import gsw
from scipy.stats import t as student_t

warnings.filterwarnings('ignore')

try:
    import koolstof as ks
    ks.parameterisations.aou_GG92
    HAVE_KS = True
except (ImportError, AttributeError):
    HAVE_KS = False
    print("koolstof not found: using gsw.O2sol_SP_pt for O2sat")

# =============================================================================
# 1 - Settings
# =============================================================================

GOBAI_FILE = "data/CONTENT_DIC_v2-3.nc"
ITO_FILE = "data/o2_NN_ship+Argo_ensmean.nc"
OUT_FILE = "data/processing_steps/01A1_O2_product_structural_uncertainty.nc"

PERIOD = ("2004-01-01", "2020-12-31")   # common period of both products
ZMAX = 1000                             # ML4O2 maximum depth (m)
MIN_VALID_FRAC = 0.9                    # pixel needs >= 90% valid months

# =============================================================================
# 2 - Helpers
# =============================================================================

def mask_mediterranean(lat, lon):
    lon180 = ((lon + 180) % 360) - 180
    return (lat >= 30) & (lat <= 46) & (lon180 >= -6) & (lon180 <= 37)

def deseason(da):
    return da.groupby("time.month") - da.groupby("time.month").mean("time")

def t_years(da):
    return ((da["time"] - da["time"].isel(time=0)) / np.timedelta64(1, "D")).values / 365.25

def trend_ar1(y, t):
    """OLS trend along axis 0, SE adjusted for lag-1 autocorrelation (NaN-aware)."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float).reshape((-1,) + (1,) * (y.ndim - 1))
    m = np.isfinite(y)
    n = m.sum(0).astype(float)
    tt = np.where(m, t, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        dt = tt - np.nanmean(tt, 0)
        ybar = np.nanmean(y, 0)
        sxx = np.nansum(dt ** 2, 0)
        slope = np.nansum(dt * (y - ybar), 0) / sxx
        resid = y - ybar - slope * dt
        r1 = np.clip(np.nansum(resid[1:] * resid[:-1], 0) / np.nansum(resid ** 2, 0), 0, 0.99)
        neff = np.maximum(n * (1 - r1) / (1 + r1), 3.0)
        se = np.sqrt(np.nansum(resid ** 2, 0) / (n - 2) / sxx * (n - 2) / (neff - 2))
        p = 2 * student_t.sf(np.abs(slope / se), neff - 2)
    bad = n < 24
    return tuple(np.where(bad, np.nan, a) for a in (slope, se, p, neff))

def o2sat(T, SP, p, lon, lat):
    """O2 solubility (umol kg-1), Garcia & Gordon (1992), as in 01A."""
    SA = gsw.SA_from_SP(SP, p, lon, lat)
    theta = gsw.pt0_from_t(SA, T, p)
    if HAVE_KS:
        return ks.parameterisations.aou_GG92(oxygen=0 * T, temperature=theta, salinity=SP)[0]
    return gsw.O2sol_SP_pt(SP, theta)

# =============================================================================
# 3 - Load data
# =============================================================================

print("Loading GOBAI-O2 (0-1250 dbar, 2004-2020)...")
gob = xr.open_dataset(GOBAI_FILE)
g = gob[["oxy", "temp", "sal"]].sel(pres=slice(0, ZMAX * 1.25), time=slice(*PERIOD))
g = g.transpose("time", "pres", "lat", "lon").load()

print("Opening ML4O2...")
ito = xr.open_dataset(ITO_FILE).sel(depth=slice(0, ZMAX), time=slice(*PERIOD))

# ML4O2 on the GOBAI horizontal grid (both 1 degree, centred on .5)
g_lon = g["lon"].values
ito_lon = ito["lon"].values % 360
ito_lon = np.where(ito_lon < g_lon.min(), ito_lon + 360, ito_lon)
ito = ito.assign_coords(lon=ito_lon).sortby("lon")
ito = ito.sel(lat=g["lat"], lon=g["lon"], method="nearest")
ito = ito.assign_coords(lat=g["lat"].values, lon=g_lon)

depth = ito["depth"].values.astype(float)
pres_eq = gsw.p_from_z(-depth, 0.0)   # pressure equivalent of ML4O2 depths

lat2d, lon2d = xr.broadcast(g["lat"], g["lon"])
med = mask_mediterranean(lat2d, lon2d).values
w2d = np.cos(np.deg2rad(lat2d.values))

def gmean(arr, mask):
    """Area-weighted mean over (lat, lon) for arr (time, lat, lon)."""
    w = np.where(mask, w2d, 0.0)[None, :, :] * np.isfinite(arr)
    return np.nansum(arr * w, axis=(1, 2)) / w.sum(axis=(1, 2))

# Align ML4O2 months (1st of month) with GOBAI months (mid-month)
ym_g = pd.DatetimeIndex(g["time"].values).to_period("M")
ym_i = pd.DatetimeIndex(ito["time"].values).to_period("M")
common = ym_g.intersection(ym_i)
ig = np.where(ym_g.isin(common))[0]
ii = np.where(ym_i.isin(common))[0]
time_c = g["time"].values[ig]

# =============================================================================
# 4 - Global-mean time series at each ML4O2 depth level
# =============================================================================

gm_aou_g, gm_aou_i, footprint_frac = [], [], []
for k in range(depth.size):
    print(f"Level {k + 1}/{depth.size}: {depth[k]:.0f} m")
    gk = g.interp(pres=pres_eq[k]).isel(time=ig)
    sat = o2sat(gk["temp"], gk["sal"], pres_eq[k], gk["lon"], gk["lat"])
    sat = np.asarray(sat.transpose("time", "lat", "lon").values if hasattr(sat, "transpose") else sat)
    aou_g = sat - gk["oxy"].values
    aou_i = sat - ito["o2"].isel(depth=k, time=ii).transpose("time", "lat", "lon").values

    valid_g = np.isfinite(aou_g).mean(0) >= MIN_VALID_FRAC
    valid_i = np.isfinite(aou_i).mean(0) >= MIN_VALID_FRAC
    fp = valid_g & valid_i & ~med
    footprint_frac.append(fp.sum() / max((valid_g & ~med).sum(), 1))

    gm_aou_g.append(gmean(aou_g, fp))
    gm_aou_i.append(gmean(aou_i, fp))

# =============================================================================
# 5 - Trends and structural uncertainty
# =============================================================================

def fit(values):
    da = xr.DataArray(np.array(values).T, dims=("time", "depth"),
                      coords={"time": time_c, "depth": depth})
    return trend_ar1(deseason(da).values, t_years(da))

tr_g, se_g, p_g, neff_g = fit(gm_aou_g)
tr_i, se_i, p_i, neff_i = fit(gm_aou_i)
diff = tr_g - tr_i

out = xr.Dataset(
    {
        "AOU_trend_GOBAI": ("depth", tr_g),
        "AOU_trend_GOBAI_se": ("depth", se_g),
        "AOU_trend_ML4O2": ("depth", tr_i),
        "AOU_trend_ML4O2_se": ("depth", se_i),
        "AOU_trend_diff_GOBAI_minus_ML4O2": ("depth", diff),
        "sigma_struct_AOU": ("depth", np.abs(diff) / np.sqrt(2)),
        "footprint_fraction": ("depth", np.array(footprint_frac)),
    },
    coords={"depth": depth, "pres": ("depth", pres_eq)},
)
out.attrs.update({
    "units": "umol kg-1 yr-1",
    "period": f"{PERIOD[0]} to {PERIOD[1]}",
    "description": "Global-mean (area-weighted, common footprint, no Mediterranean) AOU trends "
                   "from GOBAI-O2 v2.3 and ML4O2 (Ito et al., 2024). ML4O2 AOU uses GOBAI O2sat. "
                   "Deseasonalised OLS, lag-1 autocorrelation-adjusted SE. "
                   "sigma_struct_AOU = |GOBAI - ML4O2| / sqrt(2).",
})
out.to_netcdf(OUT_FILE)
print(f"\nSaved: {OUT_FILE}")

# =============================================================================
# 6 - Summary
# =============================================================================

edges = np.concatenate([[0], 0.5 * (depth[1:] + depth[:-1]), [depth[-1]]])
dz = np.diff(edges)

def layer_mean(x, z0, z1):
    s = (depth >= z0) & (depth <= z1) & np.isfinite(x)
    return (x[s] * dz[s]).sum() / dz[s].sum()

print("\nThickness-weighted means, 2004-2020 (umol kg-1 yr-1):")
print(f"  {'':22s} {'0-1000 m':>10s} {'100-1000 m':>11s}")
for lab, x in [("AOU trend GOBAI-O2", tr_g), ("AOU trend ML4O2", tr_i),
               ("sigma_struct_AOU", out["sigma_struct_AOU"].values)]:
    print(f"  {lab:22s} {layer_mean(x, 0, 1000):+10.3f} {layer_mean(x, 100, 1000):+11.3f}")
print(f"\nsigma_struct_AOU at deepest level ({depth[-1]:.0f} m): {out['sigma_struct_AOU'].values[-1]:.3f}"
      " (applied below this depth in 01A)")
print(f"Footprint retained (min over depth): {min(footprint_frac):.2f}")
print("Processing complete!")
