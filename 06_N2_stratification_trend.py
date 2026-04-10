"""
06_N2_stratification_trend.py

Computes the depth-weighted mean Brunt-Vaisala frequency (N^2) over 0-200 m
and its linear trend at each grid cell.

Vectorised implementation: loops over time steps but computes N^2 across all
(lat, lon) simultaneously within each step, avoiding the slow profile-by-
profile apply_ufunc approach.

Input:  data/CONTENT_DIC_v2-3.nc
Output: data/processing_steps/06_N2_stratification_trend.nc

Variables saved:
  - trend_N2bar_0_200_peryr : linear trend of annual-mean N^2 (s^-2 yr^-1)
  - N2bar_0_200_mean        : time-mean N^2 over 0-200 m (s^-2)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
import gsw
import os

# =============================================================================
# 1 - Paths
# =============================================================================
INPUT_PATH = "data/CONTENT_DIC_v2-3.nc"
OUTPUT_DIR = "data/processing_steps"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "06_N2_stratification_trend.nc")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 2 - Load dataset
# =============================================================================
print("Loading dataset...")
ds = xr.open_dataset(INPUT_PATH)
print(f"  Dimensions: {dict(ds.dims)}")

sal = ds["sal"].values    # (time, pres, lat, lon)
temp = ds["temp"].values
pres = ds["pres"].values  # (pres,)
lon = ds["lon"].values    # (lon,)
lat = ds["lat"].values    # (lat,)
time = ds["time"].values

n_time, n_pres, n_lat, n_lon = sal.shape
print(f"  Shape: ({n_time}, {n_pres}, {n_lat}, {n_lon})")

# =============================================================================
# 3 - Prepare broadcasting arrays for gsw
# =============================================================================
# For a single time step, arrays have shape (pres, lat, lon).
# gsw.Nsquared operates along axis=0 (pressure).

pres_3d = pres[:, np.newaxis, np.newaxis]                          # (pres, 1, 1)
lon_3d = lon[np.newaxis, np.newaxis, :]                            # (1, 1, lon)
lat_3d = lat[np.newaxis, :, np.newaxis]                            # (1, lat, 1)

# Pressure midpoints and layer thicknesses for averaging
p_mid_1d = 0.5 * (pres[:-1] + pres[1:])          # (pres-1,)
dp = np.diff(pres)                                 # (pres-1,)

# Mask for midpoints within 0-200 m
mask_200 = p_mid_1d <= 200.0
dp_200 = dp[mask_200]                              # layer thicknesses in band
dp_bc = dp_200[:, np.newaxis, np.newaxis]           # (n_band, 1, 1)

n_band = int(np.sum(mask_200))
print(f"  N^2 levels in 0-200 m band: {n_band}")

# =============================================================================
# 4 - Compute depth-weighted mean N^2 (0-200 m) at each time step
# =============================================================================
print("Computing N^2 mean (0-200 m) for each time step...")

N2_mean_all = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float64)

for t in range(n_time):
    if (t + 1) % 24 == 0 or t == 0:
        print(f"  Time step {t + 1}/{n_time}")

    sal_t = sal[t]   # (pres, lat, lon)
    temp_t = temp[t]

    # Absolute Salinity and Conservative Temperature
    SA = gsw.SA_from_SP(sal_t, pres_3d, lon_3d, lat_3d)
    CT = gsw.CT_from_t(SA, temp_t, pres_3d)

    # Brunt-Vaisala frequency squared along pressure axis
    N2_t, _ = gsw.Nsquared(SA, CT, pres_3d, lat=lat_3d, axis=0)
    # N2_t shape: (pres-1, lat, lon)

    # Select 0-200 m band
    N2_band = N2_t[mask_200, :, :]   # (n_band, lat, lon)

    # Depth-weighted mean, handling NaNs
    valid = np.isfinite(N2_band)
    N2_weighted = np.where(valid, N2_band * dp_bc, 0.0)
    dp_valid = np.where(valid, dp_bc, 0.0)

    sum_weights = np.sum(dp_valid, axis=0)
    with np.errstate(invalid="ignore"):
        N2_mean_all[t] = np.where(
            sum_weights > 0,
            np.sum(N2_weighted, axis=0) / sum_weights,
            np.nan,
        )

print("  N^2 computation complete.")

# =============================================================================
# 5 - Wrap into xarray and compute annual means
# =============================================================================
print("Computing annual means...")

N2_da = xr.DataArray(
    N2_mean_all,
    coords={"time": ds["time"], "lat": ds["lat"], "lon": ds["lon"]},
    dims=["time", "lat", "lon"],
    name="N2bar_0_200",
)

N2_annual = N2_da.groupby("time.year").mean("time", skipna=True)

# =============================================================================
# 6 - Linear trend per grid cell
# =============================================================================
print("Fitting linear trends...")

fit = N2_annual.polyfit(dim="year", deg=1, skipna=True)
trend_N2 = fit["polyfit_coefficients"].sel(degree=1)
trend_N2.name = "trend_N2bar_0_200_peryr"

# Time-mean N^2 (for reference / plotting)
N2_mean = N2_da.mean(dim="time", skipna=True)
N2_mean.name = "N2bar_0_200_mean"

# =============================================================================
# 7 - Save
# =============================================================================
out = xr.Dataset({
    "trend_N2bar_0_200_peryr": trend_N2,
    "N2bar_0_200_mean": N2_mean,
})

# Add metadata
out["trend_N2bar_0_200_peryr"].attrs = {
    "long_name": "Linear trend of depth-weighted mean N^2 (0-200 m)",
    "units": "s^-2 yr^-1",
    "depth_range": "0-200 m",
    "method": "polyfit on annual means of pressure-weighted N^2",
}
out["N2bar_0_200_mean"].attrs = {
    "long_name": "Time-mean depth-weighted N^2 (0-200 m)",
    "units": "s^-2",
    "depth_range": "0-200 m",
}

print(f"Saving: {OUTPUT_PATH}")
out.to_netcdf(OUTPUT_PATH)
print("Done.")
