"""
09_OCIM_ideal_age.py

Computes band-mean ideal age from the OCIM (Ocean Circulation Inverse Model)
product on its native grid.

Bands:
  - 0-200 m
  - 200-1000 m
  - 1000-2000 m

Input:
  - data/115838.nc (OCIM ideal age; download from OCIM product)

Output:
  - data/processing_steps/09_OCIM_ideal_age.nc (on native OCIM grid)
"""
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import xarray as xr

# =============================================================================
# 1 - Paths
# =============================================================================
OCIM_PATH = "data/115838.nc"
OUTPUT_DIR = "data/processing_steps"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "09_OCIM_ideal_age.nc")

os.makedirs(OUTPUT_DIR, exist_ok=True)

BANDS = [(0, 200), (200, 1000), (1000, 2000)]

print("=" * 60)
print("OCIM ideal age band means (native grid)")
print("=" * 60)

# =============================================================================
# 2 - Load and prepare OCIM ideal age
# =============================================================================
print("Loading OCIM dataset...")
dsO = xr.open_dataset(OCIM_PATH)

# Rename dimensions for clarity
age = dsO["ideal_age"].rename({
    "Longitude": "lon",
    "Latitude": "lat",
    "Depth": "Depth",
})
age = age.assign_coords({
    "lon": ("lon", dsO["xt"].values),
    "lat": ("lat", dsO["yt"].values),
    "Depth": ("Depth", dsO["zt"].values),
})

print(f"  OCIM grid: {age.lat.size} lat x {age.lon.size} lon x {age.Depth.size} depth")
print(f"  Longitude range: {age.lon.min().values:.1f} to {age.lon.max().values:.1f}°")

# =============================================================================
# 3 - Compute depth-weighted band means
# =============================================================================
dz = age["Depth"].diff("Depth")
dz3 = dz.broadcast_like(age.isel(Depth=slice(1, None)))

def ideal_age_mean_band(age_da, dz3, zmin, zmax):
    """Depth-weighted mean ideal age within a depth band."""
    band = age_da.sel(Depth=slice(zmin, zmax))
    band_mid = band.isel(Depth=slice(1, None))
    w_mid = dz3.sel(Depth=band_mid["Depth"])
    return (band_mid * w_mid).sum("Depth", skipna=True) / w_mid.sum("Depth")

print("Computing ideal age band means...")
out = xr.Dataset(coords={"lat": age["lat"], "lon": age["lon"]})

for zmin, zmax in BANDS:
    tag = f"{zmin}_{zmax}"
    name = f"ideal_age_mean_{tag}"
    out[name] = ideal_age_mean_band(age, dz3, zmin, zmax)
    out[name].attrs = {
        "long_name": f"Depth-weighted mean ideal age ({zmin}-{zmax} m)",
        "units": "years",
        "source": "OCIM ideal age product",
    }
    print(f"  {name}: done")

# =============================================================================
# 4 - Save
# =============================================================================
if os.path.exists(OUTPUT_PATH):
    os.remove(OUTPUT_PATH)

print(f"Saving: {OUTPUT_PATH}")
out.to_netcdf(OUTPUT_PATH)
print("Done.")