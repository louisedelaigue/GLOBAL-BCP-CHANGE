import sys
import warnings
import os
import re
import numpy as np
import xarray as xr
import gsw
import koolstof as ks
import matplotlib.dates as mdates

import geopandas as gpd
from shapely.geometry import Point
from shapely.prepared import prep

warnings.filterwarnings("ignore")

print("Starting script...")

# =============================================================================
# 2 - Load Dataset
# =============================================================================
print("Loading dataset...")
ds = xr.open_dataset("data/CONTENT_DIC_v2-3.nc")
print("Dataset loaded.")

print("Loading alkalinity dataset...")
TA = xr.open_dataset("data/CONTENT_AT_v2-3.nc")
ds["TA"] = TA["AT"]
ds["TA_sigma"] = TA["AT_sigma"]
print("Alkalinity data loaded.")

# =============================================================================
# Make sure lon matches shapefile convention (-180..180) for point-in-polygon
# =============================================================================
if ds["lon"].max() > 180:
    print("Converting longitude from 0..360 to -180..180 for shapefile join...")
    ds = ds.assign_coords(lon=(((ds["lon"] + 180) % 360) - 180)).sortby("lon")

# =============================================================================
# Compute Absolute Salinity and Potential Temperature
# =============================================================================
print("Calculating absolute salinity and potential temperature...")
ds["salinity_absolute"] = gsw.conversions.SA_from_SP(ds["sal"], ds["pres"], ds["lon"], ds["lat"])
ds["theta"] = gsw.conversions.pt0_from_t(ds["salinity_absolute"], ds["temp"], ds["pres"])
ds["aou"] = ks.parameterisations.aou_GG92(
    oxygen=ds["oxy"], temperature=ds["theta"], salinity=ds["sal"]
)[0]
print("Salinity and temperature calculations complete.")

# Rename Variables
ds = ds.rename(
    {
        "oxy": "oxygen",
        "temp": "temperature",
        "sal": "salinity",
        "aou": "AOU",
        "uncer": "AOU_uncertainty",
        "CT": "DIC",
        "CT_sigma": "DIC_uncertainty",
        "TA_sigma": "TA_uncertainty",
    }
)

ds = ds[["AOU", "AOU_uncertainty", "DIC", "DIC_uncertainty", "TA", "TA_uncertainty"]]

# =============================================================================
# 3 - Convert Time to Matplotlib Numeric Format (Before Analysis)
# =============================================================================
print("Converting time to numeric format...")
ds = ds.assign_coords(time=("time", mdates.date2num(ds["time"].values)))  # days
print("Time conversion complete.")

# =============================================================================
# 4 - Build region attribution using ONLY GeoPandas (point-in-polygon)
#     Using the *exact* polygons/names from the shapefile.
# =============================================================================
print("Loading shapefile with geopandas...")
gdf = gpd.read_file("/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/data/ocean_mask/goas_v01.shp")

# Ensure geographic CRS
if gdf.crs is None:
    print("Shapefile CRS is None; assuming EPSG:4326 (lon/lat).")
    gdf = gdf.set_crs("EPSG:4326")
else:
    try:
        if str(gdf.crs).lower() not in ["epsg:4326", "wgs84", "crs84"]:
            print(f"Reprojecting shapefile from {gdf.crs} to EPSG:4326...")
            gdf = gdf.to_crs("EPSG:4326")
    except Exception:
        print("Could not interpret shapefile CRS cleanly; reprojecting to EPSG:4326...")
        gdf = gdf.to_crs("EPSG:4326")

if "name" not in gdf.columns:
    raise ValueError("Expected a 'name' column in shapefile, but it wasn't found.")

# (Optional but recommended) keep only the ocean-basin polygons you listed.
# Comment this out if you truly want every feature (Mediterranean, Baltic, etc.) as separate output files.
keep_names = {
    "Southern Ocean",
    "South Atlantic Ocean",
    "North Atlantic Ocean",
    "South Pacific Ocean",
    "North Pacific Ocean",
    "Indian Ocean",
    "Arctic Ocean",
}
gdf = gdf[gdf["name"].isin(keep_names)].copy()

if len(gdf) == 0:
    raise ValueError("No shapefile rows match keep_names. Check the spelling/capitalization in gdf['name'].")

# Prepare geometries for faster contains checks
gdf = gdf.reset_index(drop=True)
prepared = [(row["name"], prep(row.geometry)) for _, row in gdf.iterrows()]

region_names = list(gdf["name"].astype(str))
region_id_map = {name: i for i, name in enumerate(region_names)}
region_id_to_name = {i: name for name, i in region_id_map.items()}

print("Assigning region_id to each (lat, lon) grid point (GeoPandas/Shapely only)...")
lats = ds["lat"].values
lons = ds["lon"].values

# Create 2D lon/lat mesh
lon2d, lat2d = np.meshgrid(lons, lats)

# Output array of region ids; -1 means "no region / unmatched"
region_id_2d = np.full(lon2d.shape, -1, dtype=np.int16)

# Point-in-polygon assignment
for i in range(lat2d.shape[0]):
    for j in range(lat2d.shape[1]):
        pt = Point(float(lon2d[i, j]), float(lat2d[i, j]))
        found = -1
        for rname, pgeom in prepared:
            if pgeom.contains(pt):
                found = region_id_map[rname]
                break
        region_id_2d[i, j] = found

# Attach as coordinates (new region field over lat/lon)
ds = ds.assign_coords(region_id=(("lat", "lon"), region_id_2d))

# Add a string region_name coordinate too (helpful for debugging / metadata)
# xarray doesn't love 2D object arrays; we store region_name mapping in attrs instead.
ds["region_id"].attrs["description"] = "Ocean region id from goas_v01.shp (point-in-polygon, EPSG:4326)"
ds["region_id"].attrs["mapping"] = region_id_to_name
print("Region attribution complete. region_id mapping:", ds["region_id"].attrs["mapping"])

# =============================================================================
# 5 - Monte Carlo Uncertainty Calculation (Including Constant Errors)
# =============================================================================
num_iterations = 1000
print(f"Running Monte Carlo with {num_iterations} iterations per region...")

# Define Constant Mean Values
sp_constant_aou_mean = 117 / -170
cp_constant_mean = 16 / -170

# Define Constant Uncertainties
sigma_sp = 0.092   # Uncertainty for soft tissue pump
sigma_cp = 0.0081  # Uncertainty for carbonate pump

def compute_slope(ds_in: xr.Dataset, var: str) -> xr.DataArray:
    """
    Compute linear trend vs time (time in matplotlib days) and convert to per-year.
    """
    try:
        polyfit = ds_in[var].polyfit(dim="time", deg=1, skipna=True)
        slope = polyfit.polyfit_coefficients.sel(degree=1) * 365.25  # per day -> per year
        return slope
    except Exception as e:
        print(f"Error computing slope for {var}: {e}")
        return xr.zeros_like(ds_in[var].isel(time=0))

def run_monte_carlo_for_ds(ds_in: xr.Dataset, niter: int, rng: np.random.Generator):
    """
    Runs Monte Carlo, returns mean_grid(lat,lon) and std_grid(lat,lon)
    for the rates/terms.
    """
    mean_ds = xr.Dataset()
    var_ds = xr.Dataset()

    for i in range(1, niter + 1):
        print(f"  MC iter {i}/{niter}")
        mc_sample = ds_in.copy(deep=True)

        mc_sample["DIC"] += rng.normal(0, ds_in["DIC_uncertainty"])
        mc_sample["TA"]  += rng.normal(0, ds_in["TA_uncertainty"])
        mc_sample["AOU"] += rng.normal(0, ds_in["AOU_uncertainty"])

        # Perturb constants
        sp_constant_aou = rng.normal(sp_constant_aou_mean, sigma_sp)
        cp_constant = rng.normal(cp_constant_mean, sigma_cp)

        dic_rate = compute_slope(mc_sample, "DIC")
        ta_rate  = compute_slope(mc_sample, "TA")
        aou_rate = compute_slope(mc_sample, "AOU")

        soft_pump_aou_change = -sp_constant_aou * aou_rate
        carb_pump_change = 0.5 * (ta_rate - cp_constant * aou_rate)
        co2_anth_aou_change = dic_rate - (soft_pump_aou_change + carb_pump_change)

        iteration_ds = xr.Dataset(
            {
                "DIC_rate": dic_rate,
                "TA_rate": ta_rate,
                "AOU_rate": aou_rate,
                "soft_pump_aou_change": soft_pump_aou_change,
                "carb_pump_change": carb_pump_change,
                "co2_anth_aou_change": co2_anth_aou_change,
            }
        )

        if i == 1:
            mean_ds = iteration_ds
            var_ds = iteration_ds * 0
        else:
            delta = iteration_ds - mean_ds
            new_mean_ds = mean_ds + delta / i  # Welford
            var_ds = ((i - 1) * var_ds + delta * (iteration_ds - new_mean_ds)) / i
            mean_ds = new_mean_ds

    std_ds = np.sqrt(var_ds)
    return mean_ds, std_ds

def area_weighted_global(mean_field: xr.Dataset, std_field: xr.Dataset, lat: xr.DataArray):
    """
    Area-weighted mean over lat/lon and uncertainty propagation in quadrature.
    """
    weights = np.cos(np.deg2rad(lat))
    mean_out = mean_field.weighted(weights).mean(dim=["lat", "lon"])
    std_out = np.sqrt((std_field ** 2).weighted(weights).mean(dim=["lat", "lon"]))
    return mean_out, std_out

# =============================================================================
# 6 - Run per region (from shapefile) and save one NetCDF per region
# =============================================================================
rng = np.random.default_rng(12345)  # reproducible seed

print("Running region loop...")

for region_name, region_id in region_id_map.items():
    print("\n============================================================")
    print(f"Processing region: {region_name} (id={region_id})")
    print("============================================================")

    ds_region = ds.where(ds["region_id"] == region_id)

    # Skip if no valid data
    valid_count = int(np.isfinite(ds_region["DIC"]).sum().values)
    if valid_count == 0:
        print(f"  Skipping {region_name}: no valid data after masking.")
        continue

    print(f"  Valid DIC points in region: {valid_count}")

    # Monte Carlo on region-masked dataset (still lat/lon gridded)
    mean_grid, std_grid = run_monte_carlo_for_ds(ds_region, num_iterations, rng)

    # Area-weighted mean over lat/lon (within region; NaNs outside region are ignored)
    mean_region, std_region = area_weighted_global(mean_grid, std_grid, ds_region["lat"])

    # Attach uncertainties into mean dataset
    for var in std_region.data_vars:
        mean_region[f"{var}_uncertainty"] = std_region[var]

    # Add region metadata
    mean_region.attrs["region_name"] = str(region_name)
    mean_region.attrs["region_id"] = int(region_id)
    mean_region.attrs["region_id_mapping"] = str(ds["region_id"].attrs["mapping"])

    # Save
    safe_name = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(region_name)).strip("_")
    out_path = f"data/processing_steps/01B_DIC_basins_averaged_{safe_name}.nc"
    
    mean_region.to_netcdf(out_path)
    print(f"Saved: {out_path}")

print("\nProcessing complete!")