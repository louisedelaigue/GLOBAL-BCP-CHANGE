import sys
import warnings
import os
import numpy as np
import pandas as pd
import xarray as xr
import gsw
import koolstof as ks
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats

warnings.filterwarnings("ignore")

print("Starting script...")

# =============================================================================
# 2 - Load Dataset
# =============================================================================

print("Loading dataset...")
ds = xr.open_dataset("data/CONTENT_DIC_v2-3.nc")
print("Dataset loaded.")

# Subset Dataset
print("Ignoring subsetting...")
# ds = ds.isel(lat=slice(0, 20), lon=slice(0, 20))

# Compute Absolute Salinity, Potential Temperature, AOU, and Density
print("Calculating absolute salinity and potential temperature...")
ds["salinity_absolute"] = gsw.conversions.SA_from_SP(
    ds["sal"], ds["pres"], ds["lon"], ds["lat"]
)
ds["theta"] = gsw.conversions.pt0_from_t(
    ds["salinity_absolute"], ds["temp"], ds["pres"]
)
ds["aou"] = ks.parameterisations.aou_GG92(
    oxygen=ds["oxy"], temperature=ds["theta"], salinity=ds["sal"]
)[0]
ds["density"] = gsw.density.rho(
    ds["salinity_absolute"], ds["theta"], ds["pres"]
)
print("Salinity, temperature, AOU, and density calculations complete.")

# Rename Variables
ds = ds.rename({
    "oxy": "oxygen",
    "temp": "temperature",
    "sal": "salinity",
    "aou": "AOU",
    "uncer": "AOU_uncertainty",
})

ds = ds[["AOU", "AOU_uncertainty", "density"]]

# =============================================================================
# 2.5 - Load & Process NEW MLD Data (Climatology)
# =============================================================================

print("Loading and interpolating NEW MLD climatology...")

mld_new = xr.open_dataset("data/Argo_mixedlayers_monthlyclim_04142022.nc")

# Annual maximum
mld_annual_max = mld_new["mld_da_max"].max(dim="iMONTH")

# Rename dims
mld_annual_max = mld_annual_max.rename({"iLAT": "lat", "iLON": "lon"})

# Assign coordinate values
mld_annual_max = mld_annual_max.assign_coords({
    "lat": mld_new["lat"].values,
    "lon": mld_new["lon"].values,
})

# Convert longitude from [-180, 180] to [0, 360)
mld_annual_max = mld_annual_max.assign_coords(
    lon=(mld_annual_max.lon % 360)
)

# Sort after conversion
mld_annual_max = mld_annual_max.sortby("lon")

# Interpolate to ds grid
mld_interp = mld_annual_max.interp(
    lat=ds["lat"],
    lon=(ds["lon"] % 360),
    method="nearest"
)

ds["MLD"] = mld_interp

# Clean up
del mld_new, mld_annual_max, mld_interp

print("New MLD climatology applied.")

# Expand MLD to match pressure dimension
MLD_expanded = ds["MLD"].expand_dims({"pres": len(ds["pres"])}, axis=1)
MLD_expanded["pres"] = ds["pres"].values

# Keep water at or below maximum climatological MLD
mask_at_or_below_mld = ds["pres"] >= MLD_expanded

del MLD_expanded

# =============================================================================
# 3 - Convert Time to Matplotlib Numeric Format (Before Analysis)
# =============================================================================

print("Converting time to numeric format...")

# Convert time to Matplotlib datenum (keeps time in days)
ds = ds.assign_coords(time=("time", mdates.date2num(ds["time"].values)))

print("Time conversion complete.")

# =============================================================================
# 4 - Monte Carlo Uncertainty Calculation (Including Constant Errors)
# =============================================================================

num_iterations = 1000
alpha = 0.05
print(f"Running Monte Carlo with {num_iterations} iterations...")

mean_ds = xr.Dataset()
var_ds = xr.Dataset()

# Per-pixel counters
significant_count_ds = xr.Dataset()
significant_positive_count_ds = xr.Dataset()
significant_negative_count_ds = xr.Dataset()
positive_count_ds = xr.Dataset()
negative_count_ds = xr.Dataset()

# Define Constant Mean Values
sp_constant_aou_mean = 117 / -170
cp_constant_mean = 16 / -170

# Define Constant Uncertainties
sigma_sp = 0.092
sigma_cp = 0.0081

# =============================================================================
# Helper functions
# =============================================================================

def _linregress_slope_pvalue(y, x):
    """
    Compute slope and p-value for a 1D time series.
    Slope is returned per year because x is in matplotlib days.
    """
    mask = np.isfinite(y) & np.isfinite(x)
    n = np.sum(mask)

    if n < 3:
        return np.nan, np.nan

    res = stats.linregress(x[mask], y[mask])
    slope_per_year = res.slope * 365.25
    pvalue = res.pvalue
    return slope_per_year, pvalue


def compute_layer_thickness_from_centers(pres):
    """
    Compute one layer thickness per pressure level from pressure-center values.
    Thickness is returned in the same units as pres (here, dbar, used as an
    approximation to meters in this workflow).
    """
    pres_vals = np.asarray(pres.values, dtype=float)

    if pres_vals.ndim != 1:
        raise ValueError("pres must be 1D")
    if len(pres_vals) < 2:
        raise ValueError("pres must have at least 2 levels")
    if not np.all(np.diff(pres_vals) > 0):
        raise ValueError("pres must be strictly increasing")

    # Midpoints between adjacent pressure centers
    midpoints = 0.5 * (pres_vals[:-1] + pres_vals[1:])

    # Extrapolate top and bottom edges
    top_edge = pres_vals[0] - 0.5 * (pres_vals[1] - pres_vals[0])
    bottom_edge = pres_vals[-1] + 0.5 * (pres_vals[-1] - pres_vals[-2])

    edges = np.concatenate([[top_edge], midpoints, [bottom_edge]])
    thickness = np.diff(edges)

    return xr.DataArray(
        thickness,
        coords={"pres": pres},
        dims=["pres"],
        name="layer_thickness",
    )


# Compute once and reuse everywhere for consistency
layer_thickness_1d = compute_layer_thickness_from_centers(ds["pres"])


def compute_slope_and_pvalue_integrated_at_or_below_mld(
    ds, var, mask_at_or_below_mld, layer_thickness_1d
):
    """
    1) Mask to water at or below maximum climatological MLD
    2) Convert from µmol kg-1 to mol m-3 using density
    3) Build the depth-integrated time series at/below MLD
    4) Regress that integrated time series against time
    5) Return slope and p-value at each pixel

    Output slope units:
        mol m-2 yr-1
    """
    try:
        # Apply at/below-MLD mask
        var_at_or_below_mld = ds[var].where(mask_at_or_below_mld)

        # Convert from µmol kg-1 to mol m-3:
        # (µmol kg-1) * (kg m-3) / 1e6 = mol m-3
        var_at_or_below_mld_mol_m3 = (var_at_or_below_mld * ds["density"]) / 1e6

        # Broadcast thickness to full variable shape
        layer_thickness = layer_thickness_1d.broadcast_like(var_at_or_below_mld_mol_m3)

        # Integrate at/below MLD only
        integrated_series_at_or_below_mld = (
            var_at_or_below_mld_mol_m3 * layer_thickness
        ).sum(dim="pres")

        # Regress integrated time series against time
        slope, pvalue = xr.apply_ufunc(
            _linregress_slope_pvalue,
            integrated_series_at_or_below_mld,
            ds["time"],
            input_core_dims=[["time"], ["time"]],
            output_core_dims=[[], []],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float, float],
        )

        return slope, pvalue

    except Exception as e:
        print(f"Error computing slope-integrated change at/below MLD for {var}: {e}")
        template = xr.full_like(ds[var].isel(time=0, pres=0), np.nan)
        return template, template


rng = np.random.default_rng()

# =============================================================================
# Monte Carlo Loop with at/below-MLD depth integration
# =============================================================================

for i in range(1, num_iterations + 1):
    print(i)

    mc_sample = ds.copy(deep=True)
    mc_sample["AOU"] += rng.normal(0, ds["AOU_uncertainty"])

    # Perturb Constants
    sp_constant_aou = rng.normal(sp_constant_aou_mean, sigma_sp)
    cp_constant = rng.normal(cp_constant_mean, sigma_cp)

    # Build soft-pump time series first
    mc_sample["soft_pump_AOU"] = -sp_constant_aou * mc_sample["AOU"]

    # Compute at/below-MLD integrated rates AND p-values
    aou_rate_at_or_below_mld, aou_pvalue_at_or_below_mld = compute_slope_and_pvalue_integrated_at_or_below_mld(
        mc_sample, "AOU", mask_at_or_below_mld, layer_thickness_1d
    )
    soft_pump_aou_change_at_or_below_mld, soft_pump_pvalue_at_or_below_mld = compute_slope_and_pvalue_integrated_at_or_below_mld(
        mc_sample, "soft_pump_AOU", mask_at_or_below_mld, layer_thickness_1d
    )

    iteration_ds = xr.Dataset({
        "AOU_rate_at_or_below_MLD": aou_rate_at_or_below_mld,
        "soft_pump_aou_change_at_or_below_MLD": soft_pump_aou_change_at_or_below_mld,
    })

    pvalue_ds = xr.Dataset({
        "AOU_rate_at_or_below_MLD_pvalue": aou_pvalue_at_or_below_mld,
        "soft_pump_aou_change_at_or_below_MLD_pvalue": soft_pump_pvalue_at_or_below_mld,
    })

    if i == 1:
        mean_ds = iteration_ds
        var_ds = iteration_ds * 0

        significant_count_ds = xr.Dataset({
            "AOU_rate_at_or_below_MLD": xr.where(
                pvalue_ds["AOU_rate_at_or_below_MLD_pvalue"] < alpha, 1, 0
            ).astype(np.int32),
            "soft_pump_aou_change_at_or_below_MLD": xr.where(
                pvalue_ds["soft_pump_aou_change_at_or_below_MLD_pvalue"] < alpha, 1, 0
            ).astype(np.int32),
        })

        significant_positive_count_ds = xr.Dataset({
            "AOU_rate_at_or_below_MLD": xr.where(
                (pvalue_ds["AOU_rate_at_or_below_MLD_pvalue"] < alpha)
                & (iteration_ds["AOU_rate_at_or_below_MLD"] > 0),
                1, 0
            ).astype(np.int32),
            "soft_pump_aou_change_at_or_below_MLD": xr.where(
                (pvalue_ds["soft_pump_aou_change_at_or_below_MLD_pvalue"] < alpha)
                & (iteration_ds["soft_pump_aou_change_at_or_below_MLD"] > 0),
                1, 0
            ).astype(np.int32),
        })

        significant_negative_count_ds = xr.Dataset({
            "AOU_rate_at_or_below_MLD": xr.where(
                (pvalue_ds["AOU_rate_at_or_below_MLD_pvalue"] < alpha)
                & (iteration_ds["AOU_rate_at_or_below_MLD"] < 0),
                1, 0
            ).astype(np.int32),
            "soft_pump_aou_change_at_or_below_MLD": xr.where(
                (pvalue_ds["soft_pump_aou_change_at_or_below_MLD_pvalue"] < alpha)
                & (iteration_ds["soft_pump_aou_change_at_or_below_MLD"] < 0),
                1, 0
            ).astype(np.int32),
        })

        positive_count_ds = xr.Dataset({
            "AOU_rate_at_or_below_MLD": xr.where(
                iteration_ds["AOU_rate_at_or_below_MLD"] > 0, 1, 0
            ).astype(np.int32),
            "soft_pump_aou_change_at_or_below_MLD": xr.where(
                iteration_ds["soft_pump_aou_change_at_or_below_MLD"] > 0, 1, 0
            ).astype(np.int32),
        })

        negative_count_ds = xr.Dataset({
            "AOU_rate_at_or_below_MLD": xr.where(
                iteration_ds["AOU_rate_at_or_below_MLD"] < 0, 1, 0
            ).astype(np.int32),
            "soft_pump_aou_change_at_or_below_MLD": xr.where(
                iteration_ds["soft_pump_aou_change_at_or_below_MLD"] < 0, 1, 0
            ).astype(np.int32),
        })

    else:
        delta = iteration_ds - mean_ds
        new_mean_ds = mean_ds + delta / i
        var_ds = ((i - 1) * var_ds + delta * (iteration_ds - new_mean_ds)) / i
        mean_ds = new_mean_ds

        significant_count_ds["AOU_rate_at_or_below_MLD"] = significant_count_ds["AOU_rate_at_or_below_MLD"] + xr.where(
            pvalue_ds["AOU_rate_at_or_below_MLD_pvalue"] < alpha, 1, 0
        ).astype(np.int32)
        significant_count_ds["soft_pump_aou_change_at_or_below_MLD"] = significant_count_ds["soft_pump_aou_change_at_or_below_MLD"] + xr.where(
            pvalue_ds["soft_pump_aou_change_at_or_below_MLD_pvalue"] < alpha, 1, 0
        ).astype(np.int32)

        significant_positive_count_ds["AOU_rate_at_or_below_MLD"] = significant_positive_count_ds["AOU_rate_at_or_below_MLD"] + xr.where(
            (pvalue_ds["AOU_rate_at_or_below_MLD_pvalue"] < alpha)
            & (iteration_ds["AOU_rate_at_or_below_MLD"] > 0),
            1, 0
        ).astype(np.int32)
        significant_positive_count_ds["soft_pump_aou_change_at_or_below_MLD"] = significant_positive_count_ds["soft_pump_aou_change_at_or_below_MLD"] + xr.where(
            (pvalue_ds["soft_pump_aou_change_at_or_below_MLD_pvalue"] < alpha)
            & (iteration_ds["soft_pump_aou_change_at_or_below_MLD"] > 0),
            1, 0
        ).astype(np.int32)

        significant_negative_count_ds["AOU_rate_at_or_below_MLD"] = significant_negative_count_ds["AOU_rate_at_or_below_MLD"] + xr.where(
            (pvalue_ds["AOU_rate_at_or_below_MLD_pvalue"] < alpha)
            & (iteration_ds["AOU_rate_at_or_below_MLD"] < 0),
            1, 0
        ).astype(np.int32)
        significant_negative_count_ds["soft_pump_aou_change_at_or_below_MLD"] = significant_negative_count_ds["soft_pump_aou_change_at_or_below_MLD"] + xr.where(
            (pvalue_ds["soft_pump_aou_change_at_or_below_MLD_pvalue"] < alpha)
            & (iteration_ds["soft_pump_aou_change_at_or_below_MLD"] < 0),
            1, 0
        ).astype(np.int32)

        positive_count_ds["AOU_rate_at_or_below_MLD"] = positive_count_ds["AOU_rate_at_or_below_MLD"] + xr.where(
            iteration_ds["AOU_rate_at_or_below_MLD"] > 0, 1, 0
        ).astype(np.int32)
        positive_count_ds["soft_pump_aou_change_at_or_below_MLD"] = positive_count_ds["soft_pump_aou_change_at_or_below_MLD"] + xr.where(
            iteration_ds["soft_pump_aou_change_at_or_below_MLD"] > 0, 1, 0
        ).astype(np.int32)

        negative_count_ds["AOU_rate_at_or_below_MLD"] = negative_count_ds["AOU_rate_at_or_below_MLD"] + xr.where(
            iteration_ds["AOU_rate_at_or_below_MLD"] < 0, 1, 0
        ).astype(np.int32)
        negative_count_ds["soft_pump_aou_change_at_or_below_MLD"] = negative_count_ds["soft_pump_aou_change_at_or_below_MLD"] + xr.where(
            iteration_ds["soft_pump_aou_change_at_or_below_MLD"] < 0, 1, 0
        ).astype(np.int32)

# Compute final standard deviation
std_ds = np.sqrt(var_ds)

# =============================================================================
# Attach uncertainty (already in mol m-2 yr-1)
# =============================================================================

for var in std_ds.data_vars:
    mean_ds[f"{var}_uncertainty"] = std_ds[var]

# =============================================================================
# Add new count statistics
# =============================================================================

for var in ["AOU_rate_at_or_below_MLD", "soft_pump_aou_change_at_or_below_MLD"]:
    mean_ds[f"{var}_significant_count"] = significant_count_ds[var]
    mean_ds[f"{var}_significant_positive_count"] = significant_positive_count_ds[var]
    mean_ds[f"{var}_significant_negative_count"] = significant_negative_count_ds[var]
    mean_ds[f"{var}_positive_count"] = positive_count_ds[var]
    mean_ds[f"{var}_negative_count"] = negative_count_ds[var]

# =============================================================================
# Save Results (per-pixel, depth-integrated at or below MLD)
# =============================================================================

output_path = (
    "data/processing_steps/"
    "03_DIC_integrated_change_at_or_below_MLD_map_monte_carlo_online_pres_diff_with_sig.nc"
)
mean_ds.to_netcdf(output_path)

print(f"Merge and save complete. File saved to: {output_path}")
print("Processing complete!")