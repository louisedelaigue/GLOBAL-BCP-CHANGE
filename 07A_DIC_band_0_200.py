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
# 1 - Band definition
# =============================================================================

P_MIN = 0
P_MAX = 200
output_path = "data/processing_steps/07A_DIC_band_0_200_monte_carlo.nc"

# =============================================================================
# 2 - Load Dataset
# =============================================================================

print("Loading dataset...")
ds = xr.open_dataset("data/CONTENT_DIC_v2-3.nc")
print("Dataset loaded.")

# Subset Dataset
print(f"Subsetting dataset to pressure band {P_MIN}-{P_MAX} m...")
ds = ds.sel(pres=slice(P_MIN, P_MAX))
print("Subsetting complete.")

# Compute Absolute Salinity and Potential Temperature
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


def compute_slope_and_pvalue_integrated(ds, var, layer_thickness_1d):
    """
    1) Convert concentration from µmol kg-1 to mol m-3 using density
    2) Build the depth-integrated time series
    3) Regress that integrated time series against time
    4) Return slope and p-value at each pixel

    Output slope units:
        mol m-2 yr-1
    """
    try:
        # Convert from µmol kg-1 to mol m-3
        # (µmol kg-1) * (kg m-3) / 1e6 = mol m-3
        var_mol_m3 = (ds[var] * ds["density"]) / 1e6

        # Broadcast layer thickness to full variable shape
        layer_thickness = layer_thickness_1d.broadcast_like(var_mol_m3)

        # Build depth-integrated time series first
        integrated_series = (var_mol_m3 * layer_thickness).sum(dim="pres")

        # Regress integrated time series against time
        slope, pvalue = xr.apply_ufunc(
            _linregress_slope_pvalue,
            integrated_series,
            ds["time"],
            input_core_dims=[["time"], ["time"]],
            output_core_dims=[[], []],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float, float],
        )

        return slope, pvalue

    except Exception as e:
        print(f"Error computing slope-integrated change for {var}: {e}")
        template = xr.full_like(ds[var].isel(time=0, pres=0), np.nan)
        return template, template


rng = np.random.default_rng()

# =============================================================================
# Monte Carlo Loop with depth integration
# =============================================================================

for i in range(1, num_iterations + 1):
    print(i)

    mc_sample = ds.copy(deep=True)
    mc_sample["AOU"] += rng.normal(0, ds["AOU_uncertainty"])

    # Perturb Constants
    sp_constant_aou = rng.normal(sp_constant_aou_mean, sigma_sp)
    cp_constant = rng.normal(cp_constant_mean, sigma_cp)

    # Build soft-pump time series
    mc_sample["soft_pump_AOU"] = -sp_constant_aou * mc_sample["AOU"]

    # Compute depth-integrated rates AND p-values
    aou_rate, aou_pvalue = compute_slope_and_pvalue_integrated(
        mc_sample, "AOU", layer_thickness_1d
    )
    soft_pump_aou_change, soft_pump_pvalue = compute_slope_and_pvalue_integrated(
        mc_sample, "soft_pump_AOU", layer_thickness_1d
    )

    iteration_ds = xr.Dataset({
        "AOU_rate": aou_rate,
        "soft_pump_aou_change": soft_pump_aou_change,
    })

    pvalue_ds = xr.Dataset({
        "AOU_rate_pvalue": aou_pvalue,
        "soft_pump_aou_change_pvalue": soft_pump_pvalue,
    })

    if i == 1:
        mean_ds = iteration_ds
        var_ds = iteration_ds * 0

        significant_count_ds = xr.Dataset({
            "AOU_rate": xr.where(
                pvalue_ds["AOU_rate_pvalue"] < alpha, 1, 0
            ).astype(np.int32),
            "soft_pump_aou_change": xr.where(
                pvalue_ds["soft_pump_aou_change_pvalue"] < alpha, 1, 0
            ).astype(np.int32),
        })

        significant_positive_count_ds = xr.Dataset({
            "AOU_rate": xr.where(
                (pvalue_ds["AOU_rate_pvalue"] < alpha)
                & (iteration_ds["AOU_rate"] > 0),
                1, 0
            ).astype(np.int32),
            "soft_pump_aou_change": xr.where(
                (pvalue_ds["soft_pump_aou_change_pvalue"] < alpha)
                & (iteration_ds["soft_pump_aou_change"] > 0),
                1, 0
            ).astype(np.int32),
        })

        significant_negative_count_ds = xr.Dataset({
            "AOU_rate": xr.where(
                (pvalue_ds["AOU_rate_pvalue"] < alpha)
                & (iteration_ds["AOU_rate"] < 0),
                1, 0
            ).astype(np.int32),
            "soft_pump_aou_change": xr.where(
                (pvalue_ds["soft_pump_aou_change_pvalue"] < alpha)
                & (iteration_ds["soft_pump_aou_change"] < 0),
                1, 0
            ).astype(np.int32),
        })

        positive_count_ds = xr.Dataset({
            "AOU_rate": xr.where(iteration_ds["AOU_rate"] > 0, 1, 0).astype(np.int32),
            "soft_pump_aou_change": xr.where(
                iteration_ds["soft_pump_aou_change"] > 0, 1, 0
            ).astype(np.int32),
        })

        negative_count_ds = xr.Dataset({
            "AOU_rate": xr.where(iteration_ds["AOU_rate"] < 0, 1, 0).astype(np.int32),
            "soft_pump_aou_change": xr.where(
                iteration_ds["soft_pump_aou_change"] < 0, 1, 0
            ).astype(np.int32),
        })

    else:
        delta = iteration_ds - mean_ds
        new_mean_ds = mean_ds + delta / i
        var_ds = ((i - 1) * var_ds + delta * (iteration_ds - new_mean_ds)) / i
        mean_ds = new_mean_ds

        significant_count_ds["AOU_rate"] = significant_count_ds["AOU_rate"] + xr.where(
            pvalue_ds["AOU_rate_pvalue"] < alpha, 1, 0
        ).astype(np.int32)
        significant_count_ds["soft_pump_aou_change"] = significant_count_ds["soft_pump_aou_change"] + xr.where(
            pvalue_ds["soft_pump_aou_change_pvalue"] < alpha, 1, 0
        ).astype(np.int32)

        significant_positive_count_ds["AOU_rate"] = significant_positive_count_ds["AOU_rate"] + xr.where(
            (pvalue_ds["AOU_rate_pvalue"] < alpha)
            & (iteration_ds["AOU_rate"] > 0),
            1, 0
        ).astype(np.int32)
        significant_positive_count_ds["soft_pump_aou_change"] = significant_positive_count_ds["soft_pump_aou_change"] + xr.where(
            (pvalue_ds["soft_pump_aou_change_pvalue"] < alpha)
            & (iteration_ds["soft_pump_aou_change"] > 0),
            1, 0
        ).astype(np.int32)

        significant_negative_count_ds["AOU_rate"] = significant_negative_count_ds["AOU_rate"] + xr.where(
            (pvalue_ds["AOU_rate_pvalue"] < alpha)
            & (iteration_ds["AOU_rate"] < 0),
            1, 0
        ).astype(np.int32)
        significant_negative_count_ds["soft_pump_aou_change"] = significant_negative_count_ds["soft_pump_aou_change"] + xr.where(
            (pvalue_ds["soft_pump_aou_change_pvalue"] < alpha)
            & (iteration_ds["soft_pump_aou_change"] < 0),
            1, 0
        ).astype(np.int32)

        positive_count_ds["AOU_rate"] = positive_count_ds["AOU_rate"] + xr.where(
            iteration_ds["AOU_rate"] > 0, 1, 0
        ).astype(np.int32)
        positive_count_ds["soft_pump_aou_change"] = positive_count_ds["soft_pump_aou_change"] + xr.where(
            iteration_ds["soft_pump_aou_change"] > 0, 1, 0
        ).astype(np.int32)

        negative_count_ds["AOU_rate"] = negative_count_ds["AOU_rate"] + xr.where(
            iteration_ds["AOU_rate"] < 0, 1, 0
        ).astype(np.int32)
        negative_count_ds["soft_pump_aou_change"] = negative_count_ds["soft_pump_aou_change"] + xr.where(
            iteration_ds["soft_pump_aou_change"] < 0, 1, 0
        ).astype(np.int32)

# Compute final standard deviation (uncertainty propagation)
std_ds = np.sqrt(var_ds)

# =============================================================================
# Attach uncertainty (already in mol m-2 yr-1)
# =============================================================================

for var in std_ds.data_vars:
    mean_ds[f"{var}_uncertainty"] = std_ds[var]

# =============================================================================
# Add new count statistics
# =============================================================================

for var in ["AOU_rate", "soft_pump_aou_change"]:
    mean_ds[f"{var}_significant_count"] = significant_count_ds[var]
    mean_ds[f"{var}_significant_positive_count"] = significant_positive_count_ds[var]
    mean_ds[f"{var}_significant_negative_count"] = significant_negative_count_ds[var]
    mean_ds[f"{var}_positive_count"] = positive_count_ds[var]
    mean_ds[f"{var}_negative_count"] = negative_count_ds[var]

# =============================================================================
# Save Results (per-pixel, depth-integrated)
# =============================================================================

mean_ds.to_netcdf(output_path)

print(f"Merge and save complete. File saved to: {output_path}")
print("Processing complete!")