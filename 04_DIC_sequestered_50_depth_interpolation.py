import sys
import warnings
import numpy as np
import xarray as xr
import gsw
import koolstof as ks
import matplotlib.dates as mdates
from scipy import stats

warnings.filterwarnings("ignore")

print("Starting script...")

# =============================================================================
# 1 - Load Dataset
# =============================================================================

print("Loading dataset...")
ds = xr.open_dataset("data/CONTENT_DIC_v2-3.nc")
print("Dataset loaded.")

# Compute Absolute Salinity, Potential Temperature, AOU, and Density
print("Calculating derived variables...")
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

ds = ds.rename({
    "oxy": "oxygen",
    "temp": "temperature",
    "sal": "salinity",
    "aou": "AOU",
    "uncer": "AOU_uncertainty",
})
ds = ds[["AOU", "AOU_uncertainty", "density"]]
print("Derived variables calculated.")

# =============================================================================
# 2 - Load & Process MLD Data
# =============================================================================

print("Loading and interpolating MLD climatology...")

mld_new = xr.open_dataset("data/Argo_mixedlayers_monthlyclim_04142022.nc")
mld_annual_max = mld_new["mld_da_max"].max(dim="iMONTH")
mld_annual_max = mld_annual_max.rename({"iLAT": "lat", "iLON": "lon"})
mld_annual_max = mld_annual_max.assign_coords({
    "lat": mld_new["lat"].values,
    "lon": mld_new["lon"].values,
})
mld_annual_max = mld_annual_max.assign_coords(lon=(mld_annual_max.lon % 360))
mld_annual_max = mld_annual_max.sortby("lon")

mld_interp = mld_annual_max.interp(
    lat=ds["lat"],
    lon=(ds["lon"] % 360),
    method="nearest"
)
ds["MLD"] = mld_interp
del mld_new, mld_annual_max, mld_interp

MLD_expanded = ds["MLD"].expand_dims({"pres": len(ds["pres"])}, axis=1)
MLD_expanded["pres"] = ds["pres"].values
mask_at_or_below_mld = ds["pres"] >= MLD_expanded
del MLD_expanded

print("MLD climatology applied.")

# =============================================================================
# 3 - Convert Time
# =============================================================================

print("Converting time to numeric format...")
ds = ds.assign_coords(time=("time", mdates.date2num(ds["time"].values)))
print("Time conversion complete.")

# =============================================================================
# 4 - Helper Functions
# =============================================================================

def _linregress_slope_pvalue(y, x):
    mask = np.isfinite(y) & np.isfinite(x)
    if np.sum(mask) < 3:
        return np.nan, np.nan
    res = stats.linregress(x[mask], y[mask])
    return res.slope * 365.25, res.pvalue


def compute_layer_thickness_from_centers(pres):
    pres_vals = np.asarray(pres.values, dtype=float)
    if pres_vals.ndim != 1:
        raise ValueError("pres must be 1D")
    if len(pres_vals) < 2:
        raise ValueError("pres must have at least 2 levels")
    if not np.all(np.diff(pres_vals) > 0):
        raise ValueError("pres must be strictly increasing")
    midpoints = 0.5 * (pres_vals[:-1] + pres_vals[1:])
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


layer_thickness_1d = compute_layer_thickness_from_centers(ds["pres"])
pres_vals = ds["pres"].values.astype(float)


def compute_column_integral_trend(ds, var, mask_at_or_below_mld, layer_thickness_1d):
    """
    Mask to at/below MLD, convert to mol m-3, depth-integrate,
    regress against time. Returns slope (mol m-2 yr-1) and p-value.
    """
    try:
        var_masked = ds[var].where(mask_at_or_below_mld)
        var_mol_m3 = (var_masked * ds["density"]) / 1e6
        layer_thickness = layer_thickness_1d.broadcast_like(var_mol_m3)
        integrated = (var_mol_m3 * layer_thickness).sum(dim="pres")

        slope, pvalue = xr.apply_ufunc(
            _linregress_slope_pvalue,
            integrated,
            ds["time"],
            input_core_dims=[["time"], ["time"]],
            output_core_dims=[[], []],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float, float],
        )
        return slope, pvalue

    except Exception as e:
        print(f"Error in compute_column_integral_trend for {var}: {e}")
        template = xr.full_like(ds[var].isel(time=0, pres=0), np.nan)
        return template, template


def _find_50pct_depth(cumsum_1d, total_scalar):
    """
    Find the pressure at which cumulative DIC reaches 50% of total.
    cumsum_1d is along the pres dimension (already finite-filtered externally).
    """
    target = 0.5 * total_scalar
    if not np.isfinite(target) or target <= 0:
        return np.nan
    finite = np.isfinite(cumsum_1d)
    if np.sum(finite) < 2:
        return np.nan
    return float(np.interp(target, cumsum_1d[finite], pres_vals[finite]))


def compute_50pct_depth_trend(ds, var, mask_at_or_below_mld, layer_thickness_1d):
    """
    At each time step and pixel:
      1. Mask to at/below MLD, convert to mol m-3
      2. Compute layer-wise content (mol m-2 per layer)
      3. Cumulative sum along pres -> find depth of 50% cumulative content
    Then regress the (lat, lon, time) depth series against time.
    Returns slope (dbar yr-1) and p-value.
    """
    try:
        var_masked = ds[var].where(mask_at_or_below_mld)
        var_mol_m3 = (var_masked * ds["density"]) / 1e6
        layer_thickness = layer_thickness_1d.broadcast_like(var_mol_m3)

        # (lat, lon, pres, time)
        dic_layer = var_mol_m3 * layer_thickness

        # Total column integral at each time step (lat, lon, time)
        dic_total = dic_layer.sum(dim="pres")

        # Cumulative sum along pres (lat, lon, pres, time)
        dic_cumsum = dic_layer.cumsum(dim="pres")

        # 50% depth at each time step (lat, lon, time)
        depth_50_series = xr.apply_ufunc(
            _find_50pct_depth,
            dic_cumsum,
            dic_total,
            input_core_dims=[["pres"], []],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        # Regress 50% depth time series against time (lat, lon)
        slope, pvalue = xr.apply_ufunc(
            _linregress_slope_pvalue,
            depth_50_series,
            ds["time"],
            input_core_dims=[["time"], ["time"]],
            output_core_dims=[[], []],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float, float],
        )
        return slope, pvalue

    except Exception as e:
        print(f"Error in compute_50pct_depth_trend for {var}: {e}")
        template = xr.full_like(ds[var].isel(time=0, pres=0), np.nan)
        return template, template


# =============================================================================
# 5 - Monte Carlo Setup
# =============================================================================

num_iterations = 1000
alpha = 0.05

sp_constant_aou_mean = 117 / -170
cp_constant_mean = 16 / -170
sigma_sp = 0.092
sigma_cp = 0.0081

rng = np.random.default_rng()

mean_ds = xr.Dataset()
var_ds = xr.Dataset()

significant_count_ds = xr.Dataset()
significant_positive_count_ds = xr.Dataset()
significant_negative_count_ds = xr.Dataset()
positive_count_ds = xr.Dataset()
negative_count_ds = xr.Dataset()

print(f"Running Monte Carlo with {num_iterations} iterations...")

# =============================================================================
# 6 - Monte Carlo Loop
# =============================================================================

for i in range(1, num_iterations + 1):
    print(i)

    mc_sample = ds.copy(deep=True)
    mc_sample["AOU"] += rng.normal(0, ds["AOU_uncertainty"])

    sp_constant_aou = rng.normal(sp_constant_aou_mean, sigma_sp)
    cp_constant = rng.normal(cp_constant_mean, sigma_cp)

    mc_sample["soft_pump_AOU"] = -sp_constant_aou * mc_sample["AOU"]

    # --- Column integral trend (mol m-2 yr-1) ---
    aou_rate, aou_pvalue = compute_column_integral_trend(
        mc_sample, "AOU", mask_at_or_below_mld, layer_thickness_1d
    )
    soft_pump_rate, soft_pump_pvalue = compute_column_integral_trend(
        mc_sample, "soft_pump_AOU", mask_at_or_below_mld, layer_thickness_1d
    )

    # --- 50% depth trend (dbar yr-1) ---
    depth_50_slope, depth_50_pvalue = compute_50pct_depth_trend(
        mc_sample, "soft_pump_AOU", mask_at_or_below_mld, layer_thickness_1d
    )

    # --- Pack iteration results ---
    iteration_ds = xr.Dataset({
        "AOU_rate_at_or_below_MLD":                  aou_rate,
        "soft_pump_aou_change_at_or_below_MLD":      soft_pump_rate,
        "soft_pump_aou_50pct_depth_trend":           depth_50_slope,
    })

    pvalue_ds = xr.Dataset({
        "AOU_rate_at_or_below_MLD_pvalue":               aou_pvalue,
        "soft_pump_aou_change_at_or_below_MLD_pvalue":   soft_pump_pvalue,
        "soft_pump_aou_50pct_depth_trend_pvalue":        depth_50_pvalue,
    })

    # --- Helper: boolean flags for counts ---
    def _sig(pvar):
        return xr.where(pvalue_ds[pvar] < alpha, 1, 0).astype(np.int32)

    def _pos(dvar):
        return xr.where(iteration_ds[dvar] > 0, 1, 0).astype(np.int32)

    def _neg(dvar):
        return xr.where(iteration_ds[dvar] < 0, 1, 0).astype(np.int32)

    all_vars = [
        ("AOU_rate_at_or_below_MLD",             "AOU_rate_at_or_below_MLD_pvalue"),
        ("soft_pump_aou_change_at_or_below_MLD", "soft_pump_aou_change_at_or_below_MLD_pvalue"),
        ("soft_pump_aou_50pct_depth_trend",      "soft_pump_aou_50pct_depth_trend_pvalue"),
    ]

    if i == 1:
        mean_ds = iteration_ds.copy(deep=True)
        var_ds = iteration_ds * 0

        significant_count_ds          = xr.Dataset({dv: _sig(pv)              for dv, pv in all_vars})
        significant_positive_count_ds = xr.Dataset({dv: _sig(pv) * _pos(dv)  for dv, pv in all_vars})
        significant_negative_count_ds = xr.Dataset({dv: _sig(pv) * _neg(dv)  for dv, pv in all_vars})
        positive_count_ds             = xr.Dataset({dv: _pos(dv)              for dv, pv in all_vars})
        negative_count_ds             = xr.Dataset({dv: _neg(dv)              for dv, pv in all_vars})

    else:
        delta = iteration_ds - mean_ds
        new_mean_ds = mean_ds + delta / i
        var_ds = ((i - 1) * var_ds + delta * (iteration_ds - new_mean_ds)) / i
        mean_ds = new_mean_ds

        for dv, pv in all_vars:
            significant_count_ds[dv]          += _sig(pv)
            significant_positive_count_ds[dv] += _sig(pv) * _pos(dv)
            significant_negative_count_ds[dv] += _sig(pv) * _neg(dv)
            positive_count_ds[dv]             += _pos(dv)
            negative_count_ds[dv]             += _neg(dv)

# =============================================================================
# 7 - Finalise and Save
# =============================================================================

std_ds = np.sqrt(var_ds)

for var in std_ds.data_vars:
    mean_ds[f"{var}_uncertainty"] = std_ds[var]

for dv, _ in all_vars:
    mean_ds[f"{dv}_significant_count"]          = significant_count_ds[dv]
    mean_ds[f"{dv}_significant_positive_count"] = significant_positive_count_ds[dv]
    mean_ds[f"{dv}_significant_negative_count"] = significant_negative_count_ds[dv]
    mean_ds[f"{dv}_positive_count"]             = positive_count_ds[dv]
    mean_ds[f"{dv}_negative_count"]             = negative_count_ds[dv]

output_path = (
    "data/processing_steps/"
    "04_DIC_sequestered_50_depth_change.nc"
)
mean_ds.to_netcdf(output_path)

print(f"Saved to: {output_path}")
print("Processing complete!")

# import os
# import warnings
# import numpy as np
# import xarray as xr
# import gsw
# import koolstof as ks
# import matplotlib.dates as mdates
# from scipy.ndimage import gaussian_filter

# warnings.filterwarnings('ignore')  # Suppress warnings

# print("Starting script...")

# # =============================================================================
# # 1. Load Dataset & Preprocess
# # =============================================================================

# print("Loading dataset...")
# ds = xr.open_dataset("data/CONTENT_DIC_v2-3.nc")
# print("Dataset loaded.")

# # Compute Absolute Salinity & Potential Temperature
# print("Calculating absolute salinity and potential temperature...")
# ds["salinity_absolute"] = gsw.conversions.SA_from_SP(ds["sal"], ds["pres"], ds["lon"], ds["lat"])
# ds["theta"] = gsw.conversions.pt0_from_t(ds["salinity_absolute"], ds["temp"], ds["pres"])

# # Compute AOU
# ds["aou"] = ks.parameterisations.aou_GG92(oxygen=ds["oxy"], temperature=ds["theta"], salinity=ds["sal"])[0]
# ds["aou_sigma"] = ds["uncer"]

# # Compute Density
# ds["density"] = gsw.density.rho(ds['salinity_absolute'], ds['theta'], ds['pres'])

# # Only keep vars of interest
# ds = ds[['aou', 'aou_sigma', 'density']]
# print("Calculating absolute salinity and potential temperature calculated.")

# # # =============================================================================
# # # 2. Load & Process MLD Data
# # # =============================================================================

# # print("Loading and interpolating MLD data...")
# # mld = xr.open_dataset(
# #     "/home/ldelaigue/Documents/Python/AoE_SVD/DATA/mld_DR003_c1m_reg2.0.nc"
# # )

# # # Adjust longitude range and interpolate MLD data
# # mld = mld.assign_coords(lon=((mld.lon + 180) % 360) - 180)
# # mld = mld.sortby('lon')
# # max_mld_interp = mld['mld'].max(dim='time').interp(lat=ds['lat'], lon=ds['lon'], method='nearest')
# # ds['MLD'] = max_mld_interp

# # # Mask dataset below MLD
# # MLD_expanded = ds['MLD'].expand_dims({'pres': len(ds['pres'])}, axis=1)
# # MLD_expanded['pres'] = ds['pres'].values
# # mask_below_mld = ds['pres'] > MLD_expanded
# # ds_below_mld = ds.where(mask_below_mld)

# # del ds, mld, max_mld_interp, MLD_expanded, mask_below_mld

# # =============================================================================
# # 2. Load & Process NEW MLD Data (Climatology)
# # =============================================================================
# print("Loading and interpolating NEW MLD climatology...")

# # Load the new MLD dataset
# mld_new = xr.open_dataset("data/Argo_mixedlayers_monthlyclim_04142022.nc")

# # Annual maximum
# mld_annual_max = mld_new["mld_da_max"].max(dim="iMONTH")  # (iLAT, iLON)

# # Rename dims
# mld_annual_max = mld_annual_max.rename({"iLAT": "lat", "iLON": "lon"})

# # Assign coordinate values
# mld_annual_max = mld_annual_max.assign_coords({
#     "lat": mld_new["lat"].values,
#     "lon": mld_new["lon"].values
# })

# # Convert longitude from [-180, 180] to [0, 360)
# mld_annual_max = mld_annual_max.assign_coords(
#     lon=(mld_annual_max.lon % 360)
# )

# # Sort after conversion
# mld_annual_max = mld_annual_max.sortby("lon")

# # Interpolate to ds grid
# mld_interp = mld_annual_max.interp(
#     lat=ds["lat"],
#     lon=(ds["lon"] % 360),
#     method="nearest"
# )

# ds["MLD"] = mld_interp

# # Clean up
# del mld_new, mld_annual_max, mld_interp

# print("New MLD climatology applied.")

# # Expand MLD to match the dimensions required for comparison (specifically along the 'pres' dimension)
# MLD_expanded = ds['MLD'].expand_dims({'pres': len(ds['pres'])}, axis=1)
# MLD_expanded['pres'] = ds['pres'].values  # Ensure that 'pres' values are correctly assigned

# # Create a mask where pressure is greater than MLD
# mask_below_mld = ds['pres'] > MLD_expanded

# del MLD_expanded

# # Apply the mask across the dataset to set values outside the mask to NaN
# ds_below_mld = ds.where(mask_below_mld)

# del ds

# # =============================================================================
# # 3. Monte Carlo Setup
# # =============================================================================

# num_iterations = 1000
# print(f"Running Monte Carlo with {num_iterations} iterations...")

# sp_constant_aou_mean = 117 / -170
# sigma_sp = 0.092
# rng = np.random.default_rng()

# # Initialize mean and variance datasets
# mean_50_depth = None
# var_50_depth = None

# # =============================================================================
# # 4. Monte Carlo Loop
# # =============================================================================

# for i in range(1, num_iterations + 1):
#     print(f"Iteration {i}/{num_iterations}")

#     mc_sample = ds_below_mld.copy(deep=True)  
#     mc_sample['aou'] += rng.normal(0, mc_sample['aou_sigma'])

#     # Perturb Constant
#     sp_constant_aou = rng.normal(sp_constant_aou_mean, sigma_sp)
#     mc_sample['DIC_nat'] = (-1 * sp_constant_aou) * mc_sample['aou']
#     mc_sample['DIC_mol_m3'] = (mc_sample['DIC_nat'] / 1e6) * mc_sample['density']
#     mc_sample['layer_thickness'] = mc_sample['pres'].diff('pres')
#     mc_sample['DIC_mol_per_m2'] = mc_sample['DIC_mol_m3'] * mc_sample['layer_thickness']

#     # Depth-Integrated DIC_nat Below MLD
#     mc_sample['DIC_integrated_below_MLD'] = mc_sample['DIC_mol_per_m2'].sum(dim='pres')
#     half_DIC = 0.5 * mc_sample['DIC_integrated_below_MLD']

#     # Compute cumulative sum along depth
#     mc_sample['DIC_cumsum'] = mc_sample['DIC_mol_per_m2'].cumsum(dim='pres')

#     # Find depth where cumulative DIC reaches 50%
#     def find_50_depth(cumsum, pres, target):
#         return np.interp(target, cumsum, pres)
    
#     mc_sample['DIC_nat_50_depth'] = xr.apply_ufunc(
#         find_50_depth, 
#         mc_sample['DIC_cumsum'], 
#         mc_sample['pres'], 
#         half_DIC, 
#         input_core_dims=[['pres'], ['pres'], []],
#         output_core_dims=[[]],
#         vectorize=True
#     )

#     if i == 1:
#         mean_50_depth = mc_sample['DIC_nat_50_depth']
#         var_50_depth = mc_sample['DIC_nat_50_depth'] * 0
#     else:
#         delta_50 = mc_sample['DIC_nat_50_depth'] - mean_50_depth
#         new_mean_50_depth = mean_50_depth + delta_50 / i
#         var_50_depth = ((i - 1) * var_50_depth + delta_50 * (mc_sample['DIC_nat_50_depth'] - new_mean_50_depth)) / i
#         mean_50_depth = new_mean_50_depth

# # Compute standard deviation
# std_50_depth = np.sqrt(var_50_depth)

# # =============================================================================
# # 5. Save Results
# # =============================================================================

# final_ds = xr.Dataset({
#     'DIC_nat_50_depth': mean_50_depth,
#     'DIC_nat_50_depth_uncertainty': std_50_depth
# })

# output_path = "data/processing_steps/04_DIC_sequestered_50_depth_1000_iterations.nc"
# final_ds.to_netcdf(output_path)

# print(f"Monte Carlo completed. Results saved: {output_path}")
