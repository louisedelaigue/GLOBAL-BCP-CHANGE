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

warnings.filterwarnings('ignore')

print("Starting script...")

# =============================================================================
# 1 - Define Paths
# =============================================================================

CONTENT_DATA_PATH = "/home/ldelaigue/Documents/Python/AoE_SVD/DATA/CONTENT_RESULTS/"
PROCESSING_PATH = "/home/ldelaigue/Documents/Python/AoE_SVD/DATA/PROCESSING/"
FIGS_PATH = "/home/ldelaigue/Documents/Python/AoE_SVD/FIGS/"

for path in [CONTENT_DATA_PATH, PROCESSING_PATH, FIGS_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

print("Directories checked.")

# =============================================================================
# 2 - Load Dataset
# =============================================================================

print("Loading dataset...")
ds = xr.open_dataset(CONTENT_DATA_PATH + 'CONTENT_DIC_v2-1.nc')
print("Dataset loaded.")

# Subset Dataset
print("Ignoring subsetting...")
# print("Subsetting dataset to 20 pixels...")
# ds = ds.isel(lat=slice(0, 20), lon=slice(0, 20))
# print("Subset complete.")

# Compute Absolute Salinity and Potential Temperature
print("Calculating absolute salinity and potential temperature...")
ds["salinity_absolute"] = gsw.conversions.SA_from_SP(ds["sal"], ds["pres"], ds["lon"], ds["lat"])
ds["theta"] = gsw.conversions.pt0_from_t(ds["salinity_absolute"], ds["temp"], ds["pres"])
ds["aou"] = ks.parameterisations.aou_GG92(oxygen=ds["oxy"], temperature=ds["theta"], salinity=ds["sal"])[0]
print("Salinity and temperature calculations complete.")

# Rename Variables
ds = ds.rename({
    'oxy': 'oxygen',
    'temp': 'temperature',
    'sal': 'salinity',
    'aou': 'AOU',
    'uncer': 'AOU_uncertainty',

ds = ds[['AOU', 'AOU_uncertainty']] # 'DIC', 'DIC_uncertainty', 'TA', 'TA_uncertainty'

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
print(f"Running Monte Carlo with {num_iterations} iterations...")

mean_ds = xr.Dataset()
var_ds = xr.Dataset()

# Define Constant Mean Values
sp_constant_aou_mean = 117 / -170
cp_constant_mean = 16 / -170

# Define Constant Uncertainties
sigma_sp = 0.092  # Uncertainty for soft tissue pump
sigma_cp = 0.0081  # Uncertainty for carbonate pump

# Function to compute slopes safely and integrate over depth
def compute_slope_integrated(ds, var):
    try:
        polyfit = ds[var].polyfit(dim='time', deg=1, skipna=True)
        slope = polyfit.polyfit_coefficients.sel(degree=1) * 365.25  # Convert from per day to per year

        # Compute layer thickness (pressure difference between adjacent levels)
        layer_thickness = ds['pres'].diff('pres').fillna(0)

        # Ensure same dimensions before multiplying
        layer_thickness = layer_thickness.broadcast_like(slope)

        # Compute depth-integrated change in mmol/m²/yr
        integrated_change = (slope * layer_thickness).sum(dim='pres')
        return integrated_change
    except Exception as e:
        print(f"Error computing slope-integrated change: {e}")
        return None

rng = np.random.default_rng()

# Monte Carlo Loop with depth integration
for i in range(1, num_iterations + 1):
    print(i)
    mc_sample = ds.copy(deep=True)
    mc_sample['AOU'] += rng.normal(0, ds['AOU_uncertainty'])

    # Perturb Constants
    sp_constant_aou = rng.normal(sp_constant_aou_mean, sigma_sp)
    cp_constant = rng.normal(cp_constant_mean, sigma_cp)

    # Compute depth-integrated rates
    aou_rate = compute_slope_integrated(mc_sample, 'AOU')

    soft_pump_aou_change = -sp_constant_aou * aou_rate

    iteration_ds = xr.Dataset({
        'AOU_rate': aou_rate,
        'soft_pump_aou_change': soft_pump_aou_change,
    })

    if i == 1:
        mean_ds = iteration_ds
        var_ds = iteration_ds * 0  # Initialize variance dataset
    else:
        delta = iteration_ds - mean_ds
        new_mean_ds = mean_ds + delta / i
        var_ds = ((i - 1) * var_ds + delta * (iteration_ds - new_mean_ds)) / i
        mean_ds = new_mean_ds

# Compute final standard deviation (uncertainty propagation)
std_ds = np.sqrt(var_ds)

# =============================================================================
# Convert from mmol/m2/yr to mol/m2/yr
# =============================================================================

for var in std_ds.data_vars:
    mean_ds[f"{var}_uncertainty"] = std_ds[var]

mean_ds = mean_ds / 1000

# =============================================================================
# Save Results (per-pixel, depth-integrated)
# =============================================================================


mean_ds.to_netcdf("/home/ldelaigue/Documents/Python/AoE_SVD/thesis/post_thesis_submission/DATA/02_DIC_integrated_change_map_monte_carlo_online_pres_diff.nc")

print("Merge and save complete.")

print("Processing complete!")
