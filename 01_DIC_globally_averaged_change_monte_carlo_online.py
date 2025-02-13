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

# Load Alkalinity Data
print("Loading alkalinity dataset...")
TA = xr.open_dataset(CONTENT_DATA_PATH + 'CONTENT_AT_v2-1.nc')
ds['TA'] = TA['AT']
ds['TA_sigma'] = TA['AT_sigma']
print("Alkalinity data loaded.")

# Subset Dataset
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
    'CT': 'DIC',
    'CT_sigma': 'DIC_uncertainty',
    'TA_sigma': 'TA_uncertainty'
})

ds = ds[['AOU', 'AOU_uncertainty', 'DIC', 'DIC_uncertainty', 'TA', 'TA_uncertainty']]

# =============================================================================
# 3 - Convert Time to Matplotlib Numeric Format (Before Analysis)
# =============================================================================

print("Converting time to numeric format...")

# Convert time to Matplotlib datenum (keeps time in days)
ds = ds.assign_coords(time=("time", mdates.date2num(ds["time"].values)))

print("Time conversion complete.")
print("New time values:", ds["time"].values)  # Debugging

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

# Function to compute slopes safely
def compute_slope(ds, var):
    try:
        polyfit = ds[var].polyfit(dim='time', deg=1, skipna=True)
        slope = polyfit.polyfit_coefficients.sel(degree=1) * 365.25  # Convert from per day to per year
        # print(f"Slope for {var}: {slope.values}")  # Debugging print
        return slope  
    except Exception as e:
        print(f"Error computing slope for {var}: {e}")
        return xr.zeros_like(ds[var].isel(time=0))

rng = np.random.default_rng()

# Monte Carlo Loop (Iterative Mean & Variance)
for i in range(1, num_iterations + 1):
    print(i)
    mc_sample = ds.copy(deep=True)
    mc_sample['DIC'] += rng.normal(0, ds['DIC_uncertainty'])
    mc_sample['TA'] += rng.normal(0, ds['TA_uncertainty'])
    mc_sample['AOU'] += rng.normal(0, ds['AOU_uncertainty'])

    # Perturb Constants
    sp_constant_aou = rng.normal(sp_constant_aou_mean, sigma_sp)
    cp_constant = rng.normal(cp_constant_mean, sigma_cp)

    dic_rate = compute_slope(mc_sample, 'DIC')
    ta_rate = compute_slope(mc_sample, 'TA')
    aou_rate = compute_slope(mc_sample, 'AOU')

    soft_pump_aou_change = -sp_constant_aou * aou_rate
    carb_pump_change = 0.5 * (ta_rate - cp_constant * aou_rate)
    co2_anth_aou_change = dic_rate - (soft_pump_aou_change + carb_pump_change)

    iteration_ds = xr.Dataset({
        'DIC_rate': dic_rate,
        'TA_rate': ta_rate,
        'AOU_rate': aou_rate,
        'soft_pump_aou_change': soft_pump_aou_change,
        'carb_pump_change': carb_pump_change,
        'co2_anth_aou_change': co2_anth_aou_change
    })

    if i == 1:
        mean_ds = iteration_ds
        var_ds = iteration_ds * 0
    else:
        delta = iteration_ds - mean_ds
        new_mean_ds = mean_ds + delta / i  # Welford’s method
        var_ds = ((i - 1) * var_ds + delta * (iteration_ds - new_mean_ds)) / i
        mean_ds = new_mean_ds

# Compute final standard deviation
std_ds = np.sqrt(var_ds)  # No division by num_iterations
print("Monte Carlo simulations complete.")

# =============================================================================
# 5 - Compute Global Mean with Monte Carlo
# =============================================================================

print("Computing global mean across lat/lon...")
weights = np.cos(np.deg2rad(ds.lat)) # to account for the Earth's geometry
mean_ds = mean_ds.weighted(weights).mean(dim=['lat', 'lon'])
std_ds = np.sqrt((std_ds ** 2).weighted(weights).mean(dim=['lat', 'lon'])) # propagate uncertainty
print("Global mean computation complete.")

# =============================================================================
# 6 - Save Results
# =============================================================================

for var in std_ds.data_vars:
    mean_ds[f"{var}_uncertainty"] = std_ds[var]

mean_ds.to_netcdf("/home/ldelaigue/Documents/Python/AoE_SVD/thesis/post_thesis_submission/DATA/01_DIC_globally_averaged_change_monte_carlo_online.nc")
print("Merge and save complete.")

print("Processing complete!")
