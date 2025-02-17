import sys
import os
import warnings
import numpy as np
import xarray as xr
import gsw
import koolstof as ks
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')

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
# print("Loading alkalinity dataset...")
# TA = xr.open_dataset(CONTENT_DATA_PATH + 'CONTENT_AT_v2-1.nc')
# ds['TA'] = TA['AT']
# ds['TA_sigma'] = TA['AT_sigma']
# print("Alkalinity data loaded.")

# Subset Dataset
# print("Subsetting dataset to 20 pixels...")
# ds = ds.isel(lat=slice(0, 20), lon=slice(0, 20))
# print("Subset complete.")

# Compute Absolute Salinity and Potential Temperature
print("Calculating absolute salinity and potential temperature...")
ds["salinity_absolute"] = gsw.conversions.SA_from_SP(ds["sal"], ds["pres"], ds["lon"], ds["lat"])
ds["theta"] = gsw.conversions.pt0_from_t(ds["salinity_absolute"], ds["temp"], ds["pres"])
ds["aou"] = ks.parameterisations.aou_GG92(oxygen=ds["oxy"], temperature=ds["theta"], salinity=ds["sal"])[0]
ds['density'] = gsw.density.rho(ds['salinity_absolute'], ds['theta'], ds['pres'])
print("Salinity and temperature calculations complete.")

# Rename Variables
ds = ds.rename({
    'uncer': 'aou_sigma',
})


ds = ds[['aou', 'aou_sigma', 'CT', 'CT_sigma', 'density']] # 'TA', 'TA_uncertainty'

# =============================================================================
# 3 - Convert Time to Matplotlib Numeric Format (Before Analysis)
# =============================================================================

print("Converting time to numeric format...")

# Convert time to Matplotlib datenum (keeps time in days)
ds = ds.assign_coords(time=("time", mdates.date2num(ds["time"].values)))

print("Time conversion complete.")
# print("New time values:", ds["time"].values)  # Debugging

# =============================================================================
# 3 - Monte Carlo Setup
# =============================================================================

num_iterations = 1000  # Number of Monte Carlo simulations
print(f"Running Monte Carlo with {num_iterations} iterations...")

mean_ds = None  # Initialize mean dataset
var_ds = None  # Initialize variance dataset

# Define Constant Mean Values
sp_constant_aou_mean = 117 / -170
cp_constant_mean = 16 / -170

# Define Constant Uncertainties
sigma_sp = 0.092  # Uncertainty for soft tissue pump
sigma_cp = 0.0081  # Uncertainty for carbonate pump

rng = np.random.default_rng()

# =============================================================================
# 4 - Monte Carlo Loop
# =============================================================================

for i in range(1, num_iterations + 1):
    print(f"Iteration {i}/{num_iterations}")
    
    # Generate perturbed dataset
    mc_sample = ds.copy(deep=True)
    mc_sample['CT'] += rng.normal(0, ds['CT_sigma'])
    # mc_sample['TA'] += rng.normal(0, ds['TA_sigma'])
    mc_sample['aou'] += rng.normal(0, ds['aou_sigma'])
    
    # Perturb Constants
    sp_constant_aou = rng.normal(sp_constant_aou_mean, sigma_sp)
    cp_constant = rng.normal(cp_constant_mean, sigma_cp)

    # Compute Linear Trends (Using datenum & Convert to Per-Year)
    trend_ds = mc_sample.polyfit(dim='time', deg=1, skipna=True).sel(degree=1) * 365.25  # Converts per-day → per-year

    # Compute DIC Components with Perturbed Constants
    trend_ds['DIC_nat'] = (-1 * sp_constant_aou) * trend_ds['aou_polyfit_coefficients']
    # trend_ds['DIC_carb'] = 0.5 * (trend_ds['TA_polyfit_coefficients'] - cp_constant * trend_ds['aou_polyfit_coefficients'])
    # trend_ds['DIC_anth'] = trend_ds['CT_polyfit_coefficients'] - (trend_ds['DIC_nat'] + trend_ds['DIC_carb'])

    # **Depth Integration Process**
    trend_ds['DIC_nat_mol_m3'] = (trend_ds['DIC_nat'] / 1e6) * ds['density']  # Convert to mol/m³

    # Compute pressure differences (fix missing surface values)
    trend_ds['layer_thickness'] = ds['pres'].diff('pres')

    # Remove first level to avoid incorrect thickness assignment
    trend_ds = trend_ds.isel(pres=slice(1, None))

    # Compute mol/m² using the correct thicknesses
    trend_ds['DIC_nat_mol_per_m2'] = trend_ds['DIC_nat_mol_m3'] * trend_ds['layer_thickness']

    # **Online Mean and Variance Update (Welford's Method)**
    if mean_ds is None:
        mean_ds = trend_ds['DIC_nat_mol_per_m2']
        var_ds = trend_ds['DIC_nat_mol_per_m2'] * 0  # Start variance at zero
    else:
        delta = trend_ds['DIC_nat_mol_per_m2'] - mean_ds
        new_mean_ds = mean_ds + delta / i
        var_ds = ((i - 1) * var_ds + delta * (trend_ds['DIC_nat_mol_per_m2'] - new_mean_ds)) / i
        mean_ds = new_mean_ds

# =============================================================================
# 5 - Compute Final Standard Deviation
# =============================================================================

total_std_ds = np.sqrt(var_ds)  # No division by num_iterations (Welford’s Method is correct)

# Sum over pressure dimension for total depth-integrated change
dic_nat_total_integrated = mean_ds.sum(dim='pres')
dic_nat_total_integrated_std = np.sqrt((total_std_ds ** 2).sum(dim='pres'))  # Error propagation

# =============================================================================
# 6 - Save Results
# =============================================================================

final_ds = xr.Dataset({
    'DIC_nat_integrated_change': dic_nat_total_integrated,
    'DIC_nat_integrated_change_uncertainty': dic_nat_total_integrated_std
})

final_ds.to_netcdf("/home/ldelaigue/Documents/Python/AoE_SVD/thesis/post_thesis_submission/DATA/02_DIC_integrated_change_map_monte_carlo_online_pres_diff.nc")

print("Monte Carlo simulations with depth integration completed successfully.")
