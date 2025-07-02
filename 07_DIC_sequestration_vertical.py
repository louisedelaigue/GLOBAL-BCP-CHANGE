import os
import warnings
import numpy as np
import xarray as xr
import gsw
import koolstof as ks
import matplotlib.dates as mdates
from tqdm import tqdm

warnings.filterwarnings('ignore')  # Suppress warnings

print("Starting script...")

# =============================================================================
# 1. Load Dataset & Preprocess
# =============================================================================

print("Loading dataset...")
ds = xr.open_dataset(
    "/home/ldelaigue/Documents/Python/AoE_SVD/DATA/CONTENT_RESULTS/CONTENT_DIC_v2-1.nc"
)
print("Dataset loaded.")

# =============================================================================
# Subset Dataset to 4 Specific Points
# =============================================================================

print("Subsetting dataset to 4 specific points...")

points = {
    'pos-pos': {'lat': -55, 'lon': 0, 'label': 'Positive-positive (○)'},
    'neg-neg': {'lat': 58, 'lon': -51, 'label': 'Negative-negative (□)'},
    'pos-neg': {'lat': 31.5, 'lon': 136.5, 'label': 'Positive-negative (△)'},
    'neg-pos': {'lat': 63.5, 'lon': -57.5, 'label': 'Negative-positive (◇)'},
}

selected_points = []
for key, val in points.items():
    point_ds = ds.sel(lat=val['lat'], lon=val['lon'], method='nearest')
    point_ds = point_ds.expand_dims({'point': [key]})
    selected_points.append(point_ds)

ds = xr.concat(selected_points, dim='point')
print("Subset complete.")

# =============================================================================
# Compute Derived Variables
# =============================================================================

print("Computing salinity, temperature, density, and AOU...")
ds["salinity_absolute"] = gsw.conversions.SA_from_SP(ds["sal"], ds["pres"], ds["lon"], ds["lat"])
ds["theta"] = gsw.conversions.pt0_from_t(ds["salinity_absolute"], ds["temp"], ds["pres"])
ds["aou"] = ks.parameterisations.aou_GG92(oxygen=ds["oxy"], temperature=ds["theta"], salinity=ds["sal"])[0]
ds["aou_sigma"] = ds["uncer"]
ds["density"] = gsw.density.rho(ds['salinity_absolute'], ds['theta'], ds['pres'])

ds = ds[['aou', 'aou_sigma', 'density']]
print("Derived variables computed.")

# =============================================================================
# 2. Monte Carlo Setup
# =============================================================================

num_iterations = 1000
print(f"Running Monte Carlo with {num_iterations} iterations...")

sp_constant_aou_mean = 117 / -170
sigma_sp = 0.092

rng = np.random.default_rng()
mean_ds = None
var_ds = None

# Convert time to numeric format (for safety in slope fitting, not strictly needed here)
ds = ds.assign_coords(time=("time", mdates.date2num(ds["time"].values)))

# =============================================================================
# 3. Monte Carlo Loop
# =============================================================================

for i in tqdm(range(1, num_iterations + 1), desc="Monte Carlo", unit="iter"):
    mc_sample = ds.copy(deep=True)
    mc_sample['aou'] += rng.normal(0, mc_sample['aou_sigma'])

    sp_constant_aou = rng.normal(sp_constant_aou_mean, sigma_sp)
    mc_sample['DIC_nat'] = -sp_constant_aou * mc_sample['aou']
    mc_sample['DIC_mol_m3'] = (mc_sample['DIC_nat'] / 1e6) * mc_sample['density']

    # Cumulative DIC change over time at each depth level
    mc_sample['DIC_nat_cumulative'] = mc_sample['DIC_mol_m3'].diff(dim='time').sum(dim='time')

    iteration_ds = xr.Dataset({
        'DIC_nat_cumulative': mc_sample['DIC_nat_cumulative']
    })

    if i == 1:
        mean_ds = iteration_ds
        var_ds = iteration_ds * 0
    else:
        delta = iteration_ds['DIC_nat_cumulative'] - mean_ds['DIC_nat_cumulative']
        new_mean = mean_ds['DIC_nat_cumulative'] + delta / i
        new_var = ((i - 1) * var_ds['DIC_nat_cumulative'] + delta * (iteration_ds['DIC_nat_cumulative'] - new_mean)) / i

        mean_ds = xr.Dataset({'DIC_nat_cumulative': new_mean})
        var_ds = xr.Dataset({'DIC_nat_cumulative': new_var})

# Final uncertainty (std deviation)
std_ds = np.sqrt(var_ds)

# =============================================================================
# 4. Save Results
# =============================================================================

final_ds = xr.Dataset({
    'DIC_nat_cumulative': mean_ds['DIC_nat_cumulative'],
    'DIC_nat_cumulative_uncertainty': std_ds['DIC_nat_cumulative']
})

output_path = "/home/ldelaigue/Documents/Python/AoE_SVD/thesis/post_thesis_submission/DATA/07_DIC_sequestration_vertical.nc"
final_ds.to_netcdf(output_path)

print(f"Monte Carlo completed and results saved:\n{output_path}")
