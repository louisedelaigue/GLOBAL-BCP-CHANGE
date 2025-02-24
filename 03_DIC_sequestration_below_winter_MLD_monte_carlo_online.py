import os
import warnings
import numpy as np
import xarray as xr
import gsw
import koolstof as ks
import matplotlib.dates as mdates
from scipy.ndimage import gaussian_filter

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

# Subset Dataset (optional, adjust as needed)
print("Ignoring subsetting...")
# print("Subsetting dataset to 20 pixels...")
# ds = ds.isel(lat=slice(0, 20), lon=slice(0, 20))
# print("Subset complete.")

# Compute Absolute Salinity & Potential Temperature
print("Calculating absolute salinity and potential temperature...")
ds["salinity_absolute"] = gsw.conversions.SA_from_SP(ds["sal"], ds["pres"], ds["lon"], ds["lat"])
ds["theta"] = gsw.conversions.pt0_from_t(ds["salinity_absolute"], ds["temp"], ds["pres"])

# Compute AOU
ds["aou"] = ks.parameterisations.aou_GG92(oxygen=ds["oxy"], temperature=ds["theta"], salinity=ds["sal"])[0]
ds["aou_sigma"] = ds["uncer"]

# Compute Density
ds["density"] = gsw.density.rho(ds['salinity_absolute'], ds['theta'], ds['pres'])

# Only keep vars of interest
ds = ds[['aou', 'aou_sigma', 'density']]

# =============================================================================
# 2. Load & Process MLD Data (Masking After Computing DIC Components)
# =============================================================================

print("Loading and interpolating MLD data...")
mld = xr.open_dataset(
    "/home/ldelaigue/Documents/Python/AoE_SVD/DATA/mld_DR003_c1m_reg2.0.nc"
)

# Shift longitudes in mld to match ds's range of [-180, 180]
mld = mld.assign_coords(lon=((mld.lon + 180) % 360) - 180)

# Sort the longitudes so they are in ascending order
mld = mld.sortby('lon')

# Now interpolate the MLD data to match ds' lat/lon grid
max_mld = mld['mld'].max(dim='time')

del mld

max_mld_interp = max_mld.interp(lat=ds['lat'], lon=ds['lon'], method='nearest')

del max_mld

# Assign the interpolated MLD values to a new variable in ds
ds['MLD'] = max_mld_interp

del max_mld_interp

# Expand MLD to match the dimensions required for comparison (specifically along the 'pres' dimension)
MLD_expanded = ds['MLD'].expand_dims({'pres': len(ds['pres'])}, axis=1)
MLD_expanded['pres'] = ds['pres'].values  # Ensure that 'pres' values are correctly assigned

# Create a mask where pressure is greater than MLD
mask_below_mld = ds['pres'] > MLD_expanded

del MLD_expanded

# Apply the mask across the dataset to set values outside the mask to NaN
ds_below_mld = ds.where(mask_below_mld)

del ds

# =============================================================================
# 3. Monte Carlo Setup
# =============================================================================

num_iterations = 1000
print(f"Running Monte Carlo with {num_iterations} iterations...")

# Define Constants & Their Uncertainties
sp_constant_aou_mean = 117 / -170
sigma_sp = 0.092  # Uncertainty for soft-tissue pump

rng = np.random.default_rng()

# Initialize mean and variance datasets
mean_ds = None
var_ds = None  

# =============================================================================
# 4. Monte Carlo Loop
# =============================================================================

for i in range(1, num_iterations + 1):
    print(f"Iteration {i}/{num_iterations}")

    # Generate perturbed dataset
    mc_sample = ds_below_mld.copy(deep=True)  
    mc_sample['aou'] += rng.normal(0, mc_sample['aou_sigma']) 

    # ==== AOU with Gaussian filter
    # Generate random noise
    # random_noise_aou = rng.normal(0, ds_below_mld['aou_sigma'])
    
    # # Apply Gaussian smoothing before adding noise
    # smoothed_noise = gaussian_filter(random_noise_aou, sigma=2)
    
    # # Add to dataset
    # mc_sample['aou'] += smoothed_noise

    # ==== CONSTANT
    # Perturb Constant
    sp_constant_aou = rng.normal(sp_constant_aou_mean, sigma_sp)

    # Compute DIC_nat with Perturbed Constant
    mc_sample['DIC_nat'] = (-1 * sp_constant_aou) * mc_sample['aou']

    # Convert & Integrate DIC_nat Below MLD
    mc_sample['DIC_mol_m3'] = (mc_sample['DIC_nat'] / 1e6) * mc_sample['density']

    # Compute pressure differences
    mc_sample['layer_thickness'] = mc_sample['pres'].diff('pres')

    # Compute mol/m² using correct thicknesses
    mc_sample['DIC_mol_per_m2'] = mc_sample['DIC_mol_m3'] * mc_sample['layer_thickness']

    # Depth-Integrated DIC_nat Below MLD
    mc_sample['DIC_integrated_below_MLD'] = mc_sample['DIC_mol_per_m2'].sum(dim='pres')

    # Compute time-based sequestration correctly
    month_to_month_diff = mc_sample['DIC_integrated_below_MLD'].diff(dim='time')
    mc_sample['DIC_nat_sequestration'] = month_to_month_diff.sum(dim='time')

    # Store iteration results in a dataset
    iteration_ds = xr.Dataset({
        'DIC_nat_sequestration': mc_sample['DIC_nat_sequestration']
    })

    # Online Mean and Variance Calculation
    if i == 1:
        mean_ds = iteration_ds
        var_ds = iteration_ds * 0  # Initialize variance dataset
    else:
        # Extract DataArrays instead of working directly with Dataset
        delta = iteration_ds['DIC_nat_sequestration'] - mean_ds['DIC_nat_sequestration']
        new_mean_ds = mean_ds['DIC_nat_sequestration'] + delta / i
        var_ds = ((i - 1) * var_ds['DIC_nat_sequestration'] + delta * (iteration_ds['DIC_nat_sequestration'] - new_mean_ds)) / i

        # Wrap back into a Dataset
        mean_ds = xr.Dataset({'DIC_nat_sequestration': new_mean_ds})
        var_ds = xr.Dataset({'DIC_nat_sequestration': var_ds})

# Normalize mean and variance after all iterations
total_std_ds = np.sqrt(var_ds)  # Compute standard deviation

# =============================================================================
# 5. Save Results
# =============================================================================

final_ds = xr.Dataset({
    'DIC_nat_sequestration': mean_ds['DIC_nat_sequestration'],
    'DIC_nat_sequestration_uncertainty': total_std_ds['DIC_nat_sequestration']
})

output_path = "/home/ldelaigue/Documents/Python/AoE_SVD/thesis/post_thesis_submission/DATA/03_DIC_sequestration_below_winter_MLD_monte_carlo_online_1000_iterations.nc"
final_ds.to_netcdf(output_path)

print(f"Monte Carlo simulations completed successfully. File saved: {output_path}")
