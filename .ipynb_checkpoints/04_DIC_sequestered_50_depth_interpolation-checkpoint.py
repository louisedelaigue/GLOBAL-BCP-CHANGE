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
print("Calculating absolute salinity and potential temperature calculated.")

# =============================================================================
# 2. Load & Process MLD Data
# =============================================================================

print("Loading and interpolating MLD data...")
mld = xr.open_dataset(
    "/home/ldelaigue/Documents/Python/AoE_SVD/DATA/mld_DR003_c1m_reg2.0.nc"
)

# Adjust longitude range and interpolate MLD data
mld = mld.assign_coords(lon=((mld.lon + 180) % 360) - 180)
mld = mld.sortby('lon')
max_mld_interp = mld['mld'].max(dim='time').interp(lat=ds['lat'], lon=ds['lon'], method='nearest')
ds['MLD'] = max_mld_interp

# Mask dataset below MLD
MLD_expanded = ds['MLD'].expand_dims({'pres': len(ds['pres'])}, axis=1)
MLD_expanded['pres'] = ds['pres'].values
mask_below_mld = ds['pres'] > MLD_expanded
ds_below_mld = ds.where(mask_below_mld)

del ds, mld, max_mld_interp, MLD_expanded, mask_below_mld

# =============================================================================
# 3. Monte Carlo Setup
# =============================================================================

num_iterations = 2
print(f"Running Monte Carlo with {num_iterations} iterations...")

sp_constant_aou_mean = 117 / -170
sigma_sp = 0.092
rng = np.random.default_rng()

# Initialize mean and variance datasets
mean_50_depth = None
var_50_depth = None

# =============================================================================
# 4. Monte Carlo Loop
# =============================================================================

for i in range(1, num_iterations + 1):
    print(f"Iteration {i}/{num_iterations}")

    mc_sample = ds_below_mld.copy(deep=True)  
    mc_sample['aou'] += rng.normal(0, mc_sample['aou_sigma'])

    # Perturb Constant
    sp_constant_aou = rng.normal(sp_constant_aou_mean, sigma_sp)
    mc_sample['DIC_nat'] = (-1 * sp_constant_aou) * mc_sample['aou']
    mc_sample['DIC_mol_m3'] = (mc_sample['DIC_nat'] / 1e6) * mc_sample['density']
    mc_sample['layer_thickness'] = mc_sample['pres'].diff('pres')
    mc_sample['DIC_mol_per_m2'] = mc_sample['DIC_mol_m3'] * mc_sample['layer_thickness']

    # Depth-Integrated DIC_nat Below MLD
    mc_sample['DIC_integrated_below_MLD'] = mc_sample['DIC_mol_per_m2'].sum(dim='pres')
    half_DIC = 0.5 * mc_sample['DIC_integrated_below_MLD']

    # Compute cumulative sum along depth
    mc_sample['DIC_cumsum'] = mc_sample['DIC_mol_per_m2'].cumsum(dim='pres')

    # Find depth where cumulative DIC reaches 50%
    def find_50_depth(cumsum, pres, target):
        return np.interp(target, cumsum, pres)
    
    mc_sample['DIC_nat_50_depth'] = xr.apply_ufunc(
        find_50_depth, 
        mc_sample['DIC_cumsum'], 
        mc_sample['pres'], 
        half_DIC, 
        input_core_dims=[['pres'], ['pres'], []],
        output_core_dims=[[]],
        vectorize=True
    )

    if i == 1:
        mean_50_depth = mc_sample['DIC_nat_50_depth']
        var_50_depth = mc_sample['DIC_nat_50_depth'] * 0
    else:
        delta_50 = mc_sample['DIC_nat_50_depth'] - mean_50_depth
        new_mean_50_depth = mean_50_depth + delta_50 / i
        var_50_depth = ((i - 1) * var_50_depth + delta_50 * (mc_sample['DIC_nat_50_depth'] - new_mean_50_depth)) / i
        mean_50_depth = new_mean_50_depth

# Compute standard deviation
std_50_depth = np.sqrt(var_50_depth)

# =============================================================================
# 5. Save Results
# =============================================================================

final_ds = xr.Dataset({
    'DIC_nat_50_depth': mean_50_depth,
    'DIC_nat_50_depth_uncertainty': std_50_depth
})

output_path = "/home/ldelaigue/Documents/Python/AoE_SVD/thesis/post_thesis_submission/DATA/04_DIC_sequestered_50_depth.nc"
final_ds.to_netcdf(output_path)

print(f"Monte Carlo completed. Results saved: {output_path}")
