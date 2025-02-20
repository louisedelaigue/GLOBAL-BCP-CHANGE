import os
import warnings
import numpy as np
import xarray as xr
import gsw  # Gibbs SeaWater (GSW) Oceanographic Toolbox of TEOS-10
import koolstof as ks

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
print("Subsetting dataset to 20 pixels...")
ds = ds.isel(lat=slice(0, 20), lon=slice(0, 20))
print("Subset complete.")

# Define custom pressure grid
print("Defining custom pressure grid for interpolation...")
pres_fine_above_500 = ds.pres.values[ds.pres.values <= 500]  # Keep original values ≤ 500 dbar
pres_fine_below_500 = np.arange(500, ds.pres.values.max(), 20)  # 20m resolution from 500 dbar onward
pres_highres = np.concatenate((pres_fine_above_500, pres_fine_below_500))  # Combine grids

# Interpolate variables to the new pressure grid
print("Interpolating dataset onto new pressure grid...")
ds = ds.interp(pres=pres_highres, method="linear")
print("Interpolation complete.")

# Compute Absolute Salinity & Potential Temperature
print("Calculating absolute salinity and potential temperature...")
ds["salinity_absolute"] = gsw.conversions.SA_from_SP(ds["sal"], ds["pres"], ds["lon"], ds["lat"])
ds["theta"] = gsw.conversions.pt0_from_t(ds["salinity_absolute"], ds["temp"], ds["pres"])

# Compute AOU
ds["aou"] = ks.parameterisations.aou_GG92(oxygen=ds["oxy"], temperature=ds["theta"], salinity=ds["sal"])[0]
ds["aou_sigma"] = ds["uncer"]

# Compute Density
ds["density"] = gsw.density.rho(ds['salinity_absolute'], ds['theta'], ds['pres'])

# Compute DIC Components
print("Calculating DIC components...")
sp_constant_aou_mean = 117 / -170  # Soft-tissue pump constant
sigma_sp = 0.092  # Uncertainty for soft-tissue pump
ds['DIC_nat'] = (-1 * sp_constant_aou_mean) * ds['aou']
print("DIC calculations complete.")

# =============================================================================
# 2. Load & Process MLD Data (Masking Above MLD)
# =============================================================================

print("Loading and interpolating MLD data...")
mld = xr.open_dataset(
    "/home/ldelaigue/Documents/Python/AoE_SVD/DATA/mld_DR003_c1m_reg2.0.nc"
)

# Shift longitudes in mld to match ds's range of [-180, 180]
mld = mld.assign_coords(lon=((mld.lon + 180) % 360) - 180).sortby('lon')

# Interpolate the MLD data to match ds' lat/lon grid
max_mld = mld['mld'].max(dim='time').interp(lat=ds['lat'], lon=ds['lon'], method='nearest')

del mld

# Assign the interpolated MLD values to a new variable in ds
ds['MLD'] = max_mld

del max_mld

# Expand MLD to match the dimensions required for comparison (along the 'pres' dimension)
MLD_expanded = ds['MLD'].expand_dims({'pres': len(ds['pres'])}, axis=1)
MLD_expanded['pres'] = ds['pres'].values  # Ensure that 'pres' values are correctly assigned

# Create a mask where pressure is greater than MLD
mask_below_mld = ds['pres'] > MLD_expanded

del MLD_expanded

# Apply the mask across the dataset to set values outside the mask to NaN
ds_below_mld = ds.where(mask_below_mld)

del ds

print("MLD masking complete. Processing values only below MLD.")

# =============================================================================
# 3. Monte Carlo Setup
# =============================================================================

num_iterations = 10
print(f"Running Monte Carlo with {num_iterations} iterations...")

rng = np.random.default_rng()

mean_50_depth = None
var_50_depth = None  

# =============================================================================
# 4. Monte Carlo Loop
# =============================================================================

for i in range(1, num_iterations + 1):
    print(f"Iteration {i}/{num_iterations}")

    # Generate perturbed dataset
    mc_sample = ds_below_mld.copy(deep=True)
    mc_sample['aou'] += rng.normal(0, mc_sample['aou_sigma'])

    # Perturb `DIC_nat` constant
    sp_constant_aou = rng.normal(sp_constant_aou_mean, sigma_sp)
    mc_sample['DIC_nat'] = (-1 * sp_constant_aou) * mc_sample['aou']

    # Compute cumulative DIC_nat below MLD
    mc_sample['DIC_cumulative'] = mc_sample['DIC_nat'].cumsum(dim='pres')

    # Compute 50% DIC sequestration depth
    def find_shallowest_depth(pres, cumulative, threshold):
        mask = cumulative >= threshold
        return np.min(pres[mask]) if np.any(mask) else np.nan

    dic_50_depth = xr.apply_ufunc(
        find_shallowest_depth,
        mc_sample['pres'],
        mc_sample['DIC_cumulative'],
        mc_sample['DIC_cumulative'].isel(pres=-1) * 0.5,
        input_core_dims=[['pres'], ['pres'], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    # Update Online Mean & Variance (Welford's Method)
    if mean_50_depth is None:
        mean_50_depth = dic_50_depth
        var_50_depth = dic_50_depth * 0  # Initialize variance as zeros
    else:
        delta = dic_50_depth - mean_50_depth
        new_mean_50_depth = mean_50_depth + delta / i
        var_50_depth = ((i - 1) * var_50_depth + delta * (dic_50_depth - new_mean_50_depth)) / i
        mean_50_depth = new_mean_50_depth

# =============================================================================
# 5. Compute Final Standard Deviation (AFTER Monte Carlo Loop)
# =============================================================================

dic_50_depth_uncertainty = np.sqrt(var_50_depth)  # Final standard deviation

# =============================================================================
# 6. Final Dataset
# =============================================================================

final_ds = xr.Dataset({
    'DIC_50_depth': mean_50_depth,
    'DIC_50_depth_uncertainty': dic_50_depth_uncertainty
})

# =============================================================================
# 7. Save Results
# =============================================================================

output_path = "/home/ldelaigue/Documents/Python/AoE_SVD/thesis/post_thesis_submission/DATA/04_DIC_sequestration_50_depth.nc"
final_ds.to_netcdf(output_path)

print(f"Monte Carlo simulations completed successfully. File saved: {output_path}")
