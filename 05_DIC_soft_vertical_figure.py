import warnings
import numpy as np
import xarray as xr
import gsw
import koolstof as ks

warnings.filterwarnings('ignore')

print("Loading dataset...")
ds = xr.open_dataset(
    "/home/ldelaigue/Documents/Python/AoE_SVD/DATA/CONTENT_RESULTS/CONTENT_DIC_v2-1.nc"
)

print("Calculating absolute salinity and potential temperature...")
ds["salinity_absolute"] = gsw.conversions.SA_from_SP(ds["sal"], ds["pres"], ds["lon"], ds["lat"])
ds["theta"] = gsw.conversions.pt0_from_t(ds["salinity_absolute"], ds["temp"], ds["pres"])

print("Computing Apparent Oxygen Utilization (AOU)...")
ds["aou"] = ks.parameterisations.aou_GG92(
    oxygen=ds["oxy"], temperature=ds["theta"], salinity=ds["sal"]
)[0]

# Load MLD climatology (monthly)
print("Loading MLD climatology...")
mld_clim = xr.open_dataset("/home/ldelaigue/Documents/Python/AoE_SVD/thesis/post_thesis_submission/DATA/Argo_mixedlayers_monthlyclim_04142022.nc")

# Rename and prepare coordinates
mld_clim = mld_clim.rename({"iLAT": "lat", "iLON": "lon", "iMONTH": "month"})
mld_clim = mld_clim.assign_coords({
    "lat": mld_clim["lat"].values,
    "lon": mld_clim["lon"].values,
    "month": np.arange(1, 13)
})

# Extract month from ds.time
ds['month'] = ds['time'].dt.month

# Create time-varying MLD by indexing climatology with ds.month
print("Interpolating MLD for each time step...")
def get_mld_for_month(month):
    return mld_clim['mld_da_max'].sel(month=month).interp(lat=ds['lat'], lon=ds['lon'], method='nearest')

mld_time_varying = xr.concat([get_mld_for_month(m.item()) for m in ds['month']], dim='time')
mld_time_varying['time'] = ds['time']  # Set proper time axis

# Drop temporary 'month' variable to avoid merge conflict
ds = ds.drop_vars('month')

# Assign interpolated MLD
ds['MLD'] = mld_time_varying

# Compute DIC_soft
sp_constant_aou = 117 / -170  # mol C / mol O2
ds["DIC_soft"] = (-1 * sp_constant_aou) * ds["aou"]

# Save dataset with MLD and DIC_soft
output_ds = ds[["DIC_soft", "MLD"]]
output_path = "/home/ldelaigue/Documents/Python/AoE_SVD/thesis/post_thesis_submission/DATA/05_DIC_soft_with_MLD.nc"
output_ds.to_netcdf(output_path)

print(f"DIC_soft and MLD saved to {output_path}")
