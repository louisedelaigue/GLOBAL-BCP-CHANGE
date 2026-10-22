import warnings
import numpy as np
import xarray as xr
import gsw
import koolstof as ks

warnings.filterwarnings("ignore")

print("Loading dataset...")
ds = xr.open_dataset("data/CONTENT_DIC_v2-3.nc")

print("Calculating absolute salinity and potential temperature...")
ds["salinity_absolute"] = gsw.conversions.SA_from_SP(
    ds["sal"], ds["pres"], ds["lon"], ds["lat"]
)
ds["theta"] = gsw.conversions.pt0_from_t(
    ds["salinity_absolute"], ds["temp"], ds["pres"]
)

print("Computing Apparent Oxygen Utilization (AOU)...")
ds["aou"] = ks.parameterisations.aou_GG92(
    oxygen=ds["oxy"],
    temperature=ds["theta"],
    salinity=ds["sal"]
)[0]

# =============================================================================
# Load and process MLD climatology to MATCH the first script
# =============================================================================
print("Loading and interpolating NEW MLD climatology...")

mld_new = xr.open_dataset("data/Argo_mixedlayers_monthlyclim_04142022.nc")

# Annual maximum over months
mld_annual_max = mld_new["mld_da_max"].max(dim="iMONTH")

# Rename dims to match target grid
mld_annual_max = mld_annual_max.rename({"iLAT": "lat", "iLON": "lon"})

# Assign coordinate values
mld_annual_max = mld_annual_max.assign_coords({
    "lat": mld_new["lat"].values,
    "lon": mld_new["lon"].values
})

# Convert longitude from [-180, 180] to [0, 360)
mld_annual_max = mld_annual_max.assign_coords(
    lon=(mld_annual_max.lon % 360)
)

# Sort longitude after conversion
mld_annual_max = mld_annual_max.sortby("lon")

# Interpolate to ds grid
mld_interp = mld_annual_max.interp(
    lat=ds["lat"],
    lon=(ds["lon"] % 360),
    method="nearest"
)

# Assign MLD to main dataset
ds["MLD"] = mld_interp

# Clean up
del mld_new, mld_annual_max, mld_interp

print("New MLD climatology applied.")

# =============================================================================
# Compute DIC_soft
# =============================================================================
sp_constant_aou = 117 / -170  # mol C / mol O2
ds["DIC_soft"] = (-1 * sp_constant_aou) * ds["aou"]

# Save dataset with MLD and DIC_soft
output_ds = ds[["DIC_soft", "MLD"]]
output_path = "data/processing_steps/05_DIC_soft_with_MLD.nc"
output_ds.to_netcdf(output_path)

print(f"DIC_soft and MLD saved to {output_path}")