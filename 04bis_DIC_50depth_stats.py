# import numpy as np
# import xarray as xr
# from scipy.stats import linregress, ttest_ind
# import pandas as pd
# from statsmodels.nonparametric.smoothers_lowess import lowess
# import matplotlib.dates as mdates

# print("Loading dataset...")
# ds = xr.open_dataset(
#     "/home/ldelaigue/Documents/Python/AoE_SVD/thesis/post_thesis_submission/DATA/04_DIC_sequestered_50_depth.nc"
# )
# print("Dataset loaded.")

# # Subset Dataset
# print("Ignoring subsetting...")
# # print("Subsetting dataset to 20 pixels...")
# # ds = ds.isel(lat=slice(0, 20), lon=slice(0, 20))
# # print("Subset complete.")

# print("Converting time to numeric format...")

# # Convert time to Matplotlib datenum (keeps time in days)
# ds = ds.assign_coords(time=("time", mdates.date2num(ds["time"].values)))

# print("Time conversion complete.")

# def loess_smooth(x, y, frac=0.5):
#     smoothed = lowess(y, x, frac=frac, return_sorted=True)
#     return smoothed[:, 0], smoothed[:, 1]

# def analyze_dic_depth(ds):
#     """Apply statistical calculations on the model dataset and return as an xarray Dataset."""
    
#     print("Starting statistical analysis...")
#     lat_vals = ds.lat.values
#     lon_vals = ds.lon.values
#     print(f"Dataset dimensions - Lat: {len(lat_vals)}, Lon: {len(lon_vals)}")
    
#     results = {
#         "Slope_per_day": ("lat", "lon"),
#         "Slope_per_year": ("lat", "lon"),
#         "Slope_uncertainty_per_year": ("lat", "lon"),
#         "R_squared": ("lat", "lon"),
#         "P_value": ("lat", "lon"),
#         "Q1_mean": ("lat", "lon"),
#         "Q1_mean_uncertainty": ("lat", "lon"),
#         "Q4_mean": ("lat", "lon"),
#         "Q4_mean_uncertainty": ("lat", "lon"),
#         "Q_difference": ("lat", "lon"),
#         "Q_difference_uncertainty": ("lat", "lon"),
#         "T_stat": ("lat", "lon"),
#         "T_test_P_value": ("lat", "lon"),
#         "Significance": ("lat", "lon"),
#         "Q4_mean_depth": ("lat", "lon")
#     }
    
#     data_vars = {key: (dims, np.full((len(lat_vals), len(lon_vals)), np.nan)) for key, dims in results.items()}
    
#     for i, lat in enumerate(lat_vals):
#         for j, lon in enumerate(lon_vals):
#             print(f"Processing lat {lat}, lon {lon}...")
#             try:
#                 time = ds.time.values
#                 dic_50_depth = ds.sel(lat=lat, lon=lon)['DIC_nat_50_depth'].values
                
#                 if np.all(np.isnan(dic_50_depth)):
#                     print(f"Skipping lat {lat}, lon {lon} - All NaN values.")
#                     continue
                
#                 smoothed_time, smoothed_dic_50_depth = loess_smooth(time, dic_50_depth, frac=0.5)
                
#                 q1_time_threshold = np.nanquantile(smoothed_time, 0.25)
#                 q3_time_threshold = np.nanquantile(smoothed_time, 0.75)
                
#                 early_values = smoothed_dic_50_depth[smoothed_time <= q1_time_threshold]
#                 late_values = smoothed_dic_50_depth[smoothed_time >= q3_time_threshold]
                
#                 early_mean = np.nanmean(early_values)
#                 late_mean = np.nanmean(late_values)
#                 diff = late_mean - early_mean
                
#                 early_std_err = np.nanstd(early_values) / np.sqrt(len(early_values))
#                 late_std_err = np.nanstd(late_values) / np.sqrt(len(late_values))
#                 diff_std_err = np.sqrt(early_std_err**2 + late_std_err**2)
                
#                 t_stat, p_val_ttest = ttest_ind(early_values, late_values, equal_var=False, nan_policy='omit')
#                 significance = 1 if p_val_ttest < 0.05 else 0
                
#                 slope, intercept, r_value, p_value, std_err = linregress(smoothed_time, smoothed_dic_50_depth)
#                 slope_per_year = slope * 365.25
#                 slope_per_year_uncertainty = std_err * 365.25
                
#                 q4_50_depth_mean = np.nanmean(late_values)
                
#                 data_vars["Slope_per_day"][1][i, j] = slope
#                 data_vars["Slope_per_year"][1][i, j] = slope_per_year
#                 data_vars["Slope_uncertainty_per_year"][1][i, j] = slope_per_year_uncertainty
#                 data_vars["R_squared"][1][i, j] = r_value**2
#                 data_vars["P_value"][1][i, j] = p_value
#                 data_vars["Q1_mean"][1][i, j] = early_mean
#                 data_vars["Q1_mean_uncertainty"][1][i, j] = early_std_err
#                 data_vars["Q4_mean"][1][i, j] = late_mean
#                 data_vars["Q4_mean_uncertainty"][1][i, j] = late_std_err
#                 data_vars["Q_difference"][1][i, j] = diff
#                 data_vars["Q_difference_uncertainty"][1][i, j] = diff_std_err
#                 data_vars["T_stat"][1][i, j] = t_stat
#                 data_vars["T_test_P_value"][1][i, j] = p_val_ttest
#                 data_vars["Significance"][1][i, j] = significance
#                 data_vars["Q4_mean_depth"][1][i, j] = q4_50_depth_mean
                
#             except Exception as e:
#                 print(f"⚠️ Error processing lat {lat}, lon {lon}: {e}")
    
#     summary_ds = xr.Dataset(
#         {key: (dims, data) for key, (dims, data) in data_vars.items()},
#         coords={"lat": lat_vals, "lon": lon_vals}
#     )
    
#     print("Statistical analysis complete.")
#     return summary_ds

# # Example usage
# print("Starting dataset analysis...")
# summary_ds = analyze_dic_depth(ds)
# print("Analysis complete. Saving dataset...")
# summary_ds.to_netcdf("/home/ldelaigue/Documents/Python/AoE_SVD/thesis/post_thesis_submission/DATA/04bis_DIC_50depth_stats.nc")
# print("Dataset saved successfully.")

import numpy as np
import xarray as xr
from scipy.stats import linregress, ttest_ind
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.dates as mdates

print("Loading dataset...")
ds = xr.open_dataset(
    "/home/ldelaigue/Documents/Python/AoE_SVD/thesis/post_thesis_submission/DATA/04_DIC_sequestered_50_depth.nc"
)
print("Dataset loaded.")

# Convert time to Matplotlib datenum (keeps time in days)
ds = ds.assign_coords(time=("time", mdates.date2num(ds["time"].values)))
print("Time conversion complete.")

def loess_smooth(x, y, frac=0.5):
    smoothed = lowess(y, x, frac=frac, return_sorted=True)
    return smoothed[:, 0], smoothed[:, 1]

def analyze_dic_depth(ds):
    print("Starting statistical analysis...")
    lat_vals = ds.lat.values
    lon_vals = ds.lon.values
    print(f"Dataset dimensions - Lat: {len(lat_vals)}, Lon: {len(lon_vals)}")
    
    results = {
        "Slope_per_day": ("lat", "lon"),
        "Slope_per_year": ("lat", "lon"),
        "Slope_uncertainty_per_year": ("lat", "lon"),
        "R_squared": ("lat", "lon"),
        "P_value": ("lat", "lon"),
        "Q1_mean": ("lat", "lon"),
        "Q1_mean_uncertainty": ("lat", "lon"),
        "Q4_mean": ("lat", "lon"),
        "Q4_mean_uncertainty": ("lat", "lon"),
        "Q_difference": ("lat", "lon"),
        "Q_difference_uncertainty": ("lat", "lon"),
        "T_stat": ("lat", "lon"),
        "T_test_P_value": ("lat", "lon"),
        "Significance": ("lat", "lon"),
        "Q4_mean_depth": ("lat", "lon")
    }
    
    data_vars = {key: (dims, np.full((len(lat_vals), len(lon_vals)), np.nan)) for key, dims in results.items()}
    
    for i, lat in enumerate(lat_vals):
        for j, lon in enumerate(lon_vals):
            print(f"Processing lat {lat}, lon {lon}...")
            try:
                time = ds.time.values
                dic_50_depth = ds.sel(lat=lat, lon=lon)['DIC_nat_50_depth'].values
                dic_50_depth_uncertainty = ds.sel(lat=lat, lon=lon)['DIC_nat_50_depth_uncertainty'].values
                
                if np.all(np.isnan(dic_50_depth)) or np.all(np.isnan(dic_50_depth_uncertainty)):
                    print(f"Skipping lat {lat}, lon {lon} - All NaN values.")
                    continue
                
                smoothed_time, smoothed_dic_50_depth = loess_smooth(time, dic_50_depth, frac=0.5)
                _, smoothed_dic_50_depth_uncertainty = loess_smooth(time, dic_50_depth_uncertainty, frac=0.5)
                
                slope, intercept, r_value, p_value, std_err = linregress(smoothed_time, smoothed_dic_50_depth)
                slope_per_year = slope * 365.25
                
                # Compute observational uncertainty contribution to the slope uncertainty
                mean_time = np.nanmean(smoothed_time)
                time_variance = np.nansum((smoothed_time - mean_time) ** 2)
                obs_slope_uncertainty = np.sqrt(np.nansum(smoothed_dic_50_depth_uncertainty ** 2) / time_variance)
                
                # Total uncertainty propagation
                slope_per_year_uncertainty = np.sqrt((std_err * 365.25) ** 2 + (obs_slope_uncertainty * 365.25) ** 2)
                
                data_vars["Slope_per_day"][1][i, j] = slope
                data_vars["Slope_per_year"][1][i, j] = slope_per_year
                data_vars["Slope_uncertainty_per_year"][1][i, j] = slope_per_year_uncertainty
                data_vars["R_squared"][1][i, j] = r_value**2
                data_vars["P_value"][1][i, j] = p_value
                
            except Exception as e:
                print(f"⚠️ Error processing lat {lat}, lon {lon}: {e}")
    
    summary_ds = xr.Dataset(
        {key: (dims, data) for key, (dims, data) in data_vars.items()},
        coords={"lat": lat_vals, "lon": lon_vals}
    )
    
    print("Statistical analysis complete.")
    return summary_ds

print("Starting dataset analysis...")
summary_ds = analyze_dic_depth(ds)
print("Analysis complete. Saving dataset...")
summary_ds.to_netcdf("/home/ldelaigue/Documents/Python/AoE_SVD/thesis/post_thesis_submission/DATA/04bis_DIC_50depth_stats.nc")
print("Dataset saved successfully.")
