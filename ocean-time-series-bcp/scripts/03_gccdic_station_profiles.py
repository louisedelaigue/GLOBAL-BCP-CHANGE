"""
Step 3. Extract GCC-DIC v2.3 trend profiles at each station (+-2 deg box mean).

Replaces the 7 notebooks in model_series_notebooks/ (ALOHA, BATS, CVOO, IS, IrS, K2, KNOT),
which differed only in the box centre and output name.

In   GCC-DIC v2.3 CONTENT_DIC_v2-3.nc and CONTENT_AT_v2-3.nc (config.GCCDIC_DIR, or set
     the GCCDIC_DIR environment variable)
Out  data/processed/gccdic_stations/<name>.nc

The calculations, including the slope-uncertainty formula, are kept exactly as in the
notebooks so that results do not change. See README (known issues) for two points to check.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gsw
import koolstof as ks
import numpy as np
import xarray as xr

import config

BOX = 2.0                     # +- degrees around the station
BASE_DATE = np.datetime64("2004-01-15")


def to_180(ds):
    return ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon")


def subset(ds, lon, lat):
    return ds.sel(lat=slice(lat - BOX, lat + BOX), lon=slice(lon - BOX, lon + BOX))


def slope_uncertainty(x, y, y_unc):
    """2-sigma slope uncertainty as implemented in the notebooks."""
    n = len(x)
    denominator = (n * np.sum(x ** 2) - np.sum(x) ** 2) ** 2
    if denominator == 0:
        return np.inf
    return 2 * np.sqrt(n * np.sum(y ** 2 * y_unc ** 2) / denominator)


def slope_unc_field(ds, var, unc, x):
    return xr.apply_ufunc(lambda y, u: slope_uncertainty(x, y, u), ds[var], ds[unc],
                          input_core_dims=[["datenum"], ["datenum"]], vectorize=True,
                          dask="parallelized", output_dtypes=[float])


def per_year_slope(da):
    return da.polyfit(dim="datenum", deg=1, skipna=True).polyfit_coefficients.sel(degree=1) * 365.25


def station_profile(dic_all, ta_all, lon, lat):
    ds = subset(dic_all, lon, lat)
    ta = subset(ta_all, lon, lat)

    sa = gsw.SA_from_SP(ds["sal"], ds["pres"], ds["lon"], ds["lat"])
    theta = gsw.pt0_from_t(sa, ds["temp"], ds["pres"])
    ds["AOU"] = ks.parameterisations.aou_GG92(oxygen=ds["oxy"], temperature=theta, salinity=ds["sal"])[0]
    ds["TA"], ds["TA_uncertainty"] = ta["AT"], ta["AT_sigma"]
    ds = ds.rename({"uncer": "AOU_uncertainty", "CT": "DIC", "CT_sigma": "DIC_uncertainty"})
    ds = ds[["AOU", "AOU_uncertainty", "DIC", "DIC_uncertainty", "TA", "TA_uncertainty"]]

    time = np.array(ds["time"].dt.strftime("%Y-%m-%d"), dtype="datetime64[D]")
    ds = ds.assign_coords(datenum=("time", (time - BASE_DATE).astype(int)))
    ds = ds.swap_dims({"time": "datenum"}).drop_vars("time")

    d_dic, d_ta, d_aou = per_year_slope(ds["DIC"]), per_year_slope(ds["TA"]), per_year_slope(ds["AOU"])
    soft = config.R_C_O2 * d_aou
    carb = 0.5 * (d_ta + config.R_N_O2 * d_aou)
    anth = d_dic - (soft + carb)

    x = ds["datenum"].values.astype(float)
    s_dic = slope_unc_field(ds, "DIC", "DIC_uncertainty", x) * 365.25
    s_aou = slope_unc_field(ds, "AOU", "AOU_uncertainty", x) * 365.25
    s_ta = slope_unc_field(ds, "TA", "TA_uncertainty", x) * 365.25
    k, k_err, c, c_err = config.R_C_O2, config.R_C_O2_ERR, -config.R_N_O2, config.R_N_O2_ERR
    s_soft = np.sqrt((k * s_aou) ** 2 + (s_aou * k_err) ** 2)
    s_carb = np.sqrt((0.5 * s_ta) ** 2 + (0.5 * c * s_aou) ** 2
                     + (0.5 * s_ta * c_err) ** 2 + (0.5 * s_aou * c_err) ** 2)
    s_anth = np.sqrt(s_dic ** 2 + s_soft ** 2 + s_carb ** 2)

    out = xr.Dataset({
        "ct_rate_of_change_per_year": d_dic, "soft_pump_aou_change": soft,
        "carb_pump_change": carb, "co2_anth_aou_change": anth,
        "sigma_ct_rate_of_change_per_year": s_dic, "sigma_soft_pump_aou_change": s_soft,
        "sigma_carb_pump_change": s_carb, "sigma_co2_anth_aou_change": s_anth,
    })
    return out.mean(dim=["lat", "lon"]).compute()


def main():
    dic_all = to_180(xr.open_dataset(config.GCCDIC_DIC))
    ta_all = to_180(xr.open_dataset(config.GCCDIC_TA))
    config.GCCDIC_STATION_DIR.mkdir(parents=True, exist_ok=True)
    for station, cfg in config.STATIONS.items():
        print(station)
        lon, lat = cfg["gccdic_centre"]
        station_profile(dic_all, ta_all, lon, lat).to_netcdf(
            config.GCCDIC_STATION_DIR / f"{cfg['gccdic_name']}.nc")


if __name__ == "__main__":
    main()
