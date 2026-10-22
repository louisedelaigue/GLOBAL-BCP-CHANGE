"""
Step 6. Depth above which 50% of the soft-tissue carbon below the winter mixed layer is stored
(DIC50 depth), its trend and its change between the first and last quarter of each record.

Replaces _SPOTS_7_sequestration_depths.py (and the superseded XX_05_sequestration_depth.py x 7).

In   data/processed/stations/<station>_profiles.nc, data/raw/mld_DR003_c1m_reg2.0.nc
Out  data/processed/sequestration_depth_summary.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gsw
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import linregress, ttest_ind
from statsmodels.nonparametric.smoothers_lowess import lowess

import config


def max_mld(lon, lat):
    mld = xr.load_dataset(config.MLD_CLIMATOLOGY, decode_times=False)  # dates unused (climatology)
    mld = mld.assign_coords(lon=((mld.lon + 180) % 360) - 180).sortby("lon")
    return mld["mld"].max(dim="time").sel(lat=lat, lon=lon, method="nearest")


def dic50_depth(ds, salinity, lon, lat):
    """Pressure above which half of the DICsoft inventory below the max winter MLD is found."""
    sa = gsw.SA_from_SP(ds[salinity], ds["CTDPRS"], lon, lat)
    theta = gsw.pt0_from_t(sa, ds["CTDTMP"], ds["CTDPRS"])
    rho = gsw.rho(sa, theta, ds["CTDPRS"])
    dicsoft = config.R_C_O2 * ds["aou"]

    below = (ds["CTDPRS"] > max_mld(lon, lat))
    mol_m3 = (dicsoft / 1e6 * rho).where(below)
    mol_m2 = mol_m3 * ds["CTDPRS"].diff("CTDPRS")
    total = mol_m2.sum("CTDPRS")
    cumulative = mol_m2.cumsum("CTDPRS")

    def shallowest(p, cum, threshold):
        mask = cum >= threshold
        return np.min(p[mask]) if np.any(mask) else np.nan

    return xr.apply_ufunc(shallowest, mol_m2["CTDPRS"], cumulative, total * 0.5,
                          input_core_dims=[["CTDPRS"], ["CTDPRS"], []], vectorize=True)


def summarise(station, cfg):
    ds = xr.load_dataset(config.STATION_DIR / f"{station}_profiles.nc")
    lon, lat = cfg["mean_coords"]
    depth = dic50_depth(ds, cfg["salinity"], lon, lat)
    t, z = depth["datenum"].values.astype(float), depth.values

    sm = lowess(z, t, frac=0.5, return_sorted=True)
    ts, zs = sm[:, 0], sm[:, 1]
    early = zs[ts <= np.nanquantile(ts, 0.25)]
    late = zs[ts >= np.nanquantile(ts, 0.75)]
    early_se = np.nanstd(early) / np.sqrt(len(early))
    late_se = np.nanstd(late) / np.sqrt(len(late))
    t_stat, p_t = ttest_ind(early, late, equal_var=False, nan_policy="omit")
    fit = linregress(ts, zs)
    return {
        "Station": station, "Slope (m/day)": fit.slope, "Slope (m/year)": fit.slope * 365.25,
        "Slope Uncertainty (m/year)": fit.stderr * 365.25, "R²": fit.rvalue ** 2, "P-value": fit.pvalue,
        "Q1 Mean": np.nanmean(early), "Q1 Mean Uncertainty": early_se,
        "Q4 Mean": np.nanmean(late), "Q4 Mean Uncertainty": late_se,
        "Q Difference": np.nanmean(late) - np.nanmean(early),
        "Q Difference Uncertainty": np.sqrt(early_se ** 2 + late_se ** 2),
        "T-stat": t_stat, "T-test P-value": p_t, "Significance": "S" if p_t < 0.05 else "IS",
        "Q4_mean_depth": np.nanmean(late),
    }


def main():
    df = pd.DataFrame([summarise(s, c) for s, c in config.STATIONS.items()])
    df.to_csv(config.PROCESSED / "sequestration_depth_summary.csv", index=False)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
