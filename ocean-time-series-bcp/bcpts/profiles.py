"""Profile selection and interpolation of bottle data onto a common pressure grid."""
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr
from scipy import interpolate
from sklearn.cluster import MeanShift

import config


def cluster_pchip(data, x, y, bandwidth):
    """Cluster the samples of one profile along `x` (MeanShift), average `y` per cluster and
    return a PCHIP interpolator through the cluster centres.

    Identical to get_cluster_pchip() in the old XX_01_pres_cluster_pchip.py scripts.
    """
    x_data, y_data = data[x], data[y]
    valid = ~(x_data.isna() | y_data.isna())
    dx, dy = x_data[valid], y_data[valid]
    out = {"x_min": x_data.min(), "x_max": x_data.max(), "pchip": None}
    if len(dx) == 0:
        return pd.Series(out)
    ms = MeanShift(bandwidth=bandwidth).fit(np.vstack([dx]).T)
    cx = ms.cluster_centers_.ravel()
    if len(cx) < 2:
        return pd.Series(out)
    cy = np.array([np.mean(dy[ms.labels_ == i]) for i in range(len(cx))])
    if dx.max() not in cx:                      # always include the deepest sample
        cx = np.append(cx, dx.max())
        cy = np.append(cy, dy[dx == dx.max()].mean())
    order = np.argsort(cx)
    out["pchip"] = interpolate.PchipInterpolator(cx[order], cy[order])
    return pd.Series(out)


def select_samples(df, station, required):
    """Station subset with the profile filters of the old XX_01 scripts.

    `required`: variables that must all be present in a sample for it to be kept.
    """
    cfg = config.STATIONS[station]
    df = df[df["TimeSeriesSite"] == station].copy()

    if cfg["profile_id_800"]:
        cast = df["DATE"].astype(str) + "_" + df["STNNBR"].astype(str) + "_" + df["CASTNO"].astype(str)
        deepest = df.groupby(cast)["CTDPRS"].max()
        df = df[cast.isin(deepest[deepest >= 800].index)]

    for col in ["CTDPRS", *required]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["CTDPRS"].notna()]
    df = df.dropna(subset=required)
    df = df[np.isfinite(df[required]).all(axis=1)]

    df["DATE"] = pd.to_datetime(df["DATE"])
    df["datenum"] = mdates.date2num(df["DATE"])
    df = df.groupby("datenum").filter(
        lambda g: g["CTDPRS"].max() >= cfg["profile_min_max"] and (g["CTDPRS"] <= 15).any())
    return df.sort_values("CTDPRS")


def interpolate_station(df, station, variables, required=None):
    """Interpolate each profile (one per sampling date) onto a common pressure grid.

    The grid spans the pressure range covered by every profile and every variable in
    `required` (default: `variables`), with config.N_LEVELS levels, as in the old XX_01 scripts.
    Returns an xarray Dataset (CTDPRS, datenum) holding `variables`.
    """
    cfg = config.STATIONS[station]
    required = list(required or variables)
    samples = select_samples(df, station, required)

    fits = {v: samples.groupby("datenum", group_keys=False).apply(
        cluster_pchip, x="CTDPRS", y=v, bandwidth=cfg["bandwidth"]) for v in required}

    level_min = max(f["x_min"].max() for f in fits.values())
    level_max = min(f["x_max"].min() for f in fits.values())
    levels = np.linspace(level_min, level_max, config.N_LEVELS)
    dates = fits[required[0]].index.to_numpy()

    data = {}
    for v in variables:
        arr = np.full((levels.size, dates.size), np.nan)
        for i, d in enumerate(dates):
            p = fits[v].loc[d]["pchip"]
            if callable(p):
                arr[:, i] = p(levels)
        data[v] = (("CTDPRS", "datenum"), arr)

    ds = xr.Dataset(data, coords={"CTDPRS": levels, "datenum": dates.astype(np.int32)})
    ds.attrs.update(station=station, n_profiles=int(dates.size),
                    first_year=int(samples["DATE"].dt.year.min()),
                    last_year=int(samples["DATE"].dt.year.max()))
    return ds
