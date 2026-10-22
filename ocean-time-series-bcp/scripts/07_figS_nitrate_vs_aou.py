"""
Step 7. Nitrate-based vs AOU-based DICsoft trends at each station (Reviewer 1, comment 4).

Replaces _SPOTS_8_nitrate_vs_aou_DICsoft.py.

    DICsoft_AOU = (117/170) dAOU/dt
    DICsoft_NO3 = (117/16)  dNO3/dt
    residual    = DICsoft_NO3 - DICsoft_AOU    (computed within each Monte Carlo iteration)

Both estimates use the same samples (DIC, TA, AOU, T, S and NO3 all present), pressure grid
and Monte Carlo draws as Fig. 4. A positive residual trend indicates increasing preformed
nitrate (in carbon units) and/or changing O2 disequilibrium at subduction.

In   data/processed/spots_bgc_pres_filtered.csv
Out  data/processed/stations/<station>_dicsoft_no3_aou.nc
     data/processed/dicsoft_no3_aou_layer_trends.csv     (includes residuals)
     ../figs/FigureS14.png                               (residuals not plotted)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import config
from bcpts import carbon, profiles, trends

LAYERS = [(100, 500), (500, 1000), (1000, 2000)]
FIG_XLIM = {"ALOHA": 2, "BATS": 2, "K2": 2, "KNOT": 2, "CVOO": 5, "IcelandSea": 6, "IrmingerSea": 6}


def run(df, station, rng):
    cfg = config.STATIONS[station]
    required = ["TCARBN", "ALKALI", "aou", "CTDTMP", cfg["salinity"], "NITRAT"]
    prof = profiles.interpolate_station(df, station, ["aou", "NITRAT"], required=required)
    mc = trends.monte_carlo_slopes(prof, ["aou", "NITRAT"], rng)

    soft_aou_it = config.R_C_O2 * mc["aou"]
    soft_no3_it = config.R_C_N * mc["NITRAT"]
    resid_it = soft_no3_it - soft_aou_it

    a_m, a_e = mc["aou"].mean(0), mc["aou"].std(0)
    n_m, n_e = mc["NITRAT"].mean(0), mc["NITRAT"].std(0)
    aou_soft, aou_soft_err = carbon.soft_from_aou(a_m, a_e)
    no3_soft, no3_soft_err = carbon.soft_from_nitrate(n_m, n_e)

    out = xr.Dataset(coords={"CTDPRS": prof["CTDPRS"].values})
    out["AOU_change"], out["AOU_error"] = ("CTDPRS", a_m), ("CTDPRS", a_e)
    out["NO3_change"], out["NO3_error"] = ("CTDPRS", n_m), ("CTDPRS", n_e)
    out["DICsoft_AOU_change"], out["DICsoft_AOU_error"] = ("CTDPRS", aou_soft), ("CTDPRS", aou_soft_err)
    out["DICsoft_NO3_change"], out["DICsoft_NO3_error"] = ("CTDPRS", no3_soft), ("CTDPRS", no3_soft_err)
    out["residual_change"] = ("CTDPRS", resid_it.mean(0))
    out["residual_error_mc"] = ("CTDPRS", resid_it.std(0))
    out["residual_error"] = ("CTDPRS", np.sqrt(aou_soft_err ** 2 + no3_soft_err ** 2))
    out = out.rolling(CTDPRS=config.ROLLING_WINDOW, center=True).mean()
    out.attrs.update(prof.attrs, units="umol kg-1 yr-1")
    out.to_netcdf(config.STATION_DIR / f"{station}_dicsoft_no3_aou.nc")
    return out


def layer_table(results):
    rows = []
    for station, ds in results.items():
        for top, bottom in LAYERS:
            lay = ds.sel(CTDPRS=slice(top, bottom))
            if lay.sizes["CTDPRS"] == 0 or float(lay.CTDPRS.max() - lay.CTDPRS.min()) < 0.5 * (bottom - top):
                continue
            row = dict(station=station, period=f"{ds.attrs['first_year']}-{ds.attrs['last_year']}",
                       n_profiles=ds.attrs["n_profiles"], layer_dbar=f"{top}-{bottom}")
            for v in ["DICsoft_AOU", "DICsoft_NO3", "residual"]:
                row[f"{v}_change"] = float(lay[f"{v}_change"].mean())
                # levels are strongly correlated after interpolation and smoothing:
                # report the mean level uncertainty (conservative)
                row[f"{v}_error"] = float(lay[f"{v}_error"].mean())
            row["residual_significant_1sigma"] = abs(row["residual_change"]) > row["residual_error"]
            rows.append(row)
    return pd.DataFrame(rows)


def plot(results):
    styles = {"DICsoft_AOU": (r"ΔDIC$_{\mathrm{soft}}$ (AOU)", "#136F63"),
              "DICsoft_NO3": (r"ΔDIC$_{\mathrm{soft}}$ (NO$_3^-$)", "#C2185B")}
    fig, axs = plt.subplots(2, 4, figsize=(16, 11), dpi=300)
    axs = axs.flatten()
    for i, (station, ds) in enumerate(results.items()):
        ax = axs[i]
        for v, (label, colour) in styles.items():
            m, e = ds[f"{v}_change"], ds[f"{v}_error"]
            ax.plot(m, ds.CTDPRS, color=colour, lw=2, label=label, zorder=2)
            ax.fill_betweenx(ds.CTDPRS, m - e, m + e, color=colour, alpha=0.15, lw=0, zorder=1)
        p = ds.CTDPRS.values[np.isfinite(ds["DICsoft_AOU_change"].values)]
        ax.set_ylim(p.max(), 0)
        ax.set_xlim(-FIG_XLIM[station], FIG_XLIM[station])
        ax.axvline(0, color="k", alpha=0.8, lw=1, zorder=0)
        ax.grid(alpha=0.3)
        ax.set_xlabel(f"Trend ({config.TREND_UNITS})")
        if i % 4 == 0:
            ax.set_ylabel("Pressure (dbar)")
        ax.text(0, 1.03, f"{'abcdefg'[i]}) {config.STATIONS[station]['label']} "
                         f"({ds.attrs['first_year']}-{ds.attrs['last_year']})",
                fontsize=12, transform=ax.transAxes)
    handles, labels = axs[0].get_legend_handles_labels()
    axs[7].legend(handles, labels, loc="center", fontsize=14, frameon=True)
    axs[7].axis("off")
    plt.subplots_adjust(wspace=0.35, hspace=0.3)
    fig.savefig(config.FIGS / "FigureS14.png", bbox_inches="tight", dpi=300)


def main():
    df = pd.read_csv(config.SPOTS_BGC_FILTERED, na_values=-999, low_memory=False)
    rng = np.random.default_rng(config.SEED)
    config.STATION_DIR.mkdir(parents=True, exist_ok=True)
    results = {s: run(df, s, rng) for s in config.STATIONS}
    table = layer_table(results)
    table.round(3).to_csv(config.PROCESSED / "dicsoft_no3_aou_layer_trends.csv", index=False)
    print(table.round(2).to_string())
    plot(results)


if __name__ == "__main__":
    main()
