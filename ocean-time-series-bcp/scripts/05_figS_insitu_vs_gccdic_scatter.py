"""
Step 5. SI figure: in situ vs GCC-DIC v2.3 DICsoft trend at each station, both averaged over
the same pressure range.

x  in situ DICsoft trend (script 02), thickness-weighted mean over the station's pressure range
   (top of the profile to 2000 dbar, or the bottom of the in situ profile if shallower).
y  GCC-DIC v2.3 DICsoft trend at the pixel nearest the station (script 03b), thickness-weighted
   mean over the same pressure range. Uncertainty as for the global mean in Fig. 1 (Reviewer 1,
   comment 2): autocorrelation-adjusted regression error + C:O2 ratio + GOBAI-O2 vs ML4O2
   structural term (01A1 in the global repo), in quadrature.

Replaces _SPOTS_6_TS_GOBAI_TOTAL_CHANGE_LINREGRESS_umolkgyr.py, which used a GCC-DIC v2.1 map
averaged over 0-2000 dbar at every station and fixed GCC-DIC errors (0.04-0.07).

In   data/processed/stations/<station>_trends.nc
     data/processed/gccdic_stations/<name>_pixel.nc
     ../data/processing_steps/01A1_O2_product_structural_uncertainty.nc (config.O2_STRUCTURAL)
Out  data/processed/insitu_vs_gccdic_dicsoft.csv
     ../figs/FigureS11.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import config
from bcpts.trends import trend_ar1

MARKERS = {"ALOHA": "o", "BATS": "s", "K2": "^", "KNOT": "D", "CVOO": "P",
           "IrmingerSea": "X", "IcelandSea": "v"}
ORDER = ["ALOHA", "BATS", "K2", "KNOT", "CVOO", "IrmingerSea", "IcelandSea"]
MAX_PRES = 2000


def thickness_weights(pres):
    """Layer thickness from level centres (as compute_layer_thickness_from_centers in the global repo)."""
    p = np.asarray(pres, dtype=float)
    edges = np.concatenate([[p[0] - 0.5 * (p[1] - p[0])], 0.5 * (p[:-1] + p[1:]),
                            [p[-1] + 0.5 * (p[-1] - p[-2])]])
    return np.diff(edges)


def insitu_mean(ins, top, bottom):
    """Thickness-weighted mean in situ DICsoft trend and error over [top, bottom] (as before)."""
    ins = ins.sel(CTDPRS=slice(top, bottom))
    w = ins["CTDPRS"].diff("CTDPRS").fillna(0)
    mean = float((ins["DICsoft_change"] * w).sum(skipna=True) / w.sum())
    err = float((ins["DICsoft_error"] * w).sum(skipna=True) / w.sum())
    return mean, err


def structural_sigma(pres):
    """sigma_struct_AOU (GOBAI-O2 vs ML4O2, 01A1) on the GCC-DIC pressure levels.

    Held constant above the shallowest and below the deepest (1000 m) level, as in Fig. 1.
    """
    st = xr.open_dataset(config.O2_STRUCTURAL)
    return np.interp(pres, st["pres"].values, st["sigma_struct_AOU"].values)


def gccdic_mean(pix, top, bottom):
    """GCC-DIC DICsoft trend averaged over [top, bottom], with three uncertainty terms
    combined in quadrature, as for the global mean in Fig. 1 (Reviewer 1, comment 2):

    regression  trend of the deseasonalised, thickness-weighted layer-mean AOU time series,
                lag-1 autocorrelation-adjusted standard error
    ratio       C:O2 uncertainty x trend
    structural  GOBAI-O2 vs ML4O2 AOU trend difference, layer-averaged
    """
    pres = pix["pres"].values
    keep = (pres >= top) & (pres <= bottom)
    w = thickness_weights(pres)[keep]
    aou = pix["AOU"].isel(pres=np.where(keep)[0])
    ok = np.isfinite(aou)
    layer = (aou * w).where(ok).sum("pres") / (ok * w).sum("pres")
    layer = layer.groupby("time.month") - layer.groupby("time.month").mean("time")
    t = ((pix["time"] - pix["time"][0]) / np.timedelta64(1, "D")).values / 365.25
    slope, se, neff = trend_ar1(layer.values, t)

    k = config.R_C_O2
    sigma_struct_aou = np.sum(structural_sigma(pres[keep]) * w) / np.sum(w)
    terms = {"regression": k * se, "ratio": config.R_C_O2_ERR * abs(slope),
             "structural": k * sigma_struct_aou}
    total = float(np.sqrt(sum(v ** 2 for v in terms.values())))
    return k * slope, total, terms, neff


def main():
    rows = []
    for station, cfg in config.STATIONS.items():
        ins = xr.open_dataset(config.STATION_DIR / f"{station}_trends.nc")
        pix = xr.open_dataset(config.GCCDIC_STATION_DIR / f"{cfg['gccdic_name']}_pixel.nc")
        # overlapping pressure range, defined as in the original script
        p_ins = ins["CTDPRS"].sel(CTDPRS=slice(0, MAX_PRES))
        top = max(float(p_ins.min()), float(pix.pres.min()))
        bottom = min(float(p_ins.max()), float(pix.pres.max()))
        x, x_err = insitu_mean(ins, top, bottom)
        y, y_err, terms, neff = gccdic_mean(pix, top, bottom)
        rows.append({
            "station": station,
            "pressure_range_dbar": f"{top:.0f}-{bottom:.0f}",
            "insitu_period": f"{ins.attrs['first_year']}-{ins.attrs['last_year']}",
            "insitu_DICsoft": x, "insitu_DICsoft_error": x_err,
            "gccdic_DICsoft": y, "gccdic_DICsoft_error": y_err,
            **{f"gccdic_error_{k}": v for k, v in terms.items()}, "gccdic_n_eff": neff,
            "gccdic_pixel": f"{pix.attrs['pixel_lat']:.1f}N {pix.attrs['pixel_lon']:.1f}E",
        })
    df = pd.DataFrame(rows)
    df.to_csv(config.PROCESSED / "insitu_vs_gccdic_dicsoft.csv", index=False)
    print(df.round(3).to_string())

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    colours = {s: cm.viridis(i / (len(ORDER) - 1)) for i, s in enumerate(ORDER)}
    for _, r in df.iterrows():
        ax.errorbar(r["insitu_DICsoft"], r["gccdic_DICsoft"], xerr=r["insitu_DICsoft_error"],
                    yerr=r["gccdic_DICsoft_error"], fmt=MARKERS[r["station"]],
                    color=colours[r["station"]], label=config.STATIONS[r["station"]]["label"],
                    capsize=3, markersize=7, markeredgecolor="black", markeredgewidth=0.6, zorder=2)
    ax.plot([-0.5, 2.5], [-0.5, 2.5], ls="--", color="black", label="1:1 line", zorder=1)
    ax.set_xlabel(rf"$\Delta$DIC$_{{\mathrm{{soft(in\ situ)}}}}$ ({config.TREND_UNITS})")
    ax.set_ylabel(rf"$\Delta$DIC$_{{\mathrm{{soft(GCC-DIC)}}}}$ ({config.TREND_UNITS})")
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.grid(alpha=0.3)
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.1), ncol=4, frameon=False)
    plt.tight_layout(rect=[0, 0.2, 1, 1])
    fig.savefig(config.FIGS / "FigureS11.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
