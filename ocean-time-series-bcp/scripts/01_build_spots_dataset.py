"""
Step 1. Build the in situ dataset from SPOTS (Lange et al., 2024) plus the BATS bottle file,
and select the profiles used in the analysis.

Replaces _SPOTS_02_calculate_BGC.py and _SPOTS_03B_pres_check_ranges.py.

In   data/raw/spots.csv, data/raw/bats_bottle.txt
Out  data/processed/spots_bgc.csv                (all samples with DIC, TA, O2 and NO3)
     data/processed/spots_bgc_pres_filtered.csv  (profiles spanning each station's pressure window)
     figs/profile_selection.png                  (SI figure: selected vs rejected samples)
"""
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gsw
import koolstof as ks
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyCO2SYS as pyco2

import config

N_STEPS = 9
_done = [0]


@contextmanager
def step(message):
    """Print '[i/N] message ... done (t s)' around each stage so progress is visible."""
    _done[0] += 1
    t0 = time.time()
    print(f"[{_done[0]}/{N_STEPS}] {message} ...", end=" ", flush=True)
    yield
    print(f"done ({time.time() - t0:.1f} s)", flush=True)


def monthly_mean_per_pressure(df):
    """Average duplicate samples at the same pressure within each calendar month.

    The DATE of each monthly profile is set to the last day of the month,
    as pandas resample('M') did in the original script.
    """
    df = df.copy()
    df["DATE"] = df["DATE"] + pd.offsets.MonthEnd(0)
    numeric = df.select_dtypes("number").columns.drop("CTDPRS")
    out = df.groupby(["DATE", "CTDPRS"], as_index=False)[list(numeric)].mean()
    return out


def load_bats():
    bats = pd.read_csv(config.BATS_BOTTLE, skiprows=57, sep=r"\s*,\s*|\s+", engine="python",
                       comment="/", na_values=-999)
    bats["NITRAT"] = bats["NO31"] - bats["NO21"]               # NO3+NO2 minus NO2
    lon, lat = config.BATS_POSITION
    bats["CTDPRS"] = gsw.p_from_z(-pd.to_numeric(bats["Depth"]), bats["latN"])
    bats = bats.rename(columns={"Temp": "CTDTMP", "CTD_S": "CTDSAL", "Sal1": "SALNTY",
                                "yyyymmdd": "DATE", "CO2": "TCARBN", "Alk": "ALKALI",
                                "O2(1)": "OXYGEN"})
    bats["DATE"] = pd.to_datetime(bats["DATE"].astype(int).astype(str), format="%Y%m%d")
    bats = bats[["DATE", "CTDPRS", "CTDTMP", "CTDSAL", "SALNTY", "TCARBN", "ALKALI", "OXYGEN", "NITRAT"]]
    bats = monthly_mean_per_pressure(bats)
    bats["TimeSeriesSite"], bats["LONGITUDE"], bats["LATITUDE"] = "BATS", lon, lat
    return bats


def fill_dic_from_ph(df):
    """Calculate DIC from TA and pH where DIC was not measured (as in _SPOTS_02_calculate_BGC.py).

    Same PyCO2SYS inputs as the original row-by-row version, passed as arrays in a single call
    (identical results to within 1e-12 umol kg-1; ~0.3 s per row before, <1 s in total now).
    """
    missing = df["TCARBN"].isna() & df["ALKALI"].notna() & df["PH_TOT"].notna()
    if missing.any():
        m = df[missing]
        df.loc[missing, "TCARBN"] = pyco2.sys(
            par1=m["ALKALI"].values, par1_type=1, par2=m["PH_TOT"].values, par2_type=3,
            salinity=m["salinity"].values, temperature=25, pressure=0,
            temperature_out=m["CTDTMP"].values, pressure_out=m["CTDPRS"].values)["dic"]
    return df, int(missing.sum())


def build():
    with step("Reading spots.csv"):
        df = pd.read_csv(config.SPOTS_CSV, na_values=-999, low_memory=False)
        df["DATE"] = pd.to_datetime(df["DATE"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")

    with step("Monthly averaging ALOHA"):
        # ALOHA: monthly mean per pressure level, fixed position
        aloha = monthly_mean_per_pressure(df[df["TimeSeriesSite"] == "ALOHA"])
        aloha["TimeSeriesSite"] = "ALOHA"
        aloha["LONGITUDE"], aloha["LATITUDE"] = config.ALOHA_POSITION

    with step("Reading BATS bottle file"):
        bats = load_bats()
        df = pd.concat([df[df["TimeSeriesSite"] != "ALOHA"], aloha, bats], ignore_index=True).copy()
        df["salinity"] = df["SALNTY"].combine_first(df["CTDSAL"])
        df["oxygen"] = df["OXYGEN"].combine_first(df["CTDOXY"])

    with step("Filling missing DIC from TA and pH"):
        df, n_filled = fill_dic_from_ph(df)
    print(f"      {n_filled} samples filled")

    with step("Keeping samples with DIC, TA, S, O2 and NO3; uncertainties"):
        df = df.dropna(subset=["TCARBN", "ALKALI"])
        df = df.dropna(subset=["SALNTY", "CTDSAL"], how="all")
        df = df.dropna(subset=["CTDOXY", "OXYGEN"], how="all")
        df = df.dropna(subset=["NITRAT"])

        # measurement uncertainty: site mean of sqrt(precision^2 + accuracy^2)
        for var in ["TCARBN", "ALKALI", "OXYGEN", "NITRAT"]:
            p, a = df.get(f"{var}_P"), df.get(f"{var}_A")
            bias = np.sqrt(p ** 2 + a ** 2) if p is not None and a is not None else np.nan
            df[f"{var}_bias"] = pd.Series(bias, index=df.index).groupby(df["TimeSeriesSite"]).transform("mean")

    with step("Calculating AOU"):
        df["aou"] = ks.parameterisations.aou_GG92(oxygen=df["oxygen"], temperature=df["CTDTMP"],
                                                   salinity=df["salinity"])[0]
        df["aou_bias"] = df["OXYGEN_bias"]
        df = df[df["CTDPRS"] >= 0]

    # merge sub-sites, drop excluded sites
    for prefix in ["GIFT", "RADCOR"]:
        df.loc[df["TimeSeriesSite"].astype(str).str.startswith(prefix), "TimeSeriesSite"] = prefix
    df = df[~df["TimeSeriesSite"].isin(config.EXCLUDED_SITES)]

    # TEOS-10 latitude range
    df.loc[df["LATITUDE"] > 60, "LATITUDE"] = 60
    df.loc[df["LATITUDE"] < -60, "LATITUDE"] = -60
    return df


def select_profiles(df):
    """Keep sampling dates whose samples span the station's pressure window."""
    keep = []
    for station, cfg in config.STATIONS.items():
        top, bottom = cfg["pres_window"]
        site = df[df["TimeSeriesSite"] == station]
        rng = site.groupby("DATE")["CTDPRS"].agg(["min", "max"])
        ok = rng[(rng["min"] <= top) & (rng["max"] >= bottom)].index
        keep.append(site[site["DATE"].isin(ok)])
    return pd.concat(keep, ignore_index=True)


def plot_selection(df, sel):
    fig, axs = plt.subplots(4, 2, figsize=(10, 14), dpi=300)
    axs = axs.flatten()
    for i, (station, cfg) in enumerate(sorted(config.STATIONS.items())):
        ax = axs[i]
        a, s = df[df["TimeSeriesSite"] == station], sel[sel["TimeSeriesSite"] == station]
        h1 = ax.scatter(a["DATE"], a["CTDPRS"], color="black", alpha=0.26, zorder=0)
        h2 = ax.scatter(s["DATE"], s["CTDPRS"], color="blue", alpha=0.7, zorder=1)
        for y in cfg["pres_window"]:
            h3 = ax.axhline(y, color="red", ls="--", alpha=0.5, lw=2)
        ax.set_title(f"{'abcdefg'[i]}) {cfg['label']}", loc="left")
        ax.set_xlabel("Year")
        ax.set_ylabel("Pressure (dbar)")
        ax.grid(True)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_ylim(2000, 0)
    axs[7].axis("off")
    axs[7].legend([h1, h2, h3], ["All data from SPOTS", "Selected for analysis", "Pressure window"],
                  loc="center", fontsize=14)
    fig.tight_layout()
    fig.savefig(config.FIGS / "profile_selection.png", dpi=300)


def main():
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    config.FIGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = build()
    with step(f"Saving {config.SPOTS_BGC.name}"):
        df.to_csv(config.SPOTS_BGC, index=False)
    with step("Selecting profiles"):
        sel = select_profiles(df)
        sel.to_csv(config.SPOTS_BGC_FILTERED, index=False)
    with step("Plotting profile selection"):
        plot_selection(df, sel)
    print(f"\nFinished in {time.time() - t0:.0f} s. {len(sel)} samples selected:")
    print(sel.groupby("TimeSeriesSite")["DATE"].agg(profiles="nunique", first="min", last="max"))


if __name__ == "__main__":
    main()
