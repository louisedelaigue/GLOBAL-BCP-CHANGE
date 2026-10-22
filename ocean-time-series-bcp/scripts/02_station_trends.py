"""
Step 2. Interpolate each station's profiles onto a pressure grid, compute Monte Carlo trends
of DIC, TA and AOU, and decompose the DIC trend into soft-tissue, carbonate and
anthropogenic components.

Replaces the 35 scripts XX_00 to XX_04 (pressure versions) for the 7 stations.

In   data/processed/spots_bgc_pres_filtered.csv
Out  data/processed/stations/<station>_profiles.nc   interpolated profiles (CTDPRS x date)
     data/processed/stations/<station>_trends.nc     trend profiles, smoothed (used in Fig. 4)

Usage: python scripts/02_station_trends.py [STATION ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xarray as xr

import config
from bcpts import carbon, profiles, trends


def station_variables(station):
    return ["TCARBN", "ALKALI", "aou", "CTDTMP", config.STATIONS[station]["salinity"]]


def run(df, station, rng):
    prof = profiles.interpolate_station(df, station, station_variables(station))
    print(f"{station}: {prof.attrs['n_profiles']} profiles, "
          f"{prof.attrs['first_year']}-{prof.attrs['last_year']}")

    mc = trends.monte_carlo_slopes(prof, ["TCARBN", "ALKALI", "aou"], rng)
    m = {v: s.mean(0) for v, s in mc.items()}
    e = {v: s.std(0) for v, s in mc.items()}
    comp = carbon.decompose(m["TCARBN"], e["TCARBN"], m["ALKALI"], e["ALKALI"], m["aou"], e["aou"])

    out = xr.Dataset(coords={"CTDPRS": prof["CTDPRS"].values})
    for v, name in [("ALKALI", "TA"), ("aou", "AOU")]:
        out[f"{name}_change"] = ("CTDPRS", m[v])
        out[f"{name}_error"] = ("CTDPRS", e[v])
    for name, (val, err) in comp.items():
        out[f"{name}_change"] = ("CTDPRS", val)
        out[f"{name}_error"] = ("CTDPRS", err)
    out = out.rolling(CTDPRS=config.ROLLING_WINDOW, center=True).mean()
    out.attrs.update(prof.attrs, units="umol kg-1 yr-1",
                     smoothing=f"centred rolling mean over {config.ROLLING_WINDOW} levels")

    config.STATION_DIR.mkdir(parents=True, exist_ok=True)
    prof.to_netcdf(config.STATION_DIR / f"{station}_profiles.nc")
    out.to_netcdf(config.STATION_DIR / f"{station}_trends.nc")


def main(stations):
    df = pd.read_csv(config.SPOTS_BGC_FILTERED, na_values=-999, low_memory=False)
    rng = np.random.default_rng(config.SEED)
    for station in stations:
        run(df, station, rng)


if __name__ == "__main__":
    main(sys.argv[1:] or list(config.STATIONS))
