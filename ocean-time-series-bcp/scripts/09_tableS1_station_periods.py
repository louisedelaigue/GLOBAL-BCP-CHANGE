"""
Step 9. Table S1: record length and period used at each station, read from the data.

Time range      first-last year of any sample at the site in SPOTS (BATS: BATS bottle file)
Period used     first-last year of the profiles used in the decomposition (script 02), i.e.
                with co-located DIC, TA and O2 and passing the profile selection
Profiles        number of profiles used (Fig. 4)

In   data/raw/spots.csv, data/raw/bats_bottle.txt, data/processed/stations/<station>_trends.nc
Out  data/processed/tableS1_station_periods.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import xarray as xr

import config


def full_record_years():
    spots = pd.read_csv(config.SPOTS_CSV, usecols=["TimeSeriesSite", "DATE"], na_values=-999)
    spots["year"] = pd.to_datetime(spots["DATE"].astype("Int64").astype(str), format="%Y%m%d",
                                   errors="coerce").dt.year
    years = spots.groupby("TimeSeriesSite")["year"].agg(["min", "max"])
    bats = pd.read_csv(config.BATS_BOTTLE, skiprows=57, sep=r"\s*,\s*|\s+", engine="python",
                       comment="/", na_values=-999, usecols=["yyyymmdd"])
    by = pd.to_datetime(bats["yyyymmdd"].astype(int).astype(str), format="%Y%m%d").dt.year
    years.loc["BATS"] = [by.min(), by.max()]
    return years


def main():
    years = full_record_years()
    rows = []
    for station, cfg in config.STATIONS.items():
        tr = xr.open_dataset(config.STATION_DIR / f"{station}_trends.nc")
        rows.append({
            "Station": cfg["label"],
            "Time range": f"{int(years.loc[station, 'min'])}–{int(years.loc[station, 'max'])}",
            "Period used in this study": f"{tr.attrs['first_year']}–{tr.attrs['last_year']}",
            "Profiles": int(tr.attrs["n_profiles"]),
        })
    table = pd.DataFrame(rows)
    table.to_csv(config.PROCESSED / "tableS1_station_periods.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
