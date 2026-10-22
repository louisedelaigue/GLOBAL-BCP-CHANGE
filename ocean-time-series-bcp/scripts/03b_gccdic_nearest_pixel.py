"""
Step 3b. Extract the GCC-DIC v2.3 AOU time series at the pixel nearest each station.

Used by the in situ vs GCC-DIC scatter (script 05). AOU is computed as in the global
analysis (GLOBAL-BCP-CHANGE-2.3 scripts 02 and 07): GG92 from O2, potential temperature
and salinity, with the O2 product uncertainty ('uncer') as AOU uncertainty.

In   CONTENT_DIC_v2-3.nc (config.GCCDIC_DIC; only the nearest water column is read)
Out  data/processed/gccdic_stations/<name>_pixel.nc   AOU, AOU_uncertainty (time x pres)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gsw
import koolstof as ks
import numpy as np
import xarray as xr

import config


def nearest_ocean_pixel(ds, lon, lat):
    """Nearest grid column with data; searches the 3 x 3 neighbourhood if the nearest is land."""
    if float(ds.lon.max()) > 180:
        lon = lon % 360
    i = int(np.abs(ds.lat.values - lat).argmin())
    j = int(np.abs(ds.lon.values - lon).argmin())
    candidates = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ii, jj = i + di, (j + dj) % ds.sizes["lon"]
            if 0 <= ii < ds.sizes["lat"]:
                dist = (ds.lat.values[ii] - lat) ** 2 + (ds.lon.values[jj] - lon) ** 2
                candidates.append((dist, ii, jj))
    for _, ii, jj in sorted(candidates):
        col = ds.isel(lat=ii, lon=jj)
        if np.isfinite(col["oxy"].isel(time=0).values).any():
            return col.load()
    raise ValueError(f"No GCC-DIC data near {lon}, {lat}")


def main():
    ds = xr.open_dataset(config.GCCDIC_DIC)
    config.GCCDIC_STATION_DIR.mkdir(parents=True, exist_ok=True)
    for station, cfg in config.STATIONS.items():
        lon, lat = cfg["coords"]
        col = nearest_ocean_pixel(ds, lon, lat)
        sa = gsw.SA_from_SP(col["sal"], col["pres"], float(col.lon), float(col.lat))
        theta = gsw.pt0_from_t(sa, col["temp"], col["pres"])
        aou = ks.parameterisations.aou_GG92(oxygen=col["oxy"], temperature=theta, salinity=col["sal"])[0]
        out = xr.Dataset({"AOU": aou, "AOU_uncertainty": col["uncer"]}).transpose("time", "pres")
        out.attrs.update(station=station, pixel_lon=float(col.lon), pixel_lat=float(col.lat),
                         station_lon=lon, station_lat=lat, product="GCC-DIC v2.3")
        out.to_netcdf(config.GCCDIC_STATION_DIR / f"{cfg['gccdic_name']}_pixel.nc")
        print(f"{station}: pixel {float(col.lat):.1f}N {float(col.lon):.1f}E "
              f"(station {lat:.2f}N {lon:.2f}E)")


if __name__ == "__main__":
    main()
