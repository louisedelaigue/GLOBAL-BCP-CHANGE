"""
Step 8. SI figure: map of the SPOTS time-series stations, with the +-2 deg boxes averaged
from GCC-DIC for the seven stations analysed.

Replaces _SPOTS_01_map_locations.py and _SPOTS_00_verify_pixel_extraction.py.

In   data/raw/spots.csv
Out  ../figs/station_map.png (not in the SI)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

import config

LABEL_OFFSETS = {   # SPOTS site -> (label, dlon, dlat), as in the original map
    "ALOHA": ("ALOHA", 2, 2), "CARIACO": ("CARIACO", 2, 2), "CVOO": ("CVOO", 2, 3),
    "DYFAMED": ("DYFAMED", 4, 0), "GIFT": ("GIFT", -15, -4), "IcelandSea": ("IceS", -5, 4),
    "IrmingerSea": ("IrS", -10, -4), "K2": ("K2", 2, 2), "KNOT": ("KNOT", -18, -6),
    "Munida": ("Munida", -20, -7), "OWSM": ("OWSM", 2, 2), "RADCOR": ("RADCOR", -26, 0),
}


def main():
    df = pd.read_csv(config.SPOTS_CSV, na_values=-999, usecols=["TimeSeriesSite", "LATITUDE", "LONGITUDE"])
    for prefix in ["GIFT", "RADCOR"]:
        df.loc[df["TimeSeriesSite"].astype(str).str.startswith(prefix), "TimeSeriesSite"] = prefix
    sites = df.groupby("TimeSeriesSite")[["LONGITUDE", "LATITUDE"]].mean()

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300, subplot_kw={"projection": ccrs.PlateCarree()})
    ax.add_feature(cfeature.LAND, edgecolor="none", facecolor="lightgrey", zorder=1)
    for site, (lon, lat) in sites.iterrows():
        analysed = site in config.STATIONS
        ax.plot(lon, lat, "o", color="red" if analysed else "grey", markersize=5,
                transform=ccrs.Geodetic(), zorder=3)
        if site in LABEL_OFFSETS:
            label, dx, dy = LABEL_OFFSETS[site]
            ax.text(lon + dx, lat + dy, label, transform=ccrs.Geodetic(), zorder=3)
    for cfg in config.STATIONS.values():
        lon, lat = cfg["gccdic_centre"]
        ax.add_patch(mpatches.Rectangle((lon - 2, lat - 2), 4, 4, fill=False, edgecolor="#136F63",
                                        lw=1, transform=ccrs.PlateCarree(), zorder=2))
    ax.set_ylim(-90, 90)
    gl = ax.gridlines(draw_labels=True, alpha=0.5)
    gl.top_labels = gl.right_labels = False
    fig.savefig(config.FIGS / "station_map.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
