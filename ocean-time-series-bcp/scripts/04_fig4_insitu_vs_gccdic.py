"""
Step 4. Fig. 4: in situ (SPOTS) vs GCC-DIC trend profiles of DIC components at each station.

Replaces _SPOTS_5_pres_subplots_with_gobai_subsets.py (and the superseded _SPOTS_4_pres_subplots.py
and FigureS.ipynb).

In   data/processed/stations/<station>_trends.nc, data/processed/gccdic_stations/<name>.nc
Out  ../figs/Figure4.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import config

GCCDIC_VARS = {   # component -> (trend, sigma) in the GCC-DIC station files
    "DICtotal": ("ct_rate_of_change_per_year", "sigma_ct_rate_of_change_per_year"),
    "DICsoft": ("soft_pump_aou_change", "sigma_soft_pump_aou_change"),
    "DICcarb": ("carb_pump_change", "sigma_carb_pump_change"),
    "DICanth": ("co2_anth_aou_change", "sigma_co2_anth_aou_change"),
}


def profile(ax, pres, mean, err, comp):
    label, colour = config.COMPONENTS[comp]
    ax.plot(mean, pres, color=colour, lw=2, label=label)
    ax.fill_betweenx(pres, mean - err, mean + err, color=colour, alpha=0.2, lw=0)


def style(ax, max_p, xlim, ylabel):
    ax.axvline(0, color="k", alpha=0.8, zorder=0)
    ax.set_ylim(max_p, 0)
    ax.set_xlim(-xlim, xlim)
    ax.grid(alpha=0.3)
    ax.set_xlabel(rf"$\Delta$DIC$_{{\mathrm{{x}}}}$ ({config.TREND_UNITS})")
    if ylabel:
        ax.set_ylabel("Pressure (dbar)")


def main():
    fig, axs = plt.subplots(4, 4, figsize=(16, 22), dpi=300)
    axs = axs.flatten()
    for i, (station, cfg) in enumerate(config.STATIONS.items()):
        ins = xr.open_dataset(config.STATION_DIR / f"{station}_trends.nc")
        mod = xr.open_dataset(config.GCCDIC_STATION_DIR / f"{cfg['gccdic_name']}.nc")
        a_ins, a_mod = axs[2 * i], axs[2 * i + 1]
        for comp, (v, s) in GCCDIC_VARS.items():
            profile(a_ins, ins["CTDPRS"], ins[f"{comp}_change"], ins[f"{comp}_error"], comp)
            profile(a_mod, mod["pres"], mod[v], mod[s], comp)
        max_p = float(np.nanmax(ins["CTDPRS"]))
        style(a_ins, max_p, cfg["xlim"], True)
        style(a_mod, max_p, cfg["xlim"], False)
        period = f"{ins.attrs['first_year']}-{ins.attrs['last_year']}"
        a_ins.text(0, 1.03, f"{'abcdefghijklmn'[2 * i]}) {cfg['label']} - in situ ({period})",
                   fontsize=12, transform=a_ins.transAxes)
        a_mod.text(0, 1.03, f"{'abcdefghijklmn'[2 * i + 1]}) {cfg['label']} - GCC-DIC (2004-2024)",
                   fontsize=12, transform=a_mod.transAxes)
    for ax in axs[-2:]:
        ax.axis("off")
    plt.subplots_adjust(wspace=0.5, hspace=0.3)
    # legend centred in the space of the two empty bottom-right panels
    left, right = axs[-2].get_position(), axs[-1].get_position()
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", fontsize=19, ncols=2, frameon=True,
               bbox_to_anchor=((left.x0 + right.x1) / 2, (left.y0 + left.y1) / 2))
    config.FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGS / "Figure4.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
