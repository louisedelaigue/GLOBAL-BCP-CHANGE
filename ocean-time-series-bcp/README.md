# ocean-time-series-bcp

In situ validation of the biological carbon pump (BCP) trends in Delaigue et al.,
*Spatial redistribution of a globally stable marine biological carbon pump* (GBC, in revision).

Bottle data from seven SPOTS time-series stations (Lange et al., 2024) plus BATS are
interpolated onto pressure, trends of DIC, TA, AOU and nitrate are computed with Monte Carlo
uncertainty, and DIC trends are decomposed into soft-tissue (DIC<sub>soft</sub>), carbonate (DIC<sub>carb</sub>) and
anthropogenic (DIC<sub>anth</sub>) components. The results are compared with GCC-DIC v2.3.

This repository replaces `ocean-times-series` (~115 scripts and 18 notebooks) with one
config file, a small shared module and ten scripts. The results are unchanged (see *Verification*).

## Layout

```
config.py        paths, station settings, Redfield ratios, uncertainties
bcpts/           shared functions
  profiles.py      profile selection, MeanShift clustering + PCHIP interpolation
  trends.py        OLS trends with Monte Carlo measurement uncertainty
  carbon.py        DICsoft / DICcarb / DICanth decomposition, nitrate-based DICsoft
scripts/         run in order
data/raw/        inputs (not tracked)
data/processed/  everything the scripts write (not tracked)
figs/            figures (not tracked)
```

## Scripts and figures

| Script | Output | Replaces |
|---|---|---|
| `01_build_spots_dataset.py` | SPOTS + BATS samples; profile selection; `figs/profile_selection.png` (SI) | `_SPOTS_02`, `_SPOTS_03B` |
| `02_station_trends.py` | interpolated profiles and DIC-component trends per station | `XX_00`–`XX_04` (35 scripts) |
| `03_gccdic_station_profiles.py` | GCC-DIC trend profiles at each station (±2° box), Fig. 4 | `model_series_notebooks/` (7 notebooks) |
| `03b_gccdic_nearest_pixel.py` | GCC-DIC v2.3 AOU time series at the pixel nearest each station | new (replaces the v2.1 map) |
| `04_fig4_insitu_vs_gccdic.py` | **Fig. 4** | `_SPOTS_5`, `_SPOTS_4_pres`, `FigureS.ipynb` |
| `05_figS_insitu_vs_gccdic_scatter.py` | SI scatter, in situ vs GCC-DIC v2.3 DIC<sub>soft</sub> over the same pressure range | `_SPOTS_6_…_umolkgyr` |
| `06_sequestration_depths.py` | trends in the DIC<sub>soft</sub> 50% sequestration depth per station | `_SPOTS_7`, `XX_05` (7 scripts) |
| `07_figS_nitrate_vs_aou.py` | SI figure + table, nitrate- vs AOU-based DIC<sub>soft</sub> (R1 comment 4) | `_SPOTS_8` |
| `08_figS_station_map.py` | SI station map with GCC-DIC boxes | `_SPOTS_01`, `_SPOTS_00` |
| `09_tableS1_station_periods.py` | Table S1: time range, period used and profiles per station, read from the data | typed by hand |

## Running

```bash
conda env create -f environment.yml && conda activate bcp-ts
python scripts/01_build_spots_dataset.py
python scripts/02_station_trends.py          # ~2 min; or: ... 02_station_trends.py K2 KNOT
python scripts/03_gccdic_station_profiles.py # only if GCC-DIC changes; needs the CONTENT v2.3 files
python scripts/03b_gccdic_nearest_pixel.py   # ~1 min; needs CONTENT_DIC_v2-3.nc
python scripts/04_fig4_insitu_vs_gccdic.py
python scripts/05_figS_insitu_vs_gccdic_scatter.py
python scripts/06_sequestration_depths.py
python scripts/07_figS_nitrate_vs_aou.py
python scripts/08_figS_station_map.py
python scripts/09_tableS1_station_periods.py
```

Every station-specific choice is in `config.STATIONS`. The Monte Carlo seed is fixed (`config.SEED`),
so results are reproducible.

## Inputs

| File | Source |
|---|---|
| `data/raw/spots.csv` | SPOTS compilation, Lange et al. (2024) |
| `data/raw/bats_bottle.txt` | BATS bottle data, BIOS |
| `data/raw/mld_DR003_c1m_reg2.0.nc` | MLD climatology, de Boyer Montégut |
| `CONTENT_DIC_v2-3.nc`, `CONTENT_AT_v2-3.nc` | GCC-DIC v2.3, in `../data/` of GLOBAL-BCP-CHANGE (or set env var `GCCDIC_DIR`) |

Neutral density (γ<sup>n</sup>) is not needed: the analysis is on pressure, so the MATLAB γ<sup>n</sup> step and the
γ<sup>n</sup> version of the station pipeline were not carried over (they remain in `ocean-times-series`).

## Verification against ocean-times-series

Run on the same inputs (22 Oct 2026):

- Interpolated profiles: identical at all stations (same profiles, dates and pressure grid).
- Fig. 4 in situ trends: ALOHA, BATS, Iceland Sea, Irminger Sea agree within Monte Carlo noise
  (≤0.02 µmol kg<sup>−1</sup> yr<sup>−1</sup>). K2, KNOT and CVOO differ by up to 0.22 µmol kg<sup>−1</sup> yr<sup>−1</sup> (upper 500 dbar),
  because the Fig. 4 files stored in the old repo predate its own current interpolation output:
  a noise-free trend from the old repo's stored profiles reproduces the new result exactly.
- Sequestration depths: identical. Nitrate vs AOU table: identical within 0.01.
- Step 1: 9775 samples, as in the old repo; profiles per station: ALOHA 136, BATS 84, K2 28, KNOT 37, CVOO 5, Iceland Sea 19, Irminger Sea 14.
