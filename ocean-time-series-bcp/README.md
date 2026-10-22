# Processing of SPOTS in-situ timeseries

In situ validation of the biological carbon pump (BCP) trends in Delaigue et al.,
*Spatial redistribution of a globally stable marine biological carbon pump* (GBC, in revision).

Bottle data from seven SPOTS time-series stations (Lange et al., 2024) plus BATS are
interpolated onto pressure, trends of DIC, TA, AOU and nitrate are computed with Monte Carlo
uncertainty, and DIC trends are decomposed into soft-tissue (DIC<sub>soft</sub>), carbonate (DIC<sub>carb</sub>) and
anthropogenic (DIC<sub>anth</sub>) components. The results are compared with GCC-DIC v2.3.

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
(figures are written to ../figs/, named as in the paper)
```

## Scripts and figures

| Script | Output | Replaces |
|---|---|---|
| `01_build_spots_dataset.py` | SPOTS + BATS samples; profile selection; **Fig. S9** | `_SPOTS_02`, `_SPOTS_03B` |
| `02_station_trends.py` | interpolated profiles and DIC-component trends per station | `XX_00`–`XX_04` (35 scripts) |
| `03_gccdic_station_profiles.py` | GCC-DIC trend profiles at each station (±2° box), Fig. 4 | `model_series_notebooks/` (7 notebooks) |
| `03b_gccdic_nearest_pixel.py` | GCC-DIC v2.3 AOU time series at the pixel nearest each station | new (replaces the v2.1 map) |
| `04_fig4_insitu_vs_gccdic.py` | **Fig. 4** | `_SPOTS_5`, `_SPOTS_4_pres`, `FigureS.ipynb` |
| `05_figS_insitu_vs_gccdic_scatter.py` | **Fig. S11**: in situ vs GCC-DIC v2.3 DIC<sub>soft</sub> over the same pressure range | `_SPOTS_6_…_umolkgyr` |
| `06_sequestration_depths.py` | **Table S2**: trends in the DIC<sub>soft</sub> 50% sequestration depth per station | `_SPOTS_7`, `XX_05` (7 scripts) |
| `07_figS_nitrate_vs_aou.py` | **Fig. S14**, **Table S3**: nitrate- vs AOU-based DIC<sub>soft</sub> (R1 comment 4) | `_SPOTS_8` |
| `08_figS_station_map.py` | station map with GCC-DIC boxes (`station_map.png`, not in the SI) | `_SPOTS_01`, `_SPOTS_00` |
| `09_tableS1_station_periods.py` | **Table S1**: time range, period used and profiles per station, read from the data | typed by hand |

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

The analysis is made in pressure space. The neutral density (γ<sup>n</sup>) analysis (MATLAB γ<sup>n</sup> step and the
γ<sup>n</sup> version of the station pipeline, Fig. S10) is computationally heavy and requires large storage, so it is not included in this repository; it is available on request.
