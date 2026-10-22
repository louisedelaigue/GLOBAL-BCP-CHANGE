[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15790561.svg)](https://doi.org/10.5281/zenodo.15790561)

## Spatial redistribution of a globally stable marine biological carbon pump

#### **L. Delaigue<sup>1, 2</sup>, R. Sauzède<sup>3</sup>, O. Sulpis<sup>4</sup>, P. W. Boyd<sup>5</sup>, L. D. Talley<sup>6</sup>, H. Claustre<sup>2</sup>, G.-J. Reichart<sup>1, 7</sup> and M. P. Humphreys<sup>1</sup>**

<details>
<summary><strong>Author affiliations</strong></summary>

<sup>1</sup>Department of Ocean Systems (OCS), NIOZ Royal Netherlands Institute for Sea Research, PO Box 59, 1790 AB Den Burg (Texel), the Netherlands  
<sup>2</sup>Sorbonne Université, CNRS, Laboratoire d’Océanographie de Villefranche, Villefranche-sur-Mer, France  
<sup>3</sup>Sorbonne Université, CNRS, Institut de la Mer de Villefranche, Villefranche-sur-Mer, France  
<sup>4</sup>CEREGE, Aix Marseille Univ, CNRS, IRD, INRAE, Collège de France, Aix-en-Provence, France  
<sup>5</sup>Institute for Marine and Antarctic Studies, University of Tasmania, Hobart, Tasmania, Australia  
<sup>6</sup>Scripps Institution of Oceanography, University of California San Diego, La Jolla, CA, USA  
<sup>7</sup>Department of Earth Sciences, Utrecht University, Utrecht, the Netherlands  

</details>

*Corresponding author: Louise Delaigue ([louise.delaigue@imev-mer.fr](mailto:louise.delaigue@imev-mer.fr))*

> [!IMPORTANT]  
> This study is under review in *Global Biogeochemical Cycles*.
> Preprint: ESS Open Archive, [https://essopenarchive.org/doi/full/10.22541/essoar.15010559/v1](https://essopenarchive.org/doi/full/10.22541/essoar.15010559/v1)

<img src="figs/Figure3.png" width="600" />

### Abstract
Marine dissolved inorganic carbon (DIC<sub>total</sub>) is a key component of the global ocean carbon cycle. Over recent decades, DIC<sub>total</sub> has increased due to rising anthropogenic CO<sub>2</sub>, but the response from the biological carbon pump (BCP), which transfers carbon from the surface to the ocean interior, remains uncertain. Here, we reconstruct global three-dimensional DIC changes from 2004 to 2024 down to 2 km depth and distinguish contributions from anthropogenic, organic matter remineralization, and calcium carbonate cycling. We find that while DIC<sub>total</sub> increased throughout the water column, driven primarily by anthropogenic CO<sub>2</sub>, the global inventory of biologically respired carbon associated with the BCP remained constant. However, this apparent stability masked pronounced regional and vertical redistribution of respired carbon within the ocean interior. In some regions, respired carbon accumulated at shallower depths, reducing its isolation from the atmosphere, whereas in others it was transported deeper, enhancing long-term storage. These contrasting patterns co-vary with changes in upper-ocean stratification and with the ventilation state of the ocean interior, indicating that both regulate where biologically respired carbon is retained. Although these redistributions did not impact the net global carbon inventory over the study period, they reveal a structural reorganization of the BCP that may influence the ocean’s capacity to sequester carbon under continued climate change.

### Approach
Monthly 1° × 1° fields of DIC, total alkalinity (TA), temperature, salinity and O<sub>2</sub> from GCC-DIC v2.3 (CONTENT applied to GOBAI-O<sub>2</sub> v2.3, 2004–2024, 0–2000 dbar) are used to decompose DIC<sub>total</sub> into:

- DIC<sub>soft</sub> = (117/170) × ΔAOU, soft-tissue carbon from organic matter remineralization (C:−O<sub>2</sub> = 117:170, Anderson and Sarmiento, 1994);
- DIC<sub>carb</sub> = 0.5 × (ΔTA + (16/170) × ΔAOU), carbonate carbon from CaCO<sub>3</sub> dissolution, with TA corrected for nitrate released by remineralization (N:−O<sub>2</sub> = 16:170);
- DIC<sub>anth</sub> = ΔDIC<sub>total</sub> − DIC<sub>soft</sub> − DIC<sub>carb</sub>, the residual, dominated by anthropogenic CO<sub>2</sub>.

Linear trends (μmol kg<sup>−1</sup> yr<sup>−1</sup>, or mol m<sup>−2</sup> yr<sup>−1</sup> when depth-integrated) are computed with a 1000-member Monte Carlo that perturbs DIC, TA and AOU by their mapping uncertainties and the two ratios by their uncertainties (±0.092 and ±0.0081). A pixel is shown as robust when ≥ 80% of iterations give a significant trend of the same sign. The structural uncertainty of AOU is estimated from the difference between GOBAI-O<sub>2</sub> and ML4O<sub>2</sub>. GCC-DIC trends are compared with in situ trends at seven time-series stations (`ocean-time-series-bcp/`).

### Repository contents

#### Scripts (global analysis, run in order)

| Script | Purpose | Output (`data/processing_steps/`) |
|---|---|---|
| `01A1_O2_product_structural_uncertainty.py` | Structural uncertainty of the global-mean AOU trend from GOBAI-O<sub>2</sub> v2.3 vs ML4O<sub>2</sub> (2004–2020, 0–1000 m) | `01A1_O2_product_structural_uncertainty.nc` |
| `01A2_DIC_globally_averaged_change_monte_carlo_online.py` | Global-mean trend profiles of DIC<sub>total</sub>, DIC<sub>soft</sub>, DIC<sub>carb</sub> and DIC<sub>anth</sub> (Fig. 1) | `01_DIC_globally_averaged_change_monte_carlo_online.nc` |
| `01B_DIC_basins_averaged_change_monte_carlo.py` | Same, averaged over each ocean basin of GOaS v1 (Fig. 2) | `01B_DIC_basins_averaged_<basin>.nc` |
| `02_DIC_integrated_change_map_monte_carlo_online_pres_diff.py` | 0–2000 dbar integrated DIC<sub>soft</sub> trend map with Monte Carlo robustness (Fig. 3) | `02_DIC_integrated_change_map_monte_carlo_online_pres_diff_with_sig_pres_ok.nc` |
| `03_DIC_sequestration_below_winter_MLD_monte_carlo_online.py` | Net DIC<sub>soft</sub> change at or below the maximum climatological mixed layer depth (MLD) (Fig. 6a) | `03_DIC_integrated_change_at_or_below_MLD_map_monte_carlo_online_pres_diff_with_sig.nc` |
| `04_DIC_sequestered_50_depth_interpolation.py` | Trend in the depth above which 50% of the DIC<sub>soft</sub> inventory sits (Fig. 6b) | `04_DIC_sequestered_50_depth_change.nc` |
| `05_DIC_soft_vertical_figure.py` | DIC<sub>soft</sub> profiles relative to the MLD (Fig. 7) | `05_DIC_soft_with_MLD.nc` |
| `06_N2_stratification_trend.py` | Trend in the 0–200 m mean Brunt–Väisälä frequency, N<sup>2</sup> (s<sup>−2</sup> yr<sup>−1</sup>) (Fig. 5) | `06_N2_stratification_trend.nc` |
| `07A_DIC_band_0_200.py`, `07B_DIC_band_200-1000.py`, `07C_DIC_band_1000_2000.py` | DIC<sub>soft</sub> trend maps in the 0–200, 200–1000 and 1000–1975 dbar bands (Fig. 5) | `07A_DIC_band_0_200_monte_carlo.nc`, `08_DIC_band_200_1000_monte_carlo.nc`, `09_DIC_band_1000_1975_monte_carlo.nc` |
| `08_OCIM_ideal_age.py` | Band-mean OCIM ideal age, a measure of ventilation (Fig. 5) | `09_OCIM_ideal_age.nc` |

#### Notebooks

| Notebook | Content |
|---|---|
| `Figures and numbers.ipynb` | Figs. 1–7 in manuscript order, and the numbers quoted in the text. Interactive Plotly versions for exploration |
| `Supplementary_figures.ipynb` | Surface to 1000 m distributions of DIC and TA (Figs. S1–S2) |
| `Compare_oxygen_products.ipynb` | GOBAI-O<sub>2</sub> vs ML4O<sub>2</sub> O<sub>2</sub> trends, sign disagreement and trend detectability (SI) |
| `n2_dicsoft_correlation.ipynb` | Sign agreement between N<sup>2</sup> and 0–200 dbar DIC<sub>soft</sub> trends (SI) |

#### Time-series stations: `ocean-time-series-bcp/`
In situ trends of DIC<sub>total</sub>, DIC<sub>soft</sub>, DIC<sub>carb</sub> and DIC<sub>anth</sub> at seven stations (ALOHA, BATS, CVOO, Iceland Sea, Irminger Sea, K2, KNOT; SPOTS, Lange et al., 2024, plus BATS bottle data) and comparison with GCC-DIC v2.3 (Fig. 4, Table S1 and SI figures). Self-contained, with its own `config.py`, shared module and scripts; see [`ocean-time-series-bcp/README.md`](ocean-time-series-bcp/README.md).

#### Figures

All figures are written to `figs/`.

| Figure | Content | Scripts |
|---|---|---|
| 1 | Global-mean trend profiles of each DIC component | `01A1`, `01A2` |
| 2 | Basin-mean trend profiles of each DIC component | `01B` |
| 3 | 0–2000 dbar integrated DIC<sub>soft</sub> change | `02` |
| 4 | In situ vs GCC-DIC trend profiles at time-series stations | `ocean-time-series-bcp/scripts/01`–`04` |
| 5 | Physical diagnostics: stratification (N<sup>2</sup>), ventilation (ideal age) and DIC<sub>soft</sub> change by depth band | `06`, `07A`–`C`, `08` |
| 6 | (a) DIC<sub>soft</sub> sequestered below the winter MLD; (b) change in the 50% sequestration depth | `03`, `04` |
| 7 | Scenarios of vertical change in DIC<sub>soft</sub> | `01A2`–`05` |

### Data availability
Input and intermediate files are not included because of GitHub size limits. Processed intermediate files are available from the corresponding author on request. To reproduce, place the inputs below in `data/`; file paths are set at the top of each script.

| File | Dataset |
|---|---|
| `CONTENT_DIC_v2-3.nc`, `CONTENT_AT_v2-3.nc` | GCC-DIC v2.3: DIC and TA from CONTENT applied to GOBAI-O<sub>2</sub> v2.3 (available on request) |
| `o2_NN_ship+Argo_ensmean.nc` | ML4O<sub>2</sub> neural-network O<sub>2</sub> product (Ito et al., 2024) |
| `Argo_mixedlayers_monthlyclim_04142022.nc` | Argo monthly mixed layer climatology (Holte et al., 2017) |
| `115838.nc` | OCIM ideal age |
| `GOaS_v1_20211214/` | Global Oceans and Seas v1 polygons (Flanders Marine Institute, 2021) |

Time-series inputs (SPOTS, BATS, MLD climatology) are listed in [`ocean-time-series-bcp/README.md`](ocean-time-series-bcp/README.md).

### License
GNU General Public License v3.0.
