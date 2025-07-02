# Spatial redistribution of a globally constant marine biological carbon pump

#### **L. Delaigue<sup>1, 2</sup>\, R. Sauzède<sup>3</sup>, O. Sulpis<sup>4</sup>, P. W. Boyd<sup>5</sup>, H. Claustre<sup>2</sup>, G-J Reichart<sup>1, 6</sup> and M. P. Humphreys<sup>1</sup>**

<details>
<summary><strong>Author Affiliations</strong></summary>

<sup>1</sup>Department of Ocean Systems (OCS), NIOZ Royal Netherlands Institute for Sea Research, PO Box 59, 1790 AB Den Burg (Texel), the Netherlands  
<sup>2</sup>Sorbonne Université, CNRS, Laboratoire d’Océanographie de Villefranche, Villefranche-Sur-Mer, France  
<sup>3</sup>Sorbonne Université, CNRS, Institut de la Mer de Villefranche, Villefranche-Sur-Mer, France  
<sup>4</sup>CEREGE, Aix Marseille Univ, CNRS, IRD, INRAE, Collège de France, Aix-en-Provence, France  
<sup>5</sup>Institute for Marine and Antarctic Studies, University of Tasmania, Hobart, Tasmania, Australia  
<sup>6</sup>Department of Earth Sciences, Utrecht University, Utrecht, the Netherlands  

</details>

*Corresponding author: Louise Delaigue ([louise.delaigue@imev-mer.fr](mailto:louise.delaigue@imev-mer.fr))*

> [!IMPORTANT]  
> This study is currently under review for publication in Nature Geoscience.

<img src="figs/Figure2a_with_uncertainty.png" width="600" height="400" />

#### This repository contains the raw data and analysis scripts used to produce the results and figures presented in the manuscript.


## Abstract
Marine dissolved inorganic carbon (DIC) is a key component of the global ocean carbon cycle. Over recent decades, DIC has increased due to rising anthropogenic CO<sub>2</sub>, but the component of DIC change due to the biological carbon pump (BCP), which transfers carbon from the surface to the deep ocean, remains highly uncertain. Using the GOBAI-O<sub>2</sub> data product and the CANYON-B and CONTENT algorithms, we reconstructed the 3-dimensional global DIC<sub>total</sub> distribution from 2004 to 2022 and decomposed it into DIC<sub>soft</sub> (resulting from organic matter degradation), DIC<sub>carb</sub> (resulting from carbonate dissolution), and DIC<sub>anth</sub> (anthropogenic CO<sub>2</sub> plus changes in air-sea disequilibrium). We found a significant DIC<sub>total</sub> change throughout the water column, with surface concentrations increasing by ~1.0 ± 0.23 μmol kg<sup>-1</sup> yr<sup>-1</sup>, driven by DIC<sub>anth</sub> (>90% contribution). Despite a globally constant signal in DIC<sub>soft</sub>, substantial regional trends emerged. Changes in circulation, particle sinking, and remineralization altered the vertical and horizontal distributions of DIC<sub>soft</sub>. In some regions, DIC<sub>soft</sub> accumulated at shallower depths, shortening residence times; in others, it was transported deeper, enhancing long-term storage. Although these widespread and divergent trends had little net effect on the global DIC<sub>soft</sub> inventory from 2004-2022, the emerging spatial reorganization of the BCP may signal an evolving instability in the ocean carbon sink under continued climate forcing.
 
## Analysis
A detailed explanation of each script is available in the "Figures and numbers" Jupyter notebook of the repository, along with the resulting figures and statistics. Additionally, interactive plots in Plotly allow for a more detailed exploration of the numbers.

> [!NOTE]  
> This repository does not include files 04_DIC_sequestered_50_depth_1000_iterations.nc and 05_DIC_soft_with_MLD.nc. This repository also does not include the application of the CANYON-B and CONTENT algorithms on the GOBAI-O<sub>2</sub> product due to the large file sizes. However, all these files can be made available upon request, along with the GCC-DIC file.

## License
This project is licensed under the GNU General Public License v3.0 – see the LICENSE file for details.

