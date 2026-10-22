"""
Single source of truth for paths, station settings and constants.

Every station-specific choice that used to be hard-coded in the 7 x 10 per-station
scripts of ocean-times-series lives in STATIONS below.
"""
from pathlib import Path
import os

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"              # inputs we do not produce (not tracked by git)
PROCESSED = ROOT / "data" / "processed"  # everything the scripts write (not tracked)
FIGS = ROOT / "figs"

SPOTS_CSV = RAW / "spots.csv"                        # Lange et al. (2024)
BATS_BOTTLE = RAW / "bats_bottle.txt"                # BATS bottle file (BIOS)
MLD_CLIMATOLOGY = RAW / "mld_DR003_c1m_reg2.0.nc"    # de Boyer Montegut MLD climatology

# GCC-DIC v2.3 (CONTENT) fields; only needed by scripts 03 and 03b (large files, kept outside this repo).
# Default: the data/ folder of GLOBAL-BCP-CHANGE (parent folder).
GCCDIC_DIR = Path(os.environ.get("GCCDIC_DIR", ROOT.parent / "data"))
GCCDIC_DIC = GCCDIC_DIR / "CONTENT_DIC_v2-3.nc"
GCCDIC_TA = GCCDIC_DIR / "CONTENT_AT_v2-3.nc"
# Structural O2-product uncertainty (GOBAI-O2 vs ML4O2), from 01A1 in the global repo (comment 2)
O2_STRUCTURAL = GCCDIC_DIR / "processing_steps" / "01A1_O2_product_structural_uncertainty.nc"

# processed outputs
SPOTS_BGC = PROCESSED / "spots_bgc.csv"
SPOTS_BGC_FILTERED = PROCESSED / "spots_bgc_pres_filtered.csv"
STATION_DIR = PROCESSED / "stations"     # <station>_profiles.nc, <station>_trends.nc
GCCDIC_STATION_DIR = PROCESSED / "gccdic_stations"

# --------------------------------------------------------------------------- constants
# Anderson & Sarmiento (1994): C:N:-O2 = 117(+-14) : 16(+-1) : 170(+-10)
R_C_O2 = 117 / 170
R_C_O2_ERR = 0.092
R_N_O2 = 16 / 170
R_N_O2_ERR = 0.0081
R_C_N = 117 / 16
R_C_N_ERR = R_C_N * ((14 / 117) ** 2 + (1 / 16) ** 2) ** 0.5   # 0.99

# Mean combined precision and accuracy over the SPOTS compilation (umol kg-1)
MEASUREMENT_UNCERTAINTY = {
    "TCARBN": 2.819550170576402,
    "ALKALI": 6.893609115190238,
    "aou": 1.2696462837686155,
    "NITRAT": 0.7448312447271866,
}

N_MONTE_CARLO = 1000
N_LEVELS = 1000          # pressure levels of the interpolated grid
ROLLING_WINDOW = 20      # levels, centred rolling mean applied to the trend profiles
SEED = 42

# --------------------------------------------------------------------------- stations
# label          : name used in figures
# pres_window    : profile must reach above `min` and below `max` dbar (profile selection, old _SPOTS_03B)
# profile_min_max: profiles (grouped by date) must reach at least this pressure (old XX_01)
# profile_id_800 : also require each cast (DATE_STNNBR_CASTNO) to reach 800 dbar (old XX_01)
# bandwidth      : MeanShift bandwidth (dbar) for clustering samples before PCHIP (old XX_01)
# salinity       : salinity column used during interpolation (old XX_01)
# coords         : official station position (lon, lat), used for the nearest GCC-DIC pixel
# mean_coords    : mean SPOTS position (lon, lat), used for the MLD lookup (old _SPOTS_7)
# gccdic_centre  : centre of the +-2 deg box averaged from GCC-DIC (lon, lat) (old model_series_notebooks)
# gccdic_name    : file stem of the GCC-DIC station profile (old data/v2.3)
# xlim           : half-width of x axis in Fig. 4 (umol kg-1 yr-1)
STATIONS = {
    "ALOHA": dict(
        label="ALOHA", pres_window=(25, 2000), profile_min_max=1000, profile_id_800=False,
        bandwidth=20, salinity="CTDSAL",
        coords=(-158.000, 22.750), mean_coords=(-158.002237, 22.752951),
        gccdic_centre=(-158.0, 22.75), gccdic_name="ALOHA", xlim=2),
    "BATS": dict(
        label="BATS", pres_window=(25, 2000), profile_min_max=200, profile_id_800=False,
        bandwidth=20, salinity="CTDSAL",
        coords=(-64.167, 31.667), mean_coords=(-64.189528, 31.665369),
        gccdic_centre=(-64.138, 31.75), gccdic_name="BATS", xlim=2),
    "K2": dict(
        label="K2", pres_window=(25, 2000), profile_min_max=200, profile_id_800=True,
        bandwidth=0.15, salinity="CTDSAL",
        coords=(160.000, 47.000), mean_coords=(160.050667, 46.968458),
        gccdic_centre=(160.0, 47.0), gccdic_name="K2", xlim=2),
    "KNOT": dict(
        label="KNOT", pres_window=(25, 2000), profile_min_max=200, profile_id_800=True,
        bandwidth=0.15, salinity="CTDSAL",
        coords=(155.000, 44.000), mean_coords=(154.998005, 44.003165),
        gccdic_centre=(155.0, 44.0), gccdic_name="KNOT", xlim=1),
    "CVOO": dict(
        label="CVOO", pres_window=(25, 2000), profile_min_max=200, profile_id_800=True,
        bandwidth=0.15, salinity="CTDSAL",
        coords=(-24.300, 17.600), mean_coords=(-24.278775, 17.594396),
        gccdic_centre=(-24.3, 17.6), gccdic_name="CVOO", xlim=5),
    "IcelandSea": dict(
        label="Iceland Sea", pres_window=(30, 1000), profile_min_max=200, profile_id_800=True,
        bandwidth=0.15, salinity="SALNTY",
        coords=(-12.670, 68.000), mean_coords=(-12.670897, 67.996233),
        gccdic_centre=(-12.7, 68.0), gccdic_name="IS", xlim=5),
    "IrmingerSea": dict(
        label="Irminger Sea", pres_window=(25, 1000), profile_min_max=200, profile_id_800=True,
        bandwidth=0.15, salinity="SALNTY",
        coords=(-29.150, 64.150), mean_coords=(-27.955002, 64.335978),
        gccdic_centre=(-29.15, 64.15), gccdic_name="IrS", xlim=6),
}

# Fixed BATS position assigned to all BATS bottles (old _SPOTS_02)
BATS_POSITION = (-64.17397567737511, 31.686662355664797)
ALOHA_POSITION = (-158.002236666436, 22.752951285954833)

# Sites in SPOTS that are not analysed
EXCLUDED_SITES = ["OWSM", "DYFAMED", "GIFT", "CARIACO", "RADCOR"]

# --------------------------------------------------------------------------- plotting
COMPONENTS = {   # variable in trends file -> (label, colour); same colours as Fig. 4
    "DICtotal": (r"ΔDIC$_{\mathrm{total}}$", "#FF934F"),
    "DICsoft": (r"ΔDIC$_{\mathrm{soft}}$", "#136F63"),
    "DICcarb": (r"ΔDIC$_{\mathrm{carb}}$", "#8FBC94"),
    "DICanth": (r"ΔDIC$_{\mathrm{anth}}$", "#0D1321"),
}
TREND_UNITS = r"μmol kg$^{-1}$ yr$^{-1}$"
