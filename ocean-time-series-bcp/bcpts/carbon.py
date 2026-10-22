"""Decomposition of DIC trends into soft-tissue, carbonate and anthropogenic components."""
import numpy as np

import config


def _rel(err, value):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (err / value) ** 2


def decompose(dic, dic_err, ta, ta_err, aou, aou_err):
    """Soft-tissue, carbonate and anthropogenic trends with propagated errors.

    Identical to the old XX_03_pres_calculate_pumps.py scripts:
        DICsoft = (117/170) dAOU
        DICcarb = 0.5 (dTA + (16/170) dAOU)
        DICanth = dDIC - DICsoft - DICcarb
    Inputs and outputs are trend profiles (umol kg-1 yr-1) and their 1-sigma errors.
    """
    soft, soft_err = soft_from_aou(aou, aou_err)

    n_term = -config.R_N_O2 * aou                       # = (16/-170) dAOU
    n_term_err = np.sqrt(_rel(config.R_N_O2_ERR, config.R_N_O2) + _rel(aou_err, aou)) * np.abs(n_term)
    carb = 0.5 * (ta - n_term)
    carb_err = np.sqrt(ta_err ** 2 + n_term_err ** 2) / 2

    anth = dic - (soft + carb)
    anth_err = np.sqrt(dic_err ** 2 + soft_err ** 2 + carb_err ** 2)
    return {
        "DICtotal": (dic, dic_err), "DICsoft": (soft, soft_err),
        "DICcarb": (carb, carb_err), "DICanth": (anth, anth_err),
    }


def soft_from_aou(aou, aou_err):
    """AOU-based soft-tissue trend, (117/170) dAOU, with propagated error."""
    soft = config.R_C_O2 * aou
    err = np.sqrt(_rel(config.R_C_O2_ERR, config.R_C_O2) + _rel(aou_err, aou)) * np.abs(soft)
    return soft, err


def soft_from_nitrate(no3, no3_err):
    """Nitrate-based soft-tissue trend, (117/16) dNO3, with propagated error."""
    soft = config.R_C_N * no3
    err = np.sqrt(_rel(config.R_C_N_ERR, config.R_C_N) + _rel(no3_err, no3)) * np.abs(soft)
    return soft, err
