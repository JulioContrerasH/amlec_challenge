"""
Metrics and Radiative Transfer Utilities.

This module contains the core physics formulas to emulate atmospheric
radiative transfer models and the error metrics used for evaluation.
"""

import numpy as np
from typing import Tuple

def retrieve_reflectance(
    Ltoa: np.ndarray,
    Y: np.ndarray,
    sza: np.ndarray,
    n_wvl: int
) -> np.ndarray:
    """
    Performs atmospheric correction to retrieve surface reflectance.
    
    Calculates surface reflectance ($\rho$) from Top-Of-Atmosphere (TOA) radiance
    using the provided transfer functions.

    Args:
        Ltoa (np.ndarray): TOA radiance, shape [n_wvl, n_samples].
        Y (np.ndarray): Transfer-function blocks, shape [6*n_wvl, n_samples].
        sza (np.ndarray): Solar zenith angle in degrees, shape [1, 1].
        n_wvl (int): Number of spectral wavelengths.

    Returns:
        np.ndarray: Surface reflectance ($\rho$), shape [n_wvl, n_samples].
    """
    # Split transfer-function data
    L0 = Y[:, 0:n_wvl]
    E_dir = Y[:, n_wvl:2*n_wvl]
    E_dif = Y[:, 2*n_wvl:3*n_wvl]
    Sa = Y[:, 3*n_wvl:4*n_wvl]
    T_dir = Y[:, 4*n_wvl:5*n_wvl]
    T_dif = Y[:, 5*n_wvl:6*n_wvl]

    # Compute total irradiance and total transmittance
    # Latex: E_{total} = E_{dir} \cdot \cos(SZA) + E_{dif}
    E_total = E_dir * np.cos(np.radians(sza)) + E_dif
    T_total = T_dir + T_dif

    # Retrieve surface reflectance
    # Latex: \rho = \frac{\pi (L_{toa} - L_0)}{E_{total} T_{total} + \pi (L_{toa} - L_0) S_a}
    numerator = np.pi * (Ltoa - L0)
    denominator = E_total * T_total + np.pi * (Ltoa - L0) * Sa
    rho = numerator / denominator
    return rho


def toa_radiance(
    rho: np.ndarray,
    Y: np.ndarray,
    sza: np.ndarray,
    n_wvl: int
) -> np.ndarray:
    """
    Computes Top-Of-Atmosphere (TOA) radiance from surface reflectance.

    Inverse operation of `retrieve_reflectance`.

    Args:
        rho (np.ndarray): Reference surface reflectance, shape [n_wvl] or broadcastable.
        Y (np.ndarray): Transfer-function blocks, shape [6*n_wvl, n_samples].
        sza (np.ndarray): Solar zenith angle, shape [1, 1].
        n_wvl (int): Number of wavelengths.

    Returns:
        np.ndarray: TOA radiance ($L_{toa}$), shape [n_wvl, n_samples].
    """
    L0 = Y[:, 0:n_wvl]
    E_dir = Y[:, n_wvl:2*n_wvl]
    E_dif = Y[:, 2*n_wvl:3*n_wvl]
    Sa = Y[:, 3*n_wvl:4*n_wvl]
    T_dir = Y[:, 4*n_wvl:5*n_wvl]
    T_dif = Y[:, 5*n_wvl:6*n_wvl]

    E_total = E_dir * np.cos(np.radians(sza)) + E_dif
    T_total = T_dir + T_dif

    numerator = E_total * T_total * rho
    denominator = 1.0 - Sa * rho
    
    # Latex: L_{toa} = L_0 + \frac{1}{\pi} \frac{E_{total} T_{total} \rho}{1 - S_a \rho}
    Ltoa = L0 + (1.0 / np.pi) * (numerator / denominator)
    return Ltoa


def error_metric_A(
    rho_ref: np.ndarray,
    rho_ret: np.ndarray,
    wvl: np.ndarray
) -> float:
    """
    Computes the error metric for Scenario A (Atmospheric Correction).

    Calculates Mean Relative Error (MRE) excluding specific absorption bands.
    
    Args:
        rho_ref (np.ndarray): Reference reflectance, shape [n_wvl].
        rho_ret (np.ndarray): Retrieved reflectance, shape [n_wvl, n_samples].
        wvl (np.ndarray): Wavelengths, shape [1, n_wvl].

    Returns:
        float: Mean relative error in percentage.
    """
    re = 100.0 * np.abs(rho_ret - rho_ref) / rho_ref
    re_mean = np.nanmean(re, axis=0, keepdims=True)
    
    # Exclude deep absorption bands (water vapor, etc.)
    mask = ~(
        ((wvl > 931) & (wvl < 945)) |
        ((wvl > 1100) & (wvl < 1160)) |
        ((wvl > 1300) & (wvl < 1500)) |
        ((wvl > 1750) & (wvl < 1980)) |
        (wvl > 2420)
    )
    return np.nanmean(re_mean[mask])


def error_metric_B(
    Ltoa_ref: np.ndarray,
    Ltoa_ret: np.ndarray
) -> float:
    """
    Computes the error metric for Scenario B (CO2 Retrieval).

    Calculates Mean Relative Error (MRE) for TOA Radiance.

    Args:
        Ltoa_ref (np.ndarray): Reference TOA radiance, [n_wvl, n_samples].
        Ltoa_ret (np.ndarray): Estimated TOA radiance, [n_wvl, n_samples].

    Returns:
        float: Mean relative error in percentage.
    """
    re = 100.0 * np.abs(Ltoa_ret - Ltoa_ref) / Ltoa_ref
    re_mean = np.nanmean(re, axis=0, keepdims=True)
    return np.nanmean(re_mean)


def compute_final_ranks(rnk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute weighted average ranking and assign standard competition ranks.

    Weights: [0.325 (A-Interp), 0.175 (A-Extrap), 0.325 (B-Interp), 0.175 (B-Extrap)].

    Args:
        rnk (np.ndarray): Array of shape [n_models, 4] with individual ranks.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - rnk_avg: Weighted average scores.
            - final_ranks: Final integer ranks (handling ties).
    """
    n_models = rnk.shape[0]
    weights = np.array([0.325, 0.175, 0.325, 0.175])
    
    # Calculate weighted score
    rnk_avg = np.dot(rnk, weights)

    sorted_scores = np.sort(rnk_avg)
    idx_sorted = np.argsort(rnk_avg)
    final_ranks = np.zeros_like(rnk_avg, dtype=int)

    # Assign ranks with standard competition ranking for ties
    i = 0
    while i < n_models:
        tie_val = sorted_scores[i]
        tied_idx = np.where(np.isclose(sorted_scores, tie_val, atol=1e-8))[0]
        tied_idx = tied_idx[tied_idx >= i]
        k = len(tied_idx)
        
        # Assign the same rank to all tied models
        final_ranks[idx_sorted[tied_idx]] = i + 1
        i += k

    return rnk_avg, final_ranks