"""
Analysis Module.

Computes the statistical metrics, errors, and rankings for the models.
"""

import re
import numpy as np
import pandas as pd
import h5py
from scipy.interpolate import InterpolatedUnivariateSpline

from src.config import BACK_RESULTS_BASE, REFERENCE_DIR, logger
from src.metrics import (
    retrieve_reflectance, toa_radiance, 
    error_metric_A, error_metric_B, compute_final_ranks
)

def run_statistics() -> pd.DataFrame:
    """
    Compute error metrics, scores, and rankings for all models.

    Returns:
        pd.DataFrame: Table with columns [Model, MRE_A1, MRE_A2, MRE_B1, MRE_B2, Score, Ranking].
    """
    logger.info("Running benchmark statistics...")

    pattern = re.compile(r"^(.*)_[AB][1-2]\.h5$")
    all_files = list(BACK_RESULTS_BASE.glob("*.h5"))
    
    # Identify unique model names
    model_names = sorted({
        pattern.match(p.name).group(1) 
        for p in all_files if pattern.match(p.name)
    })
    
    if not model_names:
        logger.warning("No valid models found in results.")
        return pd.DataFrame()

    n_models = len(model_names)
    run_time_dic = {name: [] for name in model_names}

    scenarios = ["A", "B"]
    tracks = [1, 2]
    track_names = ["refInterp", "refExtrap"]
    n_metrics = len(scenarios) * len(tracks)

    # Arrays to hold results: Rows=Models, Cols=Metrics
    mre = np.full((n_models, n_metrics), np.nan)
    rnk = np.full((n_models, n_metrics), np.nan)

    for s_i, scenario in enumerate(scenarios):
        for t_i, track_num in enumerate(tracks):
            col_idx = t_i + s_i * len(tracks)

            # Load Reference Data
            ref_file = REFERENCE_DIR / f"scenario{scenario}" / "reference" / f"{track_names[t_i]}.h5"
            
            with h5py.File(ref_file, "r") as f:
                L_ref = np.array(f["/LUTdata"], dtype=float)
                sza_all = np.array(f["/LUTheader"], dtype=float)
                wvl = np.array(f["/wvl"], dtype=float)

            n_wvl = wvl.shape[1]
            # SZA index depends on scenario header structure
            sza = sza_all[:, [6]] if scenario == "A" else sza_all[:, [4]]

            # If Scenario A: Prepare Ref Reflectance (Spline interpolation)
            rho_ref_array = None
            if scenario == "A":
                ref_rho_file = REFERENCE_DIR / "scenarioA" / "reference" / "refldb.txt"
                rho_data = np.loadtxt(ref_rho_file, delimiter=",")
                rho_spline = InterpolatedUnivariateSpline(rho_data[:, 0], rho_data[:, 1])
                rho_ref_array = rho_spline(wvl)

            # Evaluate each model
            for model_idx, model in enumerate(model_names):
                candidate_path = BACK_RESULTS_BASE / f"{model}_{scenario}{track_num}.h5"
                
                if not candidate_path.is_file():
                    continue

                with h5py.File(candidate_path, "r") as f:
                    runtime = float(f.attrs.get('runtime', 0.0))
                    Y_pred = np.array(f["/LUTdata"], dtype=float)
                    run_time_dic[model].append(runtime)

                if scenario == "A":
                    rho_pred = retrieve_reflectance(L_ref, Y_pred, sza, n_wvl)
                    mre[model_idx, col_idx] = error_metric_A(rho_ref_array, rho_pred, wvl)
                else:
                    L_toa_est = toa_radiance(rho_ref_array, Y_pred, sza, n_wvl)
                    mre[model_idx, col_idx] = error_metric_B(L_ref, L_toa_est)

            # Rank based on current column (Ignore NaNs)
            valid_mask = ~np.isnan(mre[:, col_idx])
            if valid_mask.any():
                sorted_indices = np.argsort(mre[valid_mask, col_idx])
                # Assign 1-based rank
                temp_ranking = np.zeros(valid_mask.sum(), dtype=int)
                temp_ranking[sorted_indices] = np.arange(1, valid_mask.sum() + 1)
                rnk[valid_mask, col_idx] = temp_ranking
            
            # Penalize missing submissions with worst rank
            rnk[~valid_mask, col_idx] = n_models

    # Compute final aggregates
    avg_times = {k: float(np.mean(v)) if v else np.nan for k, v in run_time_dic.items()}
    score, final_rank = compute_final_ranks(rnk)

    # Build DataFrame
    cols = ["MRE_A1", "MRE_A2", "MRE_B1", "MRE_B2"]
    df = pd.DataFrame(mre, columns=cols)
    df.insert(0, "Model", model_names)
    df["Score"] = score
    df["Runtime"] = [avg_times[model] for model in df["Model"]]
    df["Ranking"] = final_rank

    df.sort_values("Ranking", inplace=True)
    return df