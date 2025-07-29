import time
import numpy as np
import pandas as pd
from scipy.stats import (norm, skewnorm, logistic, t, johnsonsu, gennorm)
from scipy.optimize import minimize_scalar
from matplotlib import colormaps
import matplotlib.pyplot as plt
import subprocess
from io import StringIO
import os
from sklearn.mixture import GaussianMixture
from matplotlib.lines import Line2D

def modality_test(df, code, year, flow, config, logger, unit_label,  
                  r_script_path="uv_modality_test.R", mod0=1, 
                  col="ln_uv", methods=["SI", "HH"], cap_size=50000):
    """
    Run R-based modality tests (Silverman and Hartigan's Dip) on a univariate column.

    This function runs the specified modality tests using an R script on a 
    numeric column of a pandas DataFrame.
    It supports capping sample size for bootstrapped tests (like Silverman's) 
    to improve computational efficiency.

    The current implementation is designed to:
    - Use only two methods: "SI" (Silverman) and "HH" (Hartigan's Dip)
    - Return a conservative decision: only if both reject unimodality is the 
      data labeled "multimodal"
    - Label cases where one rejects and the other fails as "borderline" but 
      conservatively mark them as "unimodal"
    - Return "unknown" if either SI or HH fails to compute 
      (e.g. missing p-values)

    Parameters:
        df : pandas.DataFrame
            The input data containing the column to test.
        r_script_path : str
            Path to the R script that runs the modality test.
        mod0 : int
            Null hypothesis: number of modes to test against (typically 1).
        col : str
            Column name in the DataFrame to test.
        methods : list of str, optional
            List of modality test methods to run. Default is ["SI", "HH"].
        cap_size : int, optional (default = 50000)
            Maximum number of observations used for bootstrapped or 
            kernel-based tests to save time.
        logger : logger object
            Optional logger to log diagnostic information and summary.

    Returns:
        report_modality : dict
            Dictionary containing all test results, p-values, decision flags, 
            and sample metadata.
        final_decision : str
            One of {"unimodal", "multimodal", "unknown"}. Conservative fallback
            if input is invalid.
        is_borderline : bool
       True if exactly one of SI/HH rejects unimodality (i.e. borderline case).
    """
    logger.info(f"🚀 Modality test (HS {code}, {year}, {flow.upper()}, " 
                f"{unit_label})")
    start_time = time.time()
    print("Running modality test on unit values...")
    
    full_df = df[col].dropna()
    original_n = len(full_df)

    boot_methods = {"SI"}
    cap_needed = methods is None or any(m in boot_methods for m in methods)

    if cap_needed and original_n > cap_size:
        sampled = full_df.sample(cap_size, random_state=42)
        was_capped = True
    else:
        sampled = full_df
        was_capped = False

    values = sampled.to_numpy()
    csv_data = "\n".join(f"{v}" for v in values)

    cmd = [config["rscript_exec"], r_script_path, str(mod0)]
    if methods:
        cmd += methods

    try:
        process = subprocess.run(
            cmd,
            input=csv_data,
            text=True,
            capture_output=True,
            check=True
        )
        pvals_df = pd.read_csv(StringIO(process.stdout))

        reject = pvals_df["P_Value"] < 0.05
        pvals_df["Decision"] = reject.map(
                          lambda x: "reject" if x else "fail to reject")
        n_reject = reject.sum()

        si_p = pvals_df.loc[pvals_df["Method"] == "SI", "P_Value"
                ].values[0] if "SI" in pvals_df["Method"].values else None
        hh_p = pvals_df.loc[pvals_df["Method"] == "HH", "P_Value"
                ].values[0] if "HH" in pvals_df["Method"].values else None

        if si_p is not None and hh_p is not None:
            if si_p < 0.05 and hh_p < 0.05:
                final_decision = "multimodal"
                is_borderline = False
            elif si_p >= 0.05 and hh_p >= 0.05:
                final_decision = "unimodal"
                is_borderline = False
            else:
                final_decision = "unimodal"
                is_borderline = True
        else:
            final_decision = "unknown"
            is_borderline = False

        report_modality = {
            "t_method": "SI+HH",
            "t_cap_size": cap_size,
            "t_modality_decision": final_decision,
            "t_modality_votes": int(n_reject),
            "t_sample_capped": was_capped,
            "t_sample_used": len(sampled),
            "t_sample_original": original_n,
            "t_borderline": is_borderline
        }

        for _, row in pvals_df.iterrows():
            method_name = row["Method"]
            report_modality[f"t_{method_name}_P"] = row["P_Value"]
            report_modality[f"t_{method_name}_decision"] = row["Decision"]

        if logger:
            logger.info("📈 Modality test summary:")
            logger.info(f"- Column tested: {col}")
            logger.info(f"- Original sample size: {original_n}")
            if was_capped:
                logger.warning(f"❗ Sample capped at {cap_size} for efficiency.")
            else:
                logger.info("✅ Full sample used.")
            logger.info(f"- Final decision: {final_decision} ({n_reject} reject out of {len(pvals_df)})")
            if final_decision == "unknown":
                logger.warning("⚠️ Modality unknown: SI or HH test missing.")
            elif is_borderline:
                logger.warning("⚠️ Borderline modality test result: SI and HH disagree.")
        
        elapsed = time.time() - start_time    
        logger.info(f"✅ Modality test (HS {code}, {year}, {flow.upper()}, "
                    f"{unit_label}) completed in {elapsed:.2f} seconds.")
        return report_modality, final_decision, is_borderline

    except subprocess.CalledProcessError as e:
        if logger:
            logger.error("❌ R script for modality test failed.")
            logger.error(f"STDERR:\n{e.stderr}")
            
        elapsed = time.time() - start_time    
        logger.info(f"✅ Modality test (HS {code}, {year}, {flow.upper()}, "
                    f"{unit_label}) completed in {elapsed:.2f} seconds.")
        return {
            "t_method": "SI+HH",
            "t_cap_size": cap_size,
            "t_modality_error": str(e),
            "t_modality_decision": "error",
            "t_modality_votes": int(n_reject),
            "t_sample_capped": was_capped,
            "t_sample_used": len(sampled),
            "t_sample_original": original_n,
            "t_borderline": is_borderline
        }, "error", False
    
def _estimate_mode(dist, args, data):
    """Estimate mode as the peak of the PDF over the data range."""
    result = minimize_scalar(
        lambda x: -dist.pdf(x, *args),
        bounds=(min(data), max(data)),
        method="bounded",
    )
    return result.x

def _fit_distribution(dist_obj, data, logger=None):
    """
    Fit a given scipy.stats distribution object to the data and return a dictionary
    of results with semantically named keys, plus the original params tuple.

    Args:
        dist_obj: A scipy.stats distribution object (e.g., scipy.stats.norm).
        data: 1D array-like of data points to fit.
        logger: Optional logger for info output.

    Returns:
        dist_name (str), result (dict), params (tuple): Name, result dict, and raw parameter tuple.
    """
    dist_name = dist_obj.name
    params = dist_obj.fit(data)
    loglik = np.sum(dist_obj.logpdf(data, *params))
    mean, var, skew, kurt = dist_obj.stats(*params, moments="mvsk")
    sample_var = np.var(data, ddof=1)
    median = dist_obj.median(*params)
    mode = _estimate_mode(dist_obj, params, data)
    n = len(data)

    aic = 2 * len(params) - 2 * loglik
    bic = len(params) * np.log(n) - 2 * loglik

    # Semantic parameter naming for reporting
    param_name_map = {
        "norm":      ["loc", "scale"],
        "skewnorm":  ["a", "loc", "scale"],
        "t":         ["df", "loc", "scale"],
        "gennorm":   ["beta", "loc", "scale"],
        "johnsonsu": ["a", "b", "loc", "scale"],
        "logistic":  ["loc", "scale"]
    }

    param_names = param_name_map.get(dist_name, [f"param{i+1}" for i in range(len(params))])
    param_dict = {f"{dist_name}_{name}": val for name, val in zip(param_names, params)}

    result = {
        **param_dict,
        f"{dist_name}_loglik": loglik,
        f"{dist_name}_mean": mean,
        f"{dist_name}_median": median,
        f"{dist_name}_mode": mode,
        f"{dist_name}_variance": var,
        f"{dist_name}_sample_variance": sample_var,
        f"{dist_name}_skew": skew,
        f"{dist_name}_kurtosis": kurt,
        f"{dist_name}_aic": aic,
        f"{dist_name}_bic": bic,
    }
    
    result = {k: round(float(v), 3) if isinstance(v, np.floating) else v 
              for k, v in result.items()}

    if logger:
        logger.info(f"Fitted {dist_name.capitalize()} Distribution")
        for k, v in result.items():
            logger.info(f"{k}: {v:.4f}" if isinstance(v, (float, int)) else f"{k}: {v}")

    return dist_name, result, params  # Note: returning raw params separately

def fit_all_unimodal_models(data, code, year, flow, logger, unit_label="USD/kg"):
    """
    Fit all candidate unimodal distributions and select the best one based on AIC and BIC.

    Args:
        data (array-like): Input data to fit.
        logger (logging.Logger, optional): Logger for info output.

    Returns:
        tuple: (best_fit_name, report_best_fit_uni, report_all_uni_fit, raw_params_dict)
    """
    logger.info(f"🚀 Fitting unimodal distribution (HS {code}, {year}, "
                f"{flow.upper()}, {unit_label})")
    start_time = time.time()
    
    model_objs = [norm, skewnorm, t, gennorm, johnsonsu, logistic]

    report_all_uni_fit = {}
    raw_params_dict = {}
    scores = {}

    for dist_obj in model_objs:
        dist_name, result, params = _fit_distribution(dist_obj, data, logger=None)
        report_all_uni_fit.update(result)
        raw_params_dict[f"{dist_name}_params"] = params
        scores[dist_name] = (result[f"{dist_name}_aic"], result[f"{dist_name}_bic"])

    best_fit_name = min(scores.items(), key=lambda x: (x[1][0], x[1][1]))[0]
    report_best_fit_uni = {
        k: v for k, v in report_all_uni_fit.items() if k.startswith(best_fit_name)
    }

    report_all_uni_fit["best_fit_name"] = best_fit_name

    logger.info(f"Best fit based on AIC/BIC: {best_fit_name.capitalize()}")
        
    elapsed = time.time() - start_time    
    logger.info(f"✅ Unimodal distribution fit (HS {code}, {year}, {flow.upper()}, "
                f"{unit_label}) completed in {elapsed:.2f} seconds.")
    
    return best_fit_name, report_best_fit_uni, report_all_uni_fit, raw_params_dict

def bootstrap_parametric_ci(
    data, code, year, flow, logger, unit_label="USD/kg",dist="skewnorm", 
    n_bootstraps=1000, confidence=0.95
):
    """
    Bootstrap confidence intervals for mean, median, mode, and variance
    of a parametric distribution.

    Args:
        data (array-like): Input data to estimate the confidence intervals.
        dist (str): Distribution name: 'normal', 'skewnorm', 'studentt',
        'gennorm', 'johnsonsu', 'logistic'.
        n_bootstraps (int): Number of bootstrap samples.
        confidence (float): Confidence level (default 0.95).

    Returns:
        tuple: CI for mean, median, mode, and variance.
    """
    logger.info(f"🚀 Bootstrapping CI (HS {code}, {year}, {flow.upper()}, {unit_label})")
    start_time = time.time()
    n = len(data)
    alpha = 1 - confidence
    boot_means, boot_medians, boot_modes, boot_vars = [], [], [], []

    for _ in range(n_bootstraps):
        sample = np.random.choice(data, size=n, replace=True)

        if dist == "norm":
            mu, sigma = norm.fit(sample)
            mean, var = norm.stats(mu, sigma, moments="mv")
            median = mu
            mode = mu

        elif dist == "skewnorm":
            a, loc, scale = skewnorm.fit(sample)
            mean, var = skewnorm.stats(a, loc=loc, scale=scale, moments="mv")
            median = skewnorm.median(a, loc=loc, scale=scale)
            mode = _estimate_mode(skewnorm, (a, loc, scale), sample)

        elif dist == "t":
            df_, loc_, scale_ = t.fit(sample)
            mean, var = t.stats(df_, loc_, scale_, moments="mv")
            median = t.median(df_, loc_, scale_)
            mode = _estimate_mode(t, (df_, loc_, scale_), sample)

        elif dist == "gennorm":
            beta_, loc_, scale_ = gennorm.fit(sample)
            mean, var = gennorm.stats(beta_, loc_, scale_, moments="mv")
            median = gennorm.median(beta_, loc_, scale_)
            mode = _estimate_mode(gennorm, (beta_, loc_, scale_), sample)

        elif dist == "johnsonsu":
            a_, b_, loc_, scale_ = johnsonsu.fit(sample)
            mean, var = johnsonsu.stats(a_, b_, loc_, scale_, moments="mv")
            median = johnsonsu.median(a_, b_, loc_, scale_)
            mode = _estimate_mode(johnsonsu, (a_, b_, loc_, scale_), sample)

        elif dist == "logistic":
            loc_, scale_ = logistic.fit(sample)
            mean, var = logistic.stats(loc_, scale_, moments="mv")
            median = logistic.median(loc_, scale_)
            mode = _estimate_mode(logistic, (loc_, scale_), sample)

        else:
            raise ValueError(f"Unsupported distribution: {dist}")

        boot_means.append(mean)
        boot_medians.append(median)
        boot_modes.append(mode)
        boot_vars.append(var)
        
    # === CI Computation ===
    def _ci_bounds(arr):
        return (
            np.percentile(arr, 100 * alpha / 2),
            np.percentile(arr, 100 * (1 - alpha / 2))
        )

    ci_mean = _ci_bounds(boot_means)
    ci_median = _ci_bounds(boot_medians)
    ci_mode = _ci_bounds(boot_modes)
    ci_var = _ci_bounds(boot_vars)
    
    logger.info(f"{confidence:.0%} CI for Mean: {ci_mean[0]:.3f} – {ci_mean[1]:.3f} (log), {np.exp(ci_mean[0]):.3f} – {np.exp(ci_mean[1]):.3f} USD/kg")
    logger.info(f"{confidence:.0%} CI for Median: {ci_median[0]:.3f} – {ci_median[1]:.3f} (log), {np.exp(ci_median[0]):.3f} – {np.exp(ci_median[1]):.3f} USD/kg")
    logger.info(f"{confidence:.0%} CI for Mode: {ci_mode[0]:.3f} – {ci_mode[1]:.3f} (log), {np.exp(ci_mode[0]):.3f} – {np.exp(ci_mode[1]):.3f} USD/kg")
    logger.info(f"{confidence:.0%} CI for Variance: {ci_var[0]:.3f} – {ci_var[1]:.3f}")
    
    elapsed = time.time() - start_time    
    logger.info(f"✅ CI bootstrapping (HS {code}, {year}, {flow.upper()}, "
                f"{unit_label}) completed in {elapsed:.2f} seconds.")
    
    return {
    "ci_ci": confidence,
    "ci_n_boot": n_bootstraps,
    "ci_mean_lower": round(float(ci_mean[0]), 3),
    "ci_mean_upper": round(float(ci_mean[1]), 3),
    "ci_median_lower": round(float(ci_median[0]), 3),
    "ci_median_upper": round(float(ci_median[1]), 3),
    "ci_mode_lower": round(float(ci_mode[0]), 3),
    "ci_mode_upper": round(float(ci_mode[1]), 3),
    "ci_variance_lower": round(float(ci_var[0]), 3),
    "ci_variance_upper": round(float(ci_var[1]), 3),
}

def find_gmm_components(
    data,
    code,
    year,
    flow,
    config,
    logger,
    unit_label="USD/kg",
    max_components=50,
    convergence_threshold=5,
    reg_covar=1e-3,
    threshold=0.2,
    n_init=10,
    plot=False,
    ax=None,
    save_path=None
):
    """
    Efficient selection of GMM components using BIC and slope-based adjustment.
    
    Select Optimal Number of GMM Components Using BIC and Tangent-Based Analysis
    This method identifies the most suitable number of Gaussian Mixture Model 
    (GMM) components for a given dataset using the Bayesian Information 
    Criterion (BIC), enhanced by slope-based (tangent) heuristics.

    BIC balances model fit and complexity:
        BIC = k * log(n) - 2 * log(L)
    where:
        - k: number of model parameters,
        - n: number of data points,
        - L: maximum likelihood of the model.

    While BIC penalizes complexity, it may not always yield the simplest 
    sufficient model. Its curve often shows:
    - **L-shape**: steep early improvement followed by plateauing,
    - **Tick-shape**: continued improvement followed by sharp worsening (overfitting).

    This method analyzes the BIC curve's shape by comparing the slope between 
    each candidate (k) and the global minimum (k_best):
        
        Slope_k = (BIC_k - BIC_best) / |k - k_best|

    Alternative candidates are flagged if their slope is significantly smaller 
    (e.g., < 20%) than the steepest slope nearby:
    - For L-shape: compares slope before the minimum,
    - For tick-shape: compares slope after the minimum.

    This allows identifying simpler yet statistically competitive models.

    Parameters:
        data : array-like
            Input 1D or 2D data array.
        max_components : int
            Maximum number of GMM components to evaluate (starting from 2).
        convergence_threshold : int
            Stop early if BIC-optimal value doesn't change for this many iterations.
        reg_covar : float
            Regularization term for GMM covariance matrices.
        threshold : float
            Slope ratio threshold for L-shape and tick-shape adjustments.
        n_init : int
            Number of initializations for GMM (robustness to local optima).
        plot : bool
            Whether to show the BIC plot.
        ax : matplotlib Axes
            Optional matplotlib axes object to draw the plot.
        save_path : str or None
            Optional path to save the plot as an image file.

    Returns:
        optimal_k : int
            Selected number of components.
        bic_values : list
            List of BIC values for each model.
        report : dict
            Diagnostic info and selection metadata.
    """
    logger.info(f"🚀 GMM component number selection (HS {code}, {year}, {flow.upper()}, {unit_label})")
    start_time = time.time()
    
    # Convert to numpy array if it's a DataFrame
    if hasattr(data, "to_numpy"):
        data = data.to_numpy()
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_samples = data.shape[0]
    bic_values = np.empty(max_components)  # BIC for k=1 to max_components
    stable_counter = 0
    prev_best = None

    for i, n_components in enumerate(range(1, max_components + 1)):
        gmm = GaussianMixture(
            n_components=n_components,
            random_state=42,
            reg_covar=reg_covar,
            n_init=n_init
        )
        gmm.fit(data)
        bic = gmm.bic(data)
        bic_values[i] = bic

        current_best = int(np.argmin(bic_values[: i + 1])) + 1
        if prev_best is None:
            prev_best = current_best
        elif current_best == prev_best:
            stable_counter += 1
        else:
            stable_counter = 0
        prev_best = current_best

        if stable_counter >= convergence_threshold:
            bic_values = bic_values[: i + 1]
            break

    best_k = int(np.argmin(bic_values)) + 1
    best_bic = np.min(bic_values)

    tangents = [
        (bic - best_bic) / abs(idx - (best_k - 1)) if idx != (best_k - 1) else 0
        for idx, bic in enumerate(bic_values)
    ]

    best_idx = best_k - 1
    l_adj = None
    tick_adj = None

    if best_idx >= 2:
        max_pre = max(tangents[:best_idx])
        max_post = max(tangents[best_idx + 1:])

        for i in range(best_idx):
            if tangents[i] < threshold * max_pre:
                l_adj = i + 1
                break
        for i in range(best_idx):
            if tangents[i] < threshold * max_post:
                tick_adj = i + 1
                break

    candidates = [best_k]
    notes = {}
    if l_adj and l_adj < best_k:
        candidates.append(l_adj)
        notes[l_adj] = "L-shape adjustment"
    if tick_adj and tick_adj < best_k:
        candidates.append(tick_adj)
        notes[tick_adj] = "Tick-shape adjustment"

    optimal_k = min(candidates)

    logger.info(f"📈 GMM selection summary: BIC-best={best_k},"
         f" L-adjust={l_adj}, Tick-adjust={tick_adj}, Selected={optimal_k}")

    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        x_vals = np.arange(1, 1 + len(bic_values))
        ax.plot(x_vals, bic_values, marker='o', label="BIC")
        ax.axvline(best_k, linestyle="--", color="red", 
                   label=f"Best BIC: {best_k}")
        for k in notes:
            ax.axvline(k, linestyle=":", label=f"{notes[k]}: {k}")
        ax.set_xticks(x_vals)
        ax.set_xlabel("Number of components")
        ax.set_ylabel("BIC")
        text_d = 'imports' if flow == 'm' else 'exports'
        ax.set_title(f"GMM component selection based on BIC for unit values "
                     f"({unit_label}) of HS {code} {text_d} in {year}")
        dummy_sample = Line2D([], [], linestyle="none")
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(0, dummy_sample)
        labels.insert(0, f"Sample size: {n_samples:,}")
        ax.legend(handles, labels)
        ax.grid(True)

        if save_path:
            plt.tight_layout()
            unit_suffix = unit_label.split("/")[-1]
            save_path = os.path.join(config["dirs"]["figures"], 
                            f"cs_{code}_{year}_{flow}_{unit_suffix}.png")
            plt.savefig(save_path, dpi=300)
            if ax is None:
                plt.close()
        else:
            plt.tight_layout()
            plt.show()

    report_gmm_cselect = {
        "cs_n_samples": n_samples,
        "cs_optimal_components": optimal_k,
        "cs_bic_best_components": best_k,
        "cs_bic_at_optimal_components": round(float(bic_values[optimal_k - 1]), 3),
        "cs_bic_at_best_components": round(float(bic_values[best_k - 1]), 3),
        "cs_l_shape_adjustment": l_adj,
        "cs_tick_shape_adjustment": tick_adj,
        "cs_converged_early": stable_counter >= convergence_threshold,
        "cs_threshold": threshold,
        "cs_n_init": n_init,
        "cs_reg_covar": reg_covar
    }
    
    elapsed = time.time() - start_time
    logger.info(f"✅ GMM component number selection (HS {code}, {year}, "
            f"{flow.upper()}, {unit_label}) completed in {elapsed:.2f} seconds.")

    return optimal_k, bic_values.tolist(), report_gmm_cselect

def fit_gmm(
    df,
    columns,
    best_component,
    code,
    year,
    flow,
    config,
    logger,
    unit_label="USD/kg",
    plot=True,
    save_path=None,
    ax=None,
    n_init=10,
    reg_covar = 1e-3,
):
    """
    Fits a Gaussian Mixture Model (GMM) to log unit value data, computes
    cluster statistics, optionally plots the distribution, and returns results.

    Parameters:
    - df: DataFrame with the data.
    - columns: List with one column name (e.g. ['ln_uv']).
    - best_component: Number of GMM components.
    - code, year, flow: Metadata for labeling.
    - cc: If True, return country composition for each cluster.
    - plot: If True, plot GMM fit and histogram.
    - save_path: File path to save plot.
    - ax: Matplotlib axis (optional).
    - n_init: Number of GMM initializations.

    Returns:
    - Dictionary with GMM statistics and metadata.
    """
    logger.info(f"🚀 GMM fit (HS {code}, {year}, {flow.upper()}, {unit_label})")
    start_time = time.time()
    
    # === Data Extraction ===
    if len(columns) != 1:
        raise ValueError(f"Expected exactly one column for 1D GMM, but got {len(columns)}: {columns}")
    
    data = df[columns[0]].values.reshape(-1, 1) # turn 1D array into a 2D column vector, required by scikit-learn's GMM implementation
    
    # === GMM Initialization and Fitting ===
    gmm = GaussianMixture(
        n_components=best_component,
        random_state=42,
        n_init=n_init,
        reg_covar=reg_covar
    )
    gmm.fit(data)
    
    # ==== Extracting GMM Parameters ===
    means = gmm.means_.flatten()
    proportions = gmm.weights_.flatten()
    covariances = gmm.covariances_.flatten()

    # ==== Confidence intervals ===
    N = len(data)
    standard_errors = np.sqrt(covariances) / np.sqrt(N * proportions)
    z_alpha_half = norm.ppf(0.975)
    lower_bound = means - z_alpha_half * standard_errors
    upper_bound = means + z_alpha_half * standard_errors

    # ==== Sorting by Cluster Size (Proportion) ===
    sorted_indices = np.argsort(proportions)[::-1]
    sorted_means = means[sorted_indices]
    sorted_proportions = proportions[sorted_indices]
    sorted_covariances = covariances[sorted_indices]
    sorted_lower_bound = lower_bound[sorted_indices]
    sorted_upper_bound = upper_bound[sorted_indices]

    # ==== Country composition (NumPy version) ===
    sorted_country_proportions = None
    component_labels = gmm.predict(data) # assign each data point to its most likely cluster using the fitted GMM.
    partner_codes = df["partnerISO"].values
    sorted_country_proportions = {}

    for new_i, old_i in enumerate(sorted_indices):
        mask = component_labels == old_i # creates a boolean mask
        selected = partner_codes[mask] # select countries with this masked component
        if len(selected) > 0:
            unique, counts = np.unique(selected, return_counts=True)
            proportions = counts / counts.sum()
            sorted_pairs = sorted(zip(unique, proportions), key=lambda x: x[1], reverse=True)
            sorted_country_proportions[new_i] = sorted_pairs
        else:
            sorted_country_proportions[new_i] = {}
        
    # ==== Plot for the total and component GMM PDF ===
    if plot:
        plt.rcParams["pdf.fonttype"] = 42
        own_figure = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            own_figure = True

        range_uv = data.max() - data.min()
        margin = 0.1 * range_uv
        x_min, x_max = data.min() - margin, data.max() + margin # 5% padding avoid cutting off GMM tails

        x_vals = np.linspace(x_min, x_max, 1000)

        cmap = colormaps.get_cmap("tab20")
        colors = [cmap(i / best_component) for i in range(best_component)]

        pdf = np.exp(gmm.score_samples(x_vals.reshape(-1, 1))) # pdf for total GMM-fitted curve

        for i in range(best_component):
            mu = sorted_means[i]
            sigma = np.sqrt(sorted_covariances[i])
            a = sorted_proportions[i]
            component_pdf = a * norm.pdf(x_vals, loc=mu, scale=sigma)

            if sorted_country_proportions:
                top_countries = sorted_country_proportions[i]
                countries_str = ", ".join(
                    [f"{k}: {v*100:.0f}%" for k, v in top_countries[:3]]
                )
                label = f"Cluster ({i+1}) {a*100:.0f}%: {np.exp(mu):.2f} {unit_label}\n({countries_str})"
            else:
                label = f"Cluster ({i+1}) {a*100:.0f}%: {np.exp(mu):.2f} {unit_label}"

            ax.plot(x_vals, component_pdf, label=label, color=colors[i])

            # Annotate cluster number
            ax.text(
                mu,
                component_pdf.max() / 2,
                f"({i+1})",
                fontsize=8,
                ha="center",
                va="center"
            )

        # Histogram
        bin_edges = np.histogram_bin_edges(data.flatten(), bins='fd') # decide bin width useing Freeman-Diaconis rule
        ax.hist(data.flatten(), bins=bin_edges, density=True, alpha=0.4, label=f"Histogram (sample size: {len(data):,})", color="gray")

        # Overall GMM
        ax.plot(x_vals, pdf, label="Overall GMM", color="black", linewidth=2)
        
        # === Evenly split all legend entries into upper left and upper right ===
        handles_all, labels_all = ax.get_legend_handles_labels()
        legend_entries = list(zip(handles_all, labels_all))

        total = len(legend_entries)
        mid = total // 2
        left_entries = legend_entries[:mid]
        right_entries = legend_entries[mid:]

        if left_entries:
            leg1 = ax.legend(*zip(*left_entries), loc="upper left", frameon=False)
            ax.add_artist(leg1)
        if right_entries:
            leg2 = ax.legend(*zip(*right_entries), loc="upper right", frameon=False)
        
        text_d = 'imports' if flow == 'm' else 'exports'
        ax.set_title(f"GMM fit of unit values ({unit_label}) for HS {code} {text_d} in {year}")
        xlabel = f"ln(Unit Value) [{unit_label}]" 
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
            
        if save_path:
            plt.tight_layout()
            unit_suffix = unit_label.split("/")[-1]  # e.g., "kg" from "USD/kg"
            save_path = os.path.join(config["dirs"]["figures"], 
                 f"fit_{code}_{year}_{flow}_{unit_suffix}_gmm_1d.png")
            plt.savefig(save_path, dpi=300)
            if ax is None:
                plt.close()   
        else:
            plt.tight_layout()
            plt.show()
            
    # === Init report with general info ===
    gmm_1d_report = {
        "gmm_components": int(best_component),
        "gmm_samples": int(N),
        "gmm_reg_covar": float(reg_covar),
        "gmm_n_init": int(n_init),
    }

    # === Unified pass per component ===
    for i in range(best_component):
        mu = float(sorted_means[i])
        p = float(sorted_proportions[i])
        cov = float(sorted_covariances[i])
        lo = float(sorted_lower_bound[i])
        hi = float(sorted_upper_bound[i])

        gmm_1d_report[f"gmm_c{i+1}_mean"] = round(mu, 4)
        gmm_1d_report[f"gmm_c{i+1}_proportion"] = round(p, 4)
        gmm_1d_report[f"gmm_c{i+1}_covariance"] = round(cov, 4)
        gmm_1d_report[f"gmm_c{i+1}_mean_ci_lower"] = round(lo, 4)
        gmm_1d_report[f"gmm_c{i+1}_mean_ci_upper"] = round(hi, 4)

        if sorted_country_proportions:
            top_items = sorted_country_proportions.get(i, [])[:3]
            for j, (country, share) in enumerate(top_items):
                gmm_1d_report[f"gmm_c{i+1}_country{j+1}"] = f"{country} ({round(share * 100, 2)}%)"
               
    elapsed = time.time() - start_time
    logger.info(f"✅ GMM fit (HS {code}, {year}, "
            f"{flow.upper()}, {unit_label}) completed in {elapsed:.2f} seconds.")
    
    return gmm_1d_report