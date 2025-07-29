"""
UVPicker - Unit Value Analysis Tool
Copyright (c) 2025 Kai Li
Licensed under LGPL v3.0 – See LICENSE file for details.
Funding Acknowledgment:
- European Horizon Project (No. 101060142) "RESOURCE – REgional project 
  development aSsistance fOr the Uptake of an aRagonese Circular Economy"
- Financial support from CML, Leiden University, for full access to the 
  UN Comtrade database 
29/07/2025
"""
import time
from uv_logger import logger_setup
from uv_config import load_config, prefix_dict_keys, save_report_dict
from uv_preparation import clean_trade, detect_outliers
from uv_analysis import (
    modality_test,
    fit_all_unimodal_models,
    bootstrap_parametric_ci,
    find_gmm_components,
    fit_gmm,
)
from uv_visualization import plot_histogram, plot_dist

# subscription_key = "4a624b220f67400c9a6ef19b1890f1f9"
# path = 'C:/Users/lik6/Data/ComtradeTariffline/merge/split_by_hs_2023_numpy'
#code = "220300"
#year = "2010"
#flow = "m"

def cmltrade_uv(code, year, flow):
    
    # === Load config (including folder paths, mappings, thresholds)
    config = load_config()

    # === Setup logger for this specific HS code, year, and flow
    logger = logger_setup(
        code=code, year=year, flow=flow, log_dir=config["dirs"]["logs"]
    )
    zero_time = time.time()  # Starting the total analysis timer
    print(f"Starting analysis for HS code {code} in year {year}...\n")
    # %% Step 1: Clean trade data
 
    (df_uv, df_q, report_clean, report_q_clean, non_kg_unit) = clean_trade(
        code, year, flow, config, logger)
 
    # %% Step 2: Detect outliers
    # === Kg-based UV ===
    df_filtered, df_outliers, report_outlier, is_valid_for_fit = detect_outliers(
        df_uv, "ln_uv", code, year, flow, config, logger, unit_label="USD/kg", 
        plot =True, save_path=True, file_format="pdf")
    
    # === Non-kg-based UV (if exists) ===
    if non_kg_unit != 'USD/kg':
        df_q_filtered, df_q_outliers, report_q_outlier, is_valid_for_fit_q = \
        detect_outliers(df_q,"ln_uv_q",code,year,flow, config, logger,
         unit_label=non_kg_unit, plot =True, save_path=True, file_format="pdf")
    
    
    if not is_valid_for_fit and non_kg_unit == 'USD/kg' or \
       not is_valid_for_fit and not is_valid_for_fit_q:
        logger.warning("❌ Skipping both kg-based and non-kg-based analysis "
                       "due to insufficient sample size.")
       
        if non_kg_unit != 'USD/kg':
            report_q_outlier = prefix_dict_keys(report_q_outlier, prefix="q_")

        report_final = {}
        report_final.update(report_clean)
        report_final.update(report_outlier)
        report_final.update(report_q_clean)
        if non_kg_unit != 'USD/kg':
            report_final.update(report_q_outlier)
        
        report_final["skip_reason"] = "Too few kg-based and non-kg-based records"
        
        save_report_dict(report_final, code, year, flow, config, logger)
        
        elapsed = time.time() - zero_time
        print(f"🟢 cmltrade_uv completed in {elapsed} seconds.")
        
        return report_final
    # %% Step 3: Histogram 
    # === Kg-based UV ===
    if is_valid_for_fit:
        plot_histogram(df_filtered["ln_uv"],code,year,flow, config, logger, 
                   unit_label="USD/kg",save_path=True, file_format="pdf")
    
    # === Non-kg-based UV (if exists) ===
    if non_kg_unit != 'USD/kg' and is_valid_for_fit_q:
        plot_histogram(df_q_filtered["ln_uv_q"],code,year,flow, config, logger,
                       unit_label=non_kg_unit,save_path=True, file_format="pdf")
    # %% Step 4: Modality test
    # === Kg-based UV ===
    if is_valid_for_fit:
        report_modality, modality_decision, is_borderline = modality_test(
          df_filtered,code, year, flow, config, logger, unit_label ="USD/kg")

    # === Non-kg-based UV (if exists) ===
    if non_kg_unit != 'USD/kg' and is_valid_for_fit_q:
        report_q_modality, modality_q_decision, is_q_borderline = \
        modality_test(df_q_filtered,code, year, flow, config, logger, 
                      unit_label=non_kg_unit, col="ln_uv_q")
    else:
        modality_q_decision = "unknown"
        is_q_borderline = False
    # %% Step 5: Distribution fit
    # %%% Kg-based UV
    if is_valid_for_fit :
        if modality_decision == "unimodal" or is_borderline:
            
            # === Fitting Unimodal distribution of kg-based UV ====
            (best_fit_name, report_best_fit_uni, report_all_uni_fit, 
             raw_params_dict) = fit_all_unimodal_models(df_filtered["ln_uv"], 
                    code, year, flow, logger, unit_label="USD/kg")
            
            # === Bootstrapping CI (kg-based) ==== 
            report_ci_uni = bootstrap_parametric_ci(
                df_filtered["ln_uv"], code, year, flow, config, logger, 
                unit_label="USD/kg", dist=best_fit_name, n_bootstraps=1000)

            # === Plotting unimodal distribution fit of kg-based UV ====
            plot_dist(
                df_filtered["ln_uv"],code,year,flow, logger, 
                unit_label="USD/kg",
                dist=None,
                best_fit_name=best_fit_name,
                report_best_fit_uni=report_best_fit_uni,
                report_all_uni_fit=report_all_uni_fit,
                raw_params_dict=raw_params_dict,
                ci=report_ci_uni,
                save_path=True,
                ax=None,
                file_format="pdf"
            )
            
            if is_borderline:
                # === Also fitting GMM on kg-based UV if borderline ===
                optimal_k, bic_values, report_gmmf_1d = find_gmm_components(
                    df_filtered[["ln_uv"]], code, year, flow, config, logger, 
                    unit_label="USD/kg", plot=True, save_path=True)
                
                report_gmm_1d = fit_gmm(
                    df_filtered,
                    ["ln_uv"],
                    optimal_k,
                    code, year, flow, config, logger, 
                    unit_label="USD/kg",
                    plot=True,
                    save_path=True,
                    n_init=10,
                    reg_covar=1e-3,
                )
                
        else:   
            optimal_k, bic_values, report_gmmf_1d = find_gmm_components(
                df_filtered[["ln_uv"]], code, year, flow, config, logger,
                plot=True, save_path=True
            )
            
            report_gmm_1d = fit_gmm(
                df_filtered,
                ["ln_uv"],
                optimal_k,
                code, year, flow, logger, 
                unit_label="USD/kg",
                plot=True,
                save_path=True,
                n_init=10,
                reg_covar=1e-3,
            )
    # %%% Non-kg-based UV
    if non_kg_unit != 'USD/kg'  and df_q is not None and not df_q.empty:
        if modality_q_decision == "unimodal" or is_q_borderline:
            
            # === Fitting Unimodal distribution of non-kg-based UV ====
            (best_fit_name_q,
                report_q_best_fit_uni,
                report_q_all_uni_fit,
                raw_params_dict_q,
            ) = fit_all_unimodal_models(df_q_filtered["ln_uv_q"], code, year, 
                                        flow, logger, unit_label=non_kg_unit)
            
            # === Bootstrapping CI (non-kg-based) ====    
            report_q_ci_uni = bootstrap_parametric_ci(
                df_q_filtered["ln_uv_q"], code, year, flow, logger, 
                unit_label=non_kg_unit, dist=best_fit_name_q, n_bootstraps=1000)
            
            # === Plotting unimodal distribution fit of non-kg-based UV ====
            plot_dist(
                df_q_filtered["ln_uv_q"],
                code,
                year,
                flow,
                logger,
                unit_label=non_kg_unit,
                dist=None,
                best_fit_name=best_fit_name_q,
                report_best_fit_uni=report_q_best_fit_uni,
                report_all_uni_fit=report_q_all_uni_fit,
                raw_params_dict=raw_params_dict_q,
                ci=report_q_ci_uni,
                save_path=True,
                ax=None,
                file_format="pdf")
                
            if is_q_borderline:
                # === Also fitting GMM on non-kg-based UV if borderline ===
                optimal_k_q, bic_values_q, report_q_gmmf_1d = find_gmm_components(
                    df_q_filtered[["ln_uv_q"]], code, year, flow, logger, 
                    unit_label=non_kg_unit, plot=True, save_path=True
                )
               
                report_q_gmm_1d = fit_gmm(
                    df_q_filtered,
                    ["ln_uv_q"],
                    optimal_k_q,
                    code,
                    year,
                    flow,
                    logger, 
                    unit_label=non_kg_unit,
                    plot=True,
                    save_path=True,
                    n_init=10,
                    reg_covar=1e-3)

        else: 
            optimal_k_q, bic_values_q, report_q_gmmf_1d = find_gmm_components(
                df_q_filtered[["ln_uv_q"]], code, year, flow, logger, 
                unit_labe=non_kg_unit, plot=True, save_path=True)
            
            report_q_gmm_1d = fit_gmm(
                df_q_filtered,
                ["ln_uv_q"],
                optimal_k_q,
                code,
                year,
                flow,
                logger,
                unit_label=non_kg_unit,
                plot=True,
                save_path=True,
                n_init=10,
                reg_covar=1e-3)
            
    # %% Step 6: Return final report
    
    report_final = {}
    # Reports to merge (non-kg ones will be prefixed)
    report_keys_kg = [
        "report_clean", "report_outlier", "report_modality", 
        "report_all_uni_fit", "report_ci_uni", "report_gmmf_1d", "report_gmm_1d"
    ]

    report_keys_q = [
        "report_q_clean", "report_q_outlier", "report_q_modality", 
        "report_q_all_uni_fit", "report_q_ci_uni", "report_q_gmmf_1d", "report_q_gmm_1d"
    ]

    # Merge kg-based reports directly
    for name in report_keys_kg:
        r = locals().get(name)
        if isinstance(r, dict):
            report_final.update(r)

    # Merge q-based reports with prefix
    for name in report_keys_q:
        r = locals().get(name)
        if isinstance(r, dict):
            r_prefixed = prefix_dict_keys(r, prefix="q_")
            report_final.update(r_prefixed)
            
    save_report_dict(report_final, code, year, flow, config, logger)
    
    elapsed = time.time() - zero_time
    print(f"🟢 cmltrade_uv completed in {elapsed} seconds.")
          
    return report_final
# %% 
if __name__ == "__main__":
    import argparse
    import pandas as pd
    import os

    parser = argparse.ArgumentParser(description="Run UVPicker unit value analysis.\n"
        "Single run: python main.py --code 370110 --year 2023 --flow m\n"
        "Batch:      python main.py --chunk task_chunks/2010/task_chunk_00000.csv")
    parser.add_argument("--code", help="HS code")
    parser.add_argument("--year", help="Year")
    parser.add_argument("--flow", help="Flow direction ('m' or 'x')")
    parser.add_argument("--chunk", help="CSV file containing columns hs_code,year,flow")

    args = parser.parse_args()

    if args.chunk is not None:
        # Batch mode: read the chunk file and process each row
        if not os.path.exists(args.chunk):
            raise FileNotFoundError(f"Chunk file not found: {args.chunk}")
        df_chunk = pd.read_csv(args.chunk, dtype={'hs_code': str, 'year': str, 'flow': str})
        for idx, row in df_chunk.iterrows():
            hs_code = str(row['hs_code']).zfill(6)
            year = str(row['year'])
            flow = str(row['flow'])
            cmltrade_uv(row['hs_code'], row['year'], row['flow'])
    else:
        # Single run mode: check required arguments
        if not (args.code and args.year and args.flow):
            parser.error("Must specify either --chunk or all of --code, --year, and --flow.")
        hs_code = str(args.code).zfill(6)
        year = str(args.year)
        flow = str(args.flow)
        cmltrade_uv(args.code, args.year, args.flow)
