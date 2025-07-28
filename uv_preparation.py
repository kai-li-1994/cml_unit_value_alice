import time
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from uv_config import load_config
config = load_config()

def clean_trade2(code, year, flow, config, logger):
    
    logger.info(f"🚀 Clean trade data (HS {code}, {year}, {flow.upper()})")
    start_time = time.time()

    # === Locate file ===
    year_folder = f"split_by_hs_{year}_numpy"
    input_subdir = os.path.join(config["dirs"]["input"], year_folder)
    matches = glob.glob(os.path.join(input_subdir, f"*_{code}_{year}.csv"))

    if not matches:
        logger.error(f"No matching file found for code {code} in path {input_subdir}")
        raise FileNotFoundError
    logger.info(f"📁 Found file: {matches[0]}")
    df = pd.read_csv(matches[0])
        
    # === Early column trimming ===
    df = df[[col for col in df.columns if col in config["cols_to_keep_early"]]]

    # === Flow filter ===
    flow = flow.lower()
    if flow not in {"m", "x"}:
        logger.error(f"Invalid flow: {flow}")
        raise ValueError("Flow must be 'm' or 'x'")
    df = df[df["flowCategory"].str.lower() == flow]
    p1 = len(df)
    
    # === HS code description mapping ===
    hs_desc_map = config["hs_desc_map"]
    str_hs_desc = hs_desc_map.get(code)


    # === ISO mapping ===
    iso_map = config["iso_map"]
    df = df.rename(columns={"reporterCode": "reporterCodeRaw", "partnerCode": "partnerCodeRaw"})
    df["reporterISO"] = df["reporterCodeRaw"].map(iso_map)
    df["partnerISO"] = df["partnerCodeRaw"].map(iso_map)

    # === Country filter ===
    lst_gp = config["lst_gp"]
    df = df[(~df["partnerISO"].isin(lst_gp)) & (~df["reporterISO"].isin(lst_gp))]
    p2 = len(df)

    # === Trade value filter ===
    df = df[((df["cifValue"].fillna(0) > 0) | (df["fobValue"].fillna(0) > 0))]
    p4 = len(df)
    
    # === Logging progress ===
    logger.info("📈 Cleaning Summary:")
    logger.info(f" - Initial rows: {p1}")
    logger.info(f" - Valid country filter: {p2} ({(p2/p1 if p1 else 0):.2%})")
    logger.info(f" - Valid trade value filter: {p4} ({(p4/p1 if p1 else 0):.2%})")

    # === Subset: kg-based UV ===
    df_uv = df[df["netWgt"].fillna(0) > 0].copy()
    p5 = len(df_uv)
    logger.info(f" - Valid kg-based UV rows: {p5} ({(p5/p1 if p1 else 0):.2%})")
    #if p5 < config.get("min_records_uv", 100):
        #logger.warning(f"⚠️ Only {p5} kg-based UV records (<{config['min_records_uv']})")
    #is_valid_kg = p5 >= config.get("min_records_uv", 100)
    
    if flow == "m":
        df_uv["uv"] = np.where(df_uv["cifValue"].fillna(0) > 0,
                               df_uv["cifValue"] / df_uv["netWgt"],
                               df_uv["fobValue"] / df_uv["netWgt"])
    else:
        df_uv["uv"] = np.where(df_uv["fobValue"].fillna(0) > 0,
                               df_uv["fobValue"] / df_uv["netWgt"],
                               df_uv["cifValue"] / df_uv["netWgt"])

    df_uv["ln_uv"] = np.log(df_uv["uv"])
    df_uv["ln_netWgt"] = np.log(df_uv["netWgt"])
    df_uv.drop(columns=["qty", "ln_qty"], errors="ignore", inplace=True)

    # === Subset: non-kg-based UV ===
    df_q_valid = df[df["qty"].fillna(0) > 0].copy()
    
    df_q = df.iloc[0:0] # Initialize an emplty placeholder
    share_pass = False
    return_unit = "USD/kg"  # Default return unit
    fail_reason_non_kg_uv = "No non-kg alternative units found"
    unit_counts = df_q_valid["qtyUnitCode"].value_counts()
    alt_units = unit_counts[~unit_counts.index.isin([-1, 8])] # Exclude kg and unknown
    
    #is_valid_q = False
    
    if not alt_units.empty: # non-kg unit exits
        top_unit = alt_units.idxmax() # keep only the top non-kg unit 
        top_count = alt_units[top_unit]
        top_share = top_count / len(df_q_valid)
        
        unit_desc = config['unit_map'].get(top_unit, f"Code {top_unit}")
        unit_abbr = config['unit_abbr_map'].get(top_unit, "N/A")
        
        non_kg_top_unit = f"{unit_desc} ({unit_abbr})"
        non_kg_top_unit_share = float(round(top_share, 4))
        # Evaluate thresholds
        share_pass = top_share >= config['q_share_threshold']
        #count_pass = top_count >= config['min_records_q']

        logger.info("🧪 Non-kg UV Subset:")
        logger.info(f" - Top unit: {unit_desc} ({unit_abbr})")
        logger.info(f" - Share of valid top non-kg UV rows: {top_share:.2%} (Required: {config['q_share_threshold']:.0%})")
        logger.info(f" - Valid top non-kg UV count: {top_count}")

        if share_pass : # Non-kg unit meets both share and size requirements
            # build df_q
            df_q = df_q_valid[df_q_valid["qtyUnitCode"] == top_unit].copy()
            p6 = len(df_q)
            #is_valid_q = True
            if flow == 'm':
                df_q['uv_q'] = np.where(df_q['cifValue'].fillna(0) > 0,
                                        df_q['cifValue'] / df_q['qty'],
                                        df_q['fobValue'] / df_q['qty'])
             
            else:
                df_q['uv_q'] = np.where(df_q['fobValue'].fillna(0) > 0,
                                        df_q['fobValue'] / df_q['qty'],
                                        df_q['cifValue'] / df_q['qty'])

            df_q["ln_uv_q"] = np.log(df_q["uv_q"])
            df_q["ln_qty"] = np.log(df_q["qty"])
            df_q.drop(columns=[
                "netWgt", "uv", "ln_uv", "ln_netWgt" ], errors="ignore", inplace=True)
            return_unit = f"USD/{unit_abbr}"  # ✅ Set return unit

            fail_reason_non_kg_uv = None
            logger.info("✅ Non-kg UV subset created.")

        else: # Non-kg unit fails to meet either share or size requirement
            fail_reason_non_kg_uv = f"share {top_share:.2%} < {config['q_share_threshold']:.0%}"
            logger.warning(f"❌ No non-kg UV subset created: {fail_reason_non_kg_uv}")
    else: # No non-kg unit found in the column "qtyUnitCode"
        logger.warning("⚠️ No non-kg units found in qty rows.")
        
    # === Restructure final report ===
    
    report_base = {
        "hs_code": code,
        "hs_code_desc": str_hs_desc,
        "year": year,
        "flow": flow,
        "c_initial_rows": p1,
        "c_valid_country_rows": p2,
        "c_valid_value_rows": p4
    }
    report_clean = {
    **report_base,
    "uv_unit": "USD/kg",
    "c_valid_kg_uv_rows": p5,
    "c_fail_reason_non_kg_uv": fail_reason_non_kg_uv
}
    report_q_clean = None     # default

    if share_pass:
        report_q_clean = {
            **report_base,
            "uv_top_non_kg_unit": f"USD/{unit_abbr}",
            "c_top_non_kg_unit": non_kg_top_unit,
            "c_top_non_kg_unit_share": non_kg_top_unit_share,
            "c_valid_top_non_kg_uv_rows": p6
        }
        
    elapsed = time.time() - start_time    
    logger.info(f"✅ Trade data cleaning completed in {elapsed:.2f} seconds.")
        
    return df_uv, df_q, report_clean, report_q_clean, return_unit

def detect_outliers2(
    df, value_column, code, year, flow, logger, unit_label="USD/kg",
    plot=False, save_path=None, file_format="png"
):
    """
    Detect outliers using modified Z-score and optional spike at log(UV) = 0 (i.e., UV = 1.0).
    """
    logger.info(f"🚀 Outlier detection (HS {code}, {year}, {flow.upper()}, {unit_label})")
    start_time = time.time()
    invalid_fit_reason = None
    # === Early exit if empty DataFrame
    if df.empty:
        logger.warning(f"⚠️ No valid UV records available for {code}-{year}-{flow} — skipping outlier detection.")
        invalid_fit_reason = "No valid records after cleaning"
        
        elapsed = time.time() - start_time    
        logger.info(f"✅ Outlier detection (HS {code}, {year}, {flow.upper()}, "
                    f"{unit_label}) completed in {elapsed:.2f} seconds.")
        
        return df.copy(), df.copy(), {
            "d_initial_rows": 0,
            "d_z_outliers": 0,
            "d_eq1_outliers": 0,
            "d_outliers_removed": 0,
            "d_outlier_rate": np.nan,
            "d_rows_after_outliers": 0,
            "d_valid_for_fit": False,
            "d_invalid_fit_reason": invalid_fit_reason
        }, False
    
    # === Step 1: Histogram-based spike detection at log(UV) = 0
    uv_raw = df[value_column]
    iqr = np.subtract(*np.percentile(uv_raw, [75, 25]))
    bin_width = 2 * iqr / (len(uv_raw) ** (1/3)) if iqr > 0 else 0.1
    bins = np.arange(uv_raw.min(), uv_raw.max() + bin_width, bin_width)
    counts, edges = np.histogram(uv_raw, bins=bins)
    bin_idx = np.digitize([0.0], edges)[0] - 1  # log(UV = 1.0) = 0.0

    if 0 <= bin_idx < len(counts):
        bar = counts[bin_idx]
        neighbors = []
        if bin_idx > 0:
            neighbors.append(counts[bin_idx - 1])
        if bin_idx < len(counts) - 1:
            neighbors.append(counts[bin_idx + 1])
        local_avg = np.mean(neighbors) if neighbors else 0

        mask_eq1 = uv_raw == 0.0  # log(1.0) = 0.0
        share_eq1 = mask_eq1.sum() / len(df)
        apply_eq1 = bar > 5 * local_avg
        n_eq1_outliers = int(mask_eq1.sum()) if apply_eq1 else 0

        if apply_eq1:
            logger.warning(f"📌 log(UV)=0 spike detected at bin {bin_idx}: {bar} vs. neighbors avg {local_avg:.1f} (share = {share_eq1:.2%}) — removed")
        else:
            logger.info(f"log(UV)=0 present but no spike: {bar} vs. neighbors avg {local_avg:.1f} (share = {share_eq1:.2%}) — kept")
    else:
        mask_eq1 = pd.Series(False, index=df.index)
        share_eq1 = 0.0
        apply_eq1 = False
        n_eq1_outliers = 0

    # === Step 2: Optional histogram plotting
    if plot and uv_raw is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(uv_raw, bins=edges, color='lightgray', edgecolor = 'white')
        ax.set_title(f"Histogram of unit values before outlier detection ({unit_label}, HS{code}, {year}, {flow})")
        ax.set_xlabel(f"ln(Unit Price) [{unit_label}]")
        ax.set_ylabel("Count")
        ax.text(0.75, 0.9, f"Sample size: {len(df):,}", transform=ax.transAxes)
        plt.tight_layout()
        if save_path:
            unit_suffix = unit_label.split("/")[-1]
            fig_path = os.path.join(config["dirs"]["figures"], f"hist_od_{code}_{year}_{flow}_{unit_suffix}.{file_format}")
            plt.savefig(fig_path, dpi=300)
            logger.info(f"📁 Saved histogram to {fig_path}")
        else:
            plt.show()

    # === Step 3: Modified Z-score detection
    raw_data = df[value_column].values
    median = np.median(raw_data)
    mad = np.median(np.abs(raw_data - median))

    if mad == 0:
        logger.warning(f"⚠️ Outlier Detection: MAD=0 for {value_column} — skipping detection.")
        invalid_fit_reason = "All values are identical (MAD=0)"
        
        elapsed = time.time() - start_time    
        logger.info(f"✅ Outlier detection (HS {code}, {year}, {flow.upper()}, "
                    f"{unit_label}) completed in {elapsed:.2f} seconds.")
        
        return df.copy().reset_index(drop=True), df.iloc[0:0].copy(), {
            "d_initial_rows": len(df),
            "d_z_outliers": 0,
            "d_eq1_outliers": 0,
            "d_outliers_removed": 0,
            "d_outlier_rate": 0,
            "d_rows_after_outliers": len(df),
            "d_valid_for_fit": False,
            "d_invalid_fit_reason": invalid_fit_reason
        }, False

    modified_z_scores = 0.6745 * (raw_data - median) / mad
    mask_z = np.abs(modified_z_scores) > 3.5

    combined_outlier_mask = mask_z | mask_eq1 if apply_eq1 else mask_z
    df_outliers = df[combined_outlier_mask]
    df_filtered = df[~combined_outlier_mask]

    # === Step 4: Summary
    n_total = len(df)
    n_z_outliers = int(mask_z.sum())
    n_combined_outliers = len(df_outliers)
    n_remaining = len(df_filtered)
    outlier_rate = (n_combined_outliers / n_total) * 100
    
    min_required_fit = config.get("min_records_uv")
    is_valid_for_fit = n_remaining >= min_required_fit

    logger.info(f"📈 Outlier detection summary (HS {code}, {year}, {flow.upper()}, {unit_label}):")
    logger.info(f"- Total rows: {n_total}")
    logger.info(f"- Z-score outliers: {n_z_outliers}")
    logger.info(f"- log(UV) = 0 rows: {mask_eq1.sum()} (share = {share_eq1:.2%}), removed: {n_eq1_outliers}")
    logger.info(f"- Total dropped: {n_combined_outliers} rows ({outlier_rate:.2f}%)")
    logger.info(f"- Remaining rows: {n_remaining}")
    
    if not is_valid_for_fit:
        logger.warning(
            f"⚠️ Insufficient sample size ({n_remaining} rows remain while {min_required_fit} required)")
        invalid_fit_reason = "Insufficient sample size"

    report_outlier = {
        "d_initial_rows": n_total,
        "d_z_outliers": n_z_outliers,
        "d_eq1_outliers": n_eq1_outliers,
        "d_outliers_removed": n_combined_outliers,
        "d_outlier_rate": outlier_rate,
        "d_rows_after_outliers": n_remaining,
        "d_valid_for_fit": is_valid_for_fit,
        "d_invalid_fit_reason": invalid_fit_reason 
    }

    elapsed = time.time() - start_time    
    logger.info(f"✅ Outlier detection (HS {code}, {year}, {flow.upper()}, "
                f"{unit_label}) completed in {elapsed:.2f} seconds.")
    
    return df_filtered, df_outliers, report_outlier, is_valid_for_fit
