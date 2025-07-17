# -*- coding: utf-8 -*-
import pandas as pd
import pickle
from pathlib import Path
import os
import platform
import shutil

def detect_environment():
    """
    Detect the environment where the code is running.
    Supports override via environment variable 'MY_ENV'.
    """
    #First check user-defined override
    env_override = os.environ.get("MY_ENV", "").lower()
    if env_override in {"alice", "claix", "windows"}:
       return env_override
    
    # Otherwise detect from hostname or OS
    node = platform.node().lower()
    if "alice" in node:
        return "alice"
    elif "claix" in node or "rwth" in node:
        return "claix"
    elif os.name == "nt":
        return "windows"
    else:
        return "unknown"
    
def get_base_dir():
    
    env = detect_environment()

    if env == "alice":
        return "/zfsstore/user/lik6/cml_unit_value"
    elif env == "claix":
        return "/home/lik6/projects/claix_data/cml_unit_value"
    elif env == "windows":
        return "C:/Users/lik6/OneDrive - Universiteit Leiden/PlasticTradeFlow/tradeflow/cml_trade/cml_unit_value"
    elif env == "linux":
        return "/home/lik6/projects/cml_unit_value"
    else:
        raise RuntimeError("❌ Unknown environment. Please define MY_ENV or extend config.py")
        
def load_config(
    country_file="./pkl/uv_mapping_country.pkl",
    unit_file="./pkl/uv_mapping_unit.pkl",
    unit_abbr_file="./pkl/uv_mapping_unitAbbr.pkl",
    ):
    """Load ISO mappings, group list, quantity unit mappings, and thresholds from pickle files."""
    
    base_dir = get_base_dir()
    
    # === ISO country mapping ===
    df_cmap = pd.read_pickle(country_file)
    iso_map = df_cmap.set_index("Code")["IsoAlpha3"].str.strip().to_dict()

    # === Group codes to filter out ===
    lst_gp = [
        "_AC", "ATA", "_X", "X1", "R91", "A49", "E29", "R20", "X2", "A79",
        "NTZ", "A59", "F49", "O19", "F19", "E19", "ZA1", "XX", "F97", "W00",
        "R4", "EUR"
    ]

    # === Quantity unit mappings ===
    with open(unit_file, "rb") as f:
        unit_map = pickle.load(f)

    with open(unit_abbr_file, "rb") as f:
        unit_abbr_map = pickle.load(f)
    

    # === Define essential columns for early-stage processing ===
    cols_to_keep_early = [
        "period", "reporterCode", "flowCategory", "partnerCode",
        "cmdCode", "qtyUnitCode", "qty", "netWgt", "cifValue", "fobValue"
    ]
    
    # === Rscript ===
    rscript_exec = None   
    possible_paths = [
        "C:/Users/lik6/AppData/Local/Programs/R/R-4.5.1/bin/x64/Rscript.exe",  # laptop
        "C:/Program Files/R/R-4.4.1/bin/x64/Rscript.exe",                      # desktop
    ]

    # First, check hardcoded Windows locations
    for path in possible_paths:
        if os.path.exists(path):
            rscript_exec = path
            break

    # Then, try from PATH (works after `module load R/...` on HPC)
    if not rscript_exec:
        rscript_exec = shutil.which("Rscript")

    # Raise a helpful error if still not found
    if not rscript_exec:
        raise RuntimeError(
            "❌ Rscript not found. On HPC, please run `module load R/...` before executing. "
            "On Windows, please check your R installation."
        )
        
    # === Define and create directories ===
    dirs = {
        "base": base_dir,
        "input": os.path.join(base_dir, "data_uncomtrade"),
        "figures": os.path.join(base_dir, "figures"),
        "logs": os.path.join(base_dir, "logs"),
        "reports": os.path.join(base_dir, "reports"),
        }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    return {
        "iso_map": iso_map,
        "lst_gp": lst_gp,
        "unit_map": unit_map,
        "unit_abbr_map": unit_abbr_map,
        "cols_to_keep_early": cols_to_keep_early,
        "q_share_threshold": 0.10,
        "min_records_q": 200,
        "min_records_uv": 200,
        "env": detect_environment(),
        "dirs": dirs,
        "rscript_exec": rscript_exec,
    }

def prefix_dict_keys(d, prefix, skip_keys=None):
    """
    Add a prefix to all keys in a dictionary, except for any in skip_keys.
    """
    if skip_keys is None:
        skip_keys = {"hs_code", "year", "flow"}
    return { (f"{prefix}{k}" if k not in skip_keys else k): v for k, v in d.items() }

def save_report_dict(report_dict, code, year, flow, config, logger=None):
    """
    Save a flat dictionary as a .parquet file in the reports folder.
    All values are converted to string to avoid ArrowTypeError.
    """
    report_path = os.path.join(
        config["dirs"]["reports"],
        f"report_{code}_{year}_{flow}.parquet"
    )
    
    # Convert all values to strings for safe saving
    df = pd.Series({k: str(v) for k, v in report_dict.items()}).to_frame("value")
    df.to_parquet(report_path)

    if logger:
        logger.info(f"✅ Saved final report to {report_path}")