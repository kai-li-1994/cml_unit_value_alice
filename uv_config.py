# -*- coding: utf-8 -*-
import pandas as pd
import pickle
import os
import platform
import shutil

def _detect_environment():
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
    
def _get_base_dir():
    env = _detect_environment()

    if env == "alice":
        return "/zfsstore/user/lik6/cml_unit_value"
    elif env == "claix":
        return "/home/lik6/projects/claix_data/cml_unit_value"
    elif env == "windows":
        # Try multiple options and pick the one that exists
        candidate_dirs = [
            "C:/Users/laptop-kl/OneDrive - Universiteit Leiden/PlasticTradeFlow/tradeflow/cml_trade/cml_unit_value_alice",
            "C:/Users/lik6/OneDrive - Universiteit Leiden/PlasticTradeFlow/tradeflow/cml_trade/cml_unit_value"
        ]
        for d in candidate_dirs:
            if os.path.exists(d):
                return d
        raise RuntimeError("❌ Windows environment detected, but none of the known base directories exist.")
    elif env == "linux":
        return "/home/lik6/projects/cml_unit_value"
    else:
        raise RuntimeError("❌ Unknown environment. Please define MY_ENV or extend config.py")
        
def load_config(
    country_file="./pkl/uv_mapping_country.pkl",
    unit_file="./pkl/uv_mapping_unit.pkl",
    unit_abbr_file="./pkl/uv_mapping_unitAbbr.pkl",
    hs_desc_file="./pkl/uv_mapping_hsdesc.pkl",
    ):
    """Load ISO mappings, group list, quantity unit mappings, and thresholds from pickle files."""
    
    base_dir = _get_base_dir()
    
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
        
    with open(hs_desc_file, "rb") as f:
        hs_desc_map = pickle.load(f)
    

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
        "C:/Program Files/R/R-4.5.1/bin/Rscript.exe"                           # rwth-aachen laptop
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

    # Optional override for missing local folder
    input_dir = "C:/Users/laptop-kl/OneDrive - Universiteit Leiden/PlasticTradeFlow/tradeflow/cml_trade/cml_unit_value/data_uncomtrade"

    # === Define and create directories ===
    dirs = {
        "base": base_dir,
        "input": input_dir,
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
        "hs_desc_map": hs_desc_map,
        "cols_to_keep_early": cols_to_keep_early,
        "q_share_threshold": 0.10,
        "min_records_q": 200,
        "min_records_uv": 200,
        "env": _detect_environment(),
        "dirs": dirs,
        "rscript_exec": rscript_exec,
    }