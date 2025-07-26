# -*- coding: utf-8 -*-
"""
Generate Comtrade reference mappings (countries, units, etc.)
Created on Wed Apr 23 2025

@author: lik6
"""
import pandas as pd
import comtradeapicall 
import pickle

# === Country mapping ===
df_p = comtradeapicall.getReference('partner').rename(columns={
    'PartnerCode': 'Code',
    'PartnerDesc': 'Desc',
    'partnerNote': 'Note',
    'PartnerCodeIsoAlpha2': 'IsoAlpha2',
    'PartnerCodeIsoAlpha3': 'IsoAlpha3'
})

df_r = comtradeapicall.getReference('reporter').rename(columns={
    'reporterCode': 'Code',
    'reporterDesc': 'Desc',
    'reporterNote': 'Note',
    'reporterCodeIsoAlpha2': 'IsoAlpha2',
    'reporterCodeIsoAlpha3': 'IsoAlpha3'
})

merge_keys = ['Code', 'Desc', 'Note', 'IsoAlpha2', 'IsoAlpha3',
              'entryEffectiveDate', 'entryExpiredDate']
df_country_mapping= pd.merge(df_p, df_r, how='outer', on=merge_keys)
df_country_mapping = df_country_mapping[merge_keys].drop_duplicates(
                     ).sort_values("Code").reset_index(drop=True)
df_country_mapping.to_pickle("uv_mapping_country.pkl")
print("✅ Saved: uv_mapping_country.pkl")

# === Unit mapping ===
units = comtradeapicall.getReference('qtyunit')
unit_map = units.set_index("qtyCode")["qtyDescription"].str.strip().to_dict()
unit_abbr_map = units.set_index("qtyCode")["qtyAbbr"].str.strip().to_dict()

with open("uv_mapping_unit.pkl", "wb") as f:
    pickle.dump(unit_map, f)

with open("uv_mapping_unitAbbr.pkl", "wb") as f:
    pickle.dump(unit_abbr_map, f)

print("✅ Saved: uv_mapping_unit.pkl and uv_mapping_unitAbbr.pkl")

# === HS code mapping ===
df_hs = comtradeapicall.getReference('cmd:HS')

# Filter to 6-digit codes (aggrLevel == 6)
df_hs["id"] = df_hs["id"].astype(str)
df_hs_leaf = df_hs[df_hs["aggrLevel"] == 6].copy()

# Extract the actual description from the 'text' field
hs_desc_map = df_hs_leaf.set_index("id")["text"].str.extract(r"-\s*(.+)$")[0].str.strip().to_dict()

# Save as pickle
with open("./pkl/uv_mapping_hsdesc.pkl", "wb") as f:
    pickle.dump(hs_desc_map, f)

print("✅ Saved: uv_mapping_hsdesc.pkl")
