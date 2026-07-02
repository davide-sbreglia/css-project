from pathlib import Path
import numpy as np
import pandas as pd

IN_PATH = Path("data/processed/brfss_panel.parquet")
OUT_PATH = Path("data/processed/brfss_panel_clean.parquet")

def recode_health_days(series): return series.replace({88: 0, 77: np.nan, 99: np.nan})
def recode_dk_refused(series): return series.replace({7: np.nan, 9: np.nan, 77: np.nan, 99: np.nan})
def recode_binary(series): return series.replace({2: 0, 7: np.nan, 9: np.nan})

def main():
    df = pd.read_parquet(IN_PATH)
    
    df["ment_health_days"] = recode_health_days(df["ment_health_days"])
    df["phys_health_days"] = recode_health_days(df["phys_health_days"])
    df["gen_health"] = recode_dk_refused(df["gen_health"])
    df["sex"] = recode_dk_refused(df["sex"])
    df["age_group"] = df["age_group"].replace({14: np.nan})
    df["marital"] = df["marital"].replace({1: "married", 6: "married", 5: "never", 3: "widowed", 2: "divsep", 4: "divsep", 9: np.nan})
    
    if "bmi" in df.columns:
        df["bmi"] = df["bmi"].replace({9999: np.nan}) / 100 
    
    binary_cols = ["smoke100", "exercise", "heart_attack", "coronary_hd", "stroke", "asthma", "diabetes"]
    for col in binary_cols:
        if col in df.columns:
            if col == "diabetes":
                df[col] = df[col].apply(lambda x: 1 if x == 1 else (0 if x in [3,4] else np.nan))
            else:
                df[col] = recode_binary(df[col])
                
    df.to_parquet(OUT_PATH, index=False)
    print("Recode completato (incluse le variabili cliniche).")

if __name__ == "__main__": main()
