from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
YEARS = range(2011, 2020)

# Aggiunte le variabili ML comportamentali e cliniche
STABLE_VARS = {
    "_STATE": "state", "MSCODE": "geo_msa", "GENHLTH": "gen_health", "MARITAL": "marital",
    "MENTHLTH": "ment_health_days", "PHYSHLTH": "phys_health_days",
    "_EDUCAG": "education", "_LLCPWT": "weight", "_AGEG5YR": "age_group",
    "IMONTH": "interview_month", "INCOME2": "income",
    "_BMI5": "bmi", "SMOKE100": "smoke100", "EXERANY2": "exercise",
    "CVDINFR4": "heart_attack", "CVDCRHD4": "coronary_hd", 
    "CVDSTRK3": "stroke", "ASTHMA3": "asthma", "DIABETE3": "diabetes"
}
SEX_VAR = {**{y: "SEX" for y in range(2011, 2018)}, 2018: "SEX1", 2019: "SEXVAR"}
EMP_VAR = {**{y: "EMPLOY" for y in (2011, 2012)}, **{y: "EMPLOY1" for y in range(2013, 2020)}}
IMPRACE_MAP = {1: "white", 2: "black", 3: "asian", 4: "native", 5: "hispanic", 6: "other"}
RACEGR3_MAP = {1: "white", 2: "black", 3: "other", 4: "other", 5: "hispanic", 9: np.nan}

def load_year(year):
    sex_var, emp_var = SEX_VAR[year], EMP_VAR[year]
    xpt_file = [p for p in RAW_DIR.iterdir() if str(year) in p.name and p.name.upper().endswith(".XPT")][0]
    cols0 = set(next(pd.read_sas(xpt_file, format="xport", chunksize=1)).columns)
    race_var, race_map = ("_IMPRACE", IMPRACE_MAP) if "_IMPRACE" in cols0 else ("_RACEGR3", RACEGR3_MAP)
    
    # Teniamo solo le variabili che esistono effettivamente in quell'anno
    keep = [v for v in STABLE_VARS if v in cols0] + [sex_var, emp_var, race_var]
    df = pd.concat([c[keep] for c in pd.read_sas(xpt_file, format="xport", chunksize=50_000)], ignore_index=True)
    
    rename = {**STABLE_VARS, sex_var: "sex", emp_var: "employment"}
    df = df.rename(columns=rename)
    df["race"] = df[race_var].map(race_map)
    df = df.drop(columns=[race_var])
    df["year"] = year
    return df

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frames = [load_year(y) for y in YEARS]
    panel = pd.concat(frames, ignore_index=True)
    panel.to_parquet(PROCESSED_DIR / "brfss_panel.parquet", index=False)
    print(f"Panel aggiornato: {len(panel):,} rows x {panel.shape[1]} cols")

if __name__ == "__main__": main()
