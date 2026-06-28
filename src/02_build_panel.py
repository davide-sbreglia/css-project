"""Build a harmonized BRFSS panel (2011-2019) from the raw .XPT files.

For each survey year this script keeps only the variables the project needs,
renames them to stable, lower-case names, and stacks all years into a single
long-format panel with an added `year` column.

Memory note: BRFSS files have 342 columns and ~400-500k rows each. To stay
within Colab's RAM, we read each file in chunks, keep only the six needed
columns from each chunk, and process one year at a time.

Cross-year harmonization: all key variables keep the same BRFSS name across
2011-2019 EXCEPT respondent sex (SEX in 2011-2017, SEX1 in 2018, SEXVAR in 2019),
which we map to a single `sex` column.

Input:  data/raw/LLCP<year>.XPT
Output: data/processed/brfss_panel.parquet
"""

from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
YEARS = range(2011, 2020)

STABLE_VARS = {
    "_STATE": "state",
    "MSCODE": "geo_msa",
    "GENHLTH": "gen_health",
    "MENTHLTH": "ment_health_days",
    "PHYSHLTH": "phys_health_days",
}

SEX_VAR_BY_YEAR = {y: "SEX" for y in range(2011, 2018)}
SEX_VAR_BY_YEAR[2018] = "SEX1"
SEX_VAR_BY_YEAR[2019] = "SEXVAR"


def xpt_path_for(year: int) -> Path:
    matches = [
        p for p in RAW_DIR.iterdir()
        if str(year) in p.name and p.name.strip().upper().endswith(".XPT")
    ]
    if not matches:
        raise FileNotFoundError(f"No .XPT found for {year} in {RAW_DIR}")
    return matches[0]


def load_year(year: int) -> pd.DataFrame:
    """Read one BRFSS year in chunks, keeping only the needed columns.

    Reading in chunks avoids holding the full 342-column file in memory; we
    subset to the six needed columns from each chunk and discard the rest.
    """
    sex_var = SEX_VAR_BY_YEAR[year]
    keep = list(STABLE_VARS.keys()) + [sex_var]

    pieces = []
    reader = pd.read_sas(xpt_path_for(year), format="xport", chunksize=50_000)
    for chunk in reader:
        pieces.append(chunk[keep])
    df = pd.concat(pieces, ignore_index=True)

    rename_map = dict(STABLE_VARS)
    rename_map[sex_var] = "sex"
    df = df.rename(columns=rename_map)
    df["year"] = year
    return df


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for year in YEARS:
        df_year = load_year(year)
        print(f"  {year}: {len(df_year):>7,} rows", flush=True)
        frames.append(df_year)

    panel = pd.concat(frames, ignore_index=True)
    out_path = PROCESSED_DIR / "brfss_panel.parquet"
    panel.to_parquet(out_path, index=False)

    print(f"\nPanel built: {len(panel):,} rows x {panel.shape[1]} columns")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
