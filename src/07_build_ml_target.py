from pathlib import Path
import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")

def main():
    df = pd.read_parquet(PROCESSED / "analytic_dataset.parquet")
    n_start = len(df)
    
    # Teniamo solo le righe dove abbiamo sia la geografia (rural) sia la salute di base
    core_vars = ['rural', 'gen_health', 'phys_health_days', 'ment_health_days']
    df = df.dropna(subset=core_vars).copy()
    
    # Salute Oggettiva (1 = Scadente, 0 = Buona) -> CDC Threshold >= 14 giorni
    df['obj_poor'] = ((df['phys_health_days'] >= 14) | (df['ment_health_days'] >= 14)).astype(int)
    
    # Salute Soggettiva (1 = Scadente, 0 = Buona) -> Fair/Poor (Codici 4, 5)
    df['subj_poor'] = df['gen_health'].isin([4, 5]).astype(int)
    
    # Definizione Target: Discordanza
    conditions = [
        (df['obj_poor'] == 0) & (df['subj_poor'] == 0),
        (df['obj_poor'] == 1) & (df['subj_poor'] == 1),
        (df['obj_poor'] == 1) & (df['subj_poor'] == 0),  # Sovra-ottimisti (Falsi Negativi)
        (df['obj_poor'] == 0) & (df['subj_poor'] == 1)   # Sovra-pessimisti (Falsi Positivi)
    ]
    choices = ['concordant_good', 'concordant_poor', 'over_optimistic', 'over_pessimistic']
    df['discordance_class'] = np.select(conditions, choices, default=np.nan)
    
    df.to_parquet(PROCESSED / "sample_ml_discordance.parquet", index=False)
    
    print(f"Righe post-pulizia target: {len(df):,}")
    print("\nDistribuzione Target Class (%):")
    print(df['discordance_class'].value_counts(normalize=True).mul(100).round(2))

if __name__ == "__main__": main()
