"""Explain the XGBoost model using SHAP to interpret Health Discordance."""
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

MODELS = Path("results/models")
PROCESSED = Path("data/processed")
RESULTS = Path("results/figures")

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    
    print("1. Caricamento dati e modello...")
    df = pd.read_parquet(PROCESSED / "sample_ml_discordance.parquet")
    with open(MODELS / "xgboost_discordance.pkl", "rb") as f:
        model = pickle.load(f)

    # Stessa preparazione usata in addestramento
    leakage_vars = ['gen_health', 'phys_health_days', 'ment_health_days', 'obj_poor', 'subj_poor']
    X = df.drop(columns=leakage_vars + ['discordance_class'])
    X = pd.get_dummies(X, drop_first=True)
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)

    target_mapping = {'concordant_good': 0, 'concordant_poor': 1, 'over_optimistic': 2, 'over_pessimistic': 3}
    y = df['discordance_class'].map(target_mapping)

    # Campionamento stratificato per SHAP (10.000 righe per computazione rapida)
    print("2. Estrazione campione stratificato per SHAP...")
    _, X_shap, _, y_shap = train_test_split(X, y, test_size=10000, random_state=42, stratify=y)

    print("3. Calcolo dei valori SHAP (potrebbe richiedere un paio di minuti)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    # In XGBoost multiclasse, shap_values è una lista di array. 
    # Indice 2 = over_optimistic, Indice 3 = over_pessimistic

    print("4. Generazione Summary Plot per i Sovra-Pessimisti...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[3], X_shap, show=False)
    plt.title("SHAP Summary: Drivers del Sovra-Pessimismo")
    plt.tight_layout()
    plt.savefig(RESULTS / "shap_over_pessimistic.png", dpi=300)
    plt.close()

    print("5. Generazione Summary Plot per i Sovra-Ottimisti...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[2], X_shap, show=False)
    plt.title("SHAP Summary: Drivers del Sovra-Ottimismo")
    plt.tight_layout()
    plt.savefig(RESULTS / "shap_over_optimistic.png", dpi=300)
    plt.close()

    print(f"Fatto! I grafici sono stati salvati nella cartella {RESULTS}/")

if __name__ == "__main__": main()
