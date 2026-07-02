"""Train the XGBoost classifier on the Discordance Target.

This script isolates the features, prevents data leakage, handles class 
imbalance via sample weighting, and trains a multi-class XGBoost model.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import pickle

PROCESSED = Path("data/processed")
MODELS = Path("results/models")

def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    
    print("1. Caricamento del dataset...")
    df = pd.read_parquet(PROCESSED / "sample_ml_discordance.parquet")
    
    # 2. Prevenzione Data Leakage
    # Rimuoviamo le variabili usate per calcolare il target e il target stesso
    leakage_vars = ['gen_health', 'phys_health_days', 'ment_health_days', 'obj_poor', 'subj_poor']
    
    # Selezioniamo le features (X)
    X = df.drop(columns=leakage_vars + ['discordance_class'])
    
    # 3. Mappatura del Target (XGBoost richiede classi numeriche da 0 a N)
    target_mapping = {
        'concordant_good': 0,
        'concordant_poor': 1,
        'over_optimistic': 2,
        'over_pessimistic': 3
    }
    y = df['discordance_class'].map(target_mapping)
    
    print(f"Features utilizzate ({X.shape[1]}): {list(X.columns)}")
    
    # 4. Train-Test Split (80% training, 20% test, stratificato)
    print("\n2. Split dei dati (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 5. Bilanciamento delle Classi (Sample Weights)
    # Diamo più peso alle classi minoritarie (over_optimistic e over_pessimistic)
    print("3. Calcolo dei pesi per bilanciare le classi minoritarie...")
    weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    # 6. Addestramento del modello XGBoost
    print("\n4. Addestramento XGBoost (potrebbe richiedere alcuni minuti)...")
    # Usiamo tree_method='hist' che è velocissimo su milioni di righe
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=4,
        tree_method='hist',
        learning_rate=0.1,
        max_depth=6,
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train, sample_weight=weights)
    
    # 7. Valutazione sul Test Set (dati mai visti)
    print("\n5. Valutazione del Modello sul Test Set:")
    y_pred = model.predict(X_test)
    
    # Report stampato con i nomi originali delle classi
    target_names = {v: k for k, v in target_mapping.items()}
    labels = [target_names[i] for i in range(4)]
    
    report = classification_report(y_test, y_pred, target_names=labels)
    print(report)
    
    # Salvataggio del modello per la fase di Explainable AI (SHAP)
    with open(MODELS / "xgboost_discordance.pkl", "wb") as f:
        pickle.dump(model, f)
    
    # Salviamo anche i nomi delle features per dopo
    with open(MODELS / "feature_names.pkl", "wb") as f:
        pickle.dump(list(X.columns), f)
        
    print(f"\nModello salvato in {MODELS}/xgboost_discordance.pkl")

if __name__ == "__main__": main()
