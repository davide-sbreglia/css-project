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
    
    # Prevenzione Data Leakage
    leakage_vars = ['gen_health', 'phys_health_days', 'ment_health_days', 'obj_poor', 'subj_poor']
    X = df.drop(columns=leakage_vars + ['discordance_class'])
    
    print("2. Trasformazione delle variabili testuali in numeriche (One-Hot Encoding)...")
    # Questa riga risolve l'errore convertendo stringhe in array binari compatibili con XGBoost
    X = pd.get_dummies(X, drop_first=True)
    
    # XGBoost richiede esplicitamente int o float, quindi convertiamo i True/False in 1/0
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
            
    # Mappatura del Target
    target_mapping = {
        'concordant_good': 0, 'concordant_poor': 1,
        'over_optimistic': 2, 'over_pessimistic': 3
    }
    y = df['discordance_class'].map(target_mapping)
    
    print(f"Features utilizzate ({X.shape[1]}): {list(X.columns)}")
    
    print("\n3. Split dei dati (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print("4. Calcolo dei pesi per bilanciare le classi minoritarie...")
    weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    print("\n5. Addestramento XGBoost (potrebbe richiedere alcuni minuti)...")
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
    
    print("\n6. Valutazione sul Test Set:")
    y_pred = model.predict(X_test)
    
    target_names = {v: k for k, v in target_mapping.items()}
    labels = [target_names[i] for i in range(4)]
    
    report = classification_report(y_test, y_pred, target_names=labels)
    print(report)
    
    with open(MODELS / "xgboost_discordance.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODELS / "feature_names.pkl", "wb") as f:
        pickle.dump(list(X.columns), f)
        
    print(f"\nModello salvato in {MODELS}/xgboost_discordance.pkl")

if __name__ == "__main__": main()
