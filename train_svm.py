import os
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

if __name__ == "__main__":
    # 1. Caricamento configurazioni dal file YAML
    config_path = "config/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Lettura del percorso dei file dal config
    features_dir = config['data']['features_dir']
    metadata_path = config['data']['metadata_path']

    print("Caricamento dei file .npy in corso...")
    X_train = np.load(os.path.join(features_dir, "X_train.npy"))
    y_train = np.load(os.path.join(features_dir, "y_train.npy"))
    X_val = np.load(os.path.join(features_dir, "X_val.npy"))
    y_val = np.load(os.path.join(features_dir, "y_val.npy"))
    X_test = np.load(os.path.join(features_dir, "X_test.npy"))
    y_test = np.load(os.path.join(features_dir, "y_test.npy"))
    print("Dati caricati correttamente!")

    # 2. Ricostruzione del LabelEncoder
    df_meta = pd.read_csv(metadata_path, sep=None, engine='python')
    label_encoder = LabelEncoder()
    label_encoder.fit(df_meta['dx'])
    target_names = label_encoder.classes_

    # 3. Unione Train e Validation per la Cross-Validation
    X_train_full = np.vstack((X_train, X_val))
    y_train_full = np.hstack((y_train, y_val))

    # 4. Feature Scaling (Standardizzazione)
    print("\nAvvio standardizzazione delle feature...")
    scaler = StandardScaler()

    # Calcolo media/varianza sul training set per evitare Data Leakage
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)

    print(f"Standardizzazione completata!")
    print(f"Dimensioni set di addestramento: {X_train_scaled.shape}")
    print("Dati pronti per la modellazione.")