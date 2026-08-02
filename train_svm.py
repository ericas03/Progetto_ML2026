import os
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

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

    # 5. Addestramento SVM
    print("\n--- ADDESTRAMENTO SVM ---")
    print(f"Dimensioni feature utilizzate: {X_train_scaled.shape[1]}")

    # Utilizzo dei migliori iperparametri
    svm_full = SVC(
        C=10,
        kernel='rbf',
        gamma='scale',
        class_weight='balanced',
        random_state=42
    )

    print("Addestramento della SVM in corso sulle 768 feature originali...")
    svm_full.fit(X_train_scaled, y_train_full)
    print("Addestramento completato!")

    # 6. Valutazione sul Test Set
    print("\nValutazione del modello sul Test Set...")
    preds_full = svm_full.predict(X_test_scaled)

    print("\n--- REPORT DI CLASSIFICAZIONE ---")
    print(classification_report(y_test, preds_full, target_names=target_names))

    # Visualizzazione grafica della Matrice di Confusione
    cm_full = confusion_matrix(y_test, preds_full)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_full, annot=True, fmt='d', cmap='Greens',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.xlabel('Predetto dal Modello')
    plt.ylabel('Malattia Reale')
    plt.title('Matrice di Confusione SVM')
    plt.tight_layout()

    # Salvataggio automatico dell'immagine nella cartella data
    plt.savefig("data/confusion_matrix_svm_full.png")
    plt.show()