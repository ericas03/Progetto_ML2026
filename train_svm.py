import os
import yaml
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, GridSearchCV

if __name__ == "__main__":
    # 1. Caricamento configurazioni dal file YAML
    config_path = "config/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Lettura del percorso dei file dal config
    features_dir = config['data']['features_dir']
    metadata_path = config['data']['metadata_path']

    print("Caricamento dei file di Addestramento e Valutazione in corso...")
    X_train = np.load(os.path.join(features_dir, "X_train.npy"))
    y_train = np.load(os.path.join(features_dir, "y_train.npy"))
    X_val = np.load(os.path.join(features_dir, "X_val.npy"))
    y_val = np.load(os.path.join(features_dir, "y_val.npy"))
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

    # Salvataggio Scaler
    joblib.dump(scaler, 'scaler.pkl')

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

    joblib.dump(svm_full, 'svm_base.pkl')
    print("Modello SVM Base salvato")

    # 6. SVM con PCA
    print("SVM CON PCA")

    # PCA a 2 dimensioni
    pca_2d = PCA(n_components=2, random_state=42)
    X_train_pca_2d = pca_2d.fit_transform(X_train_scaled)

    # Dizionario colori per le 7 classi
    colours = ['#e74c3c', '#2980b9', '#27ae60', '#f39c12', '#8e44ad', '#2c3e50', '#d35400']

    # Plot Scatter 2D PCA
    fig, ax = plt.subplots(figsize=(8, 6))
    for label_idx, name in enumerate(target_names):
        mask = y_train_full == label_idx
        ax.scatter(X_train_pca_2d[mask, 0], X_train_pca_2d[mask, 1],
                   s=35, alpha=0.7, color=colours[label_idx], label=name, edgecolors='white', linewidth=0.4)

    ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} varianza)')
    ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} varianza)')
    ax.set_title('PCA 2D - Feature ViT')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("data/pca_2d_scatter.png")
    plt.show()

    # Plot (Calcolato sulle componenti che spiegano il 90% della varianza)
    pca_90 = PCA(n_components=0.90, random_state=42)
    X_train_pca = pca_90.fit_transform(X_train_scaled)
    joblib.dump(pca_90, 'pca_90.pkl') # Salvataggio trasformatore PCA
    print(f"Dimensioni feature ridotte (PCA 90%): {X_train_pca.shape[1]}")

    # Scree Plot
    cumulative = np.cumsum(pca_90.explained_variance_ratio_)
    n_comp = np.arange(1, len(pca_90.explained_variance_ratio_) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(n_comp, pca_90.explained_variance_ratio_,
           color='#5dade2', edgecolor='#2e86c1', label='Individuale', alpha=0.7)
    ax.step(n_comp, cumulative, where='mid', color='#c0392b', linewidth=2, label='Cumulativa')
    ax.axhline(y=0.90, color='gray', linestyle='--', linewidth=1.5, label='Soglia 90%')

    ax.set_xlabel('Componenti Principali')
    ax.set_ylabel('Rapporto Varianza Spiegata')
    ax.set_title('Scree Plot (PCA 90% - ViT Features)')
    ax.set_xticks(np.arange(0, len(n_comp) + 1, step=10))
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("data/pca_scree_plot.png")
    plt.show()

    # 7. Ottimizzazione SVM su PCA
    param_grid = {'C': [0.1, 1, 10],
                  'gamma': ['scale', 0.01, 0.1],
                  'kernel': ['rbf'],
                  'class_weight': ['balanced']
                  }
    cv_stratified = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search_pca = GridSearchCV(SVC(
                                random_state=42),
                                param_grid=param_grid,
                                scoring='f1_macro',
                                cv=cv_stratified,
                                n_jobs=-1,
                                verbose=1)
    grid_search_pca.fit(X_train_pca, y_train_full)

    best_svm_pca_model = grid_search_pca.best_estimator_
    joblib.dump(best_svm_pca_model, 'svm_pca.pkl')
    print("Modello SVM+PCA salvato (svm_pca.pkl)")

    # 8. SVM con LDA
    print("LDA con SVM")

    lda_2d = LinearDiscriminantAnalysis(n_components=2, solver='eigen')
    X_train_lda_2d = lda_2d.fit_transform(X_train_scaled, y_train_full)

    # Plot  2D LDA
    fig, ax = plt.subplots(figsize=(8, 6))
    for label_idx, name in enumerate(target_names):
        mask = y_train_full == label_idx
        ax.scatter(X_train_lda_2d[mask, 0], X_train_lda_2d[mask, 1],
               s=35, alpha=0.7, color=colours[label_idx], label=name, edgecolors='white', linewidth=0.4)

    ax.set_xlabel('LD1')
    ax.set_ylabel('LD2')
    ax.set_title('LDA - Feature ViT')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("data/lda_2d_scatter.png")
    plt.show()

    lda_full = LinearDiscriminantAnalysis(solver='eigen')
    X_train_lda = lda_full.fit_transform(X_train_scaled, y_train_full)
    joblib.dump(lda_full, 'lda_full.pkl')  # Salvataggio trasformatore LDA
    print(f"Dimensioni feature ridotte (LDA): {X_train_lda.shape[1]}")

    # 9. Ottimizzazione SVM su LDA
    grid_search_lda = GridSearchCV(SVC(
        random_state=42),
        param_grid=param_grid,
        scoring='f1_macro',
        cv=cv_stratified,
        n_jobs=-1,
        verbose=1)
    grid_search_lda.fit(X_train_lda, y_train_full)

    best_svm_lda_model = grid_search_lda.best_estimator_
    joblib.dump(best_svm_lda_model, 'svm_lda.pkl')
    print("Modello SVM+LDA salvato (svm_lda.pkl)")

    print("\nAddestramento completato! Eseguire il test_svm.py per la valutazione.")