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

    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold, GridSearchCV

    # 7. SVM con PCA
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
    pca_90.fit(X_train_scaled)
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
    plt.savefig("data/pca_scree_plot.ax.set_xticks(np.arange)

    # 8. Addestramento PCA + SVM
    print("ADDESTRAMENTO PCA + SVM")

    X_train_pca = pca_90.transform(X_train_scaled)
    X_test_pca = pca_90.transform(X_test_scaled)

    print(f"Dimensioni feature originali: {X_train_scaled.shape[1]}")
    print(f"Dimensioni feature ridotte (PCA 90%): {X_train_pca.shape[1]}")
    print(f"Varianza totale spiegata: {np.sum(pca_90.explained_variance_ratio_) * 100:.2f}%")

    print("\nConfigurazione della Grid Search ottimizzata per l'SVM")
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.01, 0.1],
        'kernel': ['rbf'],
        'class_weight': ['balanced']
    }

    cv_stratified = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid_search_pca = GridSearchCV(
        estimator=SVC(random_state=42),
        param_grid=param_grid,
        scoring='f1_macro',
        cv=cv_stratified,
        n_jobs=-1,  # Sfrutta tutti i core per fare prima
        verbose=1
    )

    print("Avvio della Grid Search")
    grid_search_pca.fit(X_train_pca, y_train_full)

    print("\nRISULTATI OTTENUTI DALL'OTTIMIZZAZIONE")
    print(f"Miglior combinazione iperparametri: {grid_search_pca.best_params_}")
    print(f"Miglior F1-Score ottenuto in CV:    {grid_search_pca.best_score_:.4f}")

    best_svm_pca_model = grid_search_pca.best_estimator_

    print("\nValutazione del modello finale")
    preds_pca = best_svm_pca_model.predict(X_test_pca)

    print("\nREPORT DI CLASSIFICAZIONE PCA + SVM")
    print(classification_report(y_test, preds_pca, target_names=target_names))

    cm_pca = confusion_matrix(y_test, preds_pca)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_pca, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.xlabel('Predetto dal Modello')
    plt.ylabel('Malattia Reale')
    plt.title('Matrice di Confusione SVM PCA')
    plt.tight_layout()
    plt.savefig("data/confusion_matrix_svm_pca.png")
    plt.show()

    # 9. SVM con LDA
    print("\n==========================================")
    print(" 3. PIPELINE LDA + SVM")
    print("==========================================")

    lda_2d = LinearDiscriminantAnalysis(n_components=2)
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

    lda_full = LinearDiscriminantAnalysis()
    X_train_lda = lda_full.fit_transform(X_train_scaled, y_train_full)
    X_test_lda = lda_full.transform(X_test_scaled)

    print(f"\nRIDUZIONE DELLA DIMENSIONALITÀ (LDA)")
    print(f"Dimensioni feature originali: {X_train_scaled.shape[1]}")
    print(f"Dimensioni feature ridotte (LDA): {X_train_lda.shape[1]}")

    print("\nConfigurazione della Grid Search per SVM (dati LDA)")
    grid_search_lda = GridSearchCV(
        estimator=SVC(random_state=42),
        param_grid=param_grid,
        scoring='f1_macro',
        cv=cv_stratified,
        n_jobs=-1,
        verbose=1
    )

    print("Avvio della Grid Search sui dati ridotti con LDA")
    grid_search_lda.fit(X_train_lda, y_train_full)

    print("\nRISULTATI OTTIMIZZAZIONE LDA + SVM")
    print(f"Miglior combinazione iperparametri: {grid_search_lda.best_params_}")
    print(f"Miglior F1-Score ottenuto in CV:    {grid_search_lda.best_score_:.4f}")

    best_svm_lda_model = grid_search_lda.best_estimator_

    print("\nValutazione del modello finale (LDA)")
    preds_lda = best_svm_lda_model.predict(X_test_lda)

    print("\nREPORT DI CLASSIFICAZIONE FINALE (LDA)")
    print(classification_report(y_test, preds_lda, target_names=target_names))

    cm_lda = confusion_matrix(y_test, preds_lda)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_lda, annot=True, fmt='d', cmap='Oranges',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.xlabel('Predetto dal Modello')
    plt.ylabel('Malattia Reale')
    plt.title('Matrice di Confusione SVM (LDA)')
    plt.tight_layout()
    plt.savefig("data/confusion_matrix_svm_lda.png")
    plt.show()

    # 10. CONFRONTO FINALE METRICHE SVM
    print("CONFRONTO FINALE METRICHE SVM")

    # 1. Baseline (Tutte le 768 feature)
    preds_base = svm_full.predict(X_test_scaled)
    f1_baseline = f1_score(y_test, preds_base, average='macro')

    # 2. PCA + SVM (197 feature circa)
    f1_pca_val = f1_score(y_test, preds_pca, average='macro')

    # 3. LDA + SVM (6 feature)
    f1_lda_val = f1_score(y_test, preds_lda, average='macro')

    # CREAZIONE DEL GRAFICO
    modelli = ['SVM Base\n(768 feature)', 'SVM + PCA\n(PCA var. 90%)', 'SVM + LDA\n(6 feature)']
    punteggi = [f1_baseline, f1_pca_val, f1_lda_val]
    colori = ['#2c3e50', '#2980b9', '#e74c3c']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(modelli, punteggi, color=colori, width=0.5, edgecolor='black', alpha=0.85)

    # Aggiungiamo i valori numerici sopra ogni barra
    for bar in bars:
        yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.01,
            f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')

    # Formattazione estetica
    ax.set_ylabel('F1-Score (Macro)', fontweight='bold')
    ax.set_title('Confronto Prestazioni SVM (Test Set)', fontweight='bold', pad=15)
    ax.set_ylim(0, 1.05)  # Mantiene l'asse Y da 0 a 1
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("data/svm_comparison_bar_chart.png")
    plt.show()