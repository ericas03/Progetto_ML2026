import os
import yaml
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

if __name__ == "__main__":

    # 1. Setup e caricamento test set
    config_path = "config/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    features_dir = config['data']['features_dir']
    metadata_path = config['data']['metadata_path']

    print("Caricamento Test Set...")
    X_test = np.load(os.path.join(features_dir, "X_test.npy"))
    y_test = np.load(os.path.join(features_dir, "y_test.npy"))

    # Ricostruzione target names
    df_meta = pd.read_csv(metadata_path, sep=None, engine='python')
    label_encoder = LabelEncoder()
    label_encoder.fit(df_meta['dx'])
    target_names = label_encoder.classes_

    # 2. Caricamento modelli
    print("Caricamento trasformatori e modelli addestrati...")
    scaler = joblib.load('scaler.pkl')
    pca_90 = joblib.load('pca_90.pkl')
    lda_full = joblib.load('lda_full.pkl')

    svm_base = joblib.load('svm_base.pkl')
    svm_pca = joblib.load('svm_pca.pkl')
    svm_lda = joblib.load('svm_lda.pkl')

    # Trasformazione dei dati di test
    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca_90.transform(X_test_scaled)
    X_test_lda = lda_full.transform(X_test_scaled)

    # 3. Valutazione SVM base
    print("\nVALUTAZIONE SVM BASE")
    preds_base = svm_base.predict(X_test_scaled)
    print(classification_report(y_test, preds_base, target_names=target_names))

    cm_base = confusion_matrix(y_test, preds_base)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Greens', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predetto')
    plt.ylabel('Reale')
    plt.title('Matrice di Confusione SVM Base (768 Feature)')
    plt.tight_layout()
    plt.savefig("data/confusion_matrix_svm_full.png")

    # 4. Valutazione SVM + PCA
    print("\nVALUTAZIONE SVM + PCA")
    preds_pca = svm_pca.predict(X_test_pca)
    print(classification_report(y_test, preds_pca, target_names=target_names))

    cm_pca = confusion_matrix(y_test, preds_pca)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_pca, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predetto')
    plt.ylabel('Reale')
    plt.title('Matrice di Confusione SVM + PCA')
    plt.tight_layout()
    plt.savefig("data/confusion_matrix_svm_pca.png")

    # 5. Valutazione SVM + LDA
    print("\nVALUTAZIONE SVM + LDA")
    preds_lda = svm_lda.predict(X_test_lda)
    print(classification_report(y_test, preds_lda, target_names=target_names))

    cm_lda = confusion_matrix(y_test, preds_lda)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_lda, annot=True, fmt='d', cmap='Oranges', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predetto')
    plt.ylabel('Reale')
    plt.title('Matrice di Confusione SVM + LDA')
    plt.tight_layout()
    plt.savefig("data/confusion_matrix_svm_lda.png")

    # 6. Confronto finale metriche
    print("\n[INFO] Generazione grafico di confronto finale...")
    f1_baseline = f1_score(y_test, preds_base, average='macro')
    f1_pca_val = f1_score(y_test, preds_pca, average='macro')
    f1_lda_val = f1_score(y_test, preds_lda, average='macro')

    modelli = ['SVM Base\n(768 feature)', 'SVM + PCA\n(PCA var. 90%)', 'SVM + LDA\n(6 feature)']
    punteggi = [f1_baseline, f1_pca_val, f1_lda_val]
    colori = ['#2c3e50', '#2980b9', '#e74c3c']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(modelli, punteggi, color=colori, width=0.5, edgecolor='black', alpha=0.85)

    # Aggiungo i valori numerici sopra ogni barra
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom',
                fontweight='bold')

    ax.set_ylabel('F1-Score (Macro)', fontweight='bold')
    ax.set_title('Confronto Prestazioni SVM (Test Set)', fontweight='bold', pad=15)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("data/svm_comparison_bar_chart.png")
    plt.show()