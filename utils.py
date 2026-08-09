import os
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Riproducibilità e ambiente
def set_seed(seed=42):
    """
    Imposta il seed per tutte le librerie per garantire
    la totale riproducibilità dei risultati in PyTorch.

    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Costringe la GPU a usare algoritmi deterministici
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Seed impostato a {seed} per la massima riproducibilità.")

# 2. Funzioni di valutazione
def compute_metrics(predictions, references):
    """
    Calcola le metriche di classificazione (Accuracy, Precision, Recall, F1-Score).
    La metrica principale per le prestazioni è il Macro F1-Score,
    scelto per gestire correttamente lo sbilanciamento delle classi nel dataset.

    """
    acc = accuracy_score(references, predictions)
    precision = precision_score(references, predictions, average='macro', zero_division=0)
    recall = recall_score(references, predictions, average='macro', zero_division=0)
    f1 = f1_score(references, predictions, average='macro', zero_division=0)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate(model, dataloader, criterion, device):
    """
    Esegue la valutazione del modello sul validation o test set.
    La funzione sospende il calcolo dei gradienti e disabilita comportamenti
    legati all'addestramento (come il Dropout) per ottenere una stima
    efficiente delle performance reali della rete.
    """
    model.eval()
    running_loss = 0.0
    predictions = []
    references = []

    with torch.no_grad():
        for batch in dataloader:
            # Trasferimento dei tensori sul device target richiesto
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # Forward pass e calcolo dell'errore (Loss)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            # Estrazione della classe predetta
            pred = torch.argmax(outputs, dim=1)
            # Spostamento sulla CPU e conversione in NumPy per compatibilità con scikit-learn
            predictions.extend(pred.cpu().numpy())
            references.extend(labels.cpu().numpy())

    # Aggregazione delle metriche
    val_metrics = compute_metrics(predictions, references)
    val_metrics['loss'] = running_loss / len(dataloader)

    return val_metrics