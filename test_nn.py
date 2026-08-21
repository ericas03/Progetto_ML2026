import os
import csv
import yaml
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from data_classes.dataset import HAM10000Dataset, HAM10000PyTorchDataset
from model_classes.cnn_models import get_model


# Funzioni di Supporto
def evaluate_test(model, dataloader, criterion, device):
    # Esegue la validazione finale sul test set.
    # Calcola Loss, Accuratezza, F1-Score e accumula tutte le predizioni per il report.
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    # Calcolo delle metriche globali
    loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')

    metrics = {
        "loss": loss,
        "accuracy": acc,
        "f1_macro": f1
    }
    return metrics, all_preds, all_targets


# Esecuzione Principale
if __name__ == "__main__":

    # 1. Configurazione
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo in uso: {device}")

    # 2. Dati
    dataset_module = HAM10000Dataset(config_path="config/config.yaml")
    # Estrazione solo il test loader (train e val vengono ignorati con il simbolo _)
    _, _, test_df, class_names = dataset_module.prepare_dataframes()

    transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_ds = HAM10000PyTorchDataset(test_df, transform=transform)
    test_dl = DataLoader(test_ds, batch_size=config['training']['batch_size'], shuffle=False)

    print("Dataset caricato correttamente. Test set pronto.")

    # 3. Modello
    model_name = config['model']['name']

    # Inizializzo l'architettura della rete vuota (pretrained=False).
    # I pesi calcolati durante l'addestramento verranno caricati
    # subito dopo estraendoli dal file .pt (checkpoint).
    model = get_model(
        model_name=model_name,
        pretrained=False,
        num_classes=config['model']['num_classes']
    ).to(device)

    checkpoint_path = os.path.join("checkpoints", f"best_{model_name}.pt")

    # Controllo di sicurezza
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint non trovato: '{checkpoint_path}'. "
            "Esegui prima train_nn.py per addestrare il modello."
        )

    # Caricamento dei pesi addestrati
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    print(f"Caricato modello: {model_name.upper()} da {checkpoint_path}")

    # 4. Valutazione
    criterion = nn.CrossEntropyLoss()
    test_metrics, all_preds, all_targets = evaluate_test(model, test_dl, criterion, device)

    print(f"\nRisultati Test - {model_name.upper()}")
    print("-" * 35)
    for key, value in test_metrics.items():
        print(f"  {key:<12}: {value:.4f}")

    # Estrazione label dal config per il report
    class_names = config.get('class_names', [str(i) for i in range(config['model']['num_classes'])])

    print("\nREPORT DI CLASSIFICAZIONE")
    print(classification_report(all_targets, all_preds, target_names=class_names))

    print("\nMATRICE DI CONFUSIONE")
    print(confusion_matrix(all_targets, all_preds))

    # 5. Salvataggio su CSV
    results_csv = "results/comparison.csv"
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)

    # Dizionario con le informazioni strutturali e le metriche
    row = {
        "model_name": model_name,
        "pretrained_base": config['model']['pretrained'],
        "epochs_trained": config['training']['epochs'],
        "batch_size": config['training']['batch_size'],
        **{k: round(v, 4) for k, v in test_metrics.items()}
    }

    # Scrittura su file CSV
    file_exists = os.path.isfile(results_csv)
    with open(results_csv, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()  # Genera l'intestazione solo la prima volta
        writer.writerow(row)

    print(f"\nRisultati aggiunti con successo in --> {results_csv}")