import os
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from data_classes.dataset import HAM10000Dataset, HAM10000PyTorchDataset
from model_classes.cnn_models import get_model

# Training e Validation Loop
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    # Esegue una singola epoca di addestramento.
    # Calcola la loss e l'accuratezza media sull'intero training set.
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    # tqdm genera una barra di caricamento nel terminale
    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    metrics = {
        "loss": running_loss / total,
        "accuracy": correct / total
    }
    return metrics


def evaluate(model, dataloader, criterion, device):
    # Esegue la validazione del modello disabilitando il calcolo dei gradienti.
    # Restituisce loss e accuratezza sul validation set.
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    metrics = {
        "loss": running_loss / total,
        "accuracy": correct / total
    }
    return metrics


def _update_best(model, metric_name, val_metrics, best_metric, best_state, lower_is_better):
    # Verifica se la metrica attuale migliora il record storico.
    # In caso affermativo, clona i pesi del modello in RAM per ottimizzare
    # le operazioni di input/output su disco.
    current = val_metrics[metric_name]
    improved = current < best_metric if lower_is_better else current > best_metric

    if improved:
        print(f"  Trovato nuovo miglior risultato - val {metric_name}: {current:.4f} - salvataggio in RAM")
        best_metric = current
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_metric, best_state, improved


# Esecuzione Principale
if __name__ == "__main__":

    # 1. Configurazione
    # Caricamento del file YAML personalizzato
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo in uso: {device}")

    # 2. Dati
    dataset_module = HAM10000Dataset(config_path="config/config.yaml")
    train_df, val_df, test_df, class_names = dataset_module.prepare_dataframes()

    # Trasformazioni base per CNN (Resize e Normalizzazione standard ImageNet)
    transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = HAM10000PyTorchDataset(train_df, transform=transform)
    val_ds = HAM10000PyTorchDataset(val_df, transform=transform)

    batch_size = config['training']['batch_size']
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Dataset caricato correttamente. Batch size: {batch_size}")

    # 3. Modello
    # Legge direttamente il modello configurato nel file YAML ("vgg16", "resnet18", ecc.)
    model_name = config['model']['name']
    pretrained = config['model']['pretrained']

    model = get_model(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=config['model']['num_classes']
    ).to(device)

    print(f"Modello selezionato: {model_name.upper()} (Preaddestrato: {pretrained})")

    # 4. Ottimizzatore e Funzione di Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])

    # 5. Ciclo di Addestramento e Interruzione Anticipata
    lower_is_better = False
    best_val_metric = float("-inf")
    best_state = None

    patience = 5
    epochs_no_improve = 0
    epochs = config['training']['epochs']

    for epoch in range(epochs):
        print(f"\nEpoca {epoch + 1}/{epochs}")

        train_metrics = train_one_epoch(model, train_dl, criterion, optimizer, device)
        val_metrics = evaluate(model, val_dl, criterion, device)

        print(f"  train  loss={train_metrics['loss']:.4f}  acc={train_metrics['accuracy']:.4f}")
        print(f"  val    loss={val_metrics['loss']:.4f}  acc={val_metrics['accuracy']:.4f}")

        best_val_metric, best_state, improved = _update_best(
            model=model,
            metric_name="accuracy",
            val_metrics=val_metrics,
            best_metric=best_val_metric,
            best_state=best_state,
            lower_is_better=lower_is_better,
        )

        if improved:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if patience > 0:
                print(f"  Nessun miglioramento per {epochs_no_improve}/{patience} epoche.")
                if epochs_no_improve >= patience:
                    print(f"  Interruzione anticipata attivata dopo {epoch + 1} epoche.")
                    break

    # 6. Salvataggio del Modello Migliore
    os.makedirs("checkpoints", exist_ok=True)
    save_path = os.path.join("checkpoints", f"best_{model_name}.pt")

    if best_state is not None:
        torch.save(best_state, save_path)
        print(f"\nMiglior modello salvato --> {save_path}")
    else:
        print("\nNessun modello salvato (nessun miglioramento registrato).")