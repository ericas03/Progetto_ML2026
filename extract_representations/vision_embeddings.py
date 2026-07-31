import os
import yaml
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel

from data_classes.dataset import HAM10000Dataset

class VisionEmbeddings:
    '''
    This class is intended to extract embeddings from vision models.
    It uses ViT (Vision Transformer) as a default model.
    '''

    def __init__(self, model_name='google/vit-base-patch16-224', device='cuda'):
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)

        self.device = device
        self.model.to(self.device)

        self.model_name = model_name

        # eval mode
        self.model.eval()

    def extract(self, image):
        '''
        Extract embeddings from an image.

        Args:
            image (PIL.Image): Image to extract embeddings from.

        Returns:
            torch.Tensor: Embeddings of the image.
        '''
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()


# ==========================================

def extract_features_from_dataframe(dataframe, extractor, desc_name):

    # Scorre il dataframe, apre le immagini e utilizza l'estrattore per generare le feature.

    features, labels = [], []
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=desc_name):
        try:
            # Apre l'immagine dal percorso mappato
            img = Image.open(row['image_path']).convert('RGB')
            emb = extractor.extract(img)
            features.append(emb.squeeze())
            labels.append(row['label'])
        except Exception as e:
            print(f"Errore caricamento immagine {row['image_path']}: {e}")

    return np.array(features), np.array(labels)


if __name__ == "__main__":
    # 1. Caricamento configurazioni dal file YAML
    config_path = "config/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    features_dir = config['data']['features_dir']
    model_name = config['model']['name']

    # Creazione della cartella di output se non esiste
    os.makedirs(features_dir, exist_ok=True)

    # 2. Configurazione Hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inizializzazione sul device: {device.upper()}")

    # 3. Ottenimento dei dataframe splittati senza Data Leakage
    data_module = HAM10000Dataset(config_path=config_path)
    train_df, val_df, test_df = data_module.prepare_dataframes()

    # 4. Inizializzazione modello
    print(f"\nCaricamento modello ViT: {model_name}")
    extractor = VisionEmbeddings(model_name=model_name, device=device)

    # 5. Estrazione vettori
    print("\nAvvio estrazione feature...")
    X_train, y_train = extract_features_from_dataframe(train_df, extractor, "Estrazione Train")
    X_val, y_val = extract_features_from_dataframe(val_df, extractor, "Estrazione Val")
    X_test, y_test = extract_features_from_dataframe(test_df, extractor, "Estrazione Test")

    # 6. Salvataggio in formato compresso binario (.npy)
    print(f"\nSalvataggio dei file .npy in: {features_dir}")
    np.save(os.path.join(features_dir, "X_train.npy"), X_train)
    np.save(os.path.join(features_dir, "y_train.npy"), y_train)
    np.save(os.path.join(features_dir, "X_val.npy"), X_val)
    np.save(os.path.join(features_dir, "y_val.npy"), y_val)
    np.save(os.path.join(features_dir, "X_test.npy"), X_test)
    np.save(os.path.join(features_dir, "y_test.npy"), y_test)

    print("\nProcesso completato con successo!")