import os
import yaml
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class HAM10000Dataset:
    # Classe per la gestione e la preparazione del dataset HAM10000.

    def __init__(self, config_path="config/config.yaml"):
        # 1. Lettura dei parametri dal file di configurazione
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.image_dirs = self.config['data']['image_dirs']
        self.tab_path = self.config['data']['metadata_path']

        # 2. Controlli sull'esistenza dei file e delle directory
        if not os.path.exists(self.tab_path):
            raise FileNotFoundError(f"Il file non è stato trovato {self.tab_path}.")

        for img_dir in self.image_dirs:
            if not os.path.exists(img_dir) or len(os.listdir(img_dir)) == 0:
                raise FileNotFoundError(f" La cartella {img_dir} è vuota o inesistente.")

    def prepare_dataframes(self):
        print("Avvio preparazione dataset e mappatura immagini...")

        # 3. Mappatura immagini iterando su tutte le cartelle
        image_cache = {}
        for img_dir in self.image_dirs:
            for root, dirs, files in os.walk(img_dir):
                if "__MACOSX" in root:
                    continue
                for file in files:
                    if file.lower().endswith('.jpg') and not file.startswith('._'):
                        img_id = os.path.splitext(file)[0].strip()
                        image_cache[img_id] = os.path.join(root, file)

        # 4. Caricamento e normalizzazione metadati
        df = pd.read_csv(self.tab_path, sep=None, engine='python')
        df.columns = df.columns.astype(str).str.replace('"', '').str.strip()

        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace('"', '').str.strip()

        # 5. Label encoding
        label_encoder = LabelEncoder()
        df['label'] = label_encoder.fit_transform(df['dx'])

        # 6. Associazione tra metadati e percorso del file
        df['image_path'] = df['image_id'].map(image_cache)
        df = df.dropna(subset=['image_path']).reset_index(drop=True)

        if len(df) == 0:
            raise ValueError("ERRORE CRITICO: Nessun match tra le immagini su disco e i metadati!")

        # 7. Splitting dei dati (Data Leakage)
        test_size = self.config['data']['test_size']
        val_test_split = self.config['data']['val_test_split']
        random_seed = self.config['data']['seed']

        # Isola i lesion_id univoci per evitare che foto della stessa lesione finiscano in set diversi
        unique_lesions = df.drop_duplicates(subset='lesion_id').copy()

        # Primo Split: Isola il Training set dal resto
        train_lesions, test_val_lesions = train_test_split(
            unique_lesions['lesion_id'],
            test_size=test_size,
            random_state=random_seed,
            stratify=unique_lesions['label']
        )

        # Secondo Split: Divide la restante parte a metà tra Validation e Test
        val_lesions, test_lesions = train_test_split(
            test_val_lesions,
            test_size=val_test_split,
            random_state=random_seed,
            stratify=unique_lesions[unique_lesions['lesion_id'].isin(test_val_lesions)]['label']
        )

        # Ricostruzione dei dataframe completi assegnando tutte le foto di un lesion_id al set corretto
        train_df = df[df['lesion_id'].isin(train_lesions)].copy()
        val_df = df[df['lesion_id'].isin(val_lesions)].copy()
        test_df = df[df['lesion_id'].isin(test_lesions)].copy()

        print(f"Split completato:")
        print(f" - Train set: {len(train_df)} immagini")
        print(f" - Validation set: {len(val_df)} immagini")
        print(f" - Test set: {len(test_df)} immagini")

        return train_df, val_df, test_df