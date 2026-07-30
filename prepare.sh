#!/bin/bash

#  Setup per il progetto (HAM10000)

echo "Creazione ambiente per il progetto..."

echo "Installazione dei pacchetti da requirements.txt..."
pip install -r requirements.txt

echo "Creazione directory..."
mkdir -p data/
mkdir -p data/HAM10000_images_part_1/
mkdir -p data/HAM10000_images_part_2/
mkdir -p data/extracted_features/

echo "Setup completato!"

echo "Prima di eseguire l'estrazione delle feature, assicurati di aver posizionato:"
echo "1.Le immagini della parte 1 in: data/HAM10000_images_part_1/"
echo "2.Le immagini della parte 2 in: data/HAM10000_images_part_2/"
echo "3. Il file metadati (HAM10000_metadata.tab) dentro la cartella: data/"