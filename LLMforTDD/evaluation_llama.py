import os
import glob
import torch
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configurations
model_name = "meta-llama/Meta-Llama-3.1-8B"  # Nom du modèle
token = "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"  # Token Hugging Face
test_dataset_folder = "/content/LLM_Test/LLMforTDD/Evaluation_Dataset/Evaluation_NoFineTuned_Prompts/Evaluation_NoFineTuned_Prompts/Csv"  # Chemin des fichiers CSV contenant les descriptions
input_column = "description"  # Colonne dans le CSV contenant les descriptions
output_dir = "/content/LLM_Test/LLMforTDD/output_llama_test_cases"  # Répertoire pour sauvegarder les résultats
batch_size = 4
beam_size = 4
max_new_tokens = 300
max_length = 1024

# Chargement du modèle et du tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)

# Ajouter un token de pad si manquant
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Fonction de génération
def generate_test_cases(model, tokenizer, inputs, attention_mask):
    outputs = model.generate(
        input_ids=inputs,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_beams=beam_size,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Création du répertoire de sortie si il n'existe pas déjà
os.makedirs(output_dir, exist_ok=True)

# Parcourir les fichiers CSV
for csv_file in glob.glob(os.path.join(test_dataset_folder, "*.csv")):
    print(f"Processing file: {csv_file}")
    
    # Charger le CSV
    test_df = pd.read_csv(csv_file)
    descriptions = list(test_df[input_column].dropna())  # Assurer qu'il n'y a pas de valeurs nulles

    # Tokenisation
    tokens = tokenizer(
        descriptions,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # Création du DataLoader
    dataset = Dataset.from_dict({"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]})
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Génération et sauvegarde
    all_predictions = []
    for batch in tqdm(dataloader, desc="Generating test cases"):
        inputs = batch["input_ids"].to("cuda" if torch.cuda.is_available() else "cpu")
        attention_mask = batch["attention_mask"].to("cuda" if torch.cuda.is_available() else "cpu")
        predictions = generate_test_cases(model, tokenizer, inputs, attention_mask)
        all_predictions.extend(predictions)

    # Sauvegarder les résultats dans le répertoire de sortie
    output_file = os.path.join(output_dir, os.path.basename(csv_file).replace(".csv", "_test_cases.txt"))
    with open(output_file, "w") as f:
        for prediction in all_predictions:
            f.write(prediction + "\n")

    print(f"Test cases saved to {output_file}")
