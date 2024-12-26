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
test_dataset_folder = "/content/LLM_Test/LLMforTDD/Evaluation_Dataset/Evaluation_NoFineTuned_Prompts/Evaluation_NoFineTuned_Prompts/Csv"  # Chemin des fichiers CSV
output_dir = "/content/LLM_Test/LLMforTDD/output_llama_test_cases"  # Répertoire pour sauvegarder les résultats
batch_size = 4
beam_size = 4
max_new_tokens = 300
max_length = 1024

def load_model_and_tokenizer():
    """Charge le modèle et le tokenizer."""
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=token)

    # Ajouter un token de pad si manquant
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def generate_test_cases(model, tokenizer, inputs, attention_mask):
    """Génère les cas de test en utilisant le modèle."""
    outputs = model.generate(
        input_ids=inputs,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_beams=beam_size,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def process_csv_files(model, tokenizer, device):
    """Parcourt les fichiers CSV, génère les cas de test, et les sauvegarde."""
    os.makedirs(output_dir, exist_ok=True)
    for csv_file in glob.glob(os.path.join(test_dataset_folder, "*.csv")):
        print(f"Processing file: {csv_file}")
        
        # Charger le contenu brut du fichier CSV
        test_df = pd.read_csv(csv_file, delimiter=";")
        
        # Récupérer les descriptions et codes de méthode
        descriptions = test_df['Description'].tolist()
        codes = test_df['Code'].tolist()
        
        # Concaténer description et code pour la génération des cas de test
        descriptions_codes = [f"Description: {desc} \nCode: {code}" for desc, code in zip(descriptions, codes)]
        
        # Tokenisation
        tokens = tokenizer(
            descriptions_codes,
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
            # Déplacer les tensors et le modèle sur le même périphérique
            inputs = torch.stack([torch.tensor(item) for item in batch["input_ids"]]).to(device)
            attention_mask = torch.stack([torch.tensor(item) for item in batch["attention_mask"]]).to(device)
            
            predictions = generate_test_cases(model, tokenizer, inputs, attention_mask)
            all_predictions.extend(predictions)

        # Sauvegarder les résultats dans le répertoire de sortie
        output_file = os.path.join(output_dir, os.path.basename(csv_file).replace(".csv", "_test_cases.txt"))
        with open(output_file, "w") as f:
            for prediction in all_predictions:
                f.write(prediction + "\n")

        print(f"Test cases saved to {output_file}")

def main():
    """Point d'entrée principal du programme."""
    # Choisir le périphérique (GPU ou CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Charger le modèle et le tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Déplacer le modèle vers le périphérique
    model.to(device)
    
    # Traiter les fichiers CSV et générer les cas de test
    process_csv_files(model, tokenizer, device)

if __name__ == "__main__":
    main()
