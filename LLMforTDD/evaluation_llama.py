import os
import glob
import torch
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configurations
model_name = "meta-llama/Meta-Llama-3.1-8B" 
#model_name = "meta-llama/Llama-2-13b-hf"  # Nom du modèle
#model_name ="codellama/CodeLlama-small"
#model_name=  "mistralai/mistral-7b"


test_dataset_folder = "/content/LLM_Test/LLMforTDD/Evaluation_Dataset/Evaluation_NoFineTuned_Prompts/Evaluation_NoFineTuned_Prompts/Csv"  # Chemin des fichiers CSV
output_dir = "/content/LLM_Test/LLMforTDD/output_llama_test_cases"  # Répertoire pour sauvegarder les résultats
api_token = "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"  # Token Hugging Face
api_url = f"https://api-inference.huggingface.co/models/{model_name}"
headers = {"Authorization": f"Bearer {api_token}"}

batch_size = 2
beam_size = 4
max_new_tokens = 300
max_length = 1024

def query_huggingface_api(payload):
    """Envoie une requête à l'API Inference de Hugging Face."""
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"API request failed: {response.status_code}, {response.text}")
    return response.json()

def generate_test_cases_with_api(inputs):
    """Utilise l'API Hugging Face pour générer des cas de test."""
    predictions = []
    for input_text in inputs:
        payload = {
            "inputs": input_text,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "num_beams": 4,
                "early_stopping": True,
            }
        }
        result = query_huggingface_api(payload)
        if "error" in result:
            print(f"Error from API: {result['error']}")
        else:
            predictions.append(result[0]["generated_text"])
    return predictions


def process_csv_files():
    """Parcourt les fichiers CSV, génère les cas de test via l'API, et les sauvegarde."""
    os.makedirs(output_dir, exist_ok=True)
    for csv_file in glob.glob(os.path.join(test_dataset_folder, "*.csv")):
        try:
            print(f"Processing file: {csv_file}")
            test_df = pd.read_csv(csv_file, delimiter=";")

            # Vérification des colonnes nécessaires
            if "Description" not in test_df.columns or "Code" not in test_df.columns:
                print(f"Skipping file {csv_file}: Missing 'Description' or 'Code' columns.")
                continue

            descriptions = test_df['Description'].tolist()
            codes = test_df['Code'].tolist()
            inputs = [f"Description: {desc} \nCode: {code}" for desc, code in zip(descriptions, codes)]

            # Diviser les inputs en batchs
            all_predictions = []
            for i in tqdm(range(0, len(inputs), batch_size), desc="Generating test cases"):
                batch_inputs = inputs[i:i + batch_size]
                predictions = generate_test_cases_with_api(batch_inputs)
                all_predictions.extend(predictions)

            # Sauvegarder les résultats
            output_file = os.path.join(output_dir, os.path.basename(csv_file).replace(".csv", "_test_cases.txt"))
            with open(output_file, "w") as f:
                for prediction in all_predictions:
                    f.write(prediction + "\n")

            print(f"Test cases saved to {output_file}")

        except Exception as e:
            print(f"Error processing file {csv_file}: {e}")

def main():
    """Point d'entrée principal du programme."""
    process_csv_files()


if __name__ == "__main__":
    main()
