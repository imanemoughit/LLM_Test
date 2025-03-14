import os
import glob
import pandas as pd
from tqdm import tqdm
import requests

# Configurations
endpoint_url = "https://d9442qsr897czmk7.us-east-1.aws.endpoints.huggingface.cloud"  # URL de votre endpoint
api_token = "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"  # Token Hugging Face (si nécessaire)

test_dataset_folder = "/content/LLM_Test/LLMforTDD/Evaluation_Dataset/Evaluation_NoFineTuned_Prompts/Evaluation_NoFineTuned_Prompts/Csv"  # Chemin des fichiers CSV
output_dir = "/content/LLM_Test/LLMforTDD/output_llama_test_cases"  # Répertoire pour sauvegarder les résultats

batch_size = 2
max_new_tokens = 300

def query_huggingface_endpoint(input_text):
    """Envoie une requête à l'endpoint Hugging Face."""
    payload = {
        "inputs": input_text,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
    headers = {"Authorization": f"Bearer {api_token}"}
    response = requests.post(endpoint_url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"API request failed: {response.status_code}, {response.text}")
    return response.json()

def generate_test_cases_with_endpoint(inputs):
    """Utilise l'endpoint Hugging Face pour générer des cas de test."""
    predictions = []
    for input_text in inputs:
        try:
            result = query_huggingface_endpoint(input_text)
            if isinstance(result, dict) and "error" in result:
                print(f"Error from API: {result['error']}")
                predictions.append("Error: Unable to generate test case.")
            else:
                predictions.append(result[0]["generated_text"])  # Accéder au texte généré
        except Exception as e:
            print(f"Error querying endpoint: {e}")
            predictions.append("Error: Unable to generate test case.")
    return predictions

def process_csv_files():
    """Parcourt les fichiers CSV, génère les cas de test via l'endpoint, et les sauvegarde."""
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
                predictions = generate_test_cases_with_endpoint(batch_inputs)
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
