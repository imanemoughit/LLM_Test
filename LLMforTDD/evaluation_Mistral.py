from huggingface_hub import snapshot_download
from pathlib import Path
import os
import glob
import torch
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Set paths and model
mistral_models_path = Path.home().joinpath('mistral_models', '7B-v0.3')
mistral_models_path.mkdir(parents=True, exist_ok=True)

snapshot_download(repo_id="mistralai/Mistral-7B-v0.3", allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"], local_dir=mistral_models_path)

# Model and tokenizer loading
model_name = "mistralai/Mistral-7B-v0.3"
token = "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"  # Hugging Face Token

tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=token)

# Pipeline for text generation
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Direct paths
test_dataset_folder = "/content/LLM_Test/LLMforTDD/Evaluation_Dataset/Evaluation_NoFineTuned_Prompts/Evaluation_NoFineTuned_Prompts/Csv"
output_dir = "/content/LLM_Test/LLMforTDD/output_llama_test_cases"
batch_size = 2
beam_size = 4
max_new_tokens = 300
max_length = 1024

def load_model_and_tokenizer():
    """Loads model and tokenizer."""
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=token)
    
    # Set pad token if none
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_test_cases(model, tokenizer, inputs, attention_mask):
    """Generate test cases using the model."""
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
    """Process CSV files, generate test cases, and save."""
    os.makedirs(output_dir, exist_ok=True)
    
    for csv_file in glob.glob(os.path.join(test_dataset_folder, "*.csv")):
        print(f"Processing file: {csv_file}")
        
        # Load CSV content
        test_df = pd.read_csv(csv_file, delimiter=";")
        descriptions = test_df['Description'].tolist()
        codes = test_df['Code'].tolist()
        
        descriptions_codes = [f"Description: {desc} \nCode: {code}" for desc, code in zip(descriptions, codes)]
        
        # Tokenization
        tokens = tokenizer(
            descriptions_codes,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # Create DataLoader
        dataset = Dataset.from_dict({"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]})
        dataloader = DataLoader(dataset, batch_size=batch_size)

        # Generate and save results
        all_predictions = []
        for batch in tqdm(dataloader, desc="Generating test cases"):
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            predictions = generate_test_cases(model, tokenizer, inputs, attention_mask)
            all_predictions.extend(predictions)

        # Save predictions to file
        output_file = os.path.join(output_dir, os.path.basename(csv_file).replace(".csv", "_test_cases.txt"))
        with open(output_file, "w") as f:
            for prediction in all_predictions:
                f.write(prediction + "\n")

        print(f"Test cases saved to {output_file}")

def main():
    """Main entry point."""
    # Set device (GPU or CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Move model to the selected device
    model.to(device)
    
    # Process CSV files and generate test cases
    process_csv_files(model, tokenizer, device)

if __name__ == "__main__":
    main()
