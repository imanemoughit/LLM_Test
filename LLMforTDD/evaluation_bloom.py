import os
import glob
import torch
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configurations
model_name = "bigscience/bloom-1b7"  # Example of using Bloom model
token = "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"  # Token Hugging Face
test_dataset_folder = "/content/LLM_Test/LLMforTDD/Evaluation_Dataset/Evaluation_NoFineTuned_Prompts/Evaluation_NoFineTuned_Prompts/Csv"  # Path to CSV files
output_dir = "/content/LLM_Test/LLMforTDD/output_bloom_test_cases"  # Output directory for the generated test cases
batch_size = 1  # Reduced batch size
beam_size = 4
max_new_tokens = 300
max_length = 1024  # Can be reduced further if needed

def load_model_and_tokenizer():
    """Load the model and tokenizer."""
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=token)

    # Add a pad token if missing
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
    """Process CSV files, generate test cases, and save them."""
    os.makedirs(output_dir, exist_ok=True)
    for csv_file in glob.glob(os.path.join(test_dataset_folder, "*.csv")):
        print(f"Processing file: {csv_file}")
        
        # Load the raw content of the CSV file
        test_df = pd.read_csv(csv_file, delimiter=";")
        
        # Get descriptions and code columns
        descriptions = test_df['Description'].tolist()
        codes = test_df['Code'].tolist()
        
        # Concatenate description and code for test case generation
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
        dataloader = DataLoader(dataset, batch_size=batch_size)  # Set batch size to 1

        # Generate and save results
        all_predictions = []
        for batch in tqdm(dataloader, desc="Generating test cases"):
            # Move tensors and model to the same device (GPU or CPU)
            inputs = torch.stack([torch.tensor(item) for item in batch["input_ids"]]).to(device).long()
            attention_mask = torch.stack([torch.tensor(item) for item in batch["attention_mask"]]).to(device).long()

            # Reduce precision (optional, FP16 precision) during generation
            inputs = inputs.half()  # Use FP16 precision only when needed
            attention_mask = attention_mask.half()  # Use FP16 precision only when needed

            # Generate predictions
            predictions = generate_test_cases(model, tokenizer, inputs, attention_mask)
            all_predictions.extend(predictions)

            # Clear cache to free up memory after each batch
            torch.cuda.empty_cache()

        # Save the results to the output directory
        output_file = os.path.join(output_dir, os.path.basename(csv_file).replace(".csv", "_test_cases.txt"))
        with open(output_file, "w") as f:
            for prediction in all_predictions:
                f.write(prediction + "\n")

        print(f"Test cases saved to {output_file}")

def main():
    """Main entry point of the program."""
    # Choose device (GPU or CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Move model to device (GPU or CPU)
    model.to(device)
    
    # Process the CSV files and generate test cases
    process_csv_files(model, tokenizer, device)

if __name__ == "__main__":
    main()
