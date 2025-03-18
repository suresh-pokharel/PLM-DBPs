"""
      Author  : Suresh Pokharel
      Email   : sp2530@rit.edu
      Affiliation: Rochester Institute of Technology
"""

import argparse
import os
import pandas as pd
import tensorflow as tf
from Bio import SeqIO
import torch
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel  # Use T5EncoderModel instead of T5Model

# Function to load ProtT5 tokenizer and encoder model
def load_prott5_model():
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50')
    model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_uniref50')  # Change here
    model = model.eval()  # Set model to evaluation mode
    return tokenizer, model

# Function to extract ProtT5 features for a protein sequence
def extract_prott5_features(sequence, tokenizer, model, device='cpu'):
    # Tokenize sequence
    input_ids = tokenizer(sequence, return_tensors='pt', padding=True).input_ids.to(device)

    # Extract embeddings using encoder model
    with torch.no_grad():
        embedding = model(input_ids=input_ids).last_hidden_state.squeeze(0)

    # Take the mean across the sequence length (axis=0) to get fixed-length representation
    embedding = torch.mean(embedding, dim=0).cpu().numpy()
    
    return embedding


# Function to load and preprocess FASTA sequences
def read_fasta(file_path):
    sequences = []
    descriptions = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
        descriptions.append(record.description)
    return descriptions, sequences

# Function to predict DNA-binding probability
def predict_dna_binding(model, features):
    # Make sure input is in the correct shape for prediction
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features)
    return prediction[0][0]  # Assuming model outputs a single probability

def main(input_file, output_file, model_path):
    # Load ProtT5 tokenizer and model
    tokenizer, prott5_model = load_prott5_model()

    # Load the DNA-binding prediction model
    dbp_model = tf.keras.models.load_model(model_path)

    # Read sequences from FASTA file
    descriptions, sequences = read_fasta(input_file)

    results = []

    # Process each sequence
    for desc, seq in zip(descriptions, sequences):
        # Extract ProtT5 features
        # features = extract_prott5_features(seq, tokenizer, prott5_model)

         # Get dummy features for now #########################################################
        features = np.random.rand(1024)

        # Predict DNA-binding probability
        probability = predict_dna_binding(dbp_model, features)

        # Assign a binary prediction based on threshold (0.5 used here)
        prediction = "DNA Binding" if probability >= 0.5 else "Non-DNA Binding"

        # Store the result
        results.append({
            "protein_description": desc,
            "probability": probability,
            "prediction": prediction
        })

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Predict DNA-binding proteins using pLM-DBPs")
    parser.add_argument("input_file", type=str, help="Input FASTA file with protein sequences")
    parser.add_argument("output_file", type=str, help="Output CSV file for predictions")

    # path to saved model
    model_path = "model/plm_dbp_ann.keras"
    args = parser.parse_args()

    # Run the main prediction pipeline
    main(args.input_file, args.output_file, model_path)


# run
# python predict.py input/example.fasta output/result.csv