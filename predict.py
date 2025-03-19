#!/usr/bin/env python 3.10.16
# coding: utf-8

"""
      Author  : Suresh Pokharel
      Email   : sp2530@rit.edu
      Affiliation: Rochester Institute of Technology
"""


# Standard Libraries
import os
import sys
import re
import argparse
import subprocess
from io import StringIO

# Data Manipulation
import pandas as pd
import numpy as np

# Bioinformatics
from Bio import SeqIO

# Machine Learning & Deep Learning
import tensorflow as tf
from tensorflow.keras.models import load_model
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Transformers
from transformers import T5Tokenizer, T5EncoderModel

# Utilities
from tqdm import tqdm
from IPython.display import clear_output

# Saprot
from assets.SaProt.model.saprot.base import SaprotBaseModel
from transformers import EsmTokenizer


# set current working directory to MAIN_DIR or you can change manually if needed
MAIN_DIR = os.getcwd()
print("MAIN_DIR:", MAIN_DIR)

# Load pLMDBPs base models
ProtT5_ann_model = load_model(os.path.join(MAIN_DIR, "assets/models/ProtT5_ann_model_1.keras"), compile=False)
SaProt_ann_model = load_model(os.path.join(MAIN_DIR, "assets/models/SaProt_ann_model_1.keras"), compile=False)

# we are now running on cpu, you can change to 'cude:0' for GPU and configure 
tf_device = tf.device('cpu')
device = torch.device('cpu')


# ### ProtT5 PLM
"""
Functions to load prott5 model, and to extract embeddings from prott5. 
You may need to configure the requirements accordingly if the given requirements does not work on your device.
"""
def load_ProtT5():
    global prott5_tokenizer, prott5_model
    if "prott5_tokenizer" not in globals():
        prott5_tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50')
    if "prott5_model" not in globals():
        prott5_model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_uniref50').eval()


def get_ProtT5_embeddings(sequence): 
    # replace rare amino acids with X
    sequence = re.sub(r"[UZOB]", "X", sequence)
    
    # Add space in between amino acids
    sequence = [ ' '.join(sequence)]
    
    # set configurations and extract features
    ids = prott5_tokenizer.batch_encode_plus(sequence, add_special_tokens=True, padding=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    
    with torch.no_grad():
        embedding = prott5_model(input_ids=input_ids,attention_mask=attention_mask)
    embedding = embedding.last_hidden_state
    
    # find length
    seq_len = (attention_mask[0] == 1).sum()
    
    # average over diemnsion to geta  single per protein embedding
    seq_emd = embedding[0][:seq_len].mean(axis=0) # shape (1024)

    return seq_emd


# ### SaProt PLM

"""
We need prostt5 model to predict AA->3Di. You can use foldseek if you have pdb for the protein that you want to predict. 
Please note that using foldseek will add computational requirement.
"""

def load_ProstT5():
    global prostt5_tokenizer, prostt5_model
    if "prostt5_tokenizer" not in globals():
        prostt5_tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5_fp16')
    if "prostt5_model" not in globals():
        prostt5_model = T5EncoderModel.from_pretrained("Rostlab/ProstT5_fp16").to(device).eval()

"""
Load the CNN model once and make it a global variable
To predict AA->3Di
"""
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 32, kernel_size=(7, 1), padding=(3, 0)),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.0),
            torch.nn.Conv2d(32, 20, kernel_size=(7, 1), padding=(3, 0))
        )

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(dim=-1)
        Yhat = self.classifier(x)
        Yhat = Yhat.squeeze(dim=-1)
        return Yhat

# Function to predict 3Di sequence
def predict_3Di(sequence):
    """
    Predict 3Di sequence from an amino acid sequence.

    Args:
        sequence (str): Amino acid sequence.

    Returns:
        str: Predicted 3Di sequence
    """
    global prostt5_model, prostt5_tokenizer, cnn_model

    # Preprocess the sequence
    prefix = "<AA2fold>"
    seq = prefix + ' ' + ' '.join(list(sequence))
    token_encoding = prostt5_tokenizer(seq, return_tensors="pt").to(device)

    # Generate embeddings using the T5 model
    with torch.no_grad():
        embedding_repr = prostt5_model(**token_encoding)
        embedding = embedding_repr.last_hidden_state[:, 1:, :]  # Skip special token
        prediction = cnn_model(embedding)
        prediction = prediction.argmax(dim=1).squeeze().cpu().numpy()

    # Map predictions to 3Di symbols
    ss_mapping = {
        0: "A", 1: "C", 2: "D", 3: "E", 4: "F", 5: "G", 6: "H", 7: "I",
        8: "K", 9: "L", 10: "M", 11: "N", 12: "P", 13: "Q", 14: "R", 15: "S",
        16: "T", 17: "V", 18: "W", 19: "Y"
    }
    predicted_3Di = "".join([ss_mapping[p] for p in prediction])
    return predicted_3Di.lower()


def get_SaProt_embeddings(Seq_AA):
    """
    Get or compute SaProt embeddings for a protein sequence and its structural information.

    Parameters:
    - accession (str): Accession ID of the protein.
    - Seq_AA (str): Amino acid sequence of the protein.
    - site (int): Position of interest in the sequence.
    - feature_folder (str): Path to the folder containing precomputed features.
    - saprot_tokenizer: Tokenizer for SaProt.
    - saprot_model: Model for generating embeddings.
    - device: PyTorch device (e.g., 'cpu' or 'cuda').

    Returns:
    - torch.Tensor: Averaged representation of protein sequence.
    """

    Seq_3Di = predict_3Di(Seq_AA) # Use the provided foldseek code if pdb available
    
    # Combine sequence and structure
    combined_AA_3Di = "".join([a + b for a, b in zip(Seq_AA, Seq_3Di)])
    
    # Tokenize sequence
    inputs = saprot_tokenizer(combined_AA_3Di, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the correct device
    
    # Generate embeddings
    embeddings_per_residue = saprot_model.get_hidden_states(inputs)[0]

    # Compute protein-level representation (mean pooling)
    protein_representation = embeddings_per_residue.mean(dim=0)
    
    return protein_representation


def load_SaProt():
    global saprot_model, saprot_tokenizer

    saprot_config = {
        "task": "base",
        "config_path": os.path.join(MAIN_DIR, "assets", "SaProt", "model", "saprot", "SaProt_650M_AF2"), # Note this is the directory path of SaProt, not the ".pt" file
        "load_pretrained": True,
    }
    
    if "saprot_tokenizer" not in globals():
        saprot_tokenizer = EsmTokenizer.from_pretrained(saprot_config["config_path"])
    if "saprot_model" not in globals():
        saprot_model = SaprotBaseModel(**saprot_config)


# Load required models. These models uses a bunch of libraries that might lead to libraries version glitches.
# Please use the given version of libraries or refer to ProtT5, SaProt, ProstT5 github repos for guidance.

# Load ProtT5
load_ProtT5()

# Load ProstT5
load_ProstT5()

# Load SaProt
load_SaProt()

# Load CNN model for 3Di Prediction
# Read more: https://github.com/mheinzinger/ProstT5/tree/main/cnn_chkpnt
cnn_model = CNN()
checkpoint_path_3Di_prediction = os.path.join(MAIN_DIR, "assets","AA_to_3Di", "AA_to_3Di_prostt5_cnn_model.pt")
state = torch.load(checkpoint_path_3Di_prediction, map_location=device)
cnn_model.load_state_dict(state["state_dict"])
cnn_model = cnn_model.to(device).eval()


### Read Input Fasta File
def make_prediction(fasta_file_path):
    results = []

    for record in tqdm(SeqIO.parse(fasta_file_path, "fasta")):
        description = record.id
        
        # Amino Acid Sequence
        AA_Seq = str(record.seq)

        # Get pLM representations
        ProtT5_embeddings = get_ProtT5_embeddings(AA_Seq).detach().cpu()
        SaProt_embeddings = get_SaProt_embeddings(AA_Seq).detach().cpu()

        # Reshape embeddings, it should be (1, 1024) and (1,1280)
        if ProtT5_embeddings.ndimension() == 1:
            ProtT5_embeddings = ProtT5_embeddings.unsqueeze(0)  # Shape becomes (1, 1024)
        
        if SaProt_embeddings.ndimension() == 1:
            SaProt_embeddings = SaProt_embeddings.unsqueeze(0)  # Shape becomes (1, 1280)

        # Get predictions
        ProtT5_ANN_prob = ProtT5_ann_model(ProtT5_embeddings).numpy().item()
        SaProt_ANN_prob = SaProt_ann_model(SaProt_embeddings).numpy().item()

        # Average prediction
        avg_prob = (ProtT5_ANN_prob + SaProt_ANN_prob) / 2

        # Final prediction (binary)
        final_prediction = avg_prob > 0.5

        # Store results
        results.append({
            "description": description,
            "ProtT5_ANN_prob": ProtT5_ANN_prob,
            "SaProt_ANN_prob": SaProt_ANN_prob,
            "avg_prob": avg_prob,
            "final_prediction": final_prediction
        })

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Create output directory if it doesn't exist
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filename
    base_filename = os.path.splitext(os.path.basename(fasta_file_path))[0]
    output_file = os.path.join(output_dir, f"{base_filename}_results.csv")

    # Save to CSV
    df.to_csv(output_file, index=False)

    print(f"Results saved to {output_file}") 

if __name__ == "__main__":
    input_file = sys.argv[1]  # Get the input file from the command-line argument
    make_prediction(input_file)

# example
#python predict.py input/example.fasta
