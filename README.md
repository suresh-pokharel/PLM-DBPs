# pLM-DBPs (DNA-Binding Protein Prediction Tool)
**An Enhanced DNA-Binding Protein Prediction in Plants Using Embeddings from Protein Language Model**


**Author**: Suresh Pokharel et al.  
**Email**: [sp2530@rit.edu](mailto:sp2530@rit.edu)  
**Affiliation**: Rochester Institute of Technology  

## Overview

This tool predicts whether a given protein sequence is a DNA-binding protein (DBP) using features extracted by the ProtT5 model. It inputs a FASTA file, processes each sequence, and outputs a CSV file with probabilities and binary predictions (DNA Binding or Non-DNA Binding).

## Features

- **ProtT5** for protein feature extraction.
- **Pre-trained pLM-DBPs** model for DNA-binding protein prediction.
- Command-line interface for easy use.
- Takes a FASTA file as input and outputs a CSV file with predictions.

## Installation

### Requirements

- Python 3.9+
- TensorFlow
- PyTorch
- transformers
- Biopython
- pandas

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/suresh-pokharel/pLM-DBPs.git
   cd pLM-DBPs
```
2. Run the predict.py (input and output paths)
```bash
python predict.py input/example.fasta output/result.csv
```

### Full training code and data will be uploaded after publication.
