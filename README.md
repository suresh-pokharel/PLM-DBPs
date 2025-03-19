# pLM-DBPs (DNA-Binding Protein Prediction Tool)
**An Enhanced DNA-Binding Protein Prediction in Plants Using Embeddings from Protein Language Model**


**Author**: Suresh Pokharel et al.  
**Email**: [sp2530@rit.edu](mailto:sp2530@rit.edu)  
**Affiliation**: Rochester Institute of Technology  

## Overview

This tool predicts whether a given protein sequence is a DNA-binding protein (DBP) using pLMs based embeddings extracted by the ProtT5 and SaProt. It inputs a FASTA file, processes each sequence, and outputs a CSV file with probabilities and binary predictions (DNA Binding or Non-DNA Binding).

## Embeddings

- **ProtT5** Sequence based protein representation.
- **SaProt** Structure aware protein representation.
- **Pre-trained pLM-DBPs** model for DNA-binding protein prediction.
- 
- Takes a FASTA file as input and outputs a CSV file with predictions.

## Installation

### Requirements
    Python 3.10.6
    biopython==1.83
    numpy==1.24.4
    pandas==2.2.3
    scikit_learn==1.4.2
    tensorflow==2.17.0
    tqdm==4.66.2
    transformers==4.28.0
    torch==1.11.0
    protobuf==4.25.3
    keras==3.9.0
    pytorch-lightning==1.8.3
    scikit-learn==1.4.2
    scipy==1.15.2
    sentencepiece==0.2.0

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/suresh-pokharel/pLM-DBPs.git
   cd pLM-DBPs
```
2. Run the predict.py (input and output paths)



```bash
python predict.py
```


### Full training code and data will be uploaded after publication.
