# PLM-DBPs (DNA-Binding Protein Prediction Tool for Plants)
**PLM-DBPs: Enhancing Plant DNA-Binding Protein Prediction by Integrating Sequence-Based and Structure-Aware Protein Language Models**

**Author**: Suresh Pokharel et al.  
**Email**: [sp2530@rit.edu](mailto:sp2530@rit.edu)  
**Affiliation**: Rochester Institute of Technology  

## Overview

PLM-DBPs is a deep-learning framework designed to accurately predict DNA-binding proteins (DBPs) in plants by integrating advanced sequence-based protein language model (ProtT5) with structure-aware representations (SaProt). Our approach effectively leverages the complementary information from sequence and structural features, surpassing previous state-of-the-art methods tailored for general DBP prediction. The repository contains the complete implementation and resources required to utilize or extend this model for further research in plant genomics and biotechnology. 

**Input:** A Fasta file with protein sequences
**Output:** A CSV file with probabilities and binary predictions (DNA Binding or Non-DNA Binding).

## Embeddings
- **ProtT5** Sequence based protein representation. (https://ieeexplore.ieee.org/abstract/document/9477085, https://github.com/agemagician/ProtTrans)
- **SaProt** Structure aware protein representation.(https://github.com/westlake-repl/SaProt)
- **Pre-trained pLM-DBPs** model for DNA-binding protein prediction.


## Installation


### Make sure to create a new environment:



```bash
conda create --name plmdbps python=3.10.16
conda activate plmdbps
```

### Requirements
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
    scikit-learn==1.4.2
    scipy==1.15.2
    sentencepiece==0.2.0
    ipython==8.12.3
    pytz==2025.1
    pytorch-lightning==1.8.3

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/suresh-pokharel/pLM-DBPs.git
   cd pLM-DBPs
2. Run the predict.py (input and output paths)
   ```bash
    python predict.py input/example.fasta

### Web-Server Coming Soon
### Note: Full training script and data will be uploaded soon.
