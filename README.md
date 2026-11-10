# A Multi-Task D-MPNN Framework for Predicting Alpha-1 Adrenergic Receptor Antagonists



This repository contains a multi-task directed message-passing neural network (D-MPNN)-based prediction pipeline for identifying potential **α1-adrenergic receptor antagonists** from molecular SMILES strings. The model predicts whether a compound is active or inactive for the three subtypes: **ADRA1A, ADRA1B, and ADRA1D**.

\---

## Features

* Predicts antagonists from SMILES
* Multi-task D-MPNN (directed message-passing neural network) Model
* Saves prediction results 

\---


## Requirements

Install the following Python packages before running the prediction script:

```
pip install numpy pandas rdkit torch torch-geometric
```

Versions:

numpy==1.26.4

pandas==2.3.3

rdkit==2025.09.6

torch==2.11.0

torch-geometric==2.7.0


With versions:

```
pip install numpy==1.26.4 pandas==2.3.3 rdkit==2025.09.6 torch==2.11.0 torch-geometric==2.7.0
```


## Input File

Create a "mols.csv" file containing a column named `SMILES`.

Example:

```csv
SMILES
CCO
CCN(CC)CC
```

Save the "mols.csv" file in the same folder as script.

\---

## Running the Prediction

1. Download the model repository.
4. Run the prediction script.


## Notes

* The input CSV must contain a valid `SMILES` column.
* All model files and scripts must be present in the same project directory before running predictions.

\---


## Publication

If you use this repository or the associated model in your research, please cite the related publication:

```text
Rashi Jain, Anju Sharma, and Prabha Garg. A Multi-Task D-MPNN Framework for Predicting Alpha-1 Adrenergic Receptor Antagonists with a Drug Repurposing Study. Journal Name. Year.
```

DOI:

```text
DOI: 
```

\---

