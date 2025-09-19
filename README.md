# DTA predict

## Description

Predicting Drug-target binding affinity between drug maolecule and target protein using KIBA Kinase inhibitor bioactivity dataset. The architecture of this project combines GCN Graphical Neural Network for drug molecules, CNN Convolutional Neural Layer for protein sequence and MLP for KIBA score prediction.

## Technologies

**Back-end**: Python, flask framework

**Front-end**: HTML, CSS

**Libraries**: Deepchem, RDKit tools, tensorflow Keras, spektral, scikit-learn, matplotlib, pandas

## Features

1. **web application user interface** : Flask-based web application
	- input section for smiles sequence
	- input section for protein aa sequence
	- Kiba affinity score prediction with color bar for score indicator visualization
	- Drug molecule visualization with RDKitools

	**Example Inputs**:
   - **Drug SMILES**: `CC1=C2C=C(C=CC2=NN1)C3=CC(=CN=C3)OCC(CC4=CC=CC=C4)N`
   - **Protein**: `MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAE`

2. **ML back-end** : 
- **SMILES Featurization**: data transformation of drug molecules into molecular graphs with deepchem featurizer
- **Protein Sequence Encoding**: protein embedding with ProteinBert transformer
- **Multimodal features integration**: Combined representation of drug graph representations + protein sequence representation for score affinity prediction.


## Files and Project Structure

```
DTA_predict/
├── app.py      
├── models.py                 
├── preprocessing.py          
├── train.py                 
├── predict.py               
├── requirements.txt       
├── Data/             
│   └── cleaned_kiba_dataset.csv
├── Models/                
│   ├── best_dta_model.h5
│   ├── drug_model.h5
│   └── protein_model.h5
├── templates/             
│   └── index.html
└── static/             
    └── styles.css
```

| File| Description
--- | ---
app.py |  It includes API routes for kiba score prediction and molecule visualization with RDKit tools.
models.py | Builds main architecture blocks: GCN, CNN, MLP and alternative dense neural network for drug feature extraction. 
preprocessing.py| Data transformation of drug molecules and protein sequences into suitable format. smiles_featurization function for smiles to ConvMol graph transformation, followed by molecular descriptors computation, protein_encoding function for protein sequence embedding with ProteinBert pretrained transformer.
train.py | training pipeline and model evaluation
predict.py | standalone predictio

## Dataset

The project uses the KIBA (Kinase Inhibitor BioActivity) dataset, which is a Benchmark dataset for drug-target binding affinity prediction, part of  MoleculeNet collection of datasets, included in DeepChem. Drug-target bioactivity matrix with 52,498 chemical compounds, 467 kinase targets and a 246,088 KIBA scores. It contains 118 253 rows and 5 columns, for each row:
- Drug ID
- Protein ID
- Drug molecules represented as SMILES
- Protein sequences
- Kiba binding affinity scores

## Model training and evaluation

Model trained with Adam optimizer, Mean Squared Error loss, Early stopping and Model checkpoint callbacks.

Model evaluated with:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Coefficient of Determination (R²)


## Usage

1. **clone the repository** and navigate to directory
```bash
   git clone https://github.com/leila7575/DTA_predict.git
   cd DTA_predict
 ```

2.**create virtual envirnment**
```bash
   python -m venv venv
   source venv/bin/activate
```

3. **Install the requirements**
```bash
   pip install -r requirements.txt
```

4. **Data preprocessing**

Generates molecular graphs and protein embedding from raw kiba data and saves numpy array in Data directory.

```bash
python preprocessing.py
```

4. **Training**

Training and evaluation pipeline

```bash
python train.py
```
Saves trained models on Models directory

5. **Flask app**

Launch the flask web interface:

```bash
python app.py
```
open your browser to `http://localhost:5000`

6. **Standalone Prediction**

```bash
python predict.py
```

## Architecture

Three blocks:

1. **Drug Model**: GCN block or alternative dense neural network for processing drug molecular graphs 
2. **Protein Model**: CNN for processing ProtBert embeddings of protein sequences
3. **DTA Model**: Multi-layer perceptron that processes concatenated representations of drug molecule and protein sequences to predict binding affinity.


## Authors
Leila Louajri

## References

- KIBA Dataset: [Tang et al., 2014](https://doi.org/10.1093/bioinformatics/btu626)
- ProtBert: [Elnaggar et al., 2020](https://doi.org/10.1109/TPAMI.2021.3095381)
- DeepChem: [Ramsundar et al., 2019](https://github.com/deepchem/deepchem)
- Tang J. KiBA - a benchmark dataset for drug target prediction [Internet]. Kaggle. Available from: https://www.kaggle.com/datasets/christang0002/davis-and-kiba?select=kiba.txt  

- Tang J, Szwajda A, Shakyawar S, Xu T, Hintsanen P, Wennerberg K, et al. Making sense of large-scale kinase inhibitor bioactivity data sets: a comparative and integrative analysis. Bioinformatics. 2014;30(12):172–180. Available from: https://researchportal.helsinki.fi/en/datasets/kiba-a-benchmark-dataset-for-drug-target-prediction  

- Öztürk H, Özgür A, Ozkirimli E. DeepDTA: Deep drug–target binding affinity prediction. Bioinformatics. 2018;34(17):i821–i829.  

- Nguyen T, Le H, Quinn TP, Venkatesh S. GraphDTA: Predicting drug–target binding affinity with graph neural networks. Bioinformatics. 2021;37(8):1140–1147.  

- Mukherjee S, Ghosh M, Basuchowdhuri P. DeepGLSTM: Deep Graph Convolutional Network and LSTM based approach for predicting drug-target binding affinity. SIAM J Sci Comput. 2022;44(3):723–737.  
