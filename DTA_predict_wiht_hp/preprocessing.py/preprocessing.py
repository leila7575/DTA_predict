#!/usr/bin/env python3


import pandas as pd
import numpy as np
import deepchem as dc
from transformers import TFBertModel, BertTokenizer
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
import os
from rdkit import RDLogger


warnings.filterwarnings("ignore", category=DeprecationWarning)
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

featurizer = dc.feat.ConvMolFeaturizer()

def smiles_featurization(smiles_sequences):
    molecular_graphs = featurizer.featurize(smiles_sequences)
    return molecular_graphs

protein_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
protbert_model = TFBertModel.from_pretrained("Rostlab/prot_bert", from_pt=True)

def protein_encoding(protein_sequence, max_len=512):
    sequence = " ".join(list(protein_sequence))
    encoded_input = protein_tokenizer(sequence, return_tensors='tf', truncation=True, padding="max_length", max_length=max_len)
    output = protbert_model(**encoded_input, training=False)

    return output.last_hidden_state[:,0,:].numpy().squeeze()

def main():
    data_dir = "Data"
    os.makedirs(data_dir, exist_ok=True)
    
    cleaned_df = pd.read_csv('Data/cleaned_kiba_dataset.csv')
    
    training_set, temp_set = train_test_split(cleaned_df, test_size=0.2, random_state=42)
    val_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=42)
    
    molgraph_train = smiles_featurization(training_set['SMILES_sequence'].tolist())
    molgraph_val = smiles_featurization(val_set['SMILES_sequence'].tolist())
    molgraph_test = smiles_featurization(test_set['SMILES_sequence'].tolist())
    
    protembedding_dict = {}
    for prot_seq in tqdm(cleaned_df['Protein_sequence'].unique()):
        protembedding_dict[prot_seq] = protein_encoding(prot_seq)
        
    protein_train = np.stack([protembedding_dict[seq] for seq in training_set['Protein_sequence']])
    protein_val   = np.stack([protembedding_dict[seq] for seq in val_set['Protein_sequence']])
    protein_test  = np.stack([protembedding_dict[seq] for seq in test_set['Protein_sequence']])
    
    score_train = training_set['affinity_score'].values
    score_val = val_set['affinity_score'].values
    score_test = test_set['affinity_score'].values
    
    np.save(os.path.join(data_dir, "molgraph_train.npy"), molgraph_train)
    np.save(os.path.join(data_dir, "molgraph_val.npy"), molgraph_val)
    np.save(os.path.join(data_dir, "molgraph_test.npy"), molgraph_test)
    
    np.save(os.path.join(data_dir, "protein_train.npy"), protein_train)
    np.save(os.path.join(data_dir, "protein_val.npy"), protein_val)
    np.save(os.path.join(data_dir, "protein_test.npy"), protein_test)
    
    np.save(os.path.join(data_dir, "score_train.npy"), score_train)
    np.save(os.path.join(data_dir, "score_val.npy"), score_val)
    np.save(os.path.join(data_dir, "score_test.npy"), score_test)
    
if __name__ == "__main__":
    main()
