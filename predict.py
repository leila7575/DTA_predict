#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf
from preprocessing import smiles_featurization, protein_encoding
from tensorflow.keras.models import load_model


MODEL_DIR = "Models"
drug_model_embedding = load_model(os.path.join(MODEL_DIR, "drug_model.h5"), compile=False)
protein_model_embedding = load_model(os.path.join(MODEL_DIR, "protein_model.h5"), compile=False)
dta_model = load_model(os.path.join(MODEL_DIR, "best_dta_model.h5"), compile=False)


if __name__ == "__main__":
	smiles = "CC1=C2C=C(C=CC2=NN1)C3=CC(=CN=C3)OCC(CC4=CC=CC=C4)N"
	protein_sequence = "MSDVAIVKEGWLHKRGEYIKTWRPRYFLLKNDGTFIGYKERPQDVDQREAPLNNFSVAQCQLMKTERPRPNTFIIRCLQWTTVIERTFHVETPEEREEWTTAIQTVADGLKKQEEEEMDFRSGSPSDNSGAEEMEVSLAKPKHRVTMNEFEYLKLLGKGTFGKVILVKEKATGRYYAMKILKKEVIVAKDEVAHTLTENRVLQNSRHPFLTALKYSFQTHDRLCFVMEYANGGELFFHLSRERVFSEDRARFYGAEIVSALDYLHSEKNVVYRDLKLENLMLDKDGHIKITDFGLCKEGIKDGATMKTFCGTPEYLAPEVLEDNDYGRAVDWWGLGVVMYEMMCGRLPFYNQDHEKLFELILMEEIRFPRTLGPEAKSLLSGLLKKDPKQRLGGGSEDAKEIMQHRFFAGIVWQHVYEKKLSPPFKPQVTSETDTRYFDEEFTAQMITITPPDQDDSMECVDSERRPHFPQFSYSASGTA"

	mol_graph_list = smiles_featurization([smiles])
	mol_graph = mol_graph_list[0]
	atom_features = mol_graph.get_atom_features()
	atom_features = np.asarray(atom_features, dtype=np.float32)
	mean = np.mean(atom_features, axis=0)
	std  = np.std(atom_features, axis=0)
	mx   = np.max(atom_features, axis=0)
	mn   = np.min(atom_features, axis=0)
	cnt  = np.array([atom_features.shape[0]], dtype=np.float32)
	drug_descriptor = np.concatenate([mean, std, mx, mn, cnt]).reshape(1, -1)

	drug_emb = drug_model_embedding.predict(drug_descriptor)
	protein_emb = protein_model_embedding.predict(protein_encoding(protein_sequence).reshape(1, -1))

	prediction = dta_model.predict([drug_emb, protein_emb])
	print("Predicted affinity:", prediction.flatten()[0])
