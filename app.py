import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from rdkit import Chem
from rdkit.Chem import Draw
import tensorflow as tf
from tensorflow.keras.models import load_model
from preprocessing import smiles_featurization, protein_encoding
from io import BytesIO
import base64


drug_model_embedding = load_model('Models/drug_model.h5', compile=False)
protein_model_embedding = load_model('Models/protein_model.h5', compile=False)
dta_model = load_model('Models/best_dta_model.h5', compile=False)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    smiles = data['smiles']
    protein_sequence = data['protein']


    mol_graph_list = smiles_featurization([smiles])
    mol_graph = mol_graph_list[0]
    atom_features = np.asarray(mol_graph.get_atom_features(), dtype=np.float32)
    mean = np.mean(atom_features, axis=0)
    std  = np.std(atom_features, axis=0)
    maximum = np.max(atom_features, axis=0)
    minimum = np.min(atom_features, axis=0)
    atoms_nb  = np.array([atom_features.shape[0]], dtype=np.float32)
    drug_descriptor = np.concatenate([mean, std, maximum, minimum, atoms_nb]).reshape(1, -1)

    drug_emb = drug_model_embedding.predict(drug_descriptor)
    embedding = protein_encoding(protein_sequence)
    protein_emb = protein_model_embedding.predict(embedding.reshape(1, -1))

    prediction = dta_model.predict([drug_emb, protein_emb])
    predicted_score = float(prediction.flatten()[0])

    return jsonify({'score': predicted_score})

@app.route("/visualize", methods=["POST"])
def visualize_molecule():
    data = request.get_json()
    smiles = data['smiles']

    molecule_graph = Chem.MolFromSmiles(smiles)
    if molecule_graph is None:
        return jsonify({'error': 'Invalid SMILES string'}), 400

    mol_image = Draw.MolsToImage([molecule_graph])
    buffer = BytesIO()
    mol_image.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return jsonify({'image': image_b64})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
