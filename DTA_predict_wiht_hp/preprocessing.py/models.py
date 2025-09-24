#!/usr/bin/env python3


from tensorflow import keras as K
import tensorflow as tf
from spektral.layers import GCNConv, GlobalAvgPool


def GCN_model(feature_matrix_nodes, max_nodes=None):

    if max_nodes is None:
        max_nodes = 100


    feature_matrix = K.layers.Input(shape=(max_nodes, feature_matrix_nodes), name="feature_matrix")
    adjacency_matrix = K.layers.Input(shape=(max_nodes, max_nodes), name="adjacency_matrix")


    x = GCNConv(128, activation='relu')([feature_matrix, adjacency_matrix])
    x = GCNConv(128, activation='relu')([x, adjacency_matrix])


    x = GlobalAvgPool()(x)

    model = K.Model(inputs=[feature_matrix, adjacency_matrix], outputs=x)
    return model


def CNN_model(protembedding_dim):

    input_layer = K.layers.Input(shape=(protembedding_dim,))
    x = K.layers.Reshape((protembedding_dim, 1))(input_layer)
    x = K.layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = K.layers.MaxPooling1D(pool_size=2)(x)
    x = K.layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = K.layers.GlobalMaxPooling1D()(x)

    model = K.Model(inputs=input_layer, outputs=x)
    return model


def DTA_MLP_model(drug_input_shape, protein_input_shape, layers_nb=2, units=256, dropout=0.3, use_batchnorm=True):

    drug_input = K.layers.Input(shape=(drug_input_shape,), name="drug_input")
    protein_input = K.layers.Input(shape=(protein_input_shape,), name="protein_input")


    x = K.layers.Concatenate()([drug_input, protein_input])

    for i in range(layers_nb):
        x = K.layers.Dense(units, activation='relu')(x)
        if use_batchnorm:
            x = K.layers.BatchNormalization()(x)
        x = K.layers.Dropout(dropout)(x)

    output = K.layers.Dense(1, activation='linear')(x)

    model = K.Model(inputs=[drug_input, protein_input], outputs=output)
    return model

def DTA_model_with_hyperparams(hyperparameters, drug_input_shape, protein_input_shape):
  layers_nb = hyperparameters.Int("layers_nb", min_value=1, max_value=5)
  units = hyperparameters.Choice("units", [128, 256, 512], default=256)
  dropout = hyperparameters.Float("dropout", min_value=0.1, max_value=0.5, step=0.1, default=0.3)
  use_batchnorm = hyperparameters.Boolean("use_batchnorm", default=True)

  model = DTA_MLP_model(
      drug_input_shape=drug_input_shape,
      protein_input_shape=protein_input_shape,
      layers_nb=layers_nb,
      units=units,
      dropout=dropout,
      use_batchnorm=use_batchnorm
  )
  return model

def alternative_drug_model(drug_feature_dim):

    input_layer = K.layers.Input(shape=(drug_feature_dim,))
    x = K.layers.Dense(256, activation='relu')(input_layer)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dropout(0.3)(x)
    x = K.layers.Dense(128, activation='relu')(x)
    x = K.layers.Dropout(0.2)(x)
    x = K.layers.Dense(128, activation='relu')(x)

    model = K.Model(inputs=input_layer, outputs=x)
    return model