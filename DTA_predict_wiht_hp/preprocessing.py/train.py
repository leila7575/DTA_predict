#!/usr/bin/env python3


import numpy as np
import tensorflow as tf
from models import alternative_drug_model, CNN_model, DTA_MLP_model, DTA_model_with_hyperparams
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import gc
import os
from scipy import stats
import keras_tuner as kt

# Define directory paths
DATA_DIR = "Data"
MODEL_DIR = "Models"
RESULTS_DIR = "Results"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

molgraph_train = np.load("Data/molgraph_train.npy", allow_pickle=True)
molgraph_val = np.load("Data/molgraph_val.npy", allow_pickle=True)
molgraph_test = np.load("Data/molgraph_test.npy", allow_pickle=True)

protein_train = np.load("Data/protein_train.npy")
protein_val = np.load("Data/protein_val.npy")
protein_test = np.load("Data/protein_test.npy")

score_train = np.load("Data/score_train.npy")
score_val = np.load("Data/score_val.npy")
score_test = np.load("Data/score_test.npy")


print(f"Train set data shapes: protein shape {protein_train.shape} , molecular graphs shape {len(molgraph_train)}, scores shape {score_train.shape}")


def extract_atom_features(molgraph):
    mol_descriptors = []
    for i, molecule in enumerate(molgraph):
        atom_features = molecule.get_atom_features()
        mol_descriptor = np.concatenate([
            np.mean(atom_features, axis=0),
            np.std(atom_features, axis=0),
            np.max(atom_features, axis=0),
            np.min(atom_features, axis=0),
            [atom_features.shape[0]]
        ])
        mol_descriptors.append(mol_descriptor)
    return np.array(mol_descriptors, dtype=np.float32)

molgraph_train_desc = extract_atom_features(molgraph_train)
molgraph_val_desc = extract_atom_features(molgraph_val)
molgraph_test_desc = extract_atom_features(molgraph_test)

del molgraph_train, molgraph_val, molgraph_test
gc.collect()

drug_feature_dim = molgraph_train_desc.shape[1]
drug_model = alternative_drug_model(drug_feature_dim)

def compute_embeddings_per_batch(model, input, batch_size):
    embeddings = []
    total_samples_nb = input.shape[0]

    for i in range(0, total_samples_nb, batch_size):
        last_batch_index = min(i + batch_size, total_samples_nb)

        data_slice = input[i:last_batch_index]
        embedding_per_batch = model(data_slice, training=False)
        embeddings.append(embedding_per_batch.numpy())

        if i % 20000 == 0:
            tf.keras.backend.clear_session()
            gc.collect()

    embeddings_matrix = np.vstack(embeddings)
    return embeddings_matrix

drug_train_embedding = compute_embeddings_per_batch(drug_model, molgraph_train_desc, batch_size=2000)
drug_val_embedding = compute_embeddings_per_batch(drug_model, molgraph_val_desc, batch_size=2000)
drug_test_embedding = compute_embeddings_per_batch(drug_model, molgraph_test_desc, batch_size=2000)

drug_model.save(os.path.join(MODEL_DIR, 'drug_model.h5'))

del molgraph_train_desc, molgraph_val_desc, molgraph_test_desc, drug_model
gc.collect()

protein_model = CNN_model(protembedding_dim=protein_train.shape[1])

protein_train_embedding = compute_embeddings_per_batch(protein_model, protein_train, batch_size=500)
protein_val_embedding = compute_embeddings_per_batch(protein_model, protein_val, batch_size=500)
protein_test_embedding = compute_embeddings_per_batch(protein_model, protein_test, batch_size=500)

protein_model.save(os.path.join(MODEL_DIR, 'protein_model.h5'))

del protein_train, protein_val, protein_test, protein_model
gc.collect()

np.save(os.path.join(DATA_DIR, "drug_train_emb.npy"), drug_train_embedding)
np.save(os.path.join(DATA_DIR, "drug_val_emb.npy"), drug_val_embedding)
np.save(os.path.join(DATA_DIR, "drug_test_emb.npy"), drug_test_embedding)
np.save(os.path.join(DATA_DIR, "protein_train_emb.npy"), protein_train_embedding)
np.save(os.path.join(DATA_DIR, "protein_val_emb.npy"), protein_val_embedding)
np.save(os.path.join(DATA_DIR, "protein_test_emb.npy"), protein_test_embedding)

drug_embedding_dim = drug_train_embedding.shape[1]
protein_embedding_dim =  protein_train_embedding.shape[1]

def build_dta_model_hparams(hyperparams):
  model = DTA_model_with_hyperparams(hyperparams, drug_embedding_dim, protein_embedding_dim)
  learning_rate = hyperparams.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
      loss='mse',
      metrics=[tf.keras.metrics.MeanAbsoluteError()]
  )
  return model

bayesian_opt = kt.BayesianOptimization(
    build_dta_model_hparams,
    objective='val_loss',
    max_trials=10,
    directory='bayesian_optimisation',
    project_name='dta_model_bayesianopt'
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
  )


bayesian_opt.search(
    x=[drug_train_embedding.astype(np.float32), protein_train_embedding.astype(np.float32)],
    y=score_train.astype(np.float32),
    validation_data=([drug_val_embedding.astype(np.float32), protein_val_embedding.astype(np.float32)], score_val.astype(np.float32)),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

best_hyperparameters = bayesian_opt.get_best_hyperparameters(num_trials=1)[0]
print(f"Best hyperparameters: {best_hyperparameters.values}")

model_hptuned = DTA_model_with_hyperparams(best_hyperparameters, drug_embedding_dim, protein_embedding_dim)

model_hptuned.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=best_hyperparameters.get("learning_rate")),
      loss='mse',
      metrics=[tf.keras.metrics.MeanAbsoluteError()]
)

model_hptuned.summary()


x_train = [drug_train_embedding.astype(np.float32), protein_train_embedding.astype(np.float32)]
x_val = [drug_val_embedding.astype(np.float32), protein_val_embedding.astype(np.float32)]
y_train = score_train.astype(np.float32)
y_val = score_val.astype(np.float32)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'best_dta_model_hp.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

history = model_hptuned.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

model_hptuned.save(os.path.join(MODEL_DIR, 'dta_predict_final_hp.h5'))

del x_train, x_val, y_train, y_val
del drug_train_embedding, drug_val_embedding, protein_train_embedding, protein_val_embedding
gc.collect()

print("Model summary, parameters and shapes:")
print(f'Drug embedding dimmension: {drug_embedding_dim}')
print(f'Protein embedding dimmension: {protein_embedding_dim}')
print(f'Training, validation and test sets lenths: {len(score_train)}, {len(score_val)}, {len(score_test)}')
total_nb_parameters = model_hptuned.count_params()
print(f'Total number of parameters: {total_nb_parameters}')

print("Model training summary:")
last_train_loss = float(history.history['loss'][-1])
last_val_loss = float(history.history['val_loss'][-1])
last_train_mae = float(history.history['mean_absolute_error'][-1])
last_val_mae = float(history.history['val_mean_absolute_error'][-1])
epochs_nb = len(history.history['loss'])
print(f'Final training loss: {last_train_loss}')
print(f'Final validation loss: {last_val_loss}')
print(f'Final training mean absolute error: {last_train_mae}')
print(f'Final validation mean absolute error: {last_val_mae}')
print(f'Total number of epochs trained: {epochs_nb}')


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(history.history['mean_absolute_error'], label='Training MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model Mean Absolute Error')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "training_history_hp.png"))
plt.show()

x_test = [drug_test_embedding.astype(np.float32), protein_test_embedding.astype(np.float32)]
y_test = score_test.astype(np.float32)

test_loss, test_mae = model_hptuned.evaluate(x_test, y_test, batch_size=32, verbose=0)
print("Evaluating the model")
print(f"Test Loss: {test_loss}")
print(f"Test MAE: {test_mae}")

def predictions_per_batches(model, drug_embedding, protein_embedding, batch_size):
    predictions = []
    total_samples_nb = len(drug_embedding)

    for i in range(0, total_samples_nb, batch_size):
        last_batch_index = min(i + batch_size, total_samples_nb)
        drug_batch = drug_embedding[i:last_batch_index]
        protein_batch = protein_embedding[i:last_batch_index]

        batch_prediction = model.predict([drug_batch, protein_batch], verbose=0)

        predictions.append(batch_prediction.flatten())

        if i % 10000 == 0:
            tf.keras.backend.clear_session()

    return np.concatenate(predictions)

predictions_test_set = predictions_per_batches(model_hptuned, drug_test_embedding, protein_test_embedding, batch_size=500)

mse = mean_squared_error(score_test, predictions_test_set)
rmse = np.sqrt(mse)
mae = mean_absolute_error(score_test, predictions_test_set)
r2 = r2_score(score_test, predictions_test_set)

print(f"Model evaluation with predicted values:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Coefficient of determination: {r2:.4f}")

plt.figure(figsize=(12, 5))
plt.scatter(score_test, predictions_test_set)
plt.plot([score_test.min(), score_test.max()], [score_test.min(), score_test.max()], 'g--')
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.xlabel('Measured Affinity Score (kiba dataset)')
plt.ylabel('Predicted Affinity Score')
plt.title('Predictions vs Measured Affinity Score')

slope, intercept, r, p, se = stats.linregress(score_test, predictions_test_set)
regression_line = slope * score_test + intercept
plt.plot(score_test, regression_line, 'y--', label=f'Regression line: slope {slope} * score_test + intercept {intercept}')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "predictions_vs_actual_hp.png"))
plt.show()
