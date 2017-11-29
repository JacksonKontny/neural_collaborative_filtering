import pandas as pd
import pickle
from numpy.linalg import norm
import numpy as np

def save_embedding(model, mapping, layer_idx, file_name):
    emb = model.layers[layer_idx]
    emb = pd.DataFrame(emb.get_weights()[0])
    mapping = pd.DataFrame.from_dict(mapping, orient='index')
    mapping.columns=['idx']
    emb = emb.join(mapping)
    emb.to_pickle('./saved_embeddings/{}'.format(file_name))

def get_nearest(df, idx, n):
    embedding = df.drop('idx', axis=1)
    distances = norm(embedding - embedding.loc[idx], axis=1, ord=2)
    sorted_distances = np.argsort(distances)
    return df.loc[sorted_distances[0: n + 1]][['idx']]
