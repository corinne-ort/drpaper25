"""
Added Corinne Orton 7-21-25

Contains functions to produce 2d embeddings using several DR methods. Embeddings are then normalized.
"""

import numpy as np
from numpy import linalg as LA
#import pandas as pd
#from scipy import optimize
#from scipy.optimize import minimize
#import matplotlib.pyplot as plt
#from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE
import pacmap

def get_embedding(X_train, method, **kwargs):
    """
    Return embedding of original data X_train according to specified DR method.
    """
    if method.lower() == "pca":
        return pca(X_train)
    
    else if method.lower() == "umap":
        return umap(X_train, **kwargs)

    else if method.lower() == "tsne":
        return tsne(X_train, **kwargs)

    else if method.lower() == "pacmap":
        return pacmap(X_train, **kwargs)

    else:
        return "Invalid transform."


def pca(X_train):
    reducer = PCA(n_components=2, random_state=42)
    embedding = reducer.fit_transform(X_train)
    embedding = embedding / LA.norm(embedding)
    return embedding


def umap(X_train, **kwargs):
    reducer = umap.UMAP(**kwargs) # default embedding dim is 2, num neighbors is 15
    embedding = reducer.fit_transform(X_train)
    embedding = embedding / LA.norm(embedding)
    return embedding


def tsne(X_train, **kwargs):
    reducer = TSNE(n_components=2, random_state=42, **kwargs) # default perplexity is 30
    embedding = reducer.fit_transform(X_train)
    embedding = embedding / LA.norm(embedding)
    return embedding

def pacmap(X_train, **kwargs):
    reducer = pacmap.PaCMAP(n_components=2)
    X_transformed = embedding.fit_transform(X_train, init="pca") # default init is PCA
    embedding = embedding / LA.norm(embedding)
    return embedding