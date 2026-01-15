# Balancing Local and Global Structure Preservation in Dimensionality Reduction

This is the repository for the UPCA algorithm introduced in the paper "Balancing Local and Global Structure Preservation in Dimensionality Reduction" by Corinne Orton, Connor R. Shrader, and Anna Little, expected to be published in _Springer Nature_ in 2026, in proceedings from the 2025 International Conference on Computational Science and Computational Intelligence (CSCI'25). UPCA is a hybrid of the PCA and UMAP dimensionality reduction methods.

## Organization
### Using UPCA
1. PCA and UMAP embeddings of all 10 datasets from the paper are stored in the __embeddings__ directory.
2. To run UPCA, open __concatenate.ipynb__ and in the third Python cell, set datasets = ['datasets you want to embed'] and alpha_list=[float(alpha values you want to select)]. This will automatically load the PCA and UMAP embeddings from __embeddings__.

_Note: The file __create_embedding.ipynb__ allows you to create embeddings with PCA, UMAP, t-SNE, or PaCMAP (any method from __transforms.py__), so you can use __concatenate.ipynb__ to make hybrid embeddings of other methods too!_

### Data
The datasets we used are all publicly available.  
CAFs: https://github.com/KPLab/SCS_CAF  
Cell Mix, TM Lung, TM Panc: https://github.com/LuyiTian/sc_mixology  
Duo 4, Duo 8, Kang, Muraro: https://github.com/hyhuang00/scRNA-DR2020  
MNIST accessed thru scikit-learn.  
FMNIST accessed thru umap-learn.  

### Things you can probably ignore
Anything with "df" in the title and a bunch of underscores is a dataframe with various results.
