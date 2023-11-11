import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

df = pd.read_excel (r"C:")
df = df.select_dtypes(include=[np.number])

# Basic Code to reduce dimension to n dimensions either with Multidimensional Scaling or PCA. Not yet used in Projekt

# Multidimensional Scaling
mds = MDS(n_components=2)
df_2d = mds.fit_transform(df)

plt.scatter(df_2d[:, 0], df_2d[:,1])
plt.show()

# Principal component analysis
pca = PCA(n_components=2)
df_2d_PCA = pca.fit_transform(df)

plt.scatter(df_2d_PCA[:, 0], df_2d_PCA[:,1])
plt.show()

# Multilinear PCA
tsne = TSNE(n_components=2)
df_2d_MPCA = tsne.fit_transform(df)

plt.scatter(df_2d_MPCA[:, 0], df_2d_MPCA[:,1])
plt.show()


