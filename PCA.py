from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel(r"C:")


# Note this computes it for every single combination of variables. PCA more efficient is better
numerical_cols = df.select_dtypes(include=['number']).columns.tolist() # Only includes numerical data

# Standardize the data
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df[numerical_cols]), columns=numerical_cols)

# Initialize a dictionary to store PCA results
pca_results = {}

# Iterate over all combinations of numerical columns
for r in range(1, len(numerical_cols) + 1):
    for subset in combinations(numerical_cols, r):
        pca = PCA(n_components=len(subset))
        principal_components = pca.fit_transform(df_standardized[list(subset)])
        pca_results[subset] = {
            'Explained Variance Ratio': pca.explained_variance_ratio_,
            'Principal Components': principal_components
        }



if pca_results:
    # Get the first combination's PCA result
    first_combination_key = next(iter(pca_results))
    pca_result = pca_results[first_combination_key]
    explained_variance_ratio = pca_result['Explained Variance Ratio']

    # Number of principal components
    n_components = len(explained_variance_ratio)

    # Plot of explained variance ratio
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.bar(range(1, n_components + 1), explained_variance_ratio, alpha=0.5, align='center')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot')

    # Cumulative explained variance plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, n_components + 1), np.cumsum(explained_variance_ratio), marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance Plot')

    plt.tight_layout()
    plt.show()
else:
    print("No PCA results to plot.")


