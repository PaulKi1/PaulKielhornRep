from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_excel(r"C:")

# Select only numerical columns
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

# Standardize the data
scaler = StandardScaler()
df_standardized = scaler.fit_transform(df[numerical_cols])

# Perform PCA on the entire dataset
pca = PCA()
principal_components = pca.fit_transform(df_standardized)


# Plot of explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
n_components = len(explained_variance_ratio)

plt.figure(figsize=(10, 4))

# Normal plot
plt.subplot(1, 2, 1)
plt.bar(range(1, n_components + 1), explained_variance_ratio, alpha=0.5, align='center')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Plot')

# Cumulative explained variance plot
plt.subplot(1, 2, 2)
plt.plot(range(1, n_components + 1), np.cumsum(explained_variance_ratio), marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance Plot')

plt.tight_layout()
plt.show()



# Trying PCA while defining the minium cumulative explained variance
# Has to be updated
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Determine the number of principal components needed for a cumulative explained variance of 0.95
num_components = np.argmax(cumulative_explained_variance >= 0.45) + 1

# Perform PCA with the determined number of components
pca = PCA(n_components=num_components)
principal_component = pca.fit_transform(principal_components)

# Print the number of retained principal components
print(f"Number of Retained Principal Components: {num_components}")

# Print the principal components
print("Principal Components:")
print(principal_component)

plt.scatter(principal_component[:, 0], principal_component[:,1])
plt.xlabel('')
plt.ylabel('')
plt.title('Plot of PCA')
plt.show()

