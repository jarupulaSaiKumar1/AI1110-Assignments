# Code to generate
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the SVHN dataset
# Assuming you have already loaded the dataset into X_train

# Perform PCA
pca = PCA()
pca.fit(X_train)

# Calculate the cumulative sum of explained variance ratio
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Find the number of eigenvectors required to keep the proportion of variance above 0.9
num_eigenvectors = np.argmax(cumulative_variance_ratio > 0.9) + 1

# Plot PoV against the number of eigenvectors
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio)
plt.xlabel('Number of Eigenvectors')
plt.ylabel('Proportion of Variance')
plt.title('PoV vs Number of Eigenvectors')
plt.show()
