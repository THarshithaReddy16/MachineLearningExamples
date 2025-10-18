# -------------------------------------------
# Dimensionality Reduction using PCA in Python
# -------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -----------------------------
# Step 1: Load the dataset
# -----------------------------
iris = load_iris()
X = iris.data   # Features
y = iris.target # Labels

# Convert into DataFrame for clarity
df = pd.DataFrame(X, columns=iris.feature_names)
print("Original Dataset Shape:", df.shape)

# -----------------------------
# Step 2: Standardize the data
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
	
# -----------------------------
# Step 3: Apply PCA
# -----------------------------
# Reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Reduced Dataset Shape (after PCA):", X_pca.shape)
print("\nExplained Variance Ratio:", pca.explained_variance_ratio_)

# -----------------------------
# Step 4: Visualize results
# -----------------------------
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="viridis", edgecolor="k", s=80)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Iris Dataset (Dimensionality Reduction)")
plt.colorbar(label="Target Classes")
plt.show()
