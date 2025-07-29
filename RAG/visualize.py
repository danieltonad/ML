import faiss
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load FAISS index
index = faiss.read_index("my_index.faiss")
vectors = index.reconstruct_n(0, index.ntotal)

# Reduce to 2D using PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(vectors)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, c='blue')
plt.title("Vector Distribution (PCA projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()
