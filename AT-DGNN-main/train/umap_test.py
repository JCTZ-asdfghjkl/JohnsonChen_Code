import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt

# Example data (40 samples, 6 features)
X = pd.read_csv('eeg_features_average.csv')
X = X.iloc[:,:6].values

# Initialize UMAP reducer
reducer = umap.UMAP(n_neighbors=5, n_components=2, random_state=42)

# Fit and transform
embedding = reducer.fit_transform(X)  # shape will be (40, 2)

# Plot
plt.close()
plt.scatter(embedding[:, 0], embedding[:, 1], c='k', s=30)
plt.tight_layout()
#plt.show()
plt.savefig('eeg_features_average_umap_plot.png', bbox_inches='tight', dpi=300)