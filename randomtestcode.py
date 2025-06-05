import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

# Install this package first: pip install python-louvain
import community as community_louvain

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Build k-NN graph
k = 10
nn = NearestNeighbors(n_neighbors=k)
nn.fit(X_scaled)
knn_graph = nn.kneighbors_graph(X_scaled, mode='connectivity')

# Convert to NetworkX graph
G = nx.from_scipy_sparse_array(knn_graph)

# Detect communities using Louvain
partition = community_louvain.best_partition(G)

# Create DataFrame for analysis
df = pd.DataFrame(X_scaled, columns=iris.feature_names)
df['true_label'] = y
df['community'] = pd.Series(partition)

# Assign majority class label per community
community_to_label = {}
for comm_id in set(partition.values()):
    members = [n for n in partition if partition[n] == comm_id]
    labels = [y[n] for n in members]
    common_label = Counter(labels).most_common(1)[0][0]
    community_to_label[comm_id] = common_label

# Predict labels
df['predicted_label'] = df['community'].map(community_to_label)

# Calculate accuracy
accuracy = np.mean(df['predicted_label'] == df['true_label'])
print(f"Louvain community-based classification accuracy: {accuracy:.2f}")

# Visualize communities (in 2D PCA for simplicity)
from sklearn.decomposition import PCA
X_pca = PCA(n_components=2).fit_transform(X_scaled)
df['x'] = X_pca[:, 0]
df['y'] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
for comm_id in set(partition.values()):
    cluster = df[df['community'] == comm_id]
    plt.scatter(cluster['x'], cluster['y'], label=f'Community {comm_id}', alpha=0.7)
plt.title("Louvain Communities in PCA-reduced Space")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend()
plt.grid(True)
plt.show()