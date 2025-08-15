import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
df = pd.read_csv('cleaned_survey.csv')
df_filtered = df[df['tech_company'] == 'Yes']
if len(df_filtered) == 0:
    
    df_filtered = df.copy()
df = df_filtered
features = ['Gender', 'self_employed', 'family_history', 'treatment',
            'work_interfere', 'remote_work', 'benefits', 'care_options',
            'wellness_program', 'seek_help', 'leave', 'mental_health_consequence',
            'coworkers', 'supervisor', 'mental_health_interview']
X = df[features]
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols),
    ('num', StandardScaler(), num_cols)
])
X_processed = preprocessor.fit_transform(X)
n_components_pca = min(10, X_processed.shape[1])
X_pca = PCA(n_components=n_components_pca, random_state=42).fit_transform(X_processed)
tsne = TSNE(n_components=2, perplexity=40, learning_rate=200,
            max_iter=2000, init='pca', random_state=42)
X_tsne = tsne.fit_transform(X_pca)
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X_tsne)
score = silhouette_score(X_tsne, labels)
print(f"Silhouette Score for k={n_clusters}: {score:.3f}")
plt.figure(figsize=(10, 7))
for i in range(n_clusters):
    plt.scatter(X_tsne[labels == i, 0], X_tsne[labels == i, 1],
                label=f'Cluster {i}', s=60, edgecolors='k', alpha=0.8)
plt.title(f"2D t-SNE Clustering (k={n_clusters}, Silhouette = {score:.3f})")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
