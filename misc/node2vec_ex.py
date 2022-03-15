#%%
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from node2vec import Node2Vec as n2v
from src.knowledge_graph.knowledge_graph import knowledge_grapher

WINDOW = 4 # Node2Vec fit window
MIN_COUNT = 1 # Node2Vec min. count
BATCH_WORDS = 10 # Node2Vec batch words

data = pd.read_csv(r'final_dataset_clean_v2 .tsv', delimiter = '\t')
grapher = knowledge_grapher(data)
grapher.load_data(r'pykeen_data\data_kgf.tsv')

g_emb = n2v(
  grapher.graph,
  dimensions=14
)

mdl = g_emb.fit(
    window=WINDOW,
    min_count=MIN_COUNT,
    batch_words=BATCH_WORDS
)

#%%
emb_df = pd.DataFrame([mdl.wv.get_vector(str(n)) for n in grapher.graph.nodes()],
        index = grapher.graph.nodes(data=False))

pca = PCA(n_components = 2, random_state = 7)
pca_mdl = pca.fit_transform(emb_df)

emb_df_PCA = pd.DataFrame(
        pca_mdl,
        columns=['x','y'],
        index = emb_df.index)

plt.clf()
fig = plt.figure(figsize=(12,5))
plt.scatter(
    x = emb_df_PCA['x'],
    y = emb_df_PCA['y'],
    s = 0.8,
    color = 'maroon',
    alpha = 0.5
)
plt.xlabel('PCA-1')
plt.ylabel('PCA-2')
plt.title('PCA Visualization')
plt.plot()

# %%
