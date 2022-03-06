import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from node2vec import Node2Vec as n2v
from src.knowledge_graph.knowledge_graph import knowledge_grapher

WINDOW = 1 # Node2Vec fit window
MIN_COUNT = 1 # Node2Vec min. count
BATCH_WORDS = 4 # Node2Vec batch words

data = pd.read_csv(r'final_dataset_clean_v2 .tsv', delimiter = '\t')

grapher = knowledge_grapher(data)
# data_kgf = grapher.extractTriples(-1)
# grapher.buildGraph(data_kgf)

grapher.load_data(r'pykeen_data\data_kgf.tsv')

g_emb = n2v(
  grapher.graph,
  dimensions=128
)

mdl = g_emb.fit(
    window=WINDOW,
    min_count=MIN_COUNT,
    batch_words=BATCH_WORDS
)

input_node = 'care'
for s in mdl.wv.most_similar(input_node, topn = 10):
    print(s)