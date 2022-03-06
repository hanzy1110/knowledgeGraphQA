#%%
import pandas as pd
import matplotlib.pyplot as plt
from src.knowledge_graph.knowledge_graph import knowledge_grapher

data = pd.read_csv(r'final_dataset_clean_v2 .tsv', delimiter = '\t')

grapher = knowledge_grapher(data)
# data_kgf = grapher.extractTriples(-1)
# grapher.buildGraph(data_kgf)

grapher.load_data(r'pykeen_data\data_kgf.tsv')

#%%
grapher.compute_centrality()
grapher.get_centers()

#%%
#grapher.prepare_data(data_kgf)
#grapher.plot_graph()
plt.plot(list(grapher.in_centrality_dict.values()), label = 'In_centrality')
# plt.plot(list(grapher.centrality_dict.values()), label = 'centrality')
plt.plot(list(grapher.out_centrality_dict.values()), label = 'out_centrality')
plt.legend()


# %%
