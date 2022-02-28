import pandas as pd
from src.knowledge_graph.knowledge_graph import knowledge_grapher

data = pd.read_excel(r'RoITD\final_dataset_clean_v2.xlsx', skiprows=[0])

grapher = knowledge_grapher(data)
data_kgf = grapher.extractTriples(-1)
grapher.buildGraph(data_kgf)
grapher.prepare_data(data_kgf)
grapher.plot_graph()



