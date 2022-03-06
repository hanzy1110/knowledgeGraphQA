
from typing import Tuple
import spacy
import pandas as pd
import numpy as np
from spacy.matcher import Matcher 
import matplotlib.pyplot as plot
from tqdm import tqdm
import networkx as ntx

class knowledge_grapher():
    def __init__(self, data, load_spacy:bool=False) -> None:
        if load_spacy:
            self.nlp = spacy.load("ro_core_news_sm")
        self.data = data
        
    def extract_entities(self, sents)->pd.DataFrame:
        # chunk one
        enti_one = ""
        enti_two = ""
        
        dep_prev_token = "" # dependency tag of previous token in sentence
        txt_prev_token = "" # previous token in sentence
        
        prefix = ""
        modifier = ""

        for tokn in self.nlp(sents):
            # chunk two
            ## move to next token if token is punctuation
            
            if tokn.dep_ != "punct":
                #  check if token is compound word or not
                if tokn.dep_ == "compound":
                    prefix = tokn.text
                    # add the current word to it if the previous word is 'compound’
                    if dep_prev_token == "compound":
                        prefix = txt_prev_token + " "+ tokn.text
                        
                # verify if token is modifier or not
                if tokn.dep_.endswith("mod") == True:
                    modifier = tokn.text
                    # add it to the current word if the previous word is 'compound'
                    if dep_prev_token == "compound":
                        modifier = txt_prev_token + " "+ tokn.text
                        
                # chunk3
                if tokn.dep_.find("subj") == True:
                    enti_one = modifier +" "+ prefix + " "+ tokn.text
                    prefix = ""
                    modifier = ""
                    dep_prev_token = ""
                    txt_prev_token = ""
                    
                # chunk4
                if tokn.dep_.find("obj") == True:
                    enti_two = modifier +" "+ prefix +" "+ tokn.text
                    
                # chunk 5
                # update variable
                dep_prev_token = tokn.dep_
                txt_prev_token = tokn.text
                
        return [enti_one.strip(), enti_two.strip()]

    def obtain_relation(self,sent):
    
        doc = self.nlp(sent)
        
        matcher = Matcher(self.nlp.vocab)
        
        pattern = [{'DEP':'ROOT'},
                {'DEP':'prep','OP':"?"},
                {'DEP':'agent','OP':"?"}, 
                {'POS':'ADJ','OP':"?"}]
        
        matcher.add(key="matching_1", patterns = [pattern])
        
        matcher = matcher(doc)
        h = len(matcher) - 1
        
        try:
            assert matcher    
            span = doc[matcher[h][1]:matcher[h][2]]
            return span.text
        except AssertionError:
            print('No match found for this entry!')
            return None
                
    def extractTriples(self, max_text) -> pd.DataFrame:
        pairs_of_entities = [self.extract_entities(i) for i in tqdm(self.data['paragraph'][:max_text])]
        relations = [self.obtain_relation(j) for j in tqdm(self.data['paragraph'][:max_text])]
        indexes = [x for x, z in enumerate(relations) if z is not None]

        relations = [x for x in relations if x is not None]
        
        # subject extraction
        source = [j[0] for j in pairs_of_entities]
        source = [source[i] for i in indexes]
        #object extraction
        target = [k[1] for k in pairs_of_entities]
        target = [target[i] for i in indexes]
        
        return pd.DataFrame({'source':source, 'target':target, 'edge':relations})

    def buildGraph(self, data_kgf, relation = None)->None:
        if relation:
            self.graph = ntx.from_pandas_edgelist(data_kgf[data_kgf['edge']==relation], "source", "target",
                         edge_attr=True, create_using=ntx.MultiDiGraph())
        else:
            self.graph = ntx.from_pandas_edgelist(data_kgf, "source", "target",
                         edge_attr=True, create_using=ntx.MultiDiGraph())

    def plot_graph(self)->None:

        plot.figure(figsize=(14, 14))
        posn = ntx.spring_layout(self.graph)
        ntx.draw(self.graph, with_labels=True, node_color='green', edge_cmap=plot.cm.Blues, pos = posn)
        plot.savefig('plots/graph_plot.png')
        plot.close()

    def compute_centrality(self,)->None:
        self.centrality_dict = ntx.degree_centrality(self.graph)
        self.in_centrality_dict = ntx.in_degree_centrality(self.graph)
        self.out_centrality_dict = ntx.out_degree_centrality(self.graph)
        # self.eigenvector_centrality_dict = ntx.katz_centrality(self.graph)

    def load_data(self, path)->None:
        data_kgf = pd.read_csv(path, delimiter='\t')
        self.buildGraph(data_kgf)
    
    def get_centers(self, max_centers:int=5)->None:
        sorted_dict = sorted(self.centrality_dict.items(), key=lambda x: x[1])[::-1]
        in_sorted_dict = sorted(self.in_centrality_dict.items(), key=lambda x: x[1])[::-1]
        out_sorted_dict = sorted(self.out_centrality_dict.items(), key=lambda x: x[1])[::-1]

        degree_centers = sorted_dict[:max_centers]
        in_degree_centers = in_sorted_dict[:max_centers]
        out_degree_centers = out_sorted_dict[:max_centers]

        self.degree_adjacency = {u:self.graph[u] for u,_ in degree_centers}
        self.in_degree_adjacency = {u:self.graph[u] for u,_ in in_degree_centers}
        self.out_degree_adjacency = {u:self.graph[u] for u,_ in out_degree_centers}

    def prepare_data(self, data_kgf:pd.DataFrame)->Tuple[pd.DataFrame]:
        
        SAMPLES = len(data_kgf.index)

        TRAIN_SPLIT = int(0.5 * SAMPLES)
        TEST_SPLIT = int(0.3 * SAMPLES)
        VALIDATION_SPLIT = int(0.2 * SAMPLES)

        train_indexes = np.random.randint(low = 0, high = len(data_kgf.index), size=TRAIN_SPLIT)
        test_indexes = np.random.randint(low = 0, high = len(data_kgf.index), size=TEST_SPLIT)
        validation_indexes = np.random.randint(low = 0, high = len(data_kgf.index), size=VALIDATION_SPLIT)

        train_df = data_kgf.iloc[train_indexes]
        test_df = data_kgf.iloc[test_indexes]
        val_df = data_kgf.iloc[validation_indexes]

        train_df.to_csv('pykeen_data/train_data.tsv', sep="\t")
        test_df.to_csv('pykeen_data/test_data.tsv', sep="\t")
        val_df.to_csv('pykeen_data/validation_data.tsv', sep="\t")
        data_kgf.to_csv('pykeen_data/data_kgf.tsv', sep="\t")
        
        return train_df, test_df, val_df