from cmath import isclose
import numpy as np
import pandas as pd
import tensorflow as tf

from itertools import combinations
from ..knowledge_graph.knowledge_graph import knowledge_grapher


def helper(arr):
    if any(arr[~np.isnan(arr)]):
        return True
    else:
        return False

class AnchorLoss():
    def __init__(self, max_output_length, batch_size) -> None:

        self.batch_size = batch_size
        self.max_output_length = max_output_length
        data = pd.read_csv('final_dataset_clean_v2 .tsv', delimiter = '\t')
        self.grapher = knowledge_grapher(data)
        self.grapher.load_data('pykeen_data/data_kgf.tsv')
        self.grapher.compute_centrality()
        self.grapher.get_centers()
        self.grapher.load_embeddings('KGWeights/weights.csv')
        self.grapher.map_centers_anchors('in_degree')
        
    def compute_denominator(self):
        centers = [_dict['center'] for _, _dict in self.grapher.mean_anchor_dict.items()]
        combinations_ = combinations(centers, 2)
        d = 0
        for arr1, arr2 in combinations_:
            arr1 = tf.convert_to_tensor(arr1, dtype=tf.float32)
            arr2 = tf.convert_to_tensor(arr2, dtype=tf.float32)
            d += tf.norm(arr1-arr2)
        return d

    def inner_loop(self, embedding):
        n = len(list(self.grapher.mean_anchor_dict.keys())) 
        tensor = np.ndarray((n,)) 
        for i, (key, arrdict) in enumerate(self.grapher.mean_anchor_dict.items()):
            if helper(arrdict['anchor']): 
                center = tf.convert_to_tensor(arrdict['center'], dtype=tf.float32)
                anchor = tf.convert_to_tensor(arrdict['anchor'], dtype=tf.float32)

                d1 = tf.norm(center-embedding)
                d2 = tf.norm(center-anchor)
                tensor[i] = d1+d2
            else:
                continue
        return tf.reduce_sum(tf.convert_to_tensor(tensor))

    def mid_loop(self, vect):
        tensor = np.ndarray((self.max_output_length,)) 
        for i, embedding in enumerate(tf.unstack(vect)):
            tensor[i] = self.inner_loop(embedding)

        return tf.reduce_sum(tf.convert_to_tensor(tensor))
    
    def lambdaBF(self, logits:tf.Tensor):
        norms = np.ndarray((self.batch_size, self.max_output_length))
        for i, out in enumerate(tf.unstack(logits)):
            for j, logit in enumerate(tf.unstack(out)):
                norms[i, j] = tf.norm(logit)
        _max = np.max(norms.sum(axis = 1)) 
        _min = np.min(norms.sum(axis = 1))
        return _min/_max if not np.isclose(_max,0, rtol=1e-5) else 1

    def loss(self, batch:tf.Tensor, logits:tf.Tensor):

        lambdaBF = self.lambdaBF(logits)
        denominator = self.compute_denominator()
        batch_loss = np.ndarray((self.batch_size,)) 
        for i, vect in enumerate(tf.unstack(batch)):
            batch_loss[i] = self.mid_loop(vect)

        batch_loss = tf.convert_to_tensor(batch_loss)

        return lambdaBF * tf.reduce_mean(batch_loss)/tf.cast(denominator, dtype=tf.float64)


