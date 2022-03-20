import tensorflow as tf
import tensorflow.keras as keras
from src.encoder_decoder.training_loop import TrainingLoop
from src.utils.dataset_creators import QADataset

path = 'final_dataset_clean_v2 .tsv'

dataset_creator = QADataset(path)
gpu = False
if gpu:
        
    with tf.device('/GPU:1'):
        optimizer = keras.optimizers.Adam()
        training_loop = TrainingLoop(dataset_creator, optimizer, D = 14, frac=1, checkpoint_folder='checkpoint_first')

        training_loop.train(10, case = 'initial')
else:
    optimizer = keras.optimizers.Adam()
    training_loop = TrainingLoop(dataset_creator, optimizer, D = 14, frac=0.5, checkpoint_folder='checkpoint_first')

    training_loop.train(10, case = 'initial')

#Expanding dataset:


if gpu:
        
    with tf.device('/GPU:0'):
        optimizer = keras.optimizers.Adam()
        training_loop = TrainingLoop(dataset_creator, optimizer, D = 14, frac=0.5, checkpoint_folder='checkpoint_KG')

        training_loop.train(10, case = 'anchor')
else:
    optimizer = keras.optimizers.Adam()
    training_loop = TrainingLoop(dataset_creator, optimizer, D = 14, frac=0.5, checkpoint_folder='checkpoint_KG')

    training_loop.train(10, case = 'anchor')

