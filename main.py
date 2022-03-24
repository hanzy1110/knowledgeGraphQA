import os
import tensorflow as tf
import tensorflow.keras as keras
from src.encoder_decoder.training_loop import TrainingLoop
from src.utils.dataset_creators import QADataset

path = 'final_dataset_clean_v2 .tsv'
#os.remove('checkpoint_first')
max_epochs = 2

dataset_creator = QADataset(path)
gpu = False
if gpu:
        
    with tf.device('/GPU:1'):
        optimizer = keras.optimizers.Adam()
        training_loop = TrainingLoop(dataset_creator, optimizer, D = 14, frac=0.1, checkpoint_folder='checkpoint_first')

        training_loop.train(max_epochs, case = 'initial')
else:
    print('No GPU!')
    optimizer = keras.optimizers.Adam()
    training_loop = TrainingLoop(dataset_creator, optimizer, D = 14, frac=1, checkpoint_folder='checkpoint_first')

    training_loop.train(10, case = 'initial')

#Expanding dataset:


if gpu:
        
    with tf.device('/GPU:1'):
        optimizer = keras.optimizers.Adam()
        training_loop = TrainingLoop(dataset_creator, optimizer, D = 14, frac=0.5, checkpoint_folder='checkpoint_KG')
        training_loop.train(max_epochs, case = 'anchor')
        EM = training_loop.exact_match()
        F1 = training_loop.F1_metric(0.5, 0.1)
        with open('output.txt', 'w+') as file:
            file.write(f'Exact Match--->{EM}')
            file.write(f'F1 Metric--->{F1}')
else:
    optimizer = keras.optimizers.Adam()
    training_loop = TrainingLoop(dataset_creator, optimizer, D = 14, frac=0.1, checkpoint_folder='checkpoint_KG')

    training_loop.train(10, case = 'anchor')
    EM = training_loop.exact_match()
    F1 = training_loop.F1_metric(0.5, 0.1)
    with open('output.txt', 'w+') as file:
        file.write(f'Exact Match--->{EM}')
        file.write(f'F1 Metric--->{F1}')

