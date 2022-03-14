# %%
import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
from src.encoder_decoder.encoder_decoder_model import AutoEncoder, AnchorLoss, beam_translate 
from tqdm import tqdm, trange

from src.utils.dataset_creators import QADataset

sequence_length = 10

# The number of dimensions used to store data passed between recurrent layers in the network.
recurrent_cell_size = 74
# The number of dimensions in our word vectorizations.
D = 14

path = 'final_dataset_clean_v2 .tsv'

dataset_creator = QADataset(path)
num_examples = -1
BUFFER_SIZE = 32000
BATCH_SIZE = 12
BETA = 0.03

train_dataset, val_dataset, lang_tokenizer = dataset_creator.call(
    num_examples, BUFFER_SIZE, BATCH_SIZE)

data_dict = next(iter(train_dataset))
print(data_dict['context'].shape,
      data_dict['question'].shape, data_dict['target'].shape)

vocab_inp_size = len(lang_tokenizer.word_index)+2
vocab_tar_size = len(lang_tokenizer.word_index)+2
max_length_input = data_dict['context'].shape[1]
max_length_input_q = data_dict['question'].shape[1]
max_length_output = data_dict['target'].shape[1]


embedding_dim = D
units = 512
steps_per_epoch = num_examples//BATCH_SIZE
#%
# Model
#%%
optimizer = keras.optimizers.Adam()
autoencoder = AutoEncoder(vocab_inp_size, D, D, BATCH_SIZE,
                            language_tokenizer=lang_tokenizer,
                            max_length_input=max_length_input,
                            max_length_output=max_length_output)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, autoencoder=autoencoder)

# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint,checkpoint_dir, 3, keep_checkpoint_every_n_hours=None,
    checkpoint_name='ckpt', step_counter=None, checkpoint_interval=None,
    init_fn=None
)
status = checkpoint_manager.restore_or_initialize()
# %%
for (batch, data_dict) in tqdm(enumerate(val_dataset.take(steps_per_epoch))):
    context = data_dict['context']
    question = data_dict['question']
    answer = data_dict['target']
    # context = lang_tokenizer.sequences_to_texts_generator(context.numpy())
    # question = lang_tokenizer.sequences_to_texts_generator(question.numpy())
    # answer = lang_tokenizer.sequences_to_texts_generator(answer.numpy())
    # print('context', context) 
    # print('question', question) 
    # print('answer', answer)
    beam_translate(context, question, answer, D,
                dataset_creator = dataset_creator, lang_tokenizer = lang_tokenizer, 
                autoencoder = autoencoder, 
                max_length_input=max_length_input, max_length_output=max_length_output)


# %%
