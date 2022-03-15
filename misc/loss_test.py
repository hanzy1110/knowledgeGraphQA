# %%
import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
from src.encoder_decoder.encoder_decoder_model import AutoEncoder, AnchorLoss
from tqdm import tqdm, trange

import src.utils.dataset_utils as du
from src.utils.dataset_creators import QADataset

sequence_length = 10

# The number of dimensions used to store data passed between recurrent layers in the network.
recurrent_cell_size = 10
# The number of dimensions in our word vectorizations.
D = 14

path = 'final_dataset_clean_v2 .tsv'

dataset_creator = QADataset(path)
num_examples = -1
BUFFER_SIZE = 32000
BATCH_SIZE = 128

train_dataset, val_dataset, lang_tokenizer = dataset_creator.call(
    num_examples, BUFFER_SIZE, BATCH_SIZE)

data_dict = next(iter(train_dataset))
print(data_dict['context'].shape,
      data_dict['question'].shape, data_dict['target'].shape)

vocab_inp_size = len(lang_tokenizer.word_index)+2
vocab_tar_size = len(lang_tokenizer.word_index)+2
max_length_input = data_dict['context'].shape[1]
max_length_output = data_dict['target'].shape[1]


embedding_dim = D
units = 1024
steps_per_epoch = num_examples//BATCH_SIZE

# Model
autoencoder = AutoEncoder(vocab_inp_size, D, D, BATCH_SIZE,
                            language_tokenizer=lang_tokenizer,
                            max_length_input=max_length_input,
                            max_length_output=max_length_output)

anchorloss = AnchorLoss(max_output_length=max_length_output, batch_size=BATCH_SIZE)

# enc_hidden = autoencoder.encoder.initialize_hidden_state()
total_loss = 0
# print(enc_hidden[0].shape, enc_hidden[1].shape)
#%%
for (batch, data_dict) in tqdm(enumerate(train_dataset.take(steps_per_epoch))):

    inp = [data_dict['context'], data_dict['question']]
    targ = data_dict['target']
    
    pred:tfa.seq2seq.BasicDecoderOutput = autoencoder([inp, targ])

    sequences = pred.sample_id
    sequences = autoencoder.encoder.embedding(sequences)

    batch_loss = anchorloss.loss(sequences)
    total_loss += batch_loss
    print(batch_loss, total_loss) 
# %%
