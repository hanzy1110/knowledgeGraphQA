# %%
import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from src.encoder_decoder.encoder_decoder_model import AutoEncoder, loss_function
from tqdm import tqdm,trange

import src.utils.dataset_utils as du
from src.utils.dataset_creators import QADataset

sequence_length = 10

# The number of dimensions used to store data passed between recurrent layers in the network.
recurrent_cell_size = 128
# The number of dimensions in our word vectorizations.
D = 128

path = 'final_dataset_clean_v2 .tsv'

dataset_creator = QADataset(path)
num_examples = -1
BUFFER_SIZE = 32000
BATCH_SIZE = 128

train_dataset, val_dataset, lang_tokenizer = dataset_creator.call(
    num_examples, BUFFER_SIZE, BATCH_SIZE)

data_dict = next(iter(train_dataset))
print(data_dict['context'].shape, data_dict['question'].shape, data_dict['target'].shape)

vocab_inp_size = len(lang_tokenizer.word_index)+2
vocab_tar_size = len(lang_tokenizer.word_index)+2
max_length_input = data_dict['context'].shape[1]
max_length_output = data_dict['target'].shape[1]


embedding_dim = D
units = 1024
steps_per_epoch = num_examples//BATCH_SIZE

BATCH_SIZE = 128
with tf.device('/GPU:0'):
    # Model
    autoencoder = AutoEncoder(vocab_inp_size, D, D, BATCH_SIZE,
                    language_tokenizer=lang_tokenizer,
                    max_length_input=max_length_input,
                    max_length_output=max_length_output)

    optimizer = keras.optimizers.Adam()


    @tf.function
    def train_step(inp, targ, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:

            pred = autoencoder([inp, targ], enc_hidden)
            real = targ[:, 1:]         # ignore <start> token

            logits = pred.rnn_output
            loss = loss_function(real, logits)

            # variables = encoder.trainable_variables + decoder.trainable_variables
            variables = autoencoder.trainable_variables

            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))

        return loss

    EPOCHS = 3

    checkpointer = keras.callbacks.ModelCheckpoint('training_checkpoints')
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir="logs")
    _callbacks = [checkpointer, tensorboard_callback]

    callbacks = keras.callbacks.CallbackList(
        _callbacks, add_history=True, model=autoencoder)

    logs = {}
    callbacks.on_train_begin(logs=logs)

    # Presentation
    epochs = trange(
        EPOCHS,
        desc="Epoch",
        unit="Epoch",
        postfix="loss = {loss:.4f}, accuracy = {accuracy:.4f}")
    epochs.set_postfix(loss=0, accuracy=0)

    for epoch in epochs:

        callbacks.on_epoch_begin(epoch, logs=logs)

        start = time.time()

        enc_hidden = autoencoder.encoder.initialize_hidden_state()
        total_loss = 0
        # print(enc_hidden[0].shape, enc_hidden[1].shape)

        for (batch, data_dict) in tqdm(enumerate(train_dataset.take(steps_per_epoch))):

            callbacks.on_batch_begin(batch, logs=logs)
            callbacks.on_train_batch_begin(batch, logs=logs)

            inp = [data_dict['context'], data_dict['question']]
            targ = data_dict['target']

            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                            batch,
                                                            batch_loss.numpy()))
            # saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                callbacks.on_epoch_end()
                
        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


    callbacks.on_train_end(logs=logs)

    # Fetch the history object we normally get from keras.fit
    history_object = None
    for cb in callbacks:
        if isinstance(cb, keras.callbacks.History):
            history_object = cb
    assert history_object is not None