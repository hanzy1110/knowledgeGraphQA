#%%
import tensorflow as tf
import tensorflow.keras as keras
from src.encoder_decoder.encoder_decoder_model import (Encoder, Decoder,
                                                     loss_function, 
                                                     beam_evaluate_sentence,
                                                     beam_answer)
from src.utils.dataset_creators import QADataset

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

D = 128
embedding_dim = 128
units = 1024
steps_per_epoch = num_examples//BATCH_SIZE

BATCH_SIZE = 128
# Model
encoder = Encoder(vocab_inp_size, D, D, BATCH_SIZE,
                language_tokenizer=lang_tokenizer)

decoder = Decoder(vocab_inp_size, D, D, BATCH_SIZE,
                language_tokenizer=lang_tokenizer,
                max_length_input=max_length_input,
                max_length_output=max_length_output)

optimizer = keras.optimizers.Adam()

#%%
checkpoint = tf.train.Checkpoint(
                                    optimizer=optimizer,
                                    encoder=encoder,
                                    decoder=decoder)
status = checkpoint.restore(tf.train.latest_checkpoint('training_checkpoints')).expect_partial()
status.assert_consumed()
# %%
