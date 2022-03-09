#%%
import pandas as pd
import numpy as np
import tensorflow as tf

# import keras
import tensorflow.keras as keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
# from keras.layers import Embedding, Input, LSTM, Attention, Concatenate, TimeDistributed, Dense
# import keras.utils
import src.utils.dataset_utils as du

vocab_size = 4096
sequence_length = 10

# The number of dimensions used to store data passed between recurrent layers in the network.
recurrent_cell_size = 128
# The number of dimensions in our word vectorizations.
D = 128 
# How quickly the network learns. Too high, and we may run into numeric instability 
# or other issues.
learning_rate = 0.005
# Dropout probabilities. For a description of dropout and what these probabilities are, 
# see Entailment with TensorFlow.
input_p, output_p = 0.5, 0.5
# How many questions we train on at a time.
batch_size = 128
# Number of passes in episodic memory. We'll get to this later.
passes = 4
# Feed Forward layer sizes: the number of dimensions used to store data passed from feed-forward layers.
ff_hidden_size = 256
weight_decay = 0.00000001
# The strength of our regularization. Increase to encourage sparsity in episodic memory, 
# but makes training slower. Don't make this larger than leraning_rate.
training_iterations_count = 400000

#%%
# data = pd.read_excel(r'RoITD\final_dataset_clean_v2.xlsx', skiprows=[0])
data = pd.read_csv('final_dataset_clean_v2 .tsv', delimiter='\t', encoding="utf-8")
data_dict = du.parse_input(data, 0.7, 0.2)

vectorizer = TextVectorization(max_tokens=vocab_size, output_sequence_length=sequence_length)
text = np.hstack([data_dict['joined_data']['train'], data_dict['answers']['train']])

#%%
text_ds = tf.data.Dataset.from_tensor_slices(text).batch(128)
vectorizer.adapt(text_ds)

voc = vectorizer.get_vocabulary()
num_tokens = len(voc) + 2
word_index = dict(zip(voc, range(len(voc))))
embedding_matrix = du.get_embedding_matrix(word_index)

# prepared_data = {}
prepared_data = {key: du.vectorize_text(aux, vectorizer) for key, aux in data_dict.items()}


# input_dict = {'input':{}}
# input_dict['input']['train'] = tf.concat([prepared_data['context']['train'], prepared_data['questions']['train']], axis = 1)    
# input_dict['input']['test'] = tf.concat([prepared_data['context']['test'], prepared_data['questions']['test']], axis = 1)    
# input_dict['input']['validate'] = tf.concat([prepared_data['context']['validate'], prepared_data['questions']['validate']], axis=1)    


# %%
#Input module
embedding_layer = keras.layers.Embedding(
    num_tokens,
    D,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False, name = "Embedder"
)

int_input_sequence = keras.Input(shape = (sequence_length,), 
                        dtype=tf.int32, name = 'int_input_sequence')

context_embedded = embedding_layer(int_input_sequence)
#encoder lstm 1
encoder_lstm = keras.layers.LSTM(D, return_sequences=True,return_state=True, name = "encoder_lstm")
encoder_output, state_h, state_c = encoder_lstm(context_embedded)

encoder_model = keras.Model(inputs = int_input_sequence, outputs = encoder_output, name="encoder_model")

keras.utils.plot_model(encoder_model, "encoder_model_with_shape_info.png", show_shapes=True)
encoder_state = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = keras.Input(shape=(sequence_length,), dtype='int32', name = "decoder_inputs")

#embedding layer
dec_emb = embedding_layer(decoder_inputs)
decoder_lstm = keras.layers.LSTM(D, return_sequences=True, return_state=True, name = "decoder_lstm")
decoder_output, decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb, initial_state=encoder_state)

# Attention layer
attn_out = keras.layers.Attention()([encoder_output, decoder_output])
# Concat attention input and decoder LSTM output
decoder_concat_input = keras.layers.Concatenate(axis=-1, name='concat_layer')([decoder_output, attn_out])
#dense layer
decoder_dense =  keras.layers.TimeDistributed(keras.layers.Dense(num_tokens, activation='softmax'),name = 'answer')
# decoder_dense =  keras.layers.Dense(num_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_output)

decoder_model = keras.Model(inputs = [decoder_inputs, int_input_sequence], outputs = decoder_outputs, name = "Decoder_model")
keras.utils.plot_model(decoder_model, "decoder_model_with_shape_info.png", show_shapes=True)

# Define the model 
# model = keras.Model([[context_seq_input, question_seq_input], decoder_inputs], decoder_outputs)
# model = keras.Model([context_seq_input, decoder_inputs], decoder_outputs)
autoencoder_input = keras.Input(shape = (sequence_length,),dtype=tf.int32, name = 'autoencoder_input')
encoded_state_sequence = encoder_model(autoencoder_input)
final_decoder_output = decoder_model([encoded_state_sequence, autoencoder_input]) 

model = keras.Model(inputs = autoencoder_input, outputs = final_decoder_output, name="Complete_model")

#%%
model.summary()

model.compile(
    loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["acc", "mae"]
)

keras.utils.plot_model(model, "complete_model_with_shape_info.png", show_shapes=True)

tensorboard_callback = keras.callbacks.TensorBoard(log_dir="QA")

# c_train = prepared_data['context']['train']
# q_train = prepared_data['questions']['train']
# a_train = prepared_data['answers']['train']

# c_validate = prepared_data['context']['validate']
# q_validate = prepared_data['questions']['validate']
# a_validate = prepared_data['answers']['validate']

c_train = prepared_data['joined_data']['train']
a_train = prepared_data['answers']['train']

c_validate = prepared_data['joined_data']['validate']
a_validate = prepared_data['answers']['validate']

# dec_input = tf.random.normal(shape = c_train.shape)
# dec_input = tf.Tensor([])
# out = encoder_model({"context":c_train ,"question": q_train})

#%%
model.fit({"context":c_train},
        {"answer": a_train},
        batch_size=128, epochs=5, 
        validation_data=({'context':c_validate},
                        {'answer': a_validate}),
        callbacks=[tensorboard_callback])

# model.fit({"context":c_train ,"question": q_train},
#         {"answer": a_train},
#         batch_size=128, epochs=5, 
#         validation_data=({'context':c_validate, 'question': q_validate},
#                         {'answer': a_validate}),
        # callbacks=[tensorboard_callback])
