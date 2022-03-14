import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
from itertools import combinations
from ..utils.dataset_utils import get_embedding_matrix
from ..utils.dataset_creators import QADataset
from ..knowledge_graph.knowledge_graph import knowledge_grapher

class Encoder(keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, language_tokenizer: keras.preprocessing.text.Tokenizer):

        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units

        self.concat = keras.layers.Concatenate(axis=-1)

        self.embedding_matrix = get_embedding_matrix(
            language_tokenizer.word_index)

        print('embedding shape', self.embedding_matrix.shape[0])
        print('vocab size', vocab_size)
        assert self.embedding_matrix.shape[0] == vocab_size

        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim,
                                                embeddings_initializer=keras.initializers.Constant(
                                                    self.embedding_matrix),
                                                trainable=True, name="Embedder")

        # self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)
        ##-------- LSTM layer in Encoder ------- ##
        self.lstm_layer = keras.layers.LSTM(self.enc_units,
                                            return_sequences=True,
                                            return_state=True,
                                            recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.concat(x)
        x = self.embedding(x)
        output, h, c = self.lstm_layer(x, initial_state=hidden)
        return output, h, c

    def initialize_hidden_state(self):
        return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))]


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, max_length_input,
                 max_length_output, language_tokenizer: keras.preprocessing.text.Tokenizer, attention_type='luong'):

        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.attention_type = attention_type
        self.max_length_output = max_length_output
        # Embedding Layer
        # voc = language_tokenizer.get_vocabulary()

        word_index = language_tokenizer.word_index

        self.embedding_matrix = get_embedding_matrix(word_index)
        print('embedding shape', self.embedding_matrix.shape[0])
        print('vocab size', vocab_size)
        assert self.embedding_matrix.shape[0] == vocab_size

        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim,
                                                embeddings_initializer=keras.initializers.Constant(
                                                    self.embedding_matrix),
                                                trainable=True, name="Embedder")

        # self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)

        # Final Dense layer on which softmax will be applied
        self.fc = keras.layers.Dense(vocab_size)

        # Define the fundamental cell for decoder recurrent structure
        self.decoder_rnn_cell = keras.layers.LSTMCell(self.dec_units)

        # Sampler
        self.sampler = tfa.seq2seq.TrainingSampler()

        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(self.dec_units,
                                                                  None, self.batch_sz*[max_length_input], self.attention_type)

        # Wrap attention mechanism with the fundamental rnn cell of decoder
        self.rnn_cell = self.build_rnn_cell(batch_sz)

        # Define the decoder with respect to fundamental rnn cell
        self.decoder = tfa.seq2seq.BasicDecoder(
            self.rnn_cell, sampler=self.sampler, output_layer=self.fc)

    def build_rnn_cell(self, batch_sz):
        rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnn_cell,
                                                self.attention_mechanism, attention_layer_size=self.dec_units)
        return rnn_cell

    def build_attention_mechanism(self, dec_units, memory, memory_sequence_length, attention_type='luong'):
        # ------------- #
        # typ: Which sort of attention (Bahdanau, Luong)
        # dec_units: final dimension of attention outputs
        # memory: encoder hidden states of shape (batch_size, max_length_input, enc_units)
        # memory_sequence_length: 1d array of shape (batch_size) with every element set to max_length_input (for masking purpose)

        if(attention_type == 'bahdanau'):
            return tfa.seq2seq.BahdanauAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)
        else:
            return tfa.seq2seq.LuongAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)

    def build_initial_state(self, batch_sz, encoder_state, Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(
            batch_size=batch_sz, dtype=Dtype)
        decoder_initial_state = decoder_initial_state.clone(
            cell_state=encoder_state)
        return decoder_initial_state

    def call(self, inputs, initial_state):

        x = self.embedding(inputs)
        outputs, _, _ = self.decoder(
            x, initial_state=initial_state, sequence_length=self.batch_sz*[self.max_length_output-1])
        return outputs

class AutoEncoder(keras.Model):
    
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, max_length_input,
                 max_length_output, language_tokenizer: keras.preprocessing.text.Tokenizer, attention_type='luong'):
        
        super(AutoEncoder, self).__init__()
        self.batch_sz = batch_sz
        self.encoder = Encoder(vocab_size, embedding_dim, dec_units, batch_sz, language_tokenizer)
        self.decoder = Decoder(vocab_size, embedding_dim, dec_units, batch_sz, max_length_input,
                 max_length_output, language_tokenizer, attention_type)

    def call(self, inputs):
        initial_state = self.encoder.initialize_hidden_state()
        enc_output, enc_h, enc_c = self.encoder(inputs[0], initial_state)

        dec_input = inputs[1][:, :-1]  # Ignore <end> token
        # real = targ[:, 1:]         # ignore <start> token

        # Set the AttentionMechanism object with encoder_outputs
        self.decoder.attention_mechanism.setup_memory(enc_output)

        # Create AttentionWrapperState as initial_state for decoder
        decoder_initial_state = self.decoder.build_initial_state(
        self.batch_sz , [enc_h, enc_c], tf.float32)
        
        return self.decoder(dec_input, decoder_initial_state)

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
        
#O(n**3) not nice
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
        
    def loss(self, batch:tf.Tensor):
        denominator = self.compute_denominator()
        batch_loss = np.ndarray((self.batch_size,)) 
        for i, vect in enumerate(tf.unstack(batch)):
            batch_loss[i] = self.mid_loop(vect)

        batch_loss = tf.convert_to_tensor(batch_loss)
        return tf.reduce_mean(batch_loss)/tf.cast(denominator, dtype=tf.float64)

def helper(arr):
    if any(arr[~np.isnan(arr)]):
        return True
    else:
        return False


def loss_function(real, pred):
    # real shape = (BATCH_SIZE, max_length_output)
    # pred shape = (BATCH_SIZE, max_length_output, tar_vocab_size )
    cross_entropy = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = cross_entropy(y_true=real, y_pred=pred)
    # output 0 for y=0 else output 1
    mask = tf.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=loss.dtype)
    loss = mask * loss
    loss = tf.reduce_mean(loss)
    return loss

def helper2(sentence:str, dataset_creator:QADataset, lang_tokenizer, max_length_input):
    question = dataset_creator.preprocess_sentence(question)

    inputs = [lang_tokenizer.word_index[i] for i in sentence.split(' ')]
    inputs = keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_input,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)
    return inputs


def beam_evaluate_sentence(context: str, question:str, units: int,
                           dataset_creator: QADataset, lang_tokenizer,
                           autoencoder:AutoEncoder,
                           max_length_input: int, max_length_output: int,
                           beam_width=3):

    # context = helper2(context, dataset_creator=dataset_creator, 
    #                 lang_tokenizer=lang_tokenizer, 
    #                 max_length_input=max_length_input)
    # question = helper2(question, dataset_creator=dataset_creator, 
    #                 lang_tokenizer=lang_tokenizer, 
    #                 max_length_input=max_length_input)
    
    inference_batch_size = context.shape[0]
    result = ''

    enc_start_state = [tf.zeros((inference_batch_size, units)), tf.zeros(
        (inference_batch_size, units))]
    inputs = [context, question]

    enc_out, enc_h, enc_c = autoencoder.encoder(inputs, enc_start_state)

    dec_h = enc_h
    dec_c = enc_c

    start_tokens = tf.fill([inference_batch_size],
                           lang_tokenizer.word_index['<start>'])
    end_token = lang_tokenizer.word_index['<end>']

    # From official documentation
    # NOTE If you are using the BeamSearchDecoder with a cell wrapped in AttentionWrapper, then you must ensure that:
    # The encoder output has been tiled to beam_width via tfa.seq2seq.tile_batch (NOT tf.tile).
    # The batch_size argument passed to the get_initial_state method of this wrapper is equal to true_batch_size * beam_width.
    # The initial state created with get_initial_state above contains a cell_state value containing properly tiled final state from the encoder.

    enc_out = tfa.seq2seq.tile_batch(enc_out, multiplier=beam_width)
    autoencoder.decoder.attention_mechanism.setup_memory(enc_out)
    print(
        "beam_with * [batch_size, max_length_input, rnn_units] :  3 * [1, 16, 1024]] :", enc_out.shape)

    # set decoder_inital_state which is an AttentionWrapperState considering beam_width
    hidden_state = tfa.seq2seq.tile_batch(
        [enc_h, enc_c], multiplier=beam_width)
    decoder_initial_state = autoencoder.decoder.rnn_cell.get_initial_state(
        batch_size=beam_width*inference_batch_size, dtype=tf.float32)
    decoder_initial_state = decoder_initial_state.clone(
        cell_state=hidden_state)

    # Instantiate BeamSearchDecoder
    decoder_instance = tfa.seq2seq.BeamSearchDecoder(
        autoencoder.decoder.rnn_cell, beam_width=beam_width, output_layer=autoencoder.decoder.fc)
    decoder_embedding_matrix = autoencoder.decoder.embedding_matrix

    # The BeamSearchDecoder object's call() function takes care of everything.
    outputs, final_state, sequence_lengths = decoder_instance(
        decoder_embedding_matrix, start_tokens=start_tokens, end_token=end_token, initial_state=decoder_initial_state)
    # outputs is tfa.seq2seq.FinalBeamSearchDecoderOutput object.
    # The final beam predictions are stored in outputs.predicted_id
    # outputs.beam_search_decoder_output is a tfa.seq2seq.BeamSearchDecoderOutput object which keep tracks of beam_scores and parent_ids while performing a beam decoding step
    # final_state = tfa.seq2seq.BeamSearchDecoderState object.
    # Sequence Length = [inference_batch_size, beam_width] details the maximum length of the beams that are generated

    # outputs.predicted_id.shape = (inference_batch_size, time_step_outputs, beam_width)
    # outputs.beam_search_decoder_output.scores.shape = (inference_batch_size, time_step_outputs, beam_width)
    # Convert the shape of outputs and beam_scores to (inference_batch_size, beam_width, time_step_outputs)
    final_outputs = tf.transpose(outputs.predicted_ids, perm=(0, 2, 1))
    beam_scores = tf.transpose(
        outputs.beam_search_decoder_output.scores, perm=(0, 2, 1))

    return final_outputs.numpy(), beam_scores.numpy()


def beam_translate(context: str, question:str,answer:str, units: int,
                           dataset_creator: QADataset, lang_tokenizer,
                           autoencoder:AutoEncoder,
                           max_length_input: int, max_length_output: int,
                           beam_width=3):

    result, beam_scores = beam_evaluate_sentence(context, question, units,
                           dataset_creator,lang_tokenizer=lang_tokenizer,
                           autoencoder=autoencoder,
                            max_length_input=max_length_input, max_length_output=max_length_output,
                           beam_width=beam_width)

    print(result.shape, beam_scores.shape)

    for beam, score in zip(result, beam_scores):

        print(beam.shape, score.shape)
        output = lang_tokenizer.sequences_to_texts(beam)
        output = [a[:a.index('<end>')] for a in output]
        beam_score = [a.sum() for a in score]

        print(f'Context : {next(lang_tokenizer.sequences_to_texts_generator(context.numpy()))}'.replace("<OOV>", ''))
        print(f'Question {next(lang_tokenizer.sequences_to_texts_generator(question.numpy()))}'.replace("<OOV>", ''))
        print(f'Expected Answer: {next(lang_tokenizer.sequences_to_texts_generator(answer.numpy()))}'.replace("<OOV>", ''))

        for i in range(len(output)):
            print('{} Predicted translation: {}  {}'.format(
                i+1, output[i], beam_score[i]))
