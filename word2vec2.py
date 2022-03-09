# word2vec from tensorflow
# %%
import io
import re
import string
import tqdm

import numpy as np
import tensorflow as tf
# from tensorflow.keras import layers
import tensorflow.keras as keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from src.utils.dataset_creators import QADataset

# %load_ext tensorboard

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.
class Word2Vec(keras.Model):
  def __init__(self, vocab_size, embedding_dim):
    super(Word2Vec, self).__init__()
    # Set the number of negative samples per positive context.
    num_ns = 4

    self.target_embedding = keras.layers.Embedding(vocab_size,
                                      embedding_dim,
                                      input_length=1,
                                      name="w2v_embedding")
    self.context_embedding = keras.layers.Embedding(vocab_size,
                                       embedding_dim,
                                       input_length=num_ns+1)

  def call(self, pair):
    target, context = pair
    # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
    # context: (batch, context)
    if len(target.shape) == 2:
      target = tf.squeeze(target, axis=1)
    # target: (batch,)
    word_emb = self.target_embedding(target)
    # word_emb: (batch, embed)
    context_emb = self.context_embedding(context)
    # context_emb: (batch, context, embed)
    dots = tf.einsum('be,bce->bc', word_emb, context_emb)
    # dots: (batch, context)
    return dots

def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
    # Elements of each training example are appended to these lists.
    targets, contexts, labels = [], [], []

    # Build the sampling table for `vocab_size` tokens.
    sampling_table = keras.preprocessing.sequence.make_sampling_table(vocab_size)

    # Iterate over all sequences (sentences) in the dataset.
    for sequence in tqdm.tqdm(sequences):
        # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = keras.preprocessing.sequence.skipgrams(sequence,
                                                                        vocabulary_size=vocab_size,
                                                                        sampling_table=sampling_table,
                                                                        window_size=window_size,
                                                                        negative_samples=0)

        # Iterate over each positive skip-gram pair to produce training examples
        # with a positive context word and negative samples.
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(tf.constant([context_word], dtype="int64"), 1)

            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                                                    true_classes=context_class,
                                                    num_true=1,
                                                    num_sampled=num_ns,
                                                    unique=True,
                                                    range_max=vocab_size,
                                                    seed=SEED,
                                                    name="negative_sampling")

            # Build context and label vectors (for one target word)
            negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)

            context = tf.concat([context_class, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0]*num_ns, dtype="int64")

            # Append each element from the training example to global lists.
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

    return targets, contexts, labels

# Now, create a custom standardization function to lowercase the text and
# remove punctuation.

path = 'final_dataset_clean_v2 .tsv'

dataset_creator = QADataset(path)

num_examples = -1
BUFFER_SIZE = 32000
BATCH_SIZE = 128

train_dataset, val_dataset, lang_tokenizer = dataset_creator.call(
    num_examples, BUFFER_SIZE, BATCH_SIZE)

sequences = []
for sequence in train_dataset.as_numpy_iterator():
    for seq in sequence['context']:
        sequences.append(seq)
    for seq in sequence['question']:
        sequences.append(seq)
    for seq in sequence['target']:
        sequences.append(seq)

vocab_size = len(lang_tokenizer.word_index) + 2

# %%
targets, contexts, labels = generate_training_data(
    sequences=sequences,
    window_size=2,
    num_ns=4,
    vocab_size=vocab_size,
    seed=SEED)

targets = np.array(targets)
contexts = np.array(contexts)[:, :, 0]
labels = np.array(labels)

print('\n')
print(f"targets.shape: {targets.shape}")
print(f"contexts.shape: {contexts.shape}")
print(f"labels.shape: {labels.shape}")

#%%
BATCH_SIZE = 1024
BUFFER_SIZE = 10000
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print(dataset)
dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
print(dataset)
#%%

def custom_loss(x_logit, y_true):
      return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)

tensorboard_callback = keras.callbacks.TensorBoard(log_dir="logs")

embedding_dim = 128
word2vec = Word2Vec(vocab_size, embedding_dim)
word2vec.compile(optimizer='adam',
                 loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])
                 
word2vec.fit(dataset, epochs=20, callbacks=[tensorboard_callback])


# %%
weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
vocab = lang_tokenizer.word_index

out_v = open('vectors.tsv', 'w', encoding='utf-8')
out_m = open('metadata.tsv', 'w', encoding='utf-8')

out_m.write('words' + '\t\n')

for index, word in enumerate(vocab):
  if index == 0:
    continue  # skip 0, it's padding.
  vec = weights[index]
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")
out_v.close()
out_m.close()

# %%
