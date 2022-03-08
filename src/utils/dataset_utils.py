import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

# import tensorflow_text as tf_text
from typing import Dict, List, Tuple
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


def get_word_embeddings():
    word_path = os.path.join('skipgramModelMetadata','metadata.tsv')
    vector_path = os.path.join('skipgramModelMetadata','vectors.tsv')
    embeddings_index = {}
    words = pd.read_csv(word_path, sep='\t')['words   '].values

    with open(vector_path, 'r') as file:
        for line, word in zip(file, words):

            coefs = np.fromstring(line, "f", sep=" ")
            embeddings_index[word] = coefs
    
    # s = np.vstack([arr for key, arr in embeddings_index.items()])
    return embeddings_index

# embeddings = get_word_embeddings()

def get_embedding_matrix(word_index:Dict[str,str])->np.ndarray:

    embeddings = get_word_embeddings()

    num_tokens = len(set(word_index.keys())) + 2
    embedding_dim = 128
    hits = 0
    misses = 0

    # Prepare embedding matrix
    # Unknown words get the zero vector they should get normal noise!
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))
    return embedding_matrix


def parse_input(data:pd.DataFrame, train_test_cut, val_cut)->Dict[str, Dict[str, np.ndarray]]:
    aux = []
    for idx in data.index:
        aux.append(str(data.iloc[idx]['paragraph']) + "question" + str(data.iloc[idx]['question']))

    aux = np.asanyarray(aux, dtype=np.str_)

    context = np.asarray(data['paragraph'].values, dtype=np.str_)
    # questions = np.asarray(data['question'].values, dtype=np.str_)
    answers = np.asarray(data['answer'].values, dtype=np.str_)

    SAMPLES = len(context)

    TRAIN_SPLIT = int(train_test_cut* SAMPLES)
    TEST_SPLIT = int(val_cut * SAMPLES + TRAIN_SPLIT)


    context_train, context_validate, context_test = np.split(aux, [TRAIN_SPLIT, TEST_SPLIT])
    # questions_train, questions_validate, questions_test = np.split(questions, [TRAIN_SPLIT, TEST_SPLIT])
    answers_train, answers_validate, answers_test = np.split(answers, [TRAIN_SPLIT, TEST_SPLIT])

    # return {
    #     'context':{'train':context_train, 'validate': context_validate, 'test':context_test},
    #     'questions':{'train':questions_train, 'validate': questions_validate, 'test':questions_test},
    #     'answers':{'train':answers_train, 'validate': answers_validate, 'test':answers_test},
    #     'joined_data':{}
    # }
    return {
        'answers':{'train':answers_train, 'validate': answers_validate, 'test':answers_test},
        'joined_data':{'train':context_train, 'validate': context_validate, 'test':context_test}
    }

def join_questions_answers(data_dict:Dict[str, Dict[str, np.ndarray]], label:str):
    contexts = data_dict['context'][label]
    questions = data_dict['questions'][label]

    out = np.ndarray(shape = contexts.shape, dtype=np.str_)
    for i, (context, question) in enumerate(zip(contexts, questions)):
       out[i] = context + " question " + question
    data_dict['joined_data'][label] = out
    return data_dict 


def vectorize_text(data:Dict[str, np.ndarray], vectorizer)->np.ndarray:
    # Define the vocabulary size and the number of words in a sequence.
    return {key:vectorizer(text) for key, text in data.items()}
    
    # return np.asarray([vectorizer(text) for text in data], dtype=np.int32)
    

def fill_unk(unk):
    global s, embeddings
    
    # Gather the distribution hyperparameters
    v = np.var(s,0) 
    m = np.mean(s,0) 
    RS = np.random.RandomState()

    embeddings[unk] = RS.multivariate_normal(m,np.diag(v))
    return embeddings[unk]

# def tf_lower_and_split_punct(text):
#   # Split accecented characters.
#   text = tf_text.normalize_utf8(text, 'NFKD')
#   text = tf.strings.lower(text)
#   # Keep space, a to z, and select punctuation.
#   text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
#   # Add spaces around punctuation.
#   text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
#   # Strip whitespace.
#   text = tf.strings.strip(text)

#   text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
#   return text