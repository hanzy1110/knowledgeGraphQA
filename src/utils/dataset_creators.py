import tensorflow as tf
import tensorflow.keras as keras

import tensorflow_addons as tfa
from typing import List
from sklearn.model_selection import train_test_split

import pandas as pd
import unicodedata
import re
import numpy as np
import os
import io
import time

class QADataset:
    def __init__(self, file_path):
        self.problem_type = 'en-spa'
        self.inp_lang_tokenizer = None
        self.targ_lang_tokenizer = None
        self.file_path = file_path

    def unicode_to_ascii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    ## Step 1 and Step 2 
    def preprocess_sentence(self, w):
        w = self.unicode_to_ascii(w.lower().strip())

        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)

        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

        w = w.strip()

        # adding a start and an end token to the sentence
        # so that the model know when to start and stop predicting.
        w = '<start> ' + w + ' <end>'
        return w

    def create_dataset(self, path, num_examples):
        # path : path to spa-eng.txt file
        # num_examples : Limit the total number of training example for faster training (set num_examples = len(lines) to use full data)
        data = pd.read_csv(path, delimiter='\t')
        aux:List[str] = []
        for idx in data.index:
            aux.append(str(data.iloc[idx]['paragraph']) + "<question>" + str(data.iloc[idx]['question']))
    
        # lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
        clean_input = [self.preprocess_sentence(w) for w in aux[:num_examples]]
        clean_anwers = [self.preprocess_sentence(w) for w in data['answer'][:num_examples]]

        return clean_input, clean_anwers

    # Step 3 and Step 4
    def tokenize(self, lang):
        # lang = list of sentences in a language

        # print(len(lang), "example sentence: {}".format(lang[0]))
        lang_tokenizer = keras.preprocessing.text.Tokenizer(filters='', oov_token='<OOV>')
        lang_tokenizer.fit_on_texts(lang)

        ## tf.keras.preprocessing.text.Tokenizer.texts_to_sequences converts string (w1, w2, w3, ......, wn) 
        ## to a list of correspoding integer ids of words (id_w1, id_w2, id_w3, ...., id_wn)
        tensor = lang_tokenizer.texts_to_sequences(lang) 

        ## tf.keras.preprocessing.sequence.pad_sequences takes argument a list of integer id sequences 
        ## and pads the sequences to match the longest sequences in the given input
        tensor = keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

        return tensor, lang_tokenizer

    def load_dataset(self, num_examples=None):
        # creating cleaned input, output pairs
        input_info, answers = self.create_dataset(self.file_path, num_examples)

        lang_tokenizer = keras.preprocessing.text.Tokenizer(filters='', oov_token='<OOV>')
        lang_tokenizer.fit_on_texts(input_info)

        input_tensor = lang_tokenizer.texts_to_sequences(input_info)
        input_tensor = keras.preprocessing.sequence.pad_sequences(input_tensor, padding='post')

        target_tensor = lang_tokenizer.texts_to_sequences(answers)
        target_tensor = keras.preprocessing.sequence.pad_sequences(target_tensor, padding='post')

        return input_tensor, target_tensor, lang_tokenizer

    def call(self, num_examples, BUFFER_SIZE, BATCH_SIZE):
        # file_path = download_nmt()

        input_tensor, target_tensor, self.inp_lang_tokenizer = self.load_dataset(num_examples)
        # print(input_tensor.shape)
        # print(target_tensor.shape)

        input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.3)

        train_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train))
        train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

        val_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val))
        val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

        return train_dataset, val_dataset, self.inp_lang_tokenizer

if __name__  == '__main__':
    import random  
    import string  

    def specific_string(length:int) -> str:  
        # Generate random string
        sample_string = 'pqrstuvwxy' # define the specific string  
        # define the condition for random string  
        result = ''.join((random.choice(sample_string)) for _ in range(length))  

        return result

    _max_word_size = 25
    _max_context_size = 10


    with open("ramdom_dataset.txt", "w") as f:

        random_context = [specific_string(random.randint(0,_max_word_size)) for _ in range(0,random.randint(0,_max_phrase_size))]
        
        random_context.append('?')
        random_context.append('\n')

        for _string in radom_context:
            f.
