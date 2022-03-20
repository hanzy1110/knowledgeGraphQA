import tensorflow as tf
import tensorflow.keras as keras

import tensorflow_addons as tfa
from typing import List
from sklearn.model_selection import train_test_split
from random import choice

import pandas as pd
import unicodedata
import re
import numpy as np
import os
import io
import time

def mix_iters(original, mixed):
    # List because otherwise mixed gets consumed in the set comprehension
    mixed = list(mixed)
    mixed_contexts = {context for context, question, answer in mixed} 
    for tup in original:
        # If they have the same context then prefer the mixed question
        if tup[0] in mixed_contexts:
            yield next(filter(lambda mixed_: mixed_[0] == tup[0], mixed)) 
        # Else yield the original
        else:
            yield tup

def mix_strings(str_arr):
    # arr = np.array(str_Arr)
    for _ in str_arr:   
        yield choice(str_arr)

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
        w = self.unicode_to_ascii(str(w).lower().strip())

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

    def create_dataset(self, path, frac):
        # path : path to spa-eng.txt file
        # num_examples : Limit the total number of training example for faster training (set num_examples = len(lines) to use full data)
        data = pd.read_csv(path, delimiter='\t')
        data = data.sample(frac=frac)
        clean_context = [self.preprocess_sentence(w) for w in data['paragraph']]
        clean_questions = [self.preprocess_sentence(w) for w in data['question']]
        clean_answers = [self.preprocess_sentence(w) for w in data['answer']]

        return clean_context, clean_questions, clean_answers

    # Step 3 and Step 4
    def tokenize(self, corpus, lang_tokenizer:keras.preprocessing.text.Tokenizer):
        # lang = list of sentences in a language

        # print(len(lang), "example sentence: {}".format(lang[0]))

        ## tf.keras.preprocessing.text.Tokenizer.texts_to_sequences converts string (w1, w2, w3, ......, wn) 
        ## to a list of correspoding integer ids of words (id_w1, id_w2, id_w3, ...., id_wn)
        tensor = lang_tokenizer.texts_to_sequences(corpus) 

        ## tf.keras.preprocessing.sequence.pad_sequences takes argument a list of integer id sequences 
        ## and pads the sequences to match the longest sequences in the given input
        tensor = keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

        return tensor

    def load_dataset(self, num_examples=None):
        # creating cleaned input, output pairs
        # input_info, answers = self.create_dataset(self.file_path, num_examples)
        clean_context, clean_questions, clean_answers = self.create_dataset(self.file_path, num_examples)


        lang_tokenizer = keras.preprocessing.text.Tokenizer(filters='', oov_token='<OOV>')
        lang_tokenizer.fit_on_texts(clean_context)

        context_tensor = self.tokenize(clean_context, lang_tokenizer)
        question_tensor = self.tokenize(clean_questions, lang_tokenizer)
        target_tensor = self.tokenize(clean_answers, lang_tokenizer)

        return context_tensor, question_tensor, target_tensor, lang_tokenizer

    def call(self, frac, BUFFER_SIZE, BATCH_SIZE):
        # file_path = download_nmt()

        context_tensor, question_tensor, target_tensor, self.inp_lang_tokenizer = self.load_dataset(frac)
        # print(input_tensor.shape)
        # print(target_tensor.shape)

        context_tensor_train, context_tensor_val, \
         question_tensor_train, question_tensor_val, \
          target_tensor_train, target_tensor_val = train_test_split(context_tensor,
                                                                    question_tensor, 
                                                                    target_tensor,
                                                                    test_size=0.3)

        aux_train = {'context':context_tensor_train, 'question':question_tensor_train, 'target':target_tensor_train}
        aux_val = {'context':context_tensor_val, 'question':question_tensor_val, 'target':target_tensor_val}

        train_dataset = tf.data.Dataset.from_tensor_slices(aux_train)

        train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

        val_dataset = tf.data.Dataset.from_tensor_slices(aux_val)
        val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

        return train_dataset, val_dataset, self.inp_lang_tokenizer

    def create_false_positives(self, frac:float, max_false_positives:float):

        clean_context, clean_questions, clean_answers = self.create_dataset(self.file_path, frac)
        n = len(clean_context)
        max_samples = int(n*max_false_positives)
        original_data = zip(clean_context, clean_questions, clean_answers) 
        # for each context, get random question which should produce no answer
        mixed_data = zip(clean_context, mix_strings(clean_questions), ['' for _ in range(max_samples)])
        self.false_positive_data = mix_iters(original_data, mixed_data)
    
    def load_dataset_post(self, frac, max_false_positives):
        self.create_false_positives(frac, max_false_positives)

        context, questions, answers = [], [], []
        for c,q,a, in self.false_positive_data:
            context.append(c)
            questions.append(q)
            answers.append(a)
        
        lang_tokenizer = keras.preprocessing.text.Tokenizer(filters='', oov_token='<OOV>')
        lang_tokenizer.fit_on_texts(context)

        context_tensor = self.tokenize(context, lang_tokenizer)
        question_tensor = self.tokenize(questions, lang_tokenizer)
        target_tensor = self.tokenize(answers, lang_tokenizer)

        return context_tensor, question_tensor, target_tensor, lang_tokenizer

    def call_post(self, frac, max_false_positives, BATCH_SIZE):

        context_tensor, question_tensor, target_tensor, self.inp_lang_tokenizer = self.load_dataset_post(frac, max_false_positives)
        aux_val = {'context':context_tensor, 'question':question_tensor, 'target':target_tensor}

        val_dataset = tf.data.Dataset.from_tensor_slices(aux_val)
        val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

        return val_dataset, self.inp_lang_tokenizer

