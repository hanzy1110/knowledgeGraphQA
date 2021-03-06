import os
import time
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa

from tqdm import tqdm, trange
from .encoder_decoder_model import AutoEncoder, beam_answer_evauate_answers, loss_function, beam_answer
from .anchor_loss import AnchorLoss

class TrainingLoop:

    def __init__(self, dataset_creator, optimizer, D, frac, checkpoint_folder) -> None:

        self.dataset_creator = dataset_creator

        vocab_inp_size, vocab_tar_size, max_length_input, \
         self.max_length_output, self.embedding_dim, units, BATCH_SIZE = self.parse_hyperparameters(D, frac)
        
        self.autoencoder = AutoEncoder(vocab_inp_size, self.embedding_dim, units, BATCH_SIZE,
         max_length_input, self.max_length_output, self.lang_tokenizer)
        self.anchorloss = AnchorLoss(self.max_length_output, BATCH_SIZE)
        self.optimizer = optimizer

        # self.checkpointer = keras.callbacks.ModelCheckpoint(filepath='training_checkpoints', save_weights_only=False)
        # tensorboard_callback = keras.callbacks.TensorBoard(log_dir="logs")
        # _callbacks = [checkpointer, tensorboard_callback]
        _callbacks = []
        checkpoint_dir = f'./{checkpoint_folder}'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                        autoencoder = self.autoencoder)

        self.checkpoint_manager = tf.train.CheckpointManager(
                                        checkpoint, checkpoint_dir, 4, checkpoint_name='ckpt', 
                                        step_counter=None, checkpoint_interval=None,
                                        init_fn=None)

        status = self.checkpoint_manager.restore_or_initialize()

        self.callbacks = keras.callbacks.CallbackList(
            _callbacks, add_history=True, model=self.autoencoder)

    def parse_hyperparameters(self, D, frac:float=1):

        BUFFER_SIZE = 32000
        self.BATCH_SIZE = 64
        self.units = 512

        self.BETA = 0.03

        self.train_dataset, self.val_dataset, self.lang_tokenizer = self.dataset_creator.call(
            frac, BUFFER_SIZE, self.BATCH_SIZE)

        data_dict = next(iter(self.train_dataset))
        print(data_dict['context'].shape,
            data_dict['question'].shape, data_dict['target'].shape)

        vocab_inp_size = len(self.lang_tokenizer.word_index)+2
        vocab_tar_size = len(self.lang_tokenizer.word_index)+2
        self.max_length_input = data_dict['context'].shape[1]
        max_length_output = data_dict['target'].shape[1]

        embedding_dim = D
        self.steps_per_epoch = int(frac*self.BATCH_SIZE)

        return vocab_inp_size, vocab_tar_size, self.max_length_input, max_length_output, embedding_dim, self.units, self.BATCH_SIZE

    def train(self,EPOCHS = 10, case = 'initial'):

        # Presentation
        self.epochs = trange(
            EPOCHS,
            desc="Epoch",
            unit="Epoch",
            postfix="loss = {loss:.4f}, accuracy = {accuracy:.4f}")
        self.epochs.set_postfix(loss=0, accuracy=0)

        logs = {}
        self.callbacks.on_train_begin(logs=logs)
    
        for epoch in self.epochs:

            self.callbacks.on_epoch_begin(epoch, logs=logs)

            start = time.time()

            # enc_hidden = autoencoder.encoder.initialize_hidden_state()
            total_loss = 0
            # print(enc_hidden[0].shape, enc_hidden[1].shape)

            for (batch, data_dict) in tqdm(enumerate(self.train_dataset.take(self.steps_per_epoch))):

                self.callbacks.on_batch_begin(batch, logs=logs)
                self.callbacks.on_train_batch_begin(batch, logs=logs)

                inp = [data_dict['context'], data_dict['question']]
                targ = data_dict['target']
                if case == 'initial':
                    batch_loss = self.train_step_initial(inp, targ, loss_function)
                    total_loss += batch_loss
                elif case =='anchor':
                    batch_loss = self.train_step(inp, targ, loss_function, BETA=self.BETA)
                    total_loss += batch_loss

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                                    batch,
                                                                    batch_loss.numpy()))
                # saving (checkpoint) the model every 2 epochs

            if (epoch + 1) % 2 == 0:
                self.checkpoint_manager.save()
                # self.callbacks.on_epoch_end(epoch=epoch, logs=logs)
                # callbacks.on_epoch_end(epoch=epoch,logs=logs, )
                # self.autoencoder.save('training_checkpoints/ckpt')

            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                total_loss / self.steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

        self.callbacks.on_train_end(logs=logs)

        # Fetch the history object we normally get from keras.fit
        history_object = None
        for cb in self.callbacks:
            if isinstance(cb, keras.callbacks.History):
                history_object = cb
        assert history_object is not None

    # @tf.function
    def train_step(self, inp, targ, loss_function, BETA):
        loss = 0

        with tf.GradientTape() as tape:

            pred = self.autoencoder([inp, targ])
            real = targ[:, 1:]         # ignore <start> token

            logits = pred.rnn_output
            pred:tfa.seq2seq.BasicDecoderOutput = self.autoencoder([inp, targ])

            sequences = pred.sample_id
            sequences = self.autoencoder.encoder.embedding(sequences)

            anchLoss = self.anchorloss.loss(sequences, logits)
            loss = loss_function(real, logits)
            loss += tf.cast(BETA * anchLoss, dtype=tf.float32) 
            # variables = encoder.trainable_variables + decoder.trainable_variables
            variables = self.autoencoder.trainable_variables

            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))

        return loss
    
    # @tf.function
    def train_step_initial(self, inp, targ, loss_function):
        loss = 0

        with tf.GradientTape() as tape:

            pred = self.autoencoder([inp, targ])
            real = targ[:, 1:]         # ignore <start> token

            logits = pred.rnn_output
            pred:tfa.seq2seq.BasicDecoderOutput = self.autoencoder([inp, targ])

            loss = loss_function(real, logits)
            # variables = encoder.trainable_variables + decoder.trainable_variables
            variables = self.autoencoder.trainable_variables

            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

    def exact_match(self):
        
        correct = 0
        total = 0

        for (batch, data_dict) in tqdm(enumerate(self.val_dataset.take(self.steps_per_epoch))):
            context = data_dict['context']
            question = data_dict['question']
            answer = data_dict['target']
            
            out, ans = beam_answer_evauate_answers(context, question, answer, self.units,
                        lang_tokenizer = self.lang_tokenizer, 
                        autoencoder = self.autoencoder, 
                          max_length_input=self.max_length_input, max_length_output=self.max_length_output)

            if any(list(map(lambda x: x == ans, out))):
                correct +=1
                total += 1
            else:
                total += 1            

        return correct/total
    
    def F1_metric(self, frac, max_false_positives):
        dataset = self.dataset_creator.call_post(frac, max_false_positives, self.BATCH_SIZE)
        
        false_positives = 0
        false_negatives = 0
        true_positives = 0

        for (batch, data_dict) in tqdm(enumerate(dataset.take(self.steps_per_epoch))):
                context = data_dict['context']
                question = data_dict['question']
                answer = data_dict['target']
                
                out, ans = beam_answer_evauate_answers(context, question, answer, self.units,
                            lang_tokenizer = self.lang_tokenizer, 
                            autoencoder = self.autoencoder, 
                            max_length_input=self.max_length_input, max_length_output=self.max_length_output)
                if ans == '':
                    # Check for false positive
                    # If the predicted answer is empty then:
                    if any(list(map(lambda x: x == ans, out))):
                        true_positives += 1
                    else:
                        # Predicted answer is plain wrong
                        false_positives +=1
                else:
                    if any(list(map(lambda x: x == ans, out))):
                        true_positives += 1
                    elif any(list(map(lambda x: x == '', out))):
                        false_negatives += 1

        P = true_positives/(true_positives + false_positives)
        R = true_positives/(true_positives + false_negatives)
        
        return 2 * P*R/(P+R)