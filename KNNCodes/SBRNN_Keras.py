"""========================================================================
This file contains the code for the sliding bidirectional recurrent
neural network detector.

@author: Nariman Farsad
========================================================================"""

import numpy as np
import tensorflow as tf
import os
import keras as keras
from tensorflow.keras.utils import to_categorical
from keras_tcn import TCN

class ModelConfig(object):
    """ The model parameters of the SBRNN detector
    """
    def __init__(self,hidden_size=80,
                 numb_layers=3,
                 peephole=True,
                 input_size=14,
                 dropout=0.2,
                 sliding=True,
                 output_size=2,
                 cell_type='LSTM',
                 activation='tanh',
                 bidirectional=True,
                 norm_gain=0.1,
                 norm_shift=0.0):
        self.hidden_size = hidden_size
        self.numb_layers = numb_layers
        self.peephole = peephole
        self.dropout = dropout
        self.sliding = sliding
        self.input_size = input_size
        self.output_size = output_size
        self.cell_type = cell_type
        self.activation = activation
        self.bidirectional = bidirectional
        self.norm_gain = norm_gain
        self.norm_shift = norm_shift

# tcn:
#   latent-dims: 63
#   encoding-blocks: 2
#   encoding-type: "mobilenet"
#   latent-type: "3d"
#   kernel-size: 2
#   filter-size: 128
#   return-sequence: False
#   nb-stacks: 1
#   nb-dilations: 4
#   activation: "wavenet" # norm_relu, wavenet, relu, ...
#   padding: "causal" # causal, same
#   use-skips: True
#   dropout-rate: 0.
class ModelConfigTCN(object):
    """ The model parameters of the TCN detector
    """
    def __init__(self,nb_filters=128,
                 kernel_size=2,
                 input_size=14,
                 output_size=2,
                 nb_stacks=1,
                 seq_len=16,
                 dilations=(1, 2, 4, 8, 16, 32),
                 padding='causal',
                 use_skip_connections=True,
                 dropout_rate=0.0,
                 return_sequences=True,
                 sliding=True,
                 activation='wavenet',
                 kernel_initializer='he_normal',
                 use_batch_norm=False,
                 use_layer_norm=False,):
        self.nb_filters = nb_filters
        self.input_size = input_size
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.output_size = output_size
        self.nb_stacks = nb_stacks
        self.sliding = sliding
        self.dilations = dilations
        self.padding = padding
        self.use_skip_connections = use_skip_connections
        self.dropout_rate = dropout_rate
        self.return_sequences = return_sequences
        self.kernel_initializer = kernel_initializer
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm


class BRNN(object):
    """ A Multilayer Bi-directional Recurrent Neural Network
    """
    def __init__(self, input, config):
        """
        Args:
            input: input to the BRNN
            input_len: the length of inputs

            config: includes model configuration parameters
        """
        self.input = input
        self.config = config
        self.all_layer_inps = []

        self.probs, self.predictions = self.build_network()



    def build_network(self):
        """Builds the BRNN network
        """
        self.all_layer_inps.append(self.input)
        for i in range(self.config.numb_layers):
            if self.config.cell_type == "LSTM":
                layer = keras.layers.LSTM(self.config.hidden_size, activation=self.config.activation,
                                          return_sequences=True, dropout=self.config.dropout)
            elif self.config.cell_type == "GRU":
                layer = keras.layers.GRU(self.config.hidden_size, activation=self.config.activation,
                                         return_sequences=True, dropout=self.config.dropout)
            elif self.config.cell_type == "SimpleRNN":
                layer = keras.layers.SimpleRNN(self.config.hidden_size, activation=self.config.activation,
                                            return_sequences=True, dropout=self.config.dropout)

            elif self.config.cell_type == "MGRU":
                from custom_recurrent import BaseCustomGRU
                layer = BaseCustomGRU(self.config.hidden_size, activation=self.config.activation,
                                      return_sequences=True, units_bn=None,
                                      units_bn_recurrent=None,
                                      cell_type='MGUCell', dropout=self.config.dropout)

            if self.config.bidirectional:
                layer_out = keras.layers.Bidirectional(layer, merge_mode='ave')(self.all_layer_inps[-1])
            else:
                layer_out = layer(self.all_layer_inps[-1])

            self.all_layer_inps.append(layer_out)

        if self.config.output_size == 1:
            probs = keras.layers.TimeDistributed(keras.layers.Dense(1, activation='sigmoid'))(self.all_layer_inps[-1])
            predictions = tf.cast(probs > 0.5, tf.int16)
        else:
            probs = keras.layers.TimeDistributed(keras.layers.Dense(self.config.output_size, activation='softmax'))(self.all_layer_inps[-1])
            predictions = tf.argmax(probs, 2)

        return probs, predictions

    def get_outputs(self):
        return self.probs, self.predictions

class BTCN(object):
    def __init__(self, input, config):
        self.input = input
        self.config = config
        self.probs, self.predictions = self.build_network()

    def build_network(self):
        """Builds the TCN network
        """
        tcn_out = TCN(nb_filters=self.config.nb_filters, kernel_size=self.config.kernel_size,
                      nb_stacks=self.config.nb_stacks, dilations=self.config.dilations,
                      padding=self.config.padding, use_skip_connections=self.config.use_skip_connections,
                      dropout_rate=self.config.dropout_rate, return_sequences=self.config.return_sequences,
                      activation=self.config.activation)(self.input)

        tcn_out = keras.layers.Lambda(lambda x: x[:, :, 0, :])(tcn_out)
        if self.config.output_size == 1:
            probs = keras.layers.TimeDistributed(keras.layers.Dense(1, activation='sigmoid'))(tcn_out)
            predictions = tf.cast(probs > 0.5, tf.int16)
        else:
            probs = keras.layers.TimeDistributed(keras.layers.Dense(self.config.output_size, activation='softmax'))(tcn_out)
            predictions = tf.argmax(probs, 2)

        return probs, predictions

    def get_outputs(self):
        return self.probs, self.predictions


class SBRNN_Detector(object):
    """ The sliding bidirectional recurrent neural network detector
    """
    def __init__(self, config, model_type='BRNN', opt='adam'):
        self.config = config
        self.model_type = model_type
        self.graph = tf.Graph()

        if model_type == 'BRNN':
            self.inps = keras.layers.Input(shape=(None, self.config.input_size), name="inp")
            self.nn = BRNN(self.inps, self.config)
        elif model_type == 'BTCN':
            self.inps = keras.layers.Input(shape=(self.config.seq_len, 1, self.config.input_size), name="inp")
            self.nn = BTCN(self.inps, self.config)

        self.probs, self.predictions = self.nn.get_outputs()
        self.model = keras.models.Model(inputs=self.inps, outputs=self.probs)
        if self.config.output_size == 1:
            loss = 'binary_crossentropy'
        else:
            loss = 'categorical_crossentropy'

        if opt == 'nadam':
            self.model.compile(loss=loss, optimizer=keras.optimizers.Nadam(), metrics=['accuracy'])
        elif opt == 'sgd':
            self.model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
        elif opt == 'sgdn':
            self.model.compile(loss=loss, optimizer=keras.optimizers.SGD(nesterov=True), metrics=['accuracy'])
        elif opt == 'rmsprop':
            self.model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
        else:
            self.model.compile(loss=loss, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

        #print(self.model.summary())
        #print('The total number of parameters are {}.'.format(self.model_size(self.model)))


    def model_size(self, model):
        return sum([np.prod(keras.backend.get_value(w).shape) for w in model.trainable_weights])

    def load_trained_model(self, sess, trained_model_path):
        """
        Loads a trained model from what was saved. Insert the trained model path
        """
        trained_model_folder = os.path.split(trained_model_path)[0]
        ckpt = tf.train.get_checkpoint_state(trained_model_folder)
        v2_path = os.path.join(trained_model_folder, os.path.split(ckpt.model_checkpoint_path)[1] + ".index")
        norm_ckpt_path = os.path.join(trained_model_folder, os.path.split(ckpt.model_checkpoint_path)[1])
        if ckpt and (tf.gfile.Exists(norm_ckpt_path) or
                         tf.gfile.Exists(v2_path)):
            print("Reading model parameters from %s" % norm_ckpt_path)
            self.saver.restore(sess, norm_ckpt_path)
        else:
            print('Error reading weights')

    def load_model(self, filename, **kwargs):
        """Load the weights from a saved file.

        Parameters:
            filename: the path and the filename
        """
        self.model = keras.models.load_model(filename + ".h5")

    def set_lr(self, lr):
        """Set the current learning rate used during optimization.

        Parameters:
            lr: the value of the learning rate.
        """
        keras.backend.set_value(self.model.optimizer.lr, lr)

    def get_lr(self):
        """Get the current learning rate used during optimization.

        Returns:
            the learning rate used for optimization.
        """
        return keras.backend.get_value(self.model.optimizer.lr)

    def clear_model(self):
        keras.backend.clear_session()

    def save_model(self, filename, **kwargs):
        """Save the current model to a h5 file.

        Parameters:
            filename: the path and the filename
        """
        self.model.save(filename + ".h5")

    def train_on_samples(self, nn_input, labels, len_ss, batch_size=32, verbose=2, use_class_weights=False):
        """
        This trains the network by creating all sub sequences from the input
        and then using those for training
        Args
            sess: Tensorflow session instance
            nn_input: A batch of input sequences
            labels: A batch target values
            lr: learning rate
            len_ss: the length of the sub sequences (i.e., the maximum block
                    length for SBRNN detector
        Returns
            loss: the loss values for the batch
            accu: the accuracy for the batch

        """
        if nn_input.shape[1]>len_ss:
            inps, _, targs = self.create_sub_seq(nn_input,len_ss,labels)
        else:
            inps = nn_input
            targs = labels

        if self.config.output_size == 1:
            targs = np.reshape(targs, (targs.shape[0], targs.shape[1], -1))
        else:
            targs = to_categorical(targs)
        if self.model_type=='BTCN':
            inps = np.expand_dims(inps, axis=2)

        cw = np.array([1, 1])
        if use_class_weights:
            from sklearn.utils import class_weight
            cw = class_weight.compute_class_weight('balanced', np.unique(labels), np.ravel(labels))

        hist = self.model.fit(inps, targs, batch_size=batch_size, verbose=verbose, class_weight=cw)
        loss = hist.history['loss'][0]
        if 'accuracy' in hist.history:
            acc = hist.history['accuracy'][0]
        elif 'acc' in hist.history:
            acc = hist.history['acc'][0]
        else:
            acc = -1
        return loss, acc


    def create_sub_seq(self, nn_input, len_ss, labels=None):
        """
        This function creates all sub sequences for the batch
        """
        n_seq = nn_input.shape[0]
        len_seq = nn_input.shape[1]

        n_ss = len_seq - len_ss + 1
        new_labels = []
        new_inp = np.zeros((n_ss*n_seq,len_ss,nn_input.shape[2]))
        if labels is not None:
            dim_labels = labels.shape
            if len(dim_labels) == 2:
                new_labels = np.zeros((n_ss*n_seq, len_ss))
            elif len(dim_labels) == 3:
                new_labels = np.zeros((n_ss * n_seq, len_ss, dim_labels[2]))
        k = 0
        for i in range(n_seq):
            for j in range(n_ss):
                new_inp[k, :, :] = nn_input[i, j:j + len_ss, :]
                if labels is not None:
                    if len(dim_labels) == 2:
                        new_labels[k, :] = labels[i, j:j + len_ss]
                    elif len(dim_labels) == 3:
                        new_labels[k, :, :] = labels[i, j:j + len_ss, :]
                k += 1

        return new_inp, n_ss, new_labels

    def preds_to_symbs(self, preds, type="mean", lookahead_delay=None):
        """
        During testing use combine all the estimates of the SBRNN and it slides
        using mean or median
        """
        diags = [preds[::-1, :].diagonal(i) for i in range(-preds.shape[0] + 1, preds.shape[1])]
        if type == "mean":
            if lookahead_delay is None:
                preds = [np.mean(x, axis=1) for _, x in enumerate(diags)]
                symbs = [np.argmax(np.mean(x, axis=1)) for _, x in enumerate(diags)]
            else:
                avg_len = lookahead_delay + 1
                preds = [np.mean(x[:, 0:np.amin([avg_len, x.shape[1]])], axis=1) for _, x in enumerate(diags)]
                symbs = [np.argmax(np.mean(x[:, 0:np.amin([avg_len, x.shape[1]])], axis=1)) for _, x in enumerate(diags)]
        else:
            if lookahead_delay is None:
                preds = [np.median(x, axis=1) for _, x in enumerate(diags)]
                symbs = [np.argmax(np.median(x, axis=1)) for _, x in enumerate(diags)]
            else:
                avg_len = lookahead_delay + 1
                preds = [np.median(x[:, 0:np.amin([avg_len, x.shape[1]])], axis=1) for _, x in enumerate(diags)]
                symbs = [np.argmax(np.median(x[:, 0:np.amin([avg_len, x.shape[1]])], axis=1)) for _, x in
                         enumerate(diags)]
        return np.array(symbs), np.array(preds)

    def test_on_sample(self, nn_input, label, len_ss, type="mean", lookahead_delay=None):
        """
        Test the trained SBRNN detector on test samples. The input is first broken into
        sub sequences of length "len_ss". The BRNN is used on each sub sequence and the
        results are then combined using mean to median.
        """
        n_seq = label.shape[0]
        if nn_input.shape[1] > len_ss:
            new_input, n_ss, _ = self.create_sub_seq(nn_input, len_ss)
        else:
            new_input = nn_input
            n_ss = 1

        if self.model_type == 'BTCN':
            new_input = np.expand_dims(new_input, axis=2)
        probs = self.model.predict(new_input)

        pred_symb = np.zeros_like(label)
        soft_vals = np.zeros((label.shape[0], label.shape[1], self.config.output_size))
        for i in range(n_seq):
            pred_symb[i], soft_vals[i] = self.preds_to_symbs(probs[i*n_ss:(i+1)*n_ss], type=type,
                                                             lookahead_delay=lookahead_delay)

        error_rate = np.mean((pred_symb != label).astype(np.float16))

        return pred_symb, error_rate, soft_vals





if __name__=="__main__":
    #config = ModelConfig(cellType='LSTM-Norm')

    #sbrnn = SBRNN_Detector(config)

    tcn_config = ModelConfigTCN()
    inps = keras.layers.Input(shape=(None, 100), name="inp")
    tcn = BTCN(inps, tcn_config)
