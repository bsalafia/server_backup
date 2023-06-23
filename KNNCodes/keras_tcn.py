# MIT License
#
# Copyright (c) 2018 Philippe Rémy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import keras.backend as K
import keras.layers
from keras import optimizers
#from keras.engine.topology import Layer
#from keras.utils.layer_utils import Layer
from keras.layers import Layer
from keras.layers import Activation, Lambda, Dense
from keras.layers import Conv1D, Convolution1D, SpatialDropout1D
from keras.layers import Conv2D, Convolution2D, SpatialDropout2D, ZeroPadding2D
from keras.layers import Cropping1D, Cropping2D, Reshape
from keras.models import Input, Model
from typing import List, Tuple

from Utils import get_numbered_name


def channel_normalization(x):
    # type: (Layer) -> Layer
    """ Normalize a layer to the maximum activation

    This keeps a layers values between zero and one.
    It helps with relu's unbounded activation

    Args:
        x: The layer to normalize

    Returns:
        A maximal normalized layer
    """
    max_values = K.max(K.abs(x), 2, keepdims=True) + 1e-5
    out = x / max_values
    return out


def wave_net_activation(x):
    # type: (Layer) -> Layer
    """This method defines the activation used for WaveNet

    described in https://deepmind.com/blog/wavenet-generative-model-raw-audio/

    Args:
        x: The layer we want to apply the activation to

    Returns:
        A new layer with the wavenet activation applied
    """
    tanh_out = Activation('tanh', name=get_numbered_name('tcn_tanh_activation'))(x)
    sigm_out = Activation('sigmoid', name=get_numbered_name('tcn_sigm_activation'))(x)
    return keras.layers.multiply([tanh_out, sigm_out])


def temporal_convolution(filters, kernel_size, dilation_rate, padding, name='', stack_index=0, use_2d_conv=True):
    # type: (int, int, int, str, str, int, bool) -> function
    """
    Return temporal convolution layer, represented with either 1D or 2D convolutions (use_2d_conv).
    When using 2D convolutions, the input is assumed to be a rank 3 representation of a rank 2 input.
    :param filters: Number of filters
    :param kernel_size: 1D kernel size
    :param dilation_rate: Dilation factor
    :param padding: Type of padding, either 'causal' or 'same'
    :param name: Name of the layer (for naming conventions)
    :param stack_index: Index of the TCN stack (for naming conventions)
    :param use_2d_conv: Use 2D or 1D convolutions
    :return: Temporal convolution layer
    """
    def block_call(_input):
        padding_type = padding

        if padding_type == 'causal' and use_2d_conv:
            k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
            left_pad = dilation_rate * (k - 1)
            pattern = ((left_pad, 0), (0, 0))
            _input = ZeroPadding2D(padding=pattern)(_input)
            padding_type = 'valid'

        if use_2d_conv:
            conv = Conv2D(filters=filters, kernel_size=(kernel_size, 1),
                          dilation_rate=(dilation_rate, 1), padding=padding_type,
                          name=name + '_d_%s_conv_%d_tanh_s%d' % (padding_type, dilation_rate, stack_index))
        else:
            conv = Conv1D(filters=filters, kernel_size=kernel_size,
                          dilation_rate=dilation_rate, padding=padding_type,
                          name=name + '_d_%s_conv_%d_tanh_s%d' % (padding_type, dilation_rate, stack_index))

        x = conv(_input)

        return x

    return block_call


def residual_block(x, s, i, activation, nb_filters, kernel_size, padding, dropout_rate=0, name='', use_2d_conv=True):
    # type: (Layer, int, int, str, int, int, str, float, str, bool) -> Tuple[Layer, Layer]
    """Defines the residual block for the WaveNet TCN

    Args:
        x: The previous layer in the model
        s: The stack index i.e. which stack in the overall TCN
        i: The dilation power of 2 we are using for this residual block
        activation: The name of the type of activation to use
        nb_filters: The number of convolutional filters to use in this block
        kernel_size: The size of the convolutional kernel
        padding: The padding used in the convolutional layers, 'same' or 'causal'.
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
        name: Name of the model. Useful when having multiple TCN.

    Returns:
        A tuple where the first element is the residual model layer, and the second
        is the skip connection.
    """

    original_x = x
    conv = temporal_convolution(filters=nb_filters, kernel_size=kernel_size,
                                dilation_rate=i, padding=padding,
                                name=name + '_d_%s_conv_%d_tanh_s%d' % (padding, i, s),
                                stack_index=s, use_2d_conv=use_2d_conv)(x)
    if activation == 'norm_relu':
        x = Activation('relu', name=get_numbered_name('tcn_relu_activation'))(conv)
        x = Lambda(channel_normalization)(x)
    elif activation == 'wavenet':
        x = wave_net_activation(conv)
    else:
        x = Activation(activation, name=get_numbered_name('tcn_linear_activation'))(conv)

    if use_2d_conv:
        x = SpatialDropout2D(dropout_rate, name=name + '_spatial_dropout1d_%d_s%d_%f' % (i, s, dropout_rate))(x)
        x = Convolution2D(nb_filters, (1, 1), padding='same')(x)
    else:
        x = SpatialDropout1D(dropout_rate, name=name + '_spatial_dropout1d_%d_s%d_%f' % (i, s, dropout_rate))(x)
        x = Convolution1D(nb_filters, 1, padding='same')(x)

    res_x = keras.layers.add([original_x, x])
    return res_x, x


def process_dilations(dilations):
    def is_power_of_two(num):
        return num != 0 and ((num & (num - 1)) == 0)

    if all([is_power_of_two(i) for i in dilations]):
        return dilations

    else:
        new_dilations = [2 ** i for i in dilations]
        # print(f'Updated dilations from {dilations} to {new_dilations} because of backwards compatibility.')
        return new_dilations


class TCN:
    """Creates a TCN layer.

        Input shape:
            A tensor of shape (batch_size, timesteps, input_dim).

        Args:
            nb_filters: The number of filters to use in the convolutional layers.
            kernel_size: The size of the kernel to use in each convolutional layer.
            dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
            nb_stacks : The number of stacks of residual blocks to use.
            activation: The activations to use (norm_relu, wavenet, relu...).
            padding: The padding to use in the convolutional layers, 'causal' or 'same'.
            use_skip_connections: Boolean. If we want to add skip connections from input to each residual block.
            return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            name: Name of the model. Useful when having multiple TCN.

        Returns:
            A TCN layer.
        """

    def __init__(self,
                 nb_filters=64,
                 kernel_size=2,
                 nb_stacks=1,
                 dilations=[1, 2, 4, 8, 16, 32],
                 activation='norm_relu',
                 padding='causal',
                 use_skip_connections=True,
                 dropout_rate=0.0,
                 return_sequences=True,
                 return_both=None,
                 input_length=None,
                 use_2d_conv=True,
                 name='tcn'):
        self.name = name
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.activation = activation
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.padding = padding
        self.input_length = input_length
        self.use_2d_conv = use_2d_conv
        self.return_both = return_both

        if padding != 'causal' and padding != 'same':
            raise ValueError("Only 'causal' or 'same' padding are compatible for this layer.")

        if not isinstance(nb_filters, int):
            print('An interface change occurred after the version 2.1.2.')
            print('Before: tcn.TCN(i, return_sequences=False, ...)')
            print('Now should be: tcn.TCN(return_sequences=False, ...)(i)')
            print('Second solution is to pip install keras-tcn==2.1.2 to downgrade.')
            raise Exception()

    def __call__(self, inputs):
        x = inputs
        x = temporal_convolution(self.nb_filters, 1, 1, padding=self.padding,
                                 name=self.name + '_initial_conv',
                                 use_2d_conv=self.use_2d_conv)(x)
        skip_connections = []
        for s in range(self.nb_stacks):
            for i in self.dilations:
                x, skip_out = residual_block(x, s, i, self.activation, self.nb_filters,
                                             self.kernel_size, self.padding, self.dropout_rate,
                                             name=self.name, use_2d_conv=self.use_2d_conv)
                skip_connections.append(skip_out)
        if self.use_skip_connections:
            x = keras.layers.add(skip_connections)
        x = Activation('relu', name=get_numbered_name('tcn_relu_activation'))(x)

        seq_feature = x

        if not self.return_sequences:
            if self.input_length:
                if self.use_2d_conv:
                    cropping = Cropping2D(cropping=((self.input_length - 1, 0), (0, 0)))
                else:
                    cropping = Cropping1D(cropping=(self.input_length - 1, 0))
                x = Reshape(target_shape=(self.nb_filters,))(cropping(x))
            else:
                output_slice_index = -1
                x = Lambda(lambda tt: tt[:, output_slice_index, :])(x)

        if self.return_both == "tnt" or self.return_both == "hover":
            return x, seq_feature

        return x


def compiled_tcn(num_feat,  # type: int
                 num_classes,  # type: int
                 nb_filters,  # type: int
                 kernel_size,  # type: int
                 dilations,  # type: List[int]
                 nb_stacks,  # type: int
                 max_len,  # type: int
                 activation='norm_relu',  # type: str
                 padding='causal',  # type: str
                 use_skip_connections=True,  # type: bool
                 return_sequences=True,
                 regression=False,  # type: bool
                 dropout_rate=0.05,  # type: float
                 name='tcn'  # type: str
                 ):
    # type: (...) -> keras.Model
    """Creates a compiled TCN model for a given task (i.e. regression or classification).

    Args:
        num_feat: The number of features of your input, i.e. the last dimension of: (batch_size, timesteps, input_dim).
        num_classes: The size of the final dense layer, how many classes we are predicting.
        nb_filters: The number of filters to use in the convolutional layers.
        kernel_size: The size of the kernel to use in each convolutional layer.
        dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
        nb_stacks : The number of stacks of residual blocks to use.
        max_len: The maximum sequence length, use None if the sequence length is dynamic.
        activation: The activations to use.
        padding: The padding to use in the convolutional layers.
        use_skip_connections: Boolean. If we want to add skip connections from input to each residual block.
        return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
        regression: Whether the output should be continuous or discrete.
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
        name: Name of the model. Useful when having multiple TCN.

    Returns:
        A compiled keras TCN.
    """

    dilations = process_dilations(dilations)

    input_layer = Input(shape=(max_len, num_feat))

    x = TCN(nb_filters, kernel_size, nb_stacks, dilations, activation,
            padding, use_skip_connections, dropout_rate, return_sequences, name)(input_layer)

    print('x.shape=', x.shape)

    if not regression:
        # classification
        x = Dense(num_classes)(x)
        x = Activation('softmax', name=get_numbered_name('tcn_softmax_activation'))(x)
        output_layer = x
        print('model.x = {}'.format(input_layer.shape))
        print('model.y = {}'.format(output_layer.shape))
        model = Model(input_layer, output_layer)

        # https://github.com/keras-team/keras/pull/11373
        # It's now in Keras@master but still not available with pip.
        # TODO To remove later.
        def accuracy(y_true, y_pred):
            # reshape in case it's in shape (num_samples, 1) instead of (num_samples,)
            if K.ndim(y_true) == K.ndim(y_pred):
                y_true = K.squeeze(y_true, -1)
            # convert dense predictions to labels
            y_pred_labels = K.argmax(y_pred, axis=-1)
            y_pred_labels = K.cast(y_pred_labels, K.floatx())
            return K.cast(K.equal(y_true, y_pred_labels), K.floatx())

        adam = optimizers.Adam(lr=0.002, clipnorm=1.)
        model.compile(adam, loss='sparse_categorical_crossentropy', metrics=[accuracy])
        print('Adam with norm clipping.')
    else:
        # regression
        x = Dense(1)(x)
        x = Activation('linear', name=get_numbered_name('tcn_linear_activation'))(x)
        output_layer = x
        print('model.x = {}'.format(input_layer.shape))
        print('model.y = {}'.format(output_layer.shape))
        model = Model(input_layer, output_layer)
        adam = optimizers.Adam(lr=0.002, clipnorm=1.)
        model.compile(adam, loss='mean_squared_error')

    return model

