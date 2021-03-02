"""A combination of Convolutional, Recurrent, and Fully-connected networks.

Authors
 * Mirco Ravanelli 2020
 * Ju-Chieh Chou 2020
 * Titouan Parcollet 2020
 * Abdel 2020
 * Peter Plantinga 2020, 2021
"""
import torch
import speechbrain as sb


class TimeFreqCRDNN(sb.nnet.containers.Sequential):
    """This model is a combination of CNNs, RNNs, and DNNs.

    Arguments
    ---------
    input_size : int
        The size of the expected input feature vectors.
    activation : torch class
        A class used for constructing the activation layers for CNN and DNN.
    dropout : float
        Neuron dropout rate as applied to CNN, RNN, and DNN.
    cnn_blocks : int
        The number of convolutional neural blocks to include.
    cnn_channels : list of ints
        A list of the number of output channels for each CNN block.
    rnn_class : torch class
        The type of RNN to use in CRDNN network (LiGRU, LSTM, GRU, RNN)
    rnn_layers : int
        The number of recurrent RNN layers to include.
    rnn_neurons : int
        Number of neurons in each layer of the RNN.
    rnn_bidirectional : bool
        Whether this model will process just forward or in both directions.
    dnn_blocks : int
        The number of linear neural blocks to include.
    dnn_neurons : int
        The number of neurons in the linear layers.

    Example
    -------
    >>> inputs = torch.rand([10, 15, 60])
    >>> model = TimeFreqCRDNN(input_size=inputs.size(-1))
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([10, 15, 512])
    """

    def __init__(
        self,
        input_size,
        activation=torch.nn.LeakyReLU,
        dropout=0.15,
        cnn_blocks=2,
        cnn_channels=[128, 256],
        rnn_class=sb.nnet.RNN.GRU,
        rnn_layers=4,
        rnn_neurons=512,
        rnn_bidirectional=True,
        dnn_blocks=2,
        dnn_neurons=512,
    ):
        super().__init__(input_shape=[None, None, input_size])

        if cnn_blocks > 0:
            self.append(sb.nnet.containers.Sequential, layer_name="CNN")
        for block_index in range(cnn_blocks):
            self.CNN.append(
                CNN_Block,
                channels=cnn_channels[block_index],
                activation=activation,
                dropout=dropout,
                causal=not rnn_bidirectional,
                layer_name=f"block_{block_index}",
            )

        self.append(
            rnn_class,
            layer_name="RNN",
            hidden_size=rnn_neurons,
            num_layers=rnn_layers,
            dropout=dropout,
            bidirectional=rnn_bidirectional,
        )

        if dnn_blocks > 0:
            self.append(sb.nnet.containers.Sequential, layer_name="DNN")
        for block_index in range(dnn_blocks):
            self.DNN.append(
                DNN_Block,
                neurons=dnn_neurons,
                activation=activation,
                dropout=dropout,
                layer_name=f"block_{block_index}",
            )


class CNN_Block(sb.nnet.containers.Sequential):
    """CNN Block, based on VGG blocks.

    Arguments
    ---------
    input_shape : tuple
        Expected shape of the input.
    channels : int
        Number of convolutional channels for the block.
    causal : bool
        Whether to use causal convolutions in the time dimension.
    activation : torch.nn.Module class
        A class to be used for instantiating an activation layer.
    downsample : bool
        Whether to use a stride of 2 in the second conv layer.
    dropout : float
        Rate to use for dropping channels.

    Example
    -------
    >>> inputs = torch.rand(10, 15, 60)
    >>> block = CNN_Block(input_shape=inputs.shape, channels=32)
    >>> outputs = block(inputs)
    >>> outputs.shape
    torch.Size([10, 15, 30, 32])
    """

    def __init__(
        self,
        input_shape,
        channels,
        causal=False,
        activation=torch.nn.LeakyReLU,
        downsample=True,
        dropout=0.15,
    ):
        super().__init__(input_shape=input_shape)
        self.append(
            Conv1d_2d, channels=channels, causal=causal, layer_name="conv_1"
        )
        self.append(sb.nnet.normalization.LayerNorm, layer_name="norm_1")
        self.append(activation(), layer_name="act_1")
        self.append(
            Conv1d_2d,
            channels=channels,
            causal=causal,
            stride=2 if downsample else 1,
            layer_name="conv_2"
        )
        self.append(sb.nnet.normalization.LayerNorm, layer_name="norm_2")
        self.append(activation(), layer_name="act_2")

        self.append(
            sb.nnet.dropout.Dropout2d(drop_rate=dropout), layer_name="drop"
        )


class Conv1d_2d(torch.nn.Module):
    "Applies conv_1d in both time and frequency"
    def __init__(
        self, input_shape, channels, kernel_size=3, stride=1, causal=False
    ):
        self.causal = causal
        if len(input_shape) == 3:
            self.unsqueeze = True
            in_channels = 1
        else:
            self.unsqueeze = False
            in_channels = input_shape[-1]

        # Use half the channels for time convolution
        self.out_channels = channels // 2
        self.time_conv = sb.nnet.CNN.Conv2d(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=(kernel_size, 1),
            stride=stride,
            padding="valid" if causal else "same",
        )

        # And half for freq convolution
        self.freq_conv = sb.nnet.CNN.Conv2d(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=(1, kernel_size),
            stride=stride,
            padding="same",
        )

    def forward(self, x):
        "Input is [batch, time, freq, channels?]"

        # Add channels if necessary
        if self.unsqueeze:
            x = x.unsqueeze(-1)

        freq_out = self.freq_conv(x)

        # Manage causal padding when necessary
        if self.causal:
            pad = (self.kernel_size - 1, 0, 0, 0, 0, 0)
            x = torch.nn.functional.pad(x, pad)

        time_out = self.time_conv(x)

        # Combine time + freq maps
        return torch.cat([freq_out, time_out], dim=-1)


class DNN_Block(sb.nnet.containers.Sequential):
    """Block for linear layers.

    Arguments
    ---------
    input_shape : tuple
        Expected shape of the input.
    neurons : int
        Size of the linear layers.
    activation : torch.nn.Module class
        Class definition to use for constructing activation layers.
    dropout : float
        Rate to use for dropping neurons.

    Example
    -------
    >>> inputs = torch.rand(10, 15, 128)
    >>> block = DNN_Block(input_shape=inputs.shape, neurons=64)
    >>> outputs = block(inputs)
    >>> outputs.shape
    torch.Size([10, 15, 64])
    """

    def __init__(
        self, input_shape, neurons, activation=torch.nn.LeakyReLU, dropout=0.15
    ):
        super().__init__(input_shape=input_shape)
        self.append(
            sb.nnet.linear.Linear, n_neurons=neurons, layer_name="linear",
        )
        self.append(sb.nnet.normalization.BatchNorm1d, layer_name="norm")
        self.append(activation(), layer_name="act")
        self.append(torch.nn.Dropout(p=dropout), layer_name="dropout")
