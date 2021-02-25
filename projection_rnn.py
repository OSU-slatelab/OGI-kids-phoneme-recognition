import torch
import speechbrain as sb


class ProjectionRNN(sb.nnet.containers.LengthCapableSequential):
    def __init__(
        self,
        input_size,
        layers=2,
        rnn_size=1024,
        projection_size=320,
        time_reduction_layer=-1,
        rnn_class=sb.nnet.RNN.GRU,
        bidirectional=False,
        normalization=sb.nnet.normalization.LayerNorm,
        dropout=0.2,
    ):
        super().__init__(input_shape=[None, None, input_size])

        self.append(
            rnn_class,
            hidden_size=rnn_size,
            bidirectional=bidirectional,
            layer_name="rnn",
        )

        for i in range(layers - 1):
            if time_reduction_layer == i + 1:
                self.append(TimeReduction(factor=2))
            self.append(
                sb.nnet.linear.Linear,
                n_neurons=projection_size,
                layer_name="linear"
            )
            self.append(torch.nn.LeakyReLU())
            self.append(normalization, layer_name="norm")
            self.append(torch.nn.Dropout(p=dropout), layer_name="dropout")
            self.append(
                rnn_class,
                hidden_size=rnn_size,
                bidirectional=bidirectional,
                layer_name="rnn",
            )


class TimeReduction(torch.nn.Module):
    "Layer to reduce time dimension by concatenating frames"
    def __init__(self, factor=2):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return time_reduce(x, self.factor)


def time_reduce(x, factor):
    "Concatenate successive frames to reduce the time dimension"

    # Ensure time-dimension is divisible by factor
    frames_to_remove = x.size(1) % factor

    # Remove frames randomly from front or back
    leftmost = torch.randint(frames_to_remove + 1, (1,)).item()
    rightmost = x.size(1) - frames_to_remove + leftmost
    x = x[:, leftmost:rightmost]

    # Reshape frames to reduced time dimension and increased features
    shape = [x.size(0), x.size(1) // factor, x.size(2) * factor]
    return x.view(shape)
