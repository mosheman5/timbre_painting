import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.stride_tricks import as_strided
from scipy.io import wavfile

# model = crepe.core.build_and_load_model('tiny')
#
# py_weights = {}
# for layer in model.layers:
#     if len(layer.weights) == 0:
#         continue
#
#     weights = layer.get_weights()
#     if 'conv' in layer.name:
#         layer_id = int(layer.name.split('/')[0].replace('conv', '').replace('-BN', ''))
#         conv_layer_id = 4 * (layer_id - 1)
#         bn_layer_id = conv_layer_id + 1
#
#         if 'BN' in layer.name:
#             #             print(layer.weights)
#             py_weights[f'convs.{bn_layer_id}.weight'] = torch.tensor(weights[0], dtype=torch.float32)
#             py_weights[f'convs.{bn_layer_id}.bias'] = torch.tensor(weights[1], dtype=torch.float32)
#             py_weights[f'convs.{bn_layer_id}.running_mean'] = torch.tensor(weights[2], dtype=torch.float32)
#             py_weights[f'convs.{bn_layer_id}.running_var'] = torch.tensor(weights[3], dtype=torch.float32)
#         else:
#             array = weights[0]
#             array = array.squeeze(1)
#             py_weights[f'convs.{conv_layer_id}.kernel.weight'] = torch.tensor(array.transpose((2, 1, 0)),
#                                                                               dtype=torch.float32).contiguous()
#             py_weights[f'convs.{conv_layer_id}.kernel.bias'] = torch.tensor(weights[1], dtype=torch.float32)
#
#     if 'classifier' == layer.name:
#         py_weights[f'classifier.weight'] = torch.tensor(weights[0], dtype=torch.float32).transpose(0, 1).contiguous()
#         py_weights[f'classifier.bias'] = torch.tensor(weights[1], dtype=torch.float32)
#
# # the model is trained on 16kHz audio
# model_srate = 16000


def conv1d_same_padding(input, weight, bias=None, stride=None, dilation=None):
    if dilation is None:
        dilation = [1]
    if stride is None:
        stride = [1]
    input_rows = input.size(2)
    filter_rows = weight.size(2)

    out_rows = (input_rows + stride[0] - 1) // stride[0]

    padding_rows = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)

    if rows_odd:
        input = F.pad(input, [0, int(rows_odd)])

    return F.conv1d(input, weight, bias, stride,
                    padding=(padding_rows // 2),
                    dilation=dilation)


class Conv1DPaddingSame(nn.Module):
    def __init__(self,
                 ch_in,
                 filters,
                 kernel_size,
                 strides,
                 padding,
                 dilation=1,
                 use_bias=True,
                 use_activation=True):
        super().__init__()
        assert padding == "SAME", "Only same padding is supported"
        self.stride = strides
        self.dilation = dilation
        self.use_bias = use_bias
        self.use_activation = use_activation

        self.kernel = nn.Conv1d(ch_in, filters, kernel_size=kernel_size,
                                stride=strides, dilation=dilation,
                                bias=use_bias)

    def forward(self, x, activation=True):
        x = conv1d_same_padding(x,
                                self.kernel.weight,
                                bias=self.kernel.bias,
                                stride=self.kernel.stride,
                                dilation=self.kernel.dilation)

        if self.use_activation and activation:
            return F.relu(x)
        else:
            return x


def build_and_load_model_pytorch(model_capacity):
    """
    Build the CNN model and load the weights
    Parameters
    ----------
    model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
        String specifying the model capacity, which determines the model's
        capacity multiplier to 4 (tiny), 8 (small), 16 (medium), 24 (large),
        or 32 (full). 'full' uses the model size specified in the paper,
        and the others use a reduced number of filters in each convolutional
        layer, resulting in a smaller model that is faster to evaluate at the
        cost of slightly reduced pitch estimation accuracy.
    Returns
    -------
    model : tensorflow.keras.models.Model
        The pre-trained keras model loaded in memory
    """
    layers = []

    capacity_multiplier = {
        'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32
    }[model_capacity]

    filters = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
    widths = [512, 64, 64, 64, 64, 64]
    strides = [4, 1, 1, 1, 1, 1]

    in_channels = 1
    for f, w, s in zip(filters, widths, strides):
        layers += [Conv1DPaddingSame(in_channels, f, kernel_size=w, strides=s, padding="SAME")]
        layers += [nn.BatchNorm1d(f)]
        layers += [nn.MaxPool1d(2)]
        layers += [nn.Dropout(0.25)]
        in_channels = f

    return nn.Sequential(*layers)


class Crepe(nn.Module):
    def __init__(self, model_capacity):
        super().__init__()
        self.convs = build_and_load_model_pytorch(model_capacity)
        self.classifier = nn.Linear(self.convs[-4].kernel.out_channels * 4, 360)

    def forward(self, x, layer=None):
        for idx, l in enumerate(self.convs):
            x = l(x)
            if layer == idx:
                x = x.transpose(1, 2).contiguous()
                x = x.view(x.shape[0], -1)
                return x

        x = x.transpose(1, 2)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        if layer == 'classifier':
            return x
        return torch.sigmoid(x)


def load_crepe(model_path, device, model_type='tiny'):
    py_model = Crepe(model_type)
    py_model.float()
    py_weights = torch.load(model_path)
    py_model.load_state_dict(py_weights)
    py_model.eval()
    py_model.to(device)
    return py_model

def main():
    py_model = Crepe('tiny')
    py_model.float()
    py_model.load_state_dict(py_weights)
    py_model.eval()

    filename = '/tmp/test.wav'
    sr, audio = wavfile.read(filename)

    center = True
    step_size = 10

    if len(audio.shape) == 2:
        audio = audio.mean(1)  # make mono
    audio = audio.astype(np.float32)
    if sr != model_srate:
        # resample audio if necessary
        from resampy import resample

        audio = resample(audio, sr, model_srate)

    # pad so that frames are centered around their timestamps (i.e. first frame
    # is zero centered).
    if center:
        audio = np.pad(audio, 512, mode='constant', constant_values=0)

    # make 1024-sample frames of the audio with hop length of 10 milliseconds
    hop_length = int(model_srate * step_size / 1000)
    n_frames = 1 + int((len(audio) - 1024) / hop_length)
    frames = as_strided(audio, shape=(1024, n_frames),
                        strides=(audio.itemsize, hop_length * audio.itemsize))
    frames = frames.transpose().copy()

    # normalize each frame -- this is expected by the model
    frames -= np.mean(frames, axis=1)[:, np.newaxis]
    frames /= np.std(frames, axis=1)[:, np.newaxis]

    ii = frames

    tf_out = model.predict(ii)
    py_out = py_model(torch.from_numpy(ii).view(ii.shape[0], 1, -1).float())
    print(py_out.shape, tf_out.shape)
    print(tf_out.squeeze()[:10])
    print(py_out.squeeze().data.numpy()[:10])
    print(np.linalg.norm(py_out.squeeze().data.numpy() - tf_out.squeeze()))
