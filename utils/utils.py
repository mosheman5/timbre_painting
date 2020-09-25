import torch
import librosa
import torch.utils.data
import math
import numpy as np
from data_utils.spectral_feats import predict, predict_voicing
from data_utils.crepe_pytorch import Crepe
from data_utils.melosynth import melosynth_direct
import soundfile as sf
from utils.sampling import resample_torch
import gdown
import os
import tarfile


PRETRAINED_MODEL_DICT = {'Violin': '1KEodWMgtWLynBlMIZdSlIjGvrPJ2TpNQ&',
                         'Saxophone': '1GNL1yCdGmcxSGdECtpUb5BqbRWSeyB6b',
                         'Trumpet': '1SMJMnw7RorAymxQpoy_e2vUJlRH3Xmcy',
                         'Cello': '1Nx4sUznH1cWUvDOdQLFd-v7ZZKKwQWZu'}


def load_audio(filepath, sr, max_val=0.9):
    audio, sr = librosa.core.load(filepath, sr)
    return norm_audio(audio, max_val)


def create_srs(sr, stop_scale, scale_factor):
    srs = []
    for i in range(0, stop_scale + 1, 1):
        scale = math.pow(scale_factor, stop_scale - i)
        sr_local = int(round(sr * scale, 5))
        srs.append(sr_local)
    return srs


def draw_f0(Gs, samplers, in_s, max_val, loudness_list):
    G_z = in_s
    with torch.no_grad():
        for G, sampler, loudness in zip(Gs, samplers, loudness_list):
            z_in = G_z
            G_z = G(z_in.detach(), loudness.detach())
            G_z = resample_torch(G_z, None, None, max_val=max_val, sampler=sampler)
    return G_z


def norm_audio(audio, max_val=0.9):
    if max_val:
        if len(audio.shape) == 2:
            audio = audio / audio.abs().max(dim=1)[0].view(-1, 1) * max_val
        elif len(audio.shape) == 1:
            audio = audio / np.max([audio.max(), abs(audio.min())]) * max_val
        else:
            raise Exception('accepts only dims == 1 or 2!')
    return audio


class BaseAudio():
    def __init__(self, model_path, device, unvoiced_flag):
        self.device = device
        self.model = self.load_model(model_path)
        self.unvoiced_flag = unvoiced_flag

    def load_model(self, model_path):
        py_model = Crepe('full')
        py_model.float()
        py_weights = torch.load(model_path)
        py_model.load_state_dict(py_weights)
        py_model.eval()
        py_model.to(self.device)
        return py_model

    def extract_f0(self, audio_in, sr, numpy_flag=False, sampler=None):
        if numpy_flag:
            audio_in = torch.from_numpy(audio_in).to(self.device)
        with torch.no_grad():
            time, frequency, confidence, _ = predict(audio_in, sr, self.model, viterbi=True, sampler=sampler)
            if self.unvoiced_flag:
                is_voiced = predict_voicing(confidence)
                frequency *= is_voiced
        return time, frequency

    def create_f0_audio(self, audio_in, sr, time, frequency, max_val, len_audio=None, sr_out=None):
        if not sr_out:
            sr_out = sr
        audio_out = melosynth_direct(time, frequency, sr_out, max_val=max_val)
        # fix audio length by concatenation last samples back in
        if len_audio:
            len_diff = len_audio - len(audio_out)
        else:
            len_diff = len(audio_in) - len(audio_out)

        if len_diff > 0:
            audio_out = np.concatenate((audio_out, audio_out[-len_diff:]))
        elif len_diff < 0:
            audio_out = audio_out[:len(audio_in)]
        return torch.from_numpy(audio_out).float()

    def forward(self, audio_in, sr, max_val, octave=1, sampler=None, return_raw=False, numpy_flag=False,
                sr_out=None, len_audio=None):
        time, frequency = self.extract_f0(audio_in, sr, sampler=sampler, numpy_flag=numpy_flag)
        audio_out = self.create_f0_audio(audio_in, sr, time, frequency * octave, max_val=max_val, sr_out=sr_out,
                                         len_audio=len_audio)
        if return_raw:
            return audio_out, time, frequency
        else:
            return audio_out


def write_torch_audio(filename, audio_tensor, sr):
    audio_tensor = audio_tensor.detach().squeeze().cpu().numpy()
    sf.write(filename, audio_tensor, sr)


class LossMeter:
    def __init__(self, name):
        self.name = name
        self.losses = []

    def reset(self):
        self.losses = []

    def add(self, val):
        self.losses.append(val)

    def summarize_epoch(self):
        if self.losses:
            return np.mean(self.losses)
        else:
            return 0

    def sum(self):
        return sum(self.losses)


def download_pretrained_model(tag, download_dir='.'):
    """Download pretrained model form google drive.
    Args:
        tag (str): Pretrained model tag.
        download_dir (str): Directory to save downloaded files.
    Returns:
        str: Path of downloaded model checkpoint.
    """
    assert tag in PRETRAINED_MODEL_DICT, f"{tag} model does not exists."
    id_ = PRETRAINED_MODEL_DICT[tag]
    output_path = f"{download_dir}/{tag}.tar.gz"
    os.makedirs(f"{download_dir}", exist_ok=True)
    if not os.path.exists(output_path):
        gdown.download(f"https://drive.google.com/uc?id={id_}", output_path, quiet=False)

    with tarfile.open(output_path, 'r:*') as tar:
        model_folder = tar.getnames()[0]
        tar.extractall(download_dir)
    model_path = os.path.join(download_dir, model_folder)
    assert os.path.exists(os.path.join(model_path, 'args.pth')), 'args.pth file nor found. please use default checkpointing process'
    assert os.path.isdir(model_path), 'Tar file contains more than main folder, please use default checkpointing process'

    return model_path
