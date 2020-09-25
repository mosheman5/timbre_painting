import librosa
import numpy as np
import torch

from data_utils.crepe_pytorch import Crepe

_NUM_SECS = 4
_AUDIO_RATE = 16000  # 16 kHz
_F0_AND_LOUDNESS_RATE = 250  # 250 Hz
_CREPE_FRAME_SIZE = 1024
_LD_N_FFT = 2048
_LD_RANGE = 120.0
_REF_DB = 20.7  # White noise, amplitude=1.0, n_fft=2048


def calc_loudness(audio, rate=_AUDIO_RATE, center=False, hop_size=16, n_fft=_LD_N_FFT):
    np.seterr(divide='ignore')

    """Compute loudness, add to example (ref is white noise, amplitude=1)."""
    # Copied from magenta/ddsp/spectral_ops.py
    # Get magnitudes.
    # hop_size = int(_AUDIO_RATE // _F0_AND_LOUDNESS_RATE)
    if center is False:
        # Add padding to the end
        n_samples_initial = int(audio.shape[-1])
        n_frames = int(np.ceil(n_samples_initial / hop_size))
        n_samples_final = (n_frames - 1) * hop_size + n_fft
        pad = n_samples_final - n_samples_initial
        audio = np.pad(audio, ((0, pad),), "constant")

    spectra = librosa.stft(
        audio, n_fft=n_fft, hop_length=hop_size, center=center).T

    # Compute power
    amplitude = np.abs(spectra)
    amin = 1e-20  # Avoid log(0) instabilities.
    power_db = np.log10(np.maximum(amin, amplitude))
    power_db *= 20.0

    # Perceptual weighting.
    frequencies = librosa.fft_frequencies(sr=rate, n_fft=n_fft)
    a_weighting = librosa.A_weighting(frequencies)[np.newaxis, :]
    loudness = power_db + a_weighting

    # Set dynamic range.
    loudness -= _REF_DB
    loudness = np.maximum(loudness, -_LD_RANGE)

    # Average over frequency bins.
    mean_loudness_db = np.mean(loudness, axis=-1)
    return mean_loudness_db.astype(np.float32)


def py_get_activation(audio, sr, model, center=False, step_size=10, layer=None, grad=False, sampler=None):
    """
    Parameters
    ----------
    audio : np.ndarray [shape=(N,) or (N, C)]
        The audio samples. Multichannel audio will be downmixed.
    sr : int
        Sample rate of the audio samples. The audio will be resampled if
        the sample rate is not 16 kHz, which is expected by the model.
    model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
        String specifying the model capacity; see the docstring of
        :func:`~crepe.core.build_and_load_model`
    center : boolean
        - If `True` (default), the signal `audio` is padded so that frame
          `D[:, t]` is centered at `audio[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `audio[t * hop_length]`
    step_size : int
        The step size in milliseconds for running pitch estimation.
    Returns
    -------
    activation : np.ndarray [shape=(T, 360)]
        The raw activation matrix
    """
    def get_activation_inner(audio):
        # assert sr == _AUDIO_RATE, f"Hardcoded to sample rate==16000, got {sr}"
        model_srate = 16000
        if sr != model_srate:
            # resample audio if necessary
            from torchaudio.compliance.kaldi import resample_waveform
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            if not sampler:
                audio = resample_waveform(audio, sr, model_srate).squeeze()
            else:
                audio = sampler(audio).squeeze()


        if audio.dim() == 3 or audio.dim() == 2:
            audio = audio.squeeze()
        # if audio.dim() == 2:
        #     audio = audio.mean(dim=0)  # make mono
        audio = audio.float()

        # pad so that frames are centered around their timestamps (i.e. first frame
        # is zero centered).
        if center:
            # audio = np.pad(audio, 512, mode='constant', constant_values=0)
            import torch.nn.functional as F
            audio = F.pad(audio, [512, 512])
            # assert False, "Center pad not supported"

        # make 1024-sample frames of the audio with hop length of 10 milliseconds
        hop_length = int(model_srate * step_size / 1000)
        n_frames = 1 + int((audio.shape[-1] - 1024) / hop_length)
        if len(audio.shape) > 1:
            frames = []
            for channel in audio:
                frames_iter = torch.as_strided(
                    channel,
                    size=(1024, n_frames),
                    stride=(1, hop_length)
                )
                frames.append(frames_iter)
            frames = torch.cat(frames, dim=1)
        else:
            frames = torch.as_strided(
                audio,
                size=(1024, n_frames),
                stride=(1, hop_length)
            )
        frames = frames.transpose(0, 1).contiguous()

        # normalize each frame -- this is expected by the model
        frames_mean = frames.mean(dim=1).unsqueeze(1)
        frames = frames - frames_mean

        frames_std = frames.std(dim=1).detach()
        frames_std_ = torch.ones(frames_std.shape).to(frames)
        frames_std_[frames_std != 0] = frames_std[frames_std != 0]
        frames_std_ = frames_std_.unsqueeze(1)
        frames = frames / frames_std_

        # run prediction and convert the frequency bin weights to Hz
        return model(frames.view(frames.shape[0], 1, -1), layer=layer)

    if grad:
        activation = get_activation_inner(audio)
        mask = (activation != activation)
        if mask.any():
            print(activation[mask])
        activation[mask].zero_()
        return activation
    else:
        with torch.no_grad():
            return get_activation_inner(audio).detach()


def to_local_average_cents(salience, center=None):
    """
    find the weighted average cents near the argmax bin
    """

    if not hasattr(to_local_average_cents, 'cents_mapping'):
        # the bin number-to-cents mapping
        to_local_average_cents.cents_mapping = (
                np.linspace(0, 7180, 360) + 1997.3794084376191)

    if salience.ndim == 1:
        if center is None:
            center = int(np.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = np.sum(
            salience * to_local_average_cents.cents_mapping[start:end])
        weight_sum = np.sum(salience)
        return product_sum / weight_sum
    if salience.ndim == 2:
        return np.array([to_local_average_cents(salience[i, :]) for i in
                         range(salience.shape[0])])

    raise Exception("label should be either 1d or 2d ndarray")


def to_viterbi_cents(salience):
    """
    Find the Viterbi path using a transition prior that induces pitch
    continuity.
    """
    from hmmlearn import hmm

    # uniform prior on the starting pitch
    starting = np.ones(360) / 360

    # transition probabilities inducing continuous pitch
    xx, yy = np.meshgrid(range(360), range(360))
    transition = np.maximum(12 - abs(xx - yy), 0)
    transition = transition / np.sum(transition, axis=1)[:, None]

    # emission probability = fixed probability for self, evenly distribute the
    # others
    self_emission = 0.1
    emission = (np.eye(360) * self_emission + np.ones(shape=(360, 360)) *
                ((1 - self_emission) / 360))

    # fix the model parameters because we are not optimizing the model
    model = hmm.MultinomialHMM(360, starting, transition)
    model.startprob_, model.transmat_, model.emissionprob_ = \
        starting, transition, emission

    # find the Viterbi path
    observations = np.argmax(salience, axis=1)
    path = model.predict(observations.reshape(-1, 1), [len(observations)])

    return np.array([to_local_average_cents(salience[i, :], path[i]) for i in
                     range(len(observations))])


def predict(audio, sr, model, viterbi=False, center=False, step_size=10, confidence_th=-1, sampler=None):
    """
    Perform pitch estimation on given audio
    Parameters
    ----------
    audio : np.ndarray [shape=(N,) or (N, C)]
        The audio samples. Multichannel audio will be downmixed.
    sr : int
        Sample rate of the audio samples. The audio will be resampled if
        the sample rate is not 16 kHz, which is expected by the model.
    model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
        String specifying the model capacity; see the docstring of
        :func:`~crepe.core.build_and_load_model`
    viterbi : bool
        Apply viterbi smoothing to the estimated pitch curve. False by default.
    center : boolean
        - If `True` (default), the signal `audio` is padded so that frame
          `D[:, t]` is centered at `audio[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `audio[t * hop_length]`
    step_size : int
        The step size in milliseconds for running pitch estimation.
    Returns
    -------
    A 4-tuple consisting of:
        time: np.ndarray [shape=(T,)]
            The timestamps on which the pitch was estimated
        frequency: np.ndarray [shape=(T,)]
            The predicted pitch values in Hz
        confidence: np.ndarray [shape=(T,)]
            The confidence of voice activity, between 0 and 1
        activation: np.ndarray [shape=(T, 360)]
            The raw activation matrix
    """
    activation = py_get_activation(audio, sr, model,
                                   center=center, step_size=step_size, sampler=sampler)

    activation = activation.squeeze().cpu().numpy()
    confidence = activation.max(axis=1)

    if confidence.mean() < confidence_th:
        return None, None, confidence, activation

    if viterbi:
        cents = to_viterbi_cents(activation)
    else:
        cents = to_local_average_cents(activation)

    frequency = 10 * 2 ** (cents / 1200)
    frequency[np.isnan(frequency)] = 0

    time = np.arange(confidence.shape[0]) * step_size / 1000.0
    return time, frequency, confidence, activation

def predict_voicing(confidence):
    """
    Find the Viterbi path for voiced versus unvoiced frames.
    Parameters
    ----------
    confidence : np.ndarray [shape=(N,)]
        voicing confidence array, i.e. the confidence in the presence of
        a pitch
    Returns
    -------
    voicing_states : np.ndarray [shape=(N,)]
        HMM predictions for each frames state, 0 if unvoiced, 1 if
        voiced
    """
    from hmmlearn import hmm

    # uniform prior on the voicing confidence
    starting = np.array([0.5, 0.5])

    # transition probabilities inducing continuous voicing state
    transition = np.array([[0.99, 0.01], [0.01, 0.99]])

    # mean and variance for unvoiced and voiced states
    means = np.array([[0.0], [1.0]])
    variances = np.array([[0.25], [0.25]])

    # fix the model parameters because we are not optimizing the model
    model = hmm.GaussianHMM(n_components=2)
    model.startprob_, model.covars_, model.transmat_, model.means_, model.n_features = \
        starting, variances, transition, means, 1

    # find the Viterbi path
    voicing_states = model.predict(confidence.reshape(-1, 1), [len(confidence)])

    return np.array(voicing_states)

def get_feats(audio, model, layer, grad=False, step_size=None):
    """Estimate the fundamental frequency using CREPE and add to example."""
    np.seterr(divide='ignore')

    if step_size is not None:
        crepe_step_size = step_size
        frame_rate = 1000 // crepe_step_size
        hop_size = _AUDIO_RATE / frame_rate
        n_samples = audio.shape[-1]
        n_frames = int(np.ceil(n_samples / hop_size))
        n_samples_padded = (n_frames - 1) * hop_size + _CREPE_FRAME_SIZE
        n_padding = (n_samples_padded - n_samples)
        assert n_padding % 1 == 0
        audio = torch.nn.functional.pad(audio, (0, int(n_padding)), mode="constant")
    else:
        # Copied from magenta/ddsp/spectral_ops.py
        # Pad end so that `num_frames = _NUM_SECS * _F0_AND_LOUDNESS_RATE`.
        hop_size = _AUDIO_RATE / _F0_AND_LOUDNESS_RATE
        n_samples = audio.shape[-1]
        n_frames = int(np.ceil(n_samples / hop_size))
        n_samples_padded = (n_frames - 1) * hop_size + _CREPE_FRAME_SIZE
        n_padding = (n_samples_padded - n_samples)
        assert n_padding % 1 == 0
        audio = torch.nn.functional.pad(audio, (0, int(n_padding)), mode="constant")
        crepe_step_size = 1000 // _F0_AND_LOUDNESS_RATE  # milliseconds

    feats = py_get_activation(
        audio,
        _AUDIO_RATE,
        model,
        center=False,
        step_size=crepe_step_size,
        layer=layer,
        grad=grad,
    )

    return feats.transpose(0, 1).contiguous().unsqueeze(0)


class CrepeLoss(torch.nn.Module):
    def __init__(self, model, layer=None, rate=16000):
        super().__init__()
        self.model = Crepe(model)
        self.model.load_state_dict(torch.load(f'/checkpoint/adampolyak/models/crepe/{model}.pth'))
        self.model.eval()
        self.layer = layer
        self.rate = rate

    def _get_feats(self, audio, grad=False):
        """Estimate the fundamental frequency using CREPE and add to example."""
        np.seterr(divide='ignore')

        # Copied from magenta/ddsp/spectral_ops.py
        crepe_step_size = 1000 // _F0_AND_LOUDNESS_RATE  # milliseconds

        feats = py_get_activation(
            audio,
            self.rate,
            self.model,
            center=True,
            step_size=crepe_step_size,
            layer=self.layer,
            grad=grad,
        )

        return feats.transpose(0, 1).contiguous().unsqueeze(0)

    def forward(self, y_, y):
        y_ = torch.clamp(y_, min=-1, max=1)

        y_feats = self._get_feats(y_.squeeze(), grad=True)
        label_feats = self._get_feats(y.squeeze(), grad=False)

        return torch.nn.functional.l1_loss(label_feats, y_feats)


def normalize_signal(signal):
    """
    Normalize float32 signal to [-1, 1] range
    """
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        return signal * 0
    else:
        return signal / max_val


def get_features(signal, sample_freq=16000,
                 window_size=20e-3, window_stride=10e-3,
                 nfilt=64, pad_to=8):
    signal = (normalize_signal(signal.astype(np.float32)) * 32767.0).astype(
        np.int16)
    audio_duration = len(signal) * 1.0 / sample_freq
    n_window_size = int(sample_freq * window_size)
    n_window_stride = int(sample_freq * window_stride)

    # making sure length of the audio is divisible by 8 (fp16 optimization)
    length = 1 + int(math.ceil(
        (1.0 * signal.shape[0] - n_window_size) / n_window_stride
    ))

    if pad_to > 0:
        if length % pad_to != 0:
            pad_size = (pad_to - length % pad_to) * n_window_stride
            signal = np.pad(signal, (0, pad_size), mode='constant')

    features = psf.logfbank(signal=signal,
                            samplerate=sample_freq,
                            winlen=window_size,
                            winstep=window_stride,
                            nfilt=nfilt,
                            nfft=512,
                            lowfreq=0, highfreq=sample_freq / 2,
                            preemph=0.97)
    if pad_to > 0:
        assert features.shape[0] % pad_to == 0

    mean = np.mean(features)
    std_dev = np.std(features)
    features = (features - mean) / std_dev
    return features, audio_duration


def get_crepe_f02(audio, rate, model):
    # Copied from magenta/ddsp/spectral_ops.py
    # Pad end so that `num_frames = _NUM_SECS * _F0_AND_LOUDNESS_RATE`.
    crepe_step_size = 1000 // _F0_AND_LOUDNESS_RATE  # milliseconds

    _, f0, confidence, _ = predict(
        audio,
        sr=rate,
        model=model,
        viterbi=True,
        step_size=crepe_step_size,
        center=True
    )

    confidence = np.nan_to_num(confidence)

    return f0, confidence
