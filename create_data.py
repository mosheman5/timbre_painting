import librosa
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import hydra
from pathlib import Path
from functools import partial
from shutil import copyfile
import torch
from data_utils import crepe_pytorch
import numpy as np
import h5py
from data_utils import spectral_feats
from data_utils.melosynth import torch_sine
import operator
import functools
import json
import os


class LoudnessMetrics(object):

    def __init__(self, srs):
        self.srs = [str(sr) for sr in srs]
        self.reset()

    def reset(self):
        self.model_ld_max = {key: [] for key in self.srs}
        self.model_ld_mean = {key: 0 for key in self.srs}

    def accum_metrics(self, loudness, key):
        loudness_max = np.max(loudness)
        self.model_ld_max[key].append(loudness_max)
        self.model_ld_mean[key] += np.mean(loudness)

    def calc_metrics(self):
        num_samples = len(self.model_ld_max[self.srs[0]])
        self.model_ld_max = {key: np.mean(self.model_ld_max[key]) for key in self.srs}
        self.model_ld_mean = {key: self.model_ld_mean[key]/num_samples for key in self.srs}
        return self.model_ld_max, self.model_ld_mean


class ProcessData():
    def __init__(self, silence_thresh_dB, srs, device, crepe_model, crepe_path, seq_len, confidence_threshold,
    loudness, max_len):
        super().__init__()
        self.silence_thresh_dB = silence_thresh_dB
        self.srs = srs
        self.sr = srs[-1]
        self.device = torch.device(device)
        self.crepe = crepe_pytorch.load_crepe(crepe_path + '/' + crepe_model + '.pth', device, crepe_model)
        self.seq_len = seq_len
        self.confidence_threshold = confidence_threshold
        self.loudness = loudness
        self.max_len = max_len

    def process_indices(self, indices: list) -> list:
        max_len = self.max_len * self.sr

        def expand_long(indices_tuple: tuple) -> list:
            if indices_tuple[1] - indices_tuple[0] > max_len:
                ret = [(start, start+max_len) for start in np.arange(indices_tuple[0], indices_tuple[1] - max_len, max_len)]
                ret.append((ret[-1][-1], min(ret[-1][-1] + max_len, indices_tuple[1])))
                return ret
            else:
                return [indices_tuple]

        new_indices = [*map(expand_long, indices)]
        new_indices = functools.reduce(operator.concat, new_indices, [])
        new_indices = [x for x in new_indices if (x[1] - x[0] > self.seq_len * self.sr)]
        return new_indices

    def extract_f0(self, audio):
        def audio2sine(audio_torch, confidence_threshold):
            time, frequency, confidence, _ = spectral_feats.predict(audio_torch, self.sr, self.crepe,
                                                                       viterbi=True, center=True)
            if confidence.mean() < confidence_threshold:
                raise ValueError('Low f0 confidence')
            signal = torch_sine(n_samples=audio_torch.shape[1], freqs=frequency, fs=self.sr)
            return signal.numpy(), frequency

        audio = torch.from_numpy(audio).unsqueeze(0).to(self.device)
        sinewav, f0 = audio2sine(audio, self.confidence_threshold)

        return sinewav

    def save_audio(self, audio_dict, sine_dict, loudness_dict, h5_dict, counter):
        for sr_str, h5f in h5_dict.items():
            audio = np.stack([audio_dict[sr_str], sine_dict[sr_str]])
            h5f.create_dataset(f'{counter}', data=audio)
            h5f.create_dataset(f'{counter}_loudness', data=loudness_dict[sr_str])
        return counter + 1

    def init_h5(self, data_dir):
        return {str(sr): h5py.File(data_dir / f'{sr}.h5', 'w') for sr in self.srs}

    def close_h5(self, h5_dict):
        [h5f.close() for h5f in h5_dict.values()]

    def downsample_audio(self, audio, sinewave):
        audio_dict = {str(sr): librosa.resample(audio, self.srs[-1], sr) for sr in self.srs}
        sine_dict = {str(sr): librosa.resample(sinewave, self.srs[-1], sr) for sr in self.srs}
        return audio_dict, sine_dict

    def calc_loudness(self, audio_dict, loudness_metrics):
        def inner_calc_loudness(sr_str, audio):
            _sr = int(sr_str)
            _loudness = spectral_feats.calc_loudness(audio, _sr, center=False, n_fft=self.loudness.nfft // (self.sr // _sr),
                                                hop_size=self.loudness.hop_size)
            loudness_metrics.accum_metrics(_loudness, sr_str)
            return _loudness
        return {sr: inner_calc_loudness(sr, audio) for sr, audio in audio_dict.items()}

    def save_loudness_stats(self, loudness_metrics, output_dir):
        mean_max, mean = loudness_metrics.calc_metrics()
        stats_dict = {f'{sr}': {'average_max_loudness': str(mean_max[str(sr)]),
                      'mean_loudness': str(mean[str(sr)])} for sr in self.srs}
        with open(output_dir / 'loudness.json', 'w') as fp:
            json.dump(stats_dict, fp)
        loudness_metrics.reset()


    def run_on_files(self, data_dir, input_dir, output_dir, loudness_metrics):
        audio_files = list((input_dir/data_dir).glob('*.wav'))
        output_dir = output_dir / data_dir
        output_dir.mkdir(exist_ok=True)
        h5_dict = self.init_h5(output_dir)
        counter = 0
        for audio_file in tqdm(audio_files):
            # load and split files
            data, sr = librosa.load(audio_file.as_posix(), sr=self.sr)
            sounds_indices = librosa.effects.split(data, top_db=self.silence_thresh_dB)
            sounds_indices = self.process_indices(sounds_indices)
            if len(sounds_indices) == 0:
                continue

            for indices in sounds_indices:
                audio = data[indices[0]:indices[1]]
                try:
                    sine_wav = self.extract_f0(audio)
                except ValueError:
                    print('wild low f0 confidence appeared!')
                    continue
                audio_dict, sine_dict = self.downsample_audio(audio, sine_wav)
                loudness_dict = self.calc_loudness(audio_dict, loudness_metrics)
                counter = self.save_audio(audio_dict, sine_dict, loudness_dict, h5_dict, counter)

        self.save_loudness_stats(loudness_metrics, output_dir)
        self.close_h5(h5_dict)

    def run_on_dirs(self, input_dir: Path, output_dir: Path, loudness_metrics: LoudnessMetrics):
        folders = [x for x in input_dir.glob('./*') if x.is_dir()]
        for folder in tqdm(folders):
            self.run_on_files(folder.name, input_dir, output_dir, loudness_metrics)


def create_mono_urmp(instrument_key, audio_files, target_dir, instruments_dict):
    target_dir = target_dir / instruments_dict[instrument_key]
    if not target_dir.exists():
        target_dir.mkdir()
    cur_audio_files = [audio_file for audio_file in audio_files if f'_{instrument_key}_' in audio_file.name]
    [copyfile(audio_file, target_dir / audio_file.name) for audio_file in cur_audio_files]


@hydra.main(config_path="conf/data_config.yaml", strict=True)
def main(args):
    # Phase 0 - copy all urmp wavs to corresponding folders
    CWD = Path(hydra.utils.get_original_cwd())
    os.chdir(CWD)

    if args.urmp is not None:
        urmp_path = CWD / args.urmp.source_folder
        urmp_audio_files = list(urmp_path.glob(f'./*/{args.urmp.mono_regex}*.wav'))
        target_dir = CWD / 'data_tmp'
        target_dir.mkdir(exist_ok=True)
        create_mono_urmp_partial = partial(create_mono_urmp,
                                           audio_files=urmp_audio_files,
                                           target_dir=target_dir,
                                           instruments_dict=args.urmp.instruments)
        thread_map(create_mono_urmp_partial, list(args.urmp.instruments.keys()))

    # create hd5 datasets for each sample rate to be used later during training
    data_processor = hydra.utils.instantiate(args.data_processor)
    loudness_metrics = LoudnessMetrics(args.srs)
    data_processor.run_on_dirs(CWD / args.input_dir, CWD / args.output_dir, loudness_metrics)


if __name__ == "__main__":
    main()