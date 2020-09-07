import torch
import hydra
from pathlib import Path
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from utils.utils import create_srs, BaseAudio, load_audio
from utils.sampling import resample_torch, create_samplers
from models.networks import ParallelWaveGANGenerator
import soundfile as sf
import librosa
from data_utils.spectral_feats import calc_loudness
import json
import os


def load_norm_dicts(loudness_path):
    norm_dicts = {}
    with open(loudness_path, 'r') as fp:
        norm_dict = json.load(fp)
    for key, value in norm_dict.items():
        for k, v in value.items():
            norm_dict[key][k] = float(norm_dict[key][k])
    return norm_dicts


def shift_ld(loudness, ld_shift=0.0):
    """Shift loudness for normalization"""
    loudness += ld_shift
    return loudness


def norm_loudness(loudness, norm_dict):
    ld_max = np.max(loudness)
    ld_diff_max = norm_dict['average_max_loudness'] - ld_max
    loudness = shift_ld(loudness, ld_diff_max)
    ld_mean = np.mean(loudness) # TODO check if needed
    ld_diff_mean = norm_dict['mean_loudness'] - ld_mean
    return shift_ld(loudness, ld_diff_mean)


def load_trained_pyramid(trained_dir, network_params, srs, device=None):

    Gs = []
    dir_list = [trained_dir.joinpath(f'{x}') for x in range(0,len(srs))]
    for scale_num, directory in enumerate(dir_list):
        G = ParallelWaveGANGenerator(**network_params).to(device)
        state_dict = torch.load(directory.joinpath('last.pth'))
        G.load_state_dict(state_dict["model"]['generator'])
        G.eval()
        Gs.append(G)
    return Gs


def f0_transfer(real_audio, loudness_list, Gs, samplers, max_val, save_all=False):

    with torch.no_grad():

        prev_audios = []
        prev_in = real_audio

        for it, (G, sampler, loudness) in enumerate(zip(Gs, samplers, loudness_list)):

            audio_curr = G(prev_in.detach(), loudness.detach())
            prev_in = audio_curr[:, :, :prev_in.shape[-1]]
            if save_all:
                prev_audios.append(audio_curr.detach())
            prev_in = resample_torch(prev_in, None, None, max_val=max_val, sampler=sampler)

        audio_curr = Gs[-1](prev_in.detach(), loudness_list[-1].detach())
        prev_audios.append(audio_curr.detach())

    return prev_audios


def calc_loudness_list(audio, srs, device, sr_in=16000, norm_dicts=None):
    loudness_list = []
    for sr in srs:
        audio_scale = librosa.resample(audio, sr_in, sr)
        loudness = calc_loudness(audio_scale, sr, center=False, n_fft=2048 // (sr_in//sr), hop_size=32)
        if norm_dicts:
            norm_loudness(loudness, norm_dicts[sr])
        loudness_list.append(torch.tensor(loudness, dtype=torch.float).view(1,1,-1).to(device))
    return loudness_list


def save_audios(output_dirpath, audio_outputs, filename, srs):

    for it, (audio_file, sr) in enumerate(zip(audio_outputs, srs)):
        audio_file = audio_file.clamp(-1, 1).squeeze().detach().cpu().numpy()
        sf.write(
            output_dirpath.joinpath(f"{filename}_scale{it}.wav"),
            audio_file,
            sr
        )
    audio_file = audio_outputs[-1].clamp(-1, 1).squeeze().detach().cpu().numpy()
    sf.write(
        output_dirpath.joinpath(f"{filename}_f0_input.wav"),
        audio_file,
        srs[-1]
    )

@hydra.main(config_path="conf/transfer_config.yaml", strict=True)
def main(args):
    CWD = Path(hydra.utils.get_original_cwd())
    os.chdir(CWD)
    # Load model args
    trained_dirpath = Path(args.trained_dirpath)
    run_args = torch.load(trained_dirpath / 'args.pth')

    # define args from trained model
    sr = run_args.sr
    num_scales = run_args.num_scales
    scale_factor = run_args.scale_factor
    max_value = run_args.max_val
    max_value_f0 = run_args.max_val_f0
    cond_freq = run_args.cond_freq

    # Convert filepaths
    input_dirpath = Path(args.input_dirpath)
    input_files = input_dirpath.glob('*.wav')

    output_dirpath = trained_dirpath.joinpath(args.exp_name)
    try:
        output_dirpath.mkdir()
    except FileExistsError:
        print('Directory already exists')

    # Pytorch device
    device = torch.device("cuda")

    #load input file
    base_audio = BaseAudio(args.crepe_path, device, args.unvoiced_flag)
    srs = create_srs(sr, num_scales, scale_factor)
    samplers = create_samplers(srs, device=device)

    if args.norm_loudness_flag:
        norm_dicts = load_norm_dicts(trained_dirpath / 'loudness.json')
    else:
        norm_dicts = None

    octave_shifts = [2**x for x in args.octaves]

    # Load Trained models
    Gs = load_trained_pyramid(trained_dirpath, network_params=run_args.generator_params, device=device, srs=srs)

    for filepath in input_files:
        for octave in octave_shifts:
            real_audio = load_audio(filepath, sr, max_value)
            loudness_hop = 8 * sr // cond_freq
            real_audio = real_audio[:len(real_audio) // loudness_hop * loudness_hop]
            loudness_list = calc_loudness_list(audio=real_audio, srs=srs, device=device,
                                               sr_in=sr, norm_dicts=norm_dicts)
            real_audio = base_audio.forward(real_audio, sr, max_value_f0, numpy_flag=True, octave=octave)

            real_audio_orig = real_audio[None, None, ...].to(device)
            # resample input to the wanted scale
            real_audio = resample_torch(real_audio_orig, sr, srs[0], max_val=max_value_f0)

            audio_outputs = f0_transfer(real_audio,loudness_list, Gs, samplers, max_val=max_value, save_all = False)

            # add f0 sine input
            audio_outputs.append(real_audio_orig)
            save_audios(output_dirpath, audio_outputs,f'{filepath.stem}_{octave}', [srs[-1]])


if __name__ == "__main__":
     main()

