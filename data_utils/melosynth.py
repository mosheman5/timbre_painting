#!/usr/bin/env python
# CREATED: 6/13/14 10:57 AM by Justin Salamon <justin.salamon@nyu.edu>

"""
@file melosynth.py
@author  Justin Salamon <www.justinsalamon.com>
@version 0.1.1
@section DESCRIPTION
MeloSynth: synthesize a melody
MeloSynth is a python script to synthesize melodies represented as a sequence of
pitch (frequency) values. It was written to synthesize the output of the
MELODIA Melody Extraction Vamp Plugin (http://mtg.upf.edu/technologies/melodia),
but can be used to synthesize any pitch sequence represented as a two-column txt
or csv file where the first column contains timestamps and the second contains
the corresponding frequency values in Hertz.
@section USAGE
usage: melosynth.py [-h] [--output OUTPUT] [--fs FS] [--nHarmonics NHARMONICS]
                    [--square] [--useneg] [--batch]
                    inputfile
positional arguments:
  inputfile             Path to input file containing the pitch sequence
optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT       Path to output wav file. If not specified a file will
                        be created with the same path/name as inputfile but
                        ending with "_melosynth.wav".
  --fs FS               Sampling frequency for the synthesized file. If not
                        specified the default value of 16000 Hz is used.
  --nHarmonics NHARMONICS
                        Number of harmonics (including the fundamental) to use
                        in the synthesis (default is 1). As the number is
                        increased the wave will become more sawtooth-like.
  --square              Converge to square wave instead of sawtooth as the
                        number of harmonics is increased.
  --useneg              By default, negative frequency values (unvoiced
                        frames) are synthesized as silence. Setting the
                        --useneg option will synthesize these frames using
                        their absolute values (i.e. as voiced frames).
  --batch               Treat inputfile as a folder and batch process every
                        file within this folder that ends with .csv or .txt.
                        If --output is specified it is expected to be a folder
                        too. If --output is not specified, all synthesized
                        files will be saved into the input folder.
@section EXAMPLES
Basic usage, without any options:
>python melosynth.py ~/Documents/daisy3_melodia.csv
This will create a file called daisy3_melodia_melosynth.wav in the same folder
as the input file (~/Documents/) and use all the default parameter values for
the synthesis.
Advanced usage, including options:
>python melosynth.py ~/Documents/daisy3_melodia.csv --output ~/Music/mynewfile.wav --fs 44100 --nHarmonics 10 --square --useneg
Here we are providing a specified path for the output instead of the default
location. Next we specify the sample rate for the output (44.1 kHz) instead of
the default value of 16000 Hz. Next, we specify the number of harmonics to use
(10) instead of the default value of 1. Normally, as the number of harmonics is
increased the waveform will converge to a sawtooth wave, however, since we
specify the --square option, it will converge to a square wave instead. Finally,
by specifying the --useneg (use negative) option we make the script use the
absolute value of the frequencies so that negative frequencies are not
synthesized as silence (which is the default behaviour).
Batch processing:
>python melosynth.py ~/Documents/melodia_pitch/ --output ~/Documents/melodia_synth/ --batch
This will batch process all files ending with .txt or .csv in the melodia_pitch
folder, and save the synthesized melodies into the melodia_synth folder. Every
synthesized file will have the same name as its corresponding input file but
with the ending _melosynth.wav.
@section INSTALLATION
Simply download the script and run it from your terminal as instructed above.
Dependencies: python (tested on 2.7) and numpy (http://www.numpy.org/)
@section LICENSE
MeloSynth: synthesize a melody
Copyright (C) 2014 Justin Salamon.
MeloSynth is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
MeloSynth is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

import argparse, os, wave, logging, glob
import numpy as np
import torch
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

short_version = '0.1'
version = '0.1.1'


def wavwrite(x, filename, fs=44100, N=16):
    '''
    Synthesize signal x into a wavefile on disk. The values of x must be in the
    range [-1,1].
    :parameters:
    - x : numpy.array
    Signal to synthesize.
    - filename: string
    Path of output wavfile.
    - fs : int
    Sampling frequency, by default 44100.
    - N : int
    Bit depth, by default 16.
    '''

    maxVol = 2**15-1.0 # maximum amplitude
    x = x * maxVol # scale x
    # convert x to string format expected by wave
    signal = b"".join((wave.struct.pack('h', int(item)) for item in x))
    wv = wave.open(filename, 'w')
    nchannels = 1
    sampwidth = int(N / 8) # in bytes
    framerate = fs
    nframe = 0 # no limit
    comptype = 'NONE'
    compname = 'not compressed'
    wv.setparams((nchannels, sampwidth, framerate, nframe, comptype, compname))
    wv.writeframes(signal)
    wv.close()


def loadmel(inputfile, delimiter=None):
    '''
    Load a pitch (frequency) time series from a file.
    The pitch file must be in the following format:
    Double-column - each line contains two values, separated by ``delimiter``:
    the first contains the timestamp, and the second contains its corresponding
    frequency value in Hz.
    :parameters:
    - inputfile : str
    Path to pitch file
    - delimiter : str
    Column separator. By default, lines will be split by any amount of
    whitespace, unless the file ending is .csv, in which case a comma ','
    is used as the delimiter.
    :returns:
    - times : np.ndarray
    array of timestamps (float)
    - freqs : np.ndarray
    array of corresponding frequency values (float)
    '''
    if os.path.splitext(inputfile)[1] == '.csv':
        delimiter = ','
    try:
        data = np.loadtxt(inputfile, 'float', '#', delimiter)
    except ValueError:
        raise ValueError('Error: could not load %s, please check if it is in \
        the correct 2 column format' % os.path.basename(inputfile))

    # Make sure the data is in the right format
    data = data.T
    if data.shape[0] != 2:
        raise ValueError('Error: %s should be of dimension (2,x), but is of \
        dimension %s' % (os.path.basename(inputfile), data.shape))
    times = data[0]
    freqs = data[1]
    return times, freqs


def melosynth_batch(inputfolder, outputfolder, fs, nHarmonics, square, useneg):
    '''
    Run melosynth on every .txt and .csv file in inputfolder, and save the
    synthesized files to outputfolder. If outputfolder is None, the files are
    saved to intputfolder instead.
    :parameters:
    - inputfolder : str
    Path to input folder containing all the files with pitch sequences.
    - outputfolder: str
    Path to output folder. If outputfolder is None all files will be saved to
    inputfolder. In either case, each output file will be created with the same
    name as its corresponding inputfile but ending with "_melosynth.wav"
    - fs : int
    Sampling frequency for the synthesized file.
    - nHarmonics : int
    Number of harmonics (including the fundamental) to use in the synthesis
    (default is 1). As the number is increased the wave will become more
    sawtooth-like.
    - square : bool
    When set to true, the waveform will converge to a square wave instead of
    a sawtooth as the number of harmonics is increased.
    - useneg : bool
    By default, negative frequency values (unvoiced frames) are synthesized as
    silence. If useneg is set to True, these frames will be synthesized using
    their absolute values (i.e. as voiced frames).
    '''

    # Load all files in input folder that end with .txt or .csv
    inputfiles = glob.glob(os.path.join(inputfolder, "*.txt"))
    inputfiles.extend(glob.glob(os.path.join(inputfolder, "*.csv")))

    for inputfile in inputfiles:

        if outputfolder is not None:
            outfolder = outputfolder
            if not os.path.isdir(outfolder):
                os.mkdir(outfolder)
        else:
            outfolder = inputfolder

        outputfilename = os.path.basename(inputfile)[:-4] + "_melosynth.wav"
        outputfile = os.path.join(outfolder, outputfilename)
        logging.info("Processing: " + inputfile)
        logging.info("Target    : " + outputfile)

        melosynth(inputfile, outputfile, fs, nHarmonics, square, useneg)


def melosynth(inputfile, outputfile, fs, nHarmonics, square, useneg):
    '''
    Load pitch sequence from  a txt/csv file and synthesize it into a .wav
    :parameters:
    - inputfile : str
    Path to input file containing the pitch sequence.
    - outputfile: str
    Path to output wav file. If outputfile is None a file will be
    created with the same path/name as inputfile but ending with
    "_melosynth.wav"
    - fs : int
    Sampling frequency for the synthesized file.
    - nHarmonics : int
    Number of harmonics (including the fundamental) to use in the synthesis
    (default is 1). As the number is increased the wave will become more
    sawtooth-like.
    - square : bool
    When set to true, the waveform will converge to a square wave instead of
    a sawtooth as the number of harmonics is increased.
    - useneg : bool
    By default, negative frequency values (unvoiced frames) are synthesized as
    silence. If useneg is set to True, these frames will be synthesized using
    their absolute values (i.e. as voiced frames).
    '''

    # Preprocess input parameters
    fs = int(float(fs))
    nHarmonics = int(nHarmonics)
    if outputfile is None:
        outputfile = inputfile[:-4] + "_melosynth.wav"

    # Load pitch sequence
    logging.info('Loading data...')
    times, freqs = loadmel(inputfile)

    # Preprocess pitch sequence
    if useneg:
        freqs = np.abs(freqs)
    else:
        freqs[freqs < 0] = 0
    # Impute silence if start time > 0
    if times[0] > 0:
        estimated_hop = np.median(np.diff(times))
        prev_time = max(times[0] - estimated_hop, 0)
        times = np.insert(times, 0, prev_time)
        freqs = np.insert(freqs, 0, 0)


    logging.info('Generating wave...')
    signal = []

    translen = 0.010 # duration (in seconds) for fade in/out and freq interp
    phase = np.zeros(nHarmonics) # start phase for all harmonics
    f_prev = 0 # previous frequency
    t_prev = 0 # previous timestamp
    for t, f in zip(times, freqs):

        # Compute number of samples to synthesize
        nsamples = int(np.round((t - t_prev) * fs))

        if nsamples > 0:
            # calculate transition length (in samples)
            translen_sm = float(min(np.round(translen*fs), nsamples))

            # Generate frequency series
            freq_series = np.ones(nsamples) * f_prev

            # Interpolate between non-zero frequencies
            if f_prev > 0 and f > 0:
                freq_series += np.minimum(np.arange(nsamples)/translen_sm, 1) *\
                               (f - f_prev)
            elif f > 0:
                freq_series = np.ones(nsamples) * f

            # Repeat for each harmonic
            samples = np.zeros(nsamples)
            for h in range(nHarmonics):
                # Determine harmonic num (h+1 for sawtooth, 2h+1 for square)
                hnum = 2*h+1 if square else h+1
                # Compute the phase of each sample
                phasors = 2 * np.pi * (hnum) * freq_series / float(fs)
                phases = phase[h] + np.cumsum(phasors)
                # Compute sample values and add
                samples += np.sin(phases) / (hnum)
                # Update phase
                phase[h] = phases[-1]

            # Fade in/out and silence
            if f_prev == 0 and f > 0:
                samples *= np.minimum(np.arange(nsamples)/translen_sm, 1)
            if f_prev > 0 and f == 0:
                samples *= np.maximum(1 - (np.arange(nsamples)/translen_sm), 0)
            if f_prev == 0 and f == 0:
                samples *= 0

            # Append samples
            signal.extend(samples)

        t_prev = t
        f_prev = f

    # Normalize signal
    signal = np.asarray(signal)
    signal *= 0.8 / float(np.max(signal))

    logging.info('Saving wav file...')
    wavwrite(np.asarray(signal), outputfile, fs)


def melosynth_direct(times, freqs, fs, nHarmonics=1, square=False, useneg=False, max_val=0.5):

    # Preprocess input parameters
    fs = int(float(fs))
    nHarmonics = int(nHarmonics)

    # Preprocess pitch sequence
    if useneg:
        freqs = np.abs(freqs)
    else:
        freqs[freqs < 0] = 0
    # Impute silence if start time > 0
    if times[0] > 0:
        estimated_hop = np.median(np.diff(times))
        prev_time = max(times[0] - estimated_hop, 0)
        times = np.insert(times, 0, prev_time)
        freqs = np.insert(freqs, 0, 0)

    signal = []

    translen = 0.010 # duration (in seconds) for fade in/out and freq interp
    phase = np.zeros(nHarmonics) # start phase for all harmonics
    f_prev = 0 # previous frequency
    t_prev = 0 # previous timestamp
    for t, f in zip(times, freqs):

        # Compute number of samples to synthesize
        nsamples = int(np.round((t - t_prev) * fs))

        if nsamples > 0:
            # calculate transition length (in samples)
            translen_sm = float(min(np.round(translen*fs), nsamples))

            # Generate frequency series
            freq_series = np.ones(nsamples) * f_prev

            # Interpolate between non-zero frequencies
            if f_prev > 0 and f > 0:
                freq_series += np.minimum(np.arange(nsamples)/translen_sm, 1) *\
                               (f - f_prev)
            elif f > 0:
                freq_series = np.ones(nsamples) * f

            # Repeat for each harmonic
            samples = np.zeros(nsamples)
            for h in range(nHarmonics):
                # Determine harmonic num (h+1 for sawtooth, 2h+1 for square)
                hnum = 2*h+1 if square else h+1
                # Compute the phase of each sample
                phasors = 2 * np.pi * (hnum) * freq_series / float(fs)
                phases = phase[h] + np.cumsum(phasors)
                # Compute sample values and add
                samples += np.sin(phases) / (hnum)
                # Update phase
                phase[h] = phases[-1]

            # Fade in/out and silence
            if f_prev == 0 and f > 0:
                samples *= np.minimum(np.arange(nsamples)/translen_sm, 1)
            if f_prev > 0 and f == 0:
                samples *= np.maximum(1 - (np.arange(nsamples)/translen_sm), 0)
            if f_prev == 0 and f == 0:
                samples *= 0

            # Append samples
            signal.extend(samples)

        t_prev = t
        f_prev = f

    # Normalize signal
    signal = np.asarray(signal)
    signal *= max_val / float(signal.max())

    return signal


def torch_sine(n_samples, freqs, fs, max_val=0.5):

    assert len(freqs.shape) == 1
    # upsample freqs
    freqs = resample(freqs, n_timesteps=n_samples)

    # calc sin wav
    omegas = freqs * (2.0 * np.pi)  # rad / sec
    omegas = omegas / float(fs)  # rad / sample

    # Accumulate phase and synthesize.
    phases = torch.cumsum(omegas, dim=0)
    signal = torch.sin(phases)
    signal*=max_val

    return signal


def resample(inputs, n_timesteps, mode='linear', align_corners=False):

    inputs = to_torch(inputs)
    is_1d = len(inputs.shape) == 1
    is_2d = len(inputs.shape) == 2

  # Ensure inputs are at least 3d.
    if is_1d:
        inputs = inputs.view(1, 1, -1)
    elif is_2d:
        inputs = inputs.unsqueeze(0)
    # Perform resampling.
    outputs = torch.nn.functional.interpolate(inputs, size=n_timesteps, mode=mode, align_corners=align_corners)

  # Return outputs to the same dimensionality of the inputs.
    if is_1d:
        outputs = outputs[0, 0, :]
    elif is_2d:
        outputs = outputs[0, :, :]

    return outputs


def to_torch(x):
    """Ensure array/tensor is a torch tensor"""
    if torch.is_tensor(x):
        return x
    else:
        return torch.from_numpy(x)
