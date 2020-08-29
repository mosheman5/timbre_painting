# Hirearchical Timbre-Painting and Articulation Generation

This repository provides an official PyTorch implementation of "Hirearchical Timbre-Painting and Articulation Generation"

Our method generates high-fidelity audio for a target instrument, based f0 and loudness signal.

During training, loudness and f0 signal are extracted from ground-truth signal, 
which enables us to convert the melody of any input instrument to the trained instrument - task also known as Timbre Transfer

  [**Audio Samples**](https://mosheman5.github.io/timbre_painting/)
| [**Paper**](https://Placeholder)


We suggest seperating the generation process into two consecutive phases:
* Articulation - We generate the backbone of the audio and the transition between notes. 
This is done on a low sample rate from the given condition, loudness and f0 inputs. 
We use a sine excitation based on the extracted f0 signal, 
hence using the generator as a Neural-Source-Filtering network rather than a classic GAN generator which is condition on random noise.
* Timbre Painting - The next phase is composed of timbre painting networks: each network gets as input the previously generated audio and serves as a 
learnable upsample network. Each timbre-painting networks adds sample-rate specific details to the audio clip.

ADD IMAGE HERE

## Dependencies
A conda environment file is available in the repository.
* Python 3.6 +
* Pytorch 1.0
* Torchvision
* librosa
* tqdm
* scipy
* soundfile

## Usage

### 1. Cloning the repository & setting up conda environment
```
$ git clone https://github.com/mosheman5/DNP.git
$ cd DNP/
```
For creating and activating the conda environment:
```
$ conda env create -f environment.yml
$ conda activate DNP
```
 
### 2. Testing

To test on the demo speech file:

```
$ python DNP.py --run_name demo --noisy_file demo.wav --samples_dir samples --save_every 50 --num_iter 5000 --LR 0.001
```

To test on any other audio file insert the file path after the ```--noisy_file``` option.

A jupyter notebook with visualization is available: ```dnp.ipynb```

## Reference
If you found this code useful, please cite the following paper:
```
@article{michelashvili2020denoising,
  title={Speech Denoising by Accumulating Per-Frequency Modeling Fluctuations},
  author={Michael Michelashvili and Lior Wolf},
  journal={arXiv preprint arXiv:1904.07612},
  year={2020}
}
```

## Acknowledgement
The implemantation of the network architecture is taken from [Wave-U-Net](https://github.com/f90/Wave-U-Net)
