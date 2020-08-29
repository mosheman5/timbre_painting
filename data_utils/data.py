from torch.utils.data import Dataset
import h5py
import numpy as np
import random
import torch
from utils.utils import norm_audio
import math
from data_utils.spectral_feats import calc_loudness

class F0Dataset(Dataset):
    def __init__(self, dataset_len, srs, input_path, duration, val_flag=False, train_ratio=0.85,
                 max_val=1, max_val_f0=0.5, max_sr=16000, noise=0,
                 rank=0, num_ranks=1, shuffle=True, epoch=0):
        self.srs = srs
        self.data_paths = [input_path.joinpath(f'{sr}.h5') for sr in srs]
        self.dataset_len = dataset_len
        self.duration = duration
        self.val_flag = val_flag
        self.train_ratio = train_ratio
        self.input_keys, self.input_f0_dict, self.input_weights = self.cache_f0(self.data_paths[0])
        self.input_data_dicts = self.cache_data(self.data_paths)
        self.max_val = max_val
        self.max_val_f0 = max_val_f0
        self.noise = noise
        self.sr_ratio = self.srs[-1] // self.srs[0]
        self.max_sr = max_sr
        # variables for distributed run
        self.epoch=epoch
        self.rank = rank
        self.num_ranks = num_ranks
        self.num_samples_input = int(math.ceil(len(self.input_keys) * 1.0 / self.num_ranks))
        self.total_size_input = self.num_samples_input * self.num_ranks
        self.shuffle = shuffle

    def cache_f0(self, f0_path):
        h5f = h5py.File(f0_path, 'r')
        cache = {}
        keys = self.return_keys(h5f)
        weights = np.empty(len(keys))
        for it, key in enumerate(keys):
            cache[key] = h5f[key][1, :]
            weights[it] = len(h5f[key][1, :]) - self.duration * self.srs[0]

        weights /= weights.min()
        h5f.close()

        return keys, cache, weights

    def cache_data(self, data_path):
        caches = []
        for file_path in data_path:
            h5f = h5py.File(file_path, 'r')
            cache = {}
            keys = self.return_keys(h5f)

            for key in keys:
                cache[key] = h5f[key][0, :]
            h5f.close()
            caches.append(cache)

        return caches

    def split_keys(self, keys, total_size, num_samples):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(keys), generator=g).tolist()
        else:
            indices = list(range(len(keys)))

        # add extra samples to make it evenly divisible
        indices += indices[:(total_size - len(indices))]
        assert len(indices) == total_size

        # subsample
        indices = indices[self.rank:total_size:self.num_ranks]
        assert len(indices) == num_samples

        return indices

    def set_epoch(self, epoch):
        self.epoch = epoch

    def split_keys_epoch(self):
        input_indices = self.split_keys(self.input_keys,
                                               self.total_size_input, self.num_samples_input)
        self.input_keys_rank = [self.input_keys[x] for x in input_indices]
        self.input_weights_rank = self.input_weights[input_indices]

    def return_keys(self, h5dataset):
        keys = list(h5dataset.keys())
        keys = [x for x in keys if 'loudness' not in x]
        keys.sort(key=int)
        len_keys = len(keys)
        thresh_ind = int(len_keys * self.train_ratio)
        if self.val_flag:
            return keys[thresh_ind:]
        else:
            return keys[:thresh_ind]

    def __getitem__(self, _):
        f0_input, data_input, loudness_list = self.choose_random_slice(self.input_keys_rank, self.input_weights_rank,
                                                        self.input_f0_dict, self.input_data_dicts)
        f0_input = norm_audio(torch.tensor(f0_input, dtype=torch.float).unsqueeze(0), max_val=self.max_val_f0)
        data_input = norm_audio(torch.tensor(data_input, dtype=torch.float).unsqueeze(0), max_val=self.max_val)
        loudness_list = [torch.tensor(loudness, dtype=torch.float).unsqueeze(0) for loudness in loudness_list]

        f0_input += torch.randn_like(f0_input) * self.noise

        return f0_input, data_input, loudness_list

    def choose_random_slice(self, keys, weights, f0_dict, data_dicts):
        rand_key = random.choices(keys, weights=weights)[0]
        f0_chosen = f0_dict[rand_key]
        data_list, loudness_list = [], []
        for data_dict in data_dicts:
            data = data_dict[rand_key]
            data_list.append(data)
        rand_start = random.randint(0, len(f0_chosen) - self.duration * self.srs[0] - 1)  # 1.05 just to be on the safe side

        f0_chosen = f0_chosen[rand_start:rand_start + self.duration * self.srs[0]]
        for it, (sr, data) in enumerate(zip(self.srs, data_list)):
            data = data[rand_start * sr // self.srs[0]:
                                  rand_start * sr // self.srs[0] + self.duration * sr]
            data_list[it] = data
            loudness = calc_loudness(data, sr, center=False, n_fft=2048 // (self.max_sr // sr), hop_size=32)
            loudness_list.append(loudness)

        return f0_chosen, data_list[-1], loudness_list

    def __len__(self):
        return self.dataset_len
