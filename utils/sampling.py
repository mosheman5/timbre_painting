import math
import torch
import torchaudio
import fractions
import utils.utils


def create_samplers(srs, device):
    samplers = []
    for sr_in, sr_out in zip(srs[:-1], srs[1:]):
        sampler = Sampler(orig_freq=sr_in, new_freq=sr_out, device=device)
        samplers.append(sampler)
    return samplers


def resample_torch(audio_tensor, orig_freq, new_freq, max_val=0.9, sampler=None):
    if not sampler:
        sampler = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq)
    if audio_tensor.dim() == 3:
        audio_tensor = audio_tensor.squeeze(1)
    out = sampler(audio_tensor)
    out = utils.utils.norm_audio(out, max_val)
    return out.unsqueeze(1)


class Sampler(torch.nn.Module):

    def __init__(self, orig_freq=16000, new_freq=16000, device='cpu'):
        super(Sampler, self).__init__()
        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.device = device
        self.init_weights()

    def forward(self, waveform, version='new'):
        return self.resample_waveform(waveform, version)

    # torchaudio shit
    def _get_LR_indices_and_weights(self, orig_freq, new_freq, output_samples_in_unit, window_width,
                                    lowpass_cutoff, lowpass_filter_width):
        r"""Based on LinearResample::SetIndexesAndWeights where it retrieves the weights for
        resampling as well as the indices in which they are valid. LinearResample (LR) means
        that the output signal is at linearly spaced intervals (i.e the output signal has a
        frequency of ``new_freq``). It uses sinc/bandlimited interpolation to upsample/downsample
        the signal.

        The reason why the same filter is not used for multiple convolutions is because the
        sinc function could sampled at different points in time. For example, suppose
        a signal is sampled at the timestamps (seconds)
        0         16        32
        and we want it to be sampled at the timestamps (seconds)
        0 5 10 15   20 25 30  35
        at the timestamp of 16, the delta timestamps are
        16 11 6 1   4  9  14  19
        at the timestamp of 32, the delta timestamps are
        32 27 22 17 12 8 2    3

        As we can see from deltas, the sinc function is sampled at different points of time
        assuming the center of the sinc function is at 0, 16, and 32 (the deltas [..., 6, 1, 4, ....]
        for 16 vs [...., 2, 3, ....] for 32)

        Example, one case is when the ``orig_freq`` and ``new_freq`` are multiples of each other then
        there needs to be one filter.

        A windowed filter function (i.e. Hanning * sinc) because the ideal case of sinc function
        has infinite support (non-zero for all values) so instead it is truncated and multiplied by
        a window function which gives it less-than-perfect rolloff [1].

        [1] Chapter 16: Windowed-Sinc Filters, https://www.dspguide.com/ch16/1.htm

        Args:
            orig_freq (float): The original frequency of the signal
            new_freq (float): The desired frequency
            output_samples_in_unit (int): The number of output samples in the smallest repeating unit:
                num_samp_out = new_freq / Gcd(orig_freq, new_freq)
            window_width (float): The width of the window which is nonzero
            lowpass_cutoff (float): The filter cutoff in Hz. The filter cutoff needs to be less
                than samp_rate_in_hz/2 and less than samp_rate_out_hz/2.
            lowpass_filter_width (int): Controls the sharpness of the filter, more == sharper but less
                efficient. We suggest around 4 to 10 for normal use

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of ``min_input_index`` (which is the minimum indices
            where the window is valid, size (``output_samples_in_unit``)) and ``weights`` (which is the weights
            which correspond with min_input_index, size (``output_samples_in_unit``, ``max_weight_width``)).
        """
        assert lowpass_cutoff < min(orig_freq, new_freq) / 2
        output_t = torch.arange(0, output_samples_in_unit, dtype=torch.get_default_dtype()) / new_freq
        min_t = output_t - window_width
        max_t = output_t + window_width

        min_input_index = torch.ceil(min_t * orig_freq)  # size (output_samples_in_unit)
        max_input_index = torch.floor(max_t * orig_freq)  # size (output_samples_in_unit)
        num_indices = max_input_index - min_input_index + 1  # size (output_samples_in_unit)

        max_weight_width = num_indices.max()
        # create a group of weights of size (output_samples_in_unit, max_weight_width)
        j = torch.arange(max_weight_width).unsqueeze(0)
        input_index = min_input_index.unsqueeze(1) + j
        delta_t = (input_index / orig_freq) - output_t.unsqueeze(1)

        weights = torch.zeros_like(delta_t)
        inside_window_indices = delta_t.abs().lt(window_width)
        # raised-cosine (Hanning) window with width `window_width`
        weights[inside_window_indices] = 0.5 * (1 + torch.cos(2 * math.pi * lowpass_cutoff /
                                                              lowpass_filter_width * delta_t[inside_window_indices]))

        t_eq_zero_indices = delta_t.eq(0.0)
        t_not_eq_zero_indices = ~t_eq_zero_indices
        # sinc filter function
        weights[t_not_eq_zero_indices] *= torch.sin(
            2 * math.pi * lowpass_cutoff * delta_t[t_not_eq_zero_indices]) / (math.pi * delta_t[t_not_eq_zero_indices])
        # limit of the function at t = 0
        weights[t_eq_zero_indices] *= 2 * lowpass_cutoff

        weights /= orig_freq  # size (output_samples_in_unit, max_weight_width)
        return min_input_index, weights

    # shit from torchaudio all the way
    def resample_waveform(self, waveform, version='new'):
        with torch.no_grad():
            assert self.first_indices.dim() == 1
            # all the weights have the same stride but have different padding.
            # Current implementation takes the input and applies the various padding before
            # doing a conv1d for that specific weight.
            conv_stride = self.input_samples_in_unit
            conv_transpose_stride = self.output_samples_in_unit
            num_channels, wave_len = waveform.size()
            window_size = self.weights.size(1)
            tot_output_samp = self._get_num_LR_output_samples(wave_len, self.orig_freq, self.new_freq)
            output = torch.zeros((num_channels, tot_output_samp),
                                 device=self.device)

            if version == 'old':
                # eye size: (num_channels, num_channels, 1)
                eye = torch.eye(num_channels, device=self.device).unsqueeze(2)
                # original option
                for i in range(self.first_indices.size(0)):
                    wave_to_conv = waveform
                    first_index = int(self.first_indices[i].item())
                    if first_index >= 0:
                        # trim the signal as the filter will not be applied before the first_index
                        wave_to_conv = wave_to_conv[..., first_index:]

                    # pad the right of the signal to allow partial convolutions meaning compute
                    # values for partial windows (e.g. end of the window is outside the signal length)
                    max_unit_index = (tot_output_samp - 1) // self.output_samples_in_unit
                    end_index_of_last_window = max_unit_index * conv_stride + window_size
                    current_wave_len = wave_len - first_index
                    right_padding = max(0, end_index_of_last_window + 1 - current_wave_len)

                    left_padding = max(0, -first_index)
                    if left_padding != 0 or right_padding != 0:
                        wave_to_conv = torch.nn.functional.pad(wave_to_conv, (left_padding, right_padding))

                    conv_wave = torch.nn.functional.conv1d(
                        wave_to_conv.unsqueeze(0), self.weights[i].repeat(num_channels, 1, 1),
                        stride=conv_stride, groups=num_channels)

                    # we want conv_wave[:, i] to be at output[:, i + n*conv_transpose_stride]
                    dilated_conv_wave = torch.nn.functional.conv_transpose1d(
                        conv_wave, eye, stride=conv_transpose_stride).squeeze(0)

                    # pad dilated_conv_wave so it reaches the output length if needed.
                    dialated_conv_wave_len = dilated_conv_wave.size(-1)
                    left_padding = i
                    right_padding = max(0, tot_output_samp - (left_padding + dialated_conv_wave_len))
                    dilated_conv_wave = torch.nn.functional.pad(
                        dilated_conv_wave, (left_padding, right_padding))[..., :tot_output_samp]

                    output += dilated_conv_wave

            elif version=='new':
                # my option
                for i in range(self.first_indices.size(0)):
                    wave_to_conv = waveform
                    first_index = int(self.first_indices[i].item())
                    if first_index >= 0:
                        # trim the signal as the filter will not be applied before the first_index
                        wave_to_conv = wave_to_conv[..., first_index:]

                    # pad the right of the signal to allow partial convolutions meaning compute
                    # values for partial windows (e.g. end of the window is outside the signal length)
                    max_unit_index = (tot_output_samp - 1) // self.output_samples_in_unit
                    end_index_of_last_window = max_unit_index * conv_stride + window_size
                    current_wave_len = wave_len - first_index
                    right_padding = max(0, end_index_of_last_window + 1 - current_wave_len)

                    left_padding = max(0, -first_index)
                    if left_padding != 0 or right_padding != 0:
                        wave_to_conv = torch.nn.functional.pad(wave_to_conv, (left_padding, right_padding))
                    if i == 0:
                        waves_to_conv = torch.zeros([self.first_indices.size(0), num_channels,
                                                     wave_to_conv.size(-1)], device=self.device)
                    waves_to_conv[i] = wave_to_conv
                waves_to_conv = waves_to_conv.permute(1, 0, 2)

                num_groups = self.first_indices.size(0)
                conv_wave = torch.nn.functional.conv1d(
                    waves_to_conv, self.weights.view(num_groups, 1, -1),
                    stride=conv_stride, groups=num_groups)

                # we want conv_wave[:, i] to be at output[:, i + n*conv_transpose_stride]
                eyes = torch.eye(1, device=self.device).unsqueeze(2)
                eyes = eyes.repeat(num_groups, 1, 1)
                dilated_conv_waves = torch.nn.functional.conv_transpose1d(
                    conv_wave, eyes, stride=conv_transpose_stride, groups=num_groups)

                dilated_conv_waves = dilated_conv_waves.permute(1, 0, 2)

                for i, dilated_conv_wave in enumerate(dilated_conv_waves):
                    # pad dilated_conv_wave so it reaches the output length if needed.
                    dialated_conv_wave_len = dilated_conv_wave.size(-1)
                    left_padding = i
                    right_padding = max(0, tot_output_samp - (left_padding + dialated_conv_wave_len))
                    dilated_conv_wave = torch.nn.functional.pad(
                        dilated_conv_wave, (left_padding, right_padding))[..., :tot_output_samp]

                    output += dilated_conv_wave

        return output

    def _get_num_LR_output_samples(self, input_num_samp, samp_rate_in, samp_rate_out):
        r"""Based on LinearResample::GetNumOutputSamples. LinearResample (LR) means that
        the output signal is at linearly spaced intervals (i.e the output signal has a
        frequency of ``new_freq``). It uses sinc/bandlimited interpolation to upsample/downsample
        the signal.

        Args:
            input_num_samp (int): The number of samples in the input
            samp_rate_in (float): The original frequency of the signal
            samp_rate_out (float): The desired frequency

        Returns:
            int: The number of output samples
        """
        # For exact computation, we measure time in "ticks" of 1.0 / tick_freq,
        # where tick_freq is the least common multiple of samp_rate_in and
        # samp_rate_out.
        samp_rate_in = int(samp_rate_in)
        samp_rate_out = int(samp_rate_out)

        tick_freq = self._lcm(samp_rate_in, samp_rate_out)
        ticks_per_input_period = tick_freq // samp_rate_in

        # work out the number of ticks in the time interval
        # [ 0, input_num_samp/samp_rate_in ).
        interval_length_in_ticks = input_num_samp * ticks_per_input_period
        if interval_length_in_ticks <= 0:
            return 0
        ticks_per_output_period = tick_freq // samp_rate_out
        # Get the last output-sample in the closed interval, i.e. replacing [ ) with
        # [ ].  Note: integer division rounds down.  See
        # http://en.wikipedia.org/wiki/Interval_(mathematics) for an explanation of
        # the notation.
        last_output_samp = interval_length_in_ticks // ticks_per_output_period
        # We need the last output-sample in the open interval, so if it takes us to
        # the end of the interval exactly, subtract one.
        if last_output_samp * ticks_per_output_period == interval_length_in_ticks:
            last_output_samp -= 1
        # First output-sample index is zero, so the number of output samples
        # is the last output-sample plus one.
        num_output_samp = last_output_samp + 1
        return num_output_samp

    def init_weights(self, lowpass_filter_width=6):

        min_freq = min(self.orig_freq, self.new_freq)
        lowpass_cutoff = 0.99 * 0.5 * min_freq

        assert lowpass_cutoff * 2 <= min_freq

        base_freq = fractions.gcd(int(self.orig_freq), int(self.new_freq))
        input_samples_in_unit = int(self.orig_freq) // base_freq
        output_samples_in_unit = int(self.new_freq) // base_freq

        self.input_samples_in_unit = input_samples_in_unit
        self.output_samples_in_unit = output_samples_in_unit

        window_width = lowpass_filter_width / (2.0 * lowpass_cutoff)
        first_indices, weights = self._get_LR_indices_and_weights(self.orig_freq, self.new_freq, output_samples_in_unit,
                                                             window_width, lowpass_cutoff, lowpass_filter_width)
        self.first_indices = first_indices
        self.weights = weights.to(self.device)

    def _lcm(self, a, b):
        return abs(a * b) // fractions.gcd(a, b)
