from pathlib import Path

import numpy as np
import scipy.io
import scipy

import torch
from torch.utils.data import TensorDataset, Dataset
from typing import Union, List

class SSVEP_our_lab_sessions(Dataset):
    """
    fs: 256Hz
    num examples: 15360 (60sec)
    only one subject
    """
    channels = ['O1', 'Oz', 'O2', 'Cz', 'Fp1', 'ObokOka', 'Kark', 'Policzek', 'Szczeka']
    target_freqs = [6, 7, 8]
    fs = 256
    num_subjects = 1
    num_targets = 3

    def __init__(self,
                 dataset_dir,
                 *,
                 file_prefixes: Union[str, List[str]] = None,
                 channels_selected=None,
                 targets_selected=None,
                 segment_length=256,
                 overlap=0,
                 transform=None,
                 target_transform=None,
                 percent_of_data=None,
                 percent_of_best_snr=None):
        file_prefixes = [file_prefixes] if isinstance(file_prefixes, str) else file_prefixes
        self.file_prefixes = file_prefixes
        self.percent_of_data = percent_of_data
        self.percent_of_best_snr = percent_of_best_snr
        self.transform = transform
        self.target_transform = target_transform
        self.channel_indices = self._get_channel_indices(channels_selected)
        self.target_indices = self._get_target_indices(targets_selected)
        dataset_dir = Path(dataset_dir)

        data, targets = self._load_data(dataset_dir, segment_length, overlap, self.percent_of_best_snr)
        self._make_dataset(data, targets)

    def _get_channel_indices(self, channels_selected):
        if channels_selected is None:
            self.channels_selected = self.channels
        else:
            self.channels_selected = channels_selected
        ch_indices = [self.channels.index(ch) for ch in self.channels_selected]
        return ch_indices

    def _get_target_indices(self, targets_selected):
            if targets_selected is None:
                self.targets_selected = self.target_freqs
            else:
                self.targets_selected = targets_selected
            target_indices = [self.target_freqs.index(target) for target in self.targets_selected]
            return target_indices

    def _make_dataset(self, data, targets):
        data = torch.Tensor(data)
        targets = torch.Tensor(targets).type(torch.LongTensor)
        self.dataset = TensorDataset(data, targets)

    def _segment_data(self, data, segment_length, overlap):
        num_segments = (data.shape[0] - segment_length) // (segment_length - overlap) + 1
        segments = np.zeros((segment_length, num_segments, data.shape[1]))
        for i in range(num_segments):
            start = i * (segment_length - overlap)
            end = start + segment_length
            segments[:, i, :] = data[start:end, :]
        return segments

    def _load_data(self, dataset_dir, segment_length, overlap, percent_of_best_snr):
        data = []
        targets = []
        print('reading files')

        for target_idx in self.target_indices:
            target = self.target_freqs[target_idx]
            file_prefix = self.file_prefixes[0] if self.file_prefixes else ''
            file_path = dataset_dir / f'{file_prefix}{target}Hz.mat'
            mat = scipy.io.loadmat(file_path)
            arr = mat['X'] # shape: (num_samples, num_channels)
            arr = arr[:, self.channel_indices] # select channels
            arr = self._segment_data(arr, segment_length, overlap)
            arr = arr.transpose(1, 2, 0) # shape: (num_segments, num_channels, num_samples)
            data.append(arr)
            targets.extend([target_idx] * arr.shape[0])
        data = np.stack(data)
        data = data.reshape(-1, data.shape[-2], data.shape[-1])
        print('data shape:', data.shape)
        print('targets shape:', len(targets))
        # moving window
        # data = self._segment_data(data, segment_length, overlap)

        targets = np.array(targets)

        if self.percent_of_data:
            num_indices = int(np.round(self.percent_of_data/100 * len(targets)))
            random_indices = np.random.choice(len(targets), size=num_indices, replace=False)
            data = data[random_indices]
            targets = targets[random_indices]

        if percent_of_best_snr:
            indices = self._select_subset_indices_with_best_snr(data, percent_of_best_snr, self.target_freqs[target_idx])
            data = data[indices]
            targets = targets[indices]

        return data, targets
    
    def _get_top_indices(self, snrs, percent):
        # Sort SNRs in descending order
        sorted_snrs = sorted(snrs, reverse=True)

        # Calculate the threshold for of SNRs
        threshold = sorted_snrs[int(percent/100 * len(snrs))]

        # Select indices of SNRs above the threshold
        top_indices = [i for i, snr in enumerate(snrs) if snr >= threshold]

        return top_indices

    def _select_subset_indices_with_best_snr(self, data, percent, freq_base): 
        snrs = []
        data_for_snr = data[:,0,:] # select one channel
        for d in data_for_snr:
            snrs.append(self._calc_snr_for_spectrum(d, freq_base))
        indices = self._get_top_indices(snrs, percent)
        return indices
    
    def _find_closest_element_index(self, array, value):
        # Compute the absolute difference between the value and each element in the array.
        absolute_differences = np.abs(array - value)

        # Find the index of the element with the smallest absolute difference.
        closest_element_index = np.argmin(absolute_differences)

        return closest_element_index

    def _calc_snr_for_spectrum(self, X, freq_base):
        # Calculate the mean FFT of the input signal X
        P1_means, freqs = self._calc_spectrum(X)
        # Find the index of the frequency closest to the base frequency
        base_idx = self._find_closest_element_index(freqs, freq_base)

        # Calculate the base power and noise power
        base_power = P1_means[base_idx]
        noise_power = (P1_means[base_idx-2] + P1_means[base_idx+2])

        # Calculate the SNR
        snr = base_power / noise_power

        return snr

    def _calc_spectrum(self, X):
        # Define the sampling frequency and number of samples
        Fs = self.fs
        L = 256

        # Calculate the frequency range
        freqs = np.linspace(0, Fs/2, L//2 + 1)

        # Initialize an empty array to store the power spectra
        # P1_all = np.zeros([X.shape[1], 129])

        # Calculate the power spectrum for each segment of the signal
        # for i in range(X.shape[1]):
        Y = np.fft.fft(X)
        P2 = np.abs(Y/L)
        P1 = P2[:L//2+1]
        P1[2:-1] *= 2
            # P1_all[i, :] = P1

        # Calculate the mean power spectrum
        # P1_means = np.mean(P1_all, axis=0)
        return P1, freqs


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]