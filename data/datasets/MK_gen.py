from pathlib import Path

import numpy as np
import os
import scipy.io
import scipy

import torch
from torch.utils.data import TensorDataset, Dataset


def butter_bandpass_filter(arr, low, high, fs, order, axis=-1):
    """
    filter between low and high Hz with zero phase shift
    """
    nyq = 0.5 * fs
    low1 = low / nyq
    high1 = high / nyq
    b, a = scipy.signal.butter(order, [low1, high1], btype='band')
    y = scipy.signal.filtfilt(b, a, arr, axis=axis)
    return y


class MK_gen(Dataset):
    """
    fs: 256Hz
    num examples: 50000
    same phase
    """
    channels = ['1']
    target_freqs = [8, 13]
    fs = 256
    num_subjects = 1
    num_targets = 2

    def __init__(self,
                 dataset_dir,
                 *,
                 file_suffix=None,
                 channels_selected=None,
                 subjects_selected=None,
                 targets_selected=None,
                 segment_length=256,
                 overlap=0,
                 transform=None,
                 target_transform=None,
                 percent_of_data=None,
                 percent_of_best_snr=None):
        self.file_suffix = file_suffix
        self.percent_of_data = percent_of_data
        self.percent_of_best_snr = percent_of_best_snr
        self.transform = transform
        self.target_transform = target_transform
        self.subject_indices = self._get_subject_indices(subjects_selected)
        self.channel_indices = self._get_channel_indices(channels_selected)
        self.target_indices = self._get_target_indices(targets_selected)
        dataset_dir = Path(dataset_dir)

        data, targets = self._load_data(dataset_dir, segment_length, overlap, self.percent_of_best_snr)
        self._make_dataset(data, targets)

    def _get_subject_indices(self, subjects_selected):
        if subjects_selected is None:
            return range(1, self.num_subjects + 1)
        else:
            return subjects_selected

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

    def _load_data(self, dataset_dir, segment_length, overlap, percent_of_best_snr):
        data = []
        targets = []
        print('reading files')

        for target_idx in self.target_indices:
            target = self.target_freqs[target_idx]
            file_suffix = '_' + self.file_suffix if self.file_suffix else ''
            file_path = dataset_dir / f'X_{target}{file_suffix}.mat'
            mat = scipy.io.loadmat(file_path)
            arr = mat[f'X_{target}']
            data.append(arr)
            targets.extend([target_idx] * arr.shape[1])
        data = np.stack(data)
        data = np.moveaxis(data, 2, 0)
        print('data = np.stack(data)', data.shape)
        # data = data[self.target_indices]
        # print('data = data[self.target_indices]', data.shape)
        # # split data into equaly size windows with overlap along the third axis
        # for count, target_idx in enumerate(self.target_indices):
        #     arr_target = arr[count]
        #     arr_target = np.array([arr_target[:, i:i + segment_length] for i in range(0, arr_target.shape[1], segment_length - overlap) if i + segment_length <= arr_target.shape[1]])
        #     data.extend(arr_target)
        #     targets.extend(np.tile(self.target_freqs[target_idx], len(arr_target)))
        
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