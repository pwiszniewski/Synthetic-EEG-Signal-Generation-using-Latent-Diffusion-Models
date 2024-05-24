from pathlib import Path

import numpy as np
import os
import scipy.io
import scipy

import torch
from torch.utils.data import TensorDataset, Dataset

from eeggan.data.datasets.BaseDatasets import BaseEEGDataset

from torchvision import transforms


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


class SSVEP_12JFPM(Dataset):
    """
    https://github.com/mnakanishi/12JFPM_SSVEP
    This dataset contains 12-class joint frequency-phase modulated steady-state visual evoked potentials (SSVEPs) acquired from 10 subjects used to estimate an online performance of brain-computer interface (BCI) in the reference study (Nakanishi et al., 2015).
    File format

    Each .mat file has a four-way tensor electroencephalogram (EEG) data for each subject.
    Please see the reference paper for the detail.

    [Number of targets, Number of channels, Number of sampling points, Number of trials] = size(eeg)

        Number of targets : 12
        Number of channels : 8
        Number of sampling points : 1114
        Number of trials : 15
        Sampling rate [Hz] : 256

    The order of the stimulus frequencies in the EEG data:
    [9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75] Hz
    (e.g., eeg(1,:,:,:) and eeg(5,:,:,:) are the EEG data while a subject was gazing at the visual stimuli flickering at 9.25 Hz and 11.75Hz, respectively.)

    The onset of visual stimulation is at 39th sample point, which means there are redundant data for 0.15 [s] before stimulus onset.

    Reference
    Masaki Nakanishi, Yijun Wang, Yu-Te Wang and Tzyy-Ping Jung, "A Comparison Study of Canonical Correlation Analysis Based Methods for Detecting Steady-State Visual Evoked Potentials," PLoS One, vol.10, no.10, e140703, 2015. http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0140703

    """
    channels = ['G2', 'G4', 'F32', 'G8', 'G12', 'G5', 'Oz', 'G9']
    target_freqs = [9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75]
    fs = 256
    num_subjects = 10
    num_targets = 12

    def __init__(self,
                 dataset_dir,
                 *,
                 channels_selected=None,
                 subjects_selected=None,
                 targets_selected=None,
                 segment_length=256,
                 overlap=0,
                 transform=None,
                 target_transform=None,
                 percent_of_best_snr=None):
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

        for subject in self.subject_indices:
            subject_file = dataset_dir / f's{subject}.mat'

            mat = scipy.io.loadmat(subject_file)
            arr = mat['eeg']
            arr = arr[self.target_indices][:, self.channel_indices]
            # filter between 6 and 80 Hz
            arr = butter_bandpass_filter(arr, 6, 80, self.fs, order=5, axis=2)
            arr = arr[:, :, 39:] # remove redundant data for 0.15 [s] before stimulus onset
            arr = np.moveaxis(arr, 2, 3)
            arr = arr.reshape(*arr.shape[:-2], -1)
            # split data into equaly size windows with overlap along the third axis
            for count, target_idx in enumerate(self.target_indices):
                arr_target = arr[count]
                arr_target = np.array([arr_target[:, i:i + segment_length] for i in range(0, arr_target.shape[1], segment_length - overlap) if i + segment_length <= arr_target.shape[1]])
                data.extend(arr_target)
                targets.extend(np.tile(self.target_freqs[target_idx], len(arr_target)))

            # for i in range(15):
            #     a = arr[0, :, :, i]
            #     a = butter_bandpass_filter(a, 6, 80, self.fs, order=5, axis=1)
            #     data.append(a)
            #     targets.append(1)

        data = np.stack(data)
        targets = np.array(targets)
        if percent_of_best_snr:
            indices = self._select_subset_indices_with_best_snr(data, percent_of_best_snr, self.target_freqs[target_idx])
            data = data[indices]
            targets = targets[indices]
        # split data into equally size windows with overlap along the third axis
        # data = np.array([data[:, :, i:i + segment_length] for i in range(0, data.shape[2], segment_length - overlap) if i + segment_length <= data.shape[2]])
        # data = data.reshape(-1, *data.shape[2:])
        
        # targets = np.tile(targets, len(data) // len(targets))
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