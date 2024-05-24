import numpy as np
import os
import scipy.io

import torch
from data.datasets.BaseDatasets import BaseEEGDataset

from torchvision import transforms


class SynthSSVEPMK(BaseEEGDataset):
    """
    SSVEP classification with a limited training dataset
    https://ieee-dataport.org/documents/ssvep-classification-limited-training-dataset#files
    """
    channels_default = ['Oz']
    target_freqs = [5, 6, 7, 8]
    fs = 256
    num_subjects_default = 5
    num_targets = 4

    def __init__(self, *args, **kwargs):
        self.is_train = kwargs.pop('is_train', True)
        self.window_size = kwargs.pop('window_size', 256)
        self.overlap = kwargs.pop('overlap', 0)
        self.num_subjects = kwargs.pop('num_subjects', self.num_subjects_default)
        self.subjects = kwargs.pop('subjects', range(1, self.num_subjects + 1))
        self.channels = kwargs.pop('channels', self.channels_default)
        self.channels_selected = kwargs.pop('channels_selected', self.channels)
        self.targets_selected = kwargs.pop('targets_selected', self.target_freqs)
        train_samples_per_target = kwargs.pop('train_samples_per_target', 20 * self.fs)
        test_samples_per_target = kwargs.pop('test_samples_per_target', 10 * self.fs)

        samples_per_target = train_samples_per_target if self.is_train else test_samples_per_target
        self.num_blocks = int((samples_per_target - self.window_size) / (self.window_size - self.overlap)) + 1
        # self.num_blocks = samples_per_target // (self.window_size - self.overlap)
        kwargs['ch_names'] = self.channels_selected
        super(SynthSSVEPMK, self).__init__(*args, **kwargs, indices=self._get_selected_indices())

    def _get_start_end(self):
        start = 0
        end = 256
        return end, start

    def _get_selected_indices(self):
        indices = []
        for subject in self.subjects:
            subject_start_idx = subject * self.num_blocks * self.num_targets
            for target in self.targets_selected:
                target_idx = self.target_freqs.index(target)
                target_offset = target_idx * self.num_blocks
                target_indices = np.arange(self.num_blocks) + target_offset + subject_start_idx
                indices.extend(target_indices)
        return indices 

    def _load_data_raw(self, dataset_dir, files):
        print('reading data files')
        data = []
        targets = []

        for subject in range(1, self.num_subjects + 1):
            subject_dir = dataset_dir / f'Subject_{subject}'
            for freq in self.target_freqs:
                file = subject_dir / f"session_{freq}Hz_{'train' if self.is_train else 'test'}_1_electrode.mat"
                mat = scipy.io.loadmat(file)
                data_mat = mat[f"X_{'train' if self.is_train else 'test'}"] # (samples, channels)
                # split_indices = np.arange(self.window_size, data_mat.shape[0], self.window_size - self.overlap)
                # split data into equaly size windows with overlap
                # data_mat = [data_mat[i:i + self.window_size] for i in range(0, data_mat.shape[0], self.window_size - self.overlap)] # not all windows are of size self.window_size
                data_mat = np.array([data_mat[i:i + self.window_size] for i in range(0, data_mat.shape[0], self.window_size - self.overlap) if i + self.window_size <= data_mat.shape[0]]) # only windows of size self.window_size
                data.extend(data_mat)
                targets.append(freq * np.ones(len(data_mat)))
        data = np.stack(data)
        # data = data * 1e6  # convert to uV
        data = np.swapaxes(data, 1, 2)
        targets = np.concatenate(targets)
        return data, targets