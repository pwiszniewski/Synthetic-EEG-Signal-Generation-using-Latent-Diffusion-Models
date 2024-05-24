import numpy as np

from data.datasets.BaseDatasets import BaseEEGDataset
import scipy.io


class DatasetFromMatFile(BaseEEGDataset):

    def __init__(self,
                dataset_dir,
                mat_file_names,
                channels,
                *,
                cache_dir=None,
                use_cache=False,
                channels_selected=None,
                indices=None,
                transform=None,
                target_transform=None):

        self.channels = channels
        self.mat_file_names = mat_file_names
        super().__init__(dataset_dir,
                        cache_dir=cache_dir,
                        use_cache=use_cache,
                        ch_names=channels_selected,
                        indices=indices,
                        transform=transform,
                        target_transform=target_transform)

    def _get_file_list(self):
        if isinstance(self.mat_file_names, str):
            return [self.mat_file_names]
        else:
            return self.mat_file_names

    def _load_data_raw(self, dataset_dir, files):
        """ load data from mat file """
        data, targets = [], []

        for file in files:
            mat = scipy.io.loadmat(dataset_dir / file)
            data.append(mat['data'])
            targets.append(np.ravel(mat['labels']))

        data = np.concatenate(data)
        targets = np.concatenate(targets)

        return data, targets

    def get_target_indices(self, target):
        return np.where(self.targets == target)[0]
