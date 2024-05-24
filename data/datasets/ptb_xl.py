from pathlib import Path

import numpy as np
import os
import scipy.io
import mne as mne

import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, TensorDataset
from torchvision import transforms
import pandas as pd
import numpy as np
import wfdb
import ast


class PTBXL(Dataset):
    def __init__(self,
                 dataset_dir,
                 *,
                 cache_dir=None,
                 use_cache=False,
                 transform=None,
                 target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        def load_raw_data(df, sampling_rate, path):
            if sampling_rate == 100:
                data = [wfdb.rdsamp(path / f) for f in df.filename_lr]
            else:
                data = [wfdb.rdsamp(path / f) for f in df.filename_hr]
            data = np.array([signal for signal, meta in data])
            return data

        path = Path(dataset_dir)
        sampling_rate = 100

        # load and convert annotation data
        Y = pd.read_csv(path / 'ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X = load_raw_data(Y, sampling_rate, path)

        # Load scp_statements.csv for diagnostic aggregation
        agg_df = pd.read_csv(path / 'scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in agg_df.index:
                    tmp.append(agg_df.loc[key].diagnostic_class)
            return list(set(tmp))

        # Apply diagnostic superclass
        Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

        # Split data into train and test
        test_fold = 10
        # Train
        X_train = X[np.where(Y.strat_fold != test_fold)]
        y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
        # Test
        X_test = X[np.where(Y.strat_fold == test_fold)]
        y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

        # plot some examples
        # for i in range(20):
        #     plt.figure(figsize=(20, 3))
        #     plt.plot(X[i, :, 0])
        #     # plt.title(Y[i])
        #     plt.show()

        data = np.expand_dims(X[:, :256, 0], 1)
        targets = list(range(len(data)))

        if self.target_transform:
            targets = self.target_transform(targets)

        self.data = torch.Tensor(data)
        self.targets = torch.Tensor(targets).type(torch.LongTensor)

    def __len__(self):
        # return len(self.dataset)
        return len(self.targets)

    def __getitem__(self, idx):
        # data, label = self.dataset[idx]
        data, label = self.data[idx], self.targets[idx]
        return data, label





