""" Script to compute the Frechet Inception Distance (FID) of the samples of the LDM.

In order to measure the quality of the samples, we use the Frechet Inception Distance (FID) metric between 1200 images
from the MIMIC-CXR dataset and 1000 images from the LDM.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from generative.metrics import FIDMetric
from monai import transforms
from monai.config import print_config
from monai.data import Dataset
from monai.utils import set_determinism
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
#from dataset.dataset import
from glob import glob

from dataset.dataset import test_dataloader
from numpy import load
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--sample_dir", help="Location of the samples to evaluate.")
    parser.add_argument("--test_ids", help="Location of file with test ids.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    parser.add_argument(
        "--path_test_ids",
        type=str,
        default="/project/data/ids_shhs/ids_shhs_test.csv",
        #default="/project/data/ids/ids_sleep_edfx_cassette_train.csv",
    )
    parser.add_argument("--autoencoderkl_config_file_path",
                        help="Path to the .pth model from the stage1.",
                        #default="/project/config/config_aekl_eeg.yaml")
                        default="/project/config/config_aekl_eeg.yaml")
    parser.add_argument(
        "--path_pre_processed",
        type=str,
        default="/data/polysomnography/shhs_numpy",
        #default="/data/pre-processed",
    )
    args = parser.parse_args()
    return args


import numpy as np
import torch
from torch import nn


def _crop_tensors_to_match(x1, x2, axis=-1):
    """Crops two tensors to their lowest-common-dimension along an axis."""
    dim_cropped = min(x1.shape[axis], x2.shape[axis])

    x1_cropped = torch.index_select(
        x1, dim=axis,
        index=torch.arange(dim_cropped).to(device=x1.device)
    )
    x2_cropped = torch.index_select(
        x2, dim=axis,
        index=torch.arange(dim_cropped).to(device=x1.device)
    )
    return x1_cropped, x2_cropped


class _EncoderBlock(nn.Module):
    """Encoding block for a timeseries x of shape (B, C, T)."""
    def __init__(self,
                 in_channels=2,
                 out_channels=2,
                 kernel_size=9,
                 downsample=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.downsample = downsample

        self.block_prepool = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      padding='same'),
            nn.ELU(),
            nn.BatchNorm1d(num_features=out_channels),
        )

        self.pad = nn.ConstantPad1d(padding=1, value=0)
        self.maxpool = nn.MaxPool1d(
            kernel_size=self.downsample, stride=self.downsample)

    def forward(self, x):
        x = self.block_prepool(x)
        residual = x
        if x.shape[-1] % 2:
            x = self.pad(x)
        x = self.maxpool(x)
        return x, residual


class _DecoderBlock(nn.Module):
    """Decoding block for a timeseries x of shape (B, C, T)."""
    def __init__(self,
                 in_channels=2,
                 out_channels=2,
                 kernel_size=9,
                 upsample=2,
                 with_skip_connection=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.with_skip_connection = with_skip_connection

        self.block_preskip = nn.Sequential(
            nn.Upsample(scale_factor=upsample),
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=2,
                      padding='same'),
            nn.ELU(),
            nn.BatchNorm1d(num_features=out_channels),
        )
        self.block_postskip = nn.Sequential(
            nn.Conv1d(
                in_channels=(
                    2 * out_channels if with_skip_connection else out_channels),
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding='same'),
            nn.ELU(),
            nn.BatchNorm1d(num_features=out_channels),
        )

    def forward(self, x, residual):
        x = self.block_preskip(x)
        if self.with_skip_connection:
            x, residual = _crop_tensors_to_match(x, residual, axis=-1)  # in case of mismatch
            x = torch.cat([x, residual], axis=1)  # (B, 2 * C, T)
        x = self.block_postskip(x)
        return x


class USleep(nn.Module):
    """Sleep staging architecture from Perslev et al 2021.

    U-Net (autoencoder with skip connections) feature-extractor for sleep
    staging described in [1]_.

    For the encoder ('down'):
        -- the temporal dimension shrinks (via maxpooling in the time-domain)
        -- the spatial dimension expands (via more conv1d filters in the
           time-domain)
    For the decoder ('up'):
        -- the temporal dimension expands (via upsampling in the time-domain)
        -- the spatial dimension shrinks (via fewer conv1d filters in the
           time-domain)
    Both do so at exponential rates.

    Parameters
    ----------
    in_chans : int
        Number of EEG or EOG channels. Set to 2 in [1]_ (1 EEG, 1 EOG).
    sfreq : float
        EEG sampling frequency. Set to 128 in [1]_.
    depth : int
        Number of conv blocks in encoding layer (number of 2x2 max pools)
        Note: each block halve the spatial dimensions of the features.
    n_time_filters : int
        Initial number of convolutional filters. Set to 5 in [1]_.
    complexity_factor : float
        Multiplicative factor for number of channels at each layer of the U-Net.
        Set to 2 in [1]_.
    with_skip_connection : bool
        If True, use skip connections in decoder blocks.
    n_classes : int
        Number of classes. Set to 5.
    input_size_s : float
        Size of the input, in seconds. Set to 30 in [1]_.
    time_conv_size_s : float
        Size of the temporal convolution kernel, in seconds. Set to 9 / 128 in
        [1]_.
    ensure_odd_conv_size : bool
        If True and the size of the convolutional kernel is an even number, one
        will be added to it to ensure it is odd, so that the decoder blocks can
        work. This can ne useful when using different sampling rates from 128
        or 100 Hz.
    apply_softmax : bool
        If True, apply softmax on output (e.g. when using nn.NLLLoss). Use
        False if using nn.CrossEntropyLoss.

    References
    ----------
    .. [1] Perslev M, Darkner S, Kempfner L, Nikolic M, Jennum PJ, Igel C.
           U-Sleep: resilient high-frequency sleep staging. npj Digit. Med. 4, 72 (2021).
           https://github.com/perslev/U-Time/blob/master/utime/models/usleep.py
    """
    def __init__(self,
                 in_chans=2,
                 sfreq=128,
                 depth=12,
                 n_time_filters=5,
                 complexity_factor=1.67,
                 with_skip_connection=True,
                 n_classes=5,
                 input_size_s=30,
                 time_conv_size_s=9 / 128,
                 ensure_odd_conv_size=False,
                 apply_softmax=False
                 ):
        super().__init__()

        self.in_chans = in_chans
        max_pool_size = 2  # Hardcoded to avoid dimensional errors
        time_conv_size = np.round(time_conv_size_s * sfreq).astype(int)
        if time_conv_size % 2 == 0:
            if ensure_odd_conv_size:
                time_conv_size += 1
            else:
                raise ValueError(
                    'time_conv_size must be an odd number to accomodate the '
                    'upsampling step in the decoder blocks.')

        # Convert between units: seconds to time-points (at sfreq)
        input_size = np.ceil(input_size_s * sfreq).astype(int)

        channels = [in_chans]
        n_filters = n_time_filters
        for _ in range(depth + 1):
            channels.append(int(n_filters * np.sqrt(complexity_factor)))
            n_filters = int(n_filters * np.sqrt(2))
        self.channels = channels

        # Instantiate encoder
        encoder = list()
        for idx in range(depth):
            encoder += [
                _EncoderBlock(in_channels=channels[idx],
                              out_channels=channels[idx + 1],
                              kernel_size=time_conv_size,
                              downsample=max_pool_size)
            ]
        self.encoder = nn.Sequential(*encoder)

        # Instantiate bottom (channels increase, temporal dim stays the same)
        self.bottom = nn.Sequential(
                    nn.Conv1d(in_channels=channels[-2],
                              out_channels=channels[-1],
                              kernel_size=time_conv_size,
                              padding=(time_conv_size - 1) // 2),  # preserves dimension
                    nn.ELU(),
                    nn.BatchNorm1d(num_features=channels[-1]),
                )

        # Instantiate decoder
        decoder = list()
        channels_reverse = channels[::-1]
        for idx in range(depth):
            decoder += [
                _DecoderBlock(in_channels=channels_reverse[idx],
                              out_channels=channels_reverse[idx + 1],
                              kernel_size=time_conv_size,
                              upsample=max_pool_size,
                              with_skip_connection=with_skip_connection)
            ]
        self.decoder = nn.Sequential(*decoder)

        # The temporal dimension remains unchanged
        # (except through the AvgPooling which collapses it to 1)
        # The spatial dimension is preserved from the end of the UNet, and is mapped to n_classes
        self.clf = nn.Sequential(
            nn.Conv1d(
                in_channels=channels[1],
                out_channels=channels[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),                         # output is (B, C, 1, S * T)
            nn.Tanh(),
            nn.AvgPool1d(input_size),  # output is (B, C, S)
            nn.Conv1d(
                in_channels=channels[1],
                out_channels=n_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),                         # output is (B, n_classes, S)
            nn.ELU(),
            nn.Conv1d(
                in_channels=n_classes,
                out_channels=n_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Softmax(dim=1) if apply_softmax else nn.Identity(),
            # output is (B, n_classes, S)
        )

    def forward(self, x):
        """If input x has shape (B, S, C, T), return y_pred of shape (B, n_classes, S).
        If input x has shape (B, C, T), return y_pred of shape (B, n_classes).
        """
        # reshape input
        if x.ndim == 4:  # input x has shape (B, S, C, T)
            x = x.permute(0, 2, 1, 3)  # (B, C, S, T)
            x = x.flatten(start_dim=2)  # (B, C, S * T)

        # encoder
        residuals = []
        for down in self.encoder:
            x, res = down(x)
            residuals.append(res)

        # bottom
        h = self.bottom(x)
        x = h
        # decoder
        residuals = residuals[::-1]  # flip order
        for up, res in zip(self.decoder, residuals):
            x = up(x, res)

        # classifier
        y_pred = self.clf(x)        # (B, n_classes, seq_length)

        if y_pred.shape[-1] == 1:  # seq_length of 1
            y_pred = y_pred[:, :, 0]

        return y_pred, h


def main(args):
    set_determinism(seed=args.seed)
    print_config()

    config_aekl = OmegaConf.load(args.autoencoderkl_config_file_path)

    # Load pretrained model
    device = torch.device("cpu")

    n_channels=2
    n_classes=5
    sfreq=100

    model = USleep(
        in_chans=n_channels,
        sfreq=sfreq,
        depth=12,
        with_skip_connection=True,
        n_classes=n_classes,
        input_size_s=30,
        apply_softmax=False
    )


    print(f"Defining the EEG Classifier.")

    params = torch.load("/project/params.pt",
                        map_location=device)

    model.load_state_dict(params)


    # Samples
    test_loader = test_dataloader(config=config_aekl, args=args, upper_limit=1000, )

    test_loader_2 = test_dataloader(config=config_aekl, args=args, upper_limit=1000, )


    samples_features_1 = []
    samples_features_2 = []

    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    for step, batch in pbar:
        x = batch['eeg'].to(device)
        for batch2 in test_loader_2:
            x2 = batch2["eeg"].to(device)
            if batch["eeg_meta_dict"]["filename_or_obj"][0] == batch2["eeg_meta_dict"]["filename_or_obj"][0]:
                continue

            x1_cropped = x[:, :, 36:-36]
            x2_cropped = x2[:, :, 36:-36]

            double_feature_space_1 = torch.concat([x1_cropped, x1_cropped], 1)
            double_feature_space_2 = torch.concat([x2_cropped, x2_cropped], 1)

            with torch.no_grad():
                _, outputs_1 = model(double_feature_space_1.to(device))
                _, outputs_2 = model(double_feature_space_2.to(device))

                outputs_1 = outputs_1.squeeze(-1)
                outputs_2 = outputs_2.squeeze(-1)

                # dense_outputs = F.adaptive_avg_pool1d(outputs.squeeze(-1).T,1).squeeze(-1)

            samples_features_1.append(outputs_1.cpu())
            samples_features_2.append(outputs_2.cpu())

        samples_features_1 = torch.cat(samples_features_1, dim=0)
        samples_features_2 = torch.cat(samples_features_2, dim=0)

        pbar.update()
    # Compute FID
    metric = FIDMetric()
    fid = metric(samples_features_1, samples_features_2)

    print(f"FID: {np.round(fid,6)}")


if __name__ == "__main__":
    args = parse_args()
    main(args)