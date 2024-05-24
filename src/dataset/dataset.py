import pandas as pd
from monai.data import DataLoader, PersistentDataset
from monai.transforms import Compose, LoadImageD, RandSpatialCropD, ScaleIntensityD, BorderPadD, EnsureChannelFirstD
from pandas import DataFrame
from torch.utils.data import DataLoader

sfreq = 100
windows_size = 30 * sfreq

def get_trans(dataset):
    if dataset == "edfx":
        transforms_list = Compose([LoadImageD(keys='eeg'),
                                  #EnsureChannelFirstD(keys='eeg'),
                               ScaleIntensityD(factor=1e6, keys='eeg'),  # Numeric stability
                               ScaleIntensityD(minv=0, maxv=1, keys='eeg'),  # Normalization
                               RandSpatialCropD(keys='eeg', roi_size=[windows_size],
                                                random_size=False, ),
                               BorderPadD(keys='eeg', spatial_border=[36], mode="constant")
                               ])
    else:
        print("option b")
        transforms_list = Compose([LoadImageD(keys='eeg'),
                           EnsureChannelFirstD(keys='eeg'),
                           ScaleIntensityD(factor=1e6, keys='eeg'),  # Numeric stability
                           ScaleIntensityD(minv=0, maxv=1, keys='eeg'),  # Normalization
                           RandSpatialCropD(keys='eeg', roi_size=[windows_size],
                                            random_size=False, ),
                           BorderPadD(keys='eeg', spatial_border=[36], mode="constant")
                           ])
    return transforms_list        


def get_datalist(
        df: DataFrame, basepath: str, dataset: str,
):
    """
    Get data dicts for data loaders.

    """
    if dataset == "edfx":
        final = ".npy"
    else:
        final = ""

    data_dicts = []
    for index, row in df.iterrows():
        data_dicts.append(
            {
                "eeg": f"{basepath}/{row['FILE_NAME_EEG']}{final}",
                "subject": float(row["subject"]),
                "night": float(row["night"]),
                "age": float(row["age"]),
                "gender": str(row["gender"]),
                "lightoff": str(row["LightsOff"]),
            }
        )

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts


def train_dataloader(config, args, transforms_list, dataset, n_subjects=None):
    train_df = pd.read_csv(args.path_train_ids)

    if n_subjects is not None:
        train_df = train_df[:n_subjects]

    train_dicts = get_datalist(train_df, basepath=args.path_pre_processed, dataset=dataset)

    train_ds = PersistentDataset(data=train_dicts,
                                 transform=transforms_list,
                                 cache_dir=None)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        drop_last=config.train.drop_last,
        pin_memory=False,
        persistent_workers=True,
    )
    return train_loader


def valid_dataloader(config, args, transforms_list, dataset, n_subjects=None):
    valid_df = pd.read_csv(args.path_valid_ids)

    if n_subjects is not None:
        valid_df = valid_df[:n_subjects]

    valid_dicts = get_datalist(valid_df, basepath=args.path_pre_processed, dataset=dataset)

    valid_ds = PersistentDataset(data=valid_dicts, transform=transforms_list,
                                 cache_dir=None)

    valid_loader = DataLoader(valid_ds, batch_size=config.train.batch_size, shuffle=True,
                              num_workers=config.train.num_workers, drop_last=config.train.drop_last,
                              pin_memory=False,
                              persistent_workers=True, )

    return valid_loader


def test_dataloader(config, args, transforms_list, dataset, upper_limit=None, n_subjects=None):
    test_df = pd.read_csv(args.path_test_ids)

    if n_subjects is not None:
        test_df = test_df[:n_subjects]

    if upper_limit is not None:
        test_df = test_df[:upper_limit]

    test_dicts = get_datalist(test_df, basepath=args.path_pre_processed, dataset=dataset)

    test_ds = PersistentDataset(data=test_dicts, transform=transforms_list,
                                 cache_dir=None)

    test_loader = DataLoader(test_ds, batch_size=config.train.batch_size, shuffle=True,
                              num_workers=config.train.num_workers, drop_last=config.train.drop_last,
                              pin_memory=False,
                              persistent_workers=True, )

    return test_loader
