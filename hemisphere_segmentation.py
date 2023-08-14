# following tutorial from BRATs segmentation
import os
import pandas as pd
import sys

sys.path.append('/data/gpfs/projects/punim1086/ctp_project/MONAI/')
sys.path.append('../MONAI/')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import time
import matplotlib.pyplot as plt
import numpy as np
import glob
from monai.data import Dataset, CacheDataset, DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet, UNet, AttentionUnet, DenseNet
from monai.networks.layers import Norm
from torch.optim import Adam
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImage,
    LoadImaged,
    NormalizeIntensityd,
    RandAffined,
    RandScaleIntensityd,
    RandShiftIntensityd,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType,
    Resized,
    SaveImaged,
)

from monai_fns import *
from densenet import DenseNetFCN
from sklearn.model_selection import train_test_split

import torch
import math
import os

def define_zvalues(ct_img):
    z_min = int(ct_img.shape[2] * .25)
    z_max = int(ct_img.shape[2] * .85)

    steps = int((z_max - z_min) / 18)

    if steps == 0:
        z_min = 0
        z_max = ct_img.shape[2]
        steps = 1

    z = list(range(z_min, z_max))

    rem = int(len(z) / steps) - 18

    if rem < 0:
        add_on = [z[-1] for n in range(abs(rem))]
        z.extend(add_on)
    elif rem % 2 == 0:
        z_min = z_min + int(rem / 2 * steps) + 1
        z_max = z_max - int(rem / 2 * steps) + 1

    elif rem % 2 != 0:
        z_min = z_min + math.ceil(rem / 2)
        z_max = z_max - math.ceil(rem / 2) + 1

    z = list(range(z_min, z_max, steps))

    if len(z) == 19:
        z = z[1:]
    elif len(z) == 20:
        z = z[1:]
        z = z[:18]

    return z


def create_image(ct_img,
                 pred,
                 savefile,
                 z,
                 ext='png',
                 save=False,
                 dpi=250):
    ct_img, pred = [np.rot90(im) for im in [ct_img, pred]]
    ct_img, pred = [np.fliplr(im) for im in [ct_img, pred]]
    pred = np.where(pred == 0, np.nan, pred)

    fig, axs = plt.subplots(6, 6, facecolor='k')
    fig.subplots_adjust(hspace=-0.1, wspace=-0.3)
    axs = axs.ravel()
    for ax in axs:
        ax.axis("off")
    for i in range(6):
        print(i)

        axs[i].imshow(ct_img[:, :, z[i]], cmap='gray',
                      interpolation='hanning', vmin=10, vmax=ct_img.max())
        axs[i + 6].imshow(ct_img[:, :, z[i]], cmap='gray',
                          interpolation='hanning', vmin=10, vmax=ct_img.max())
        im = axs[i + 6].imshow(pred[:, :, z[i]], cmap='Reds', interpolation='hanning', alpha=0.5, vmin=0, vmax=1)

    if 12 > len(z):
        max2 = len(z)
    else:
        max2 = 12
    for i in range(6, max2):
        print(i)
        axs[i + 6].imshow(ct_img[:, :, z[i]], cmap='gray',
                          interpolation='hanning', vmin=10, vmax=ct_img.max())
        axs[i + 12].imshow(ct_img[:, :, z[i]], cmap='gray',
                           interpolation='hanning', vmin=10, vmax=ct_img.max())
        im = axs[i + 12].imshow(pred[:, :, z[i]], cmap='Reds', interpolation='hanning', alpha=0.5, vmin=0, vmax=1)

    if not 12 > len(z):
        if len(z) > 18:
            max3 = 18
        else:
            max3 = len(z)
        for i in range(12, max3):
            print(i)
            axs[i + 12].imshow(ct_img[:, :, z[i]], cmap='gray',
                               interpolation='hanning', vmin=10, vmax=ct_img.max())
            axs[i + 18].imshow(ct_img[:, :, z[i]], cmap='gray',
                               interpolation='hanning', vmin=10, vmax=ct_img.max())
            axs[i + 18].imshow(pred[:, :, z[i]], cmap='Reds', interpolation='hanning', alpha=0.5, vmin=0, vmax=1)

    if savefile:
        plt.savefig(savefile, facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=dpi, format=ext)
        plt.close()


def main():
    HOMEDIR = os.path.expanduser('~/')
    if os.path.exists(HOMEDIR + 'mediaflux/'):
        directory = HOMEDIR + 'mediaflux/data_freda/ctp_project/CTP_DL_Data/'
        ctp_dl_df = pd.read_csv(HOMEDIR + 'PycharmProjects/study_design/study_lists/data_for_ctp_dl.csv',
                                usecols=['subject', 'segmentation_type', 'dl_id'])
    elif os.path.exists('Z:/data_freda'):
        directory = 'Z:/data_freda/ctp_project/CTP_DL_Data/'
        ctp_dl_df = pd.read_csv(HOMEDIR + 'PycharmProjects/study_design/study_lists/data_for_ctp_dl.csv',
                                usecols=['subject', 'segmentation_type', 'dl_id'])
    elif os.path.exists('/data/gpfs/projects/punim1086/ctp_project'):
        directory = '/data/gpfs/projects/punim1086/ctp_project/CTP_DL_Data/'
        ctp_dl_df = pd.read_csv('/data/gpfs/projects/punim1086/study_design/study_lists/data_for_ctp_dl.csv',
                                usecols=['subject', 'segmentation_type', 'dl_id'])

    data_dir = os.path.join(directory, 'DATA')
    out_tag = 'right_hemisphere_mask'
    all_image_paths = glob.glob(os.path.join(data_dir, 'ncct', '*'))
    all_image_paths.sort()
    mask_paths = glob.glob(os.path.join(data_dir, 'right_hemisphere_mask', '*'))
    mask_paths.sort()

    ids = [os.path.basename(path).split('.nii.gz')[0].split('_')[1] for path in mask_paths]
    image_paths = [path for path in all_image_paths
                   if os.path.basename(path).split('.nii.gz')[0].split('_')[1] in ids]
    test_images = [path for path in all_image_paths if path not in image_paths]

    num_train = int(np.round(0.8 * len(mask_paths)))
    num_validation = int(np.round(0.2 * len(mask_paths)))

    random_state = 42

    train_ids, val_ids = train_test_split(ids, train_size=num_train,
                                          test_size=num_validation,
                                          random_state=random_state,
                                          shuffle=False)
    all_ids = [str(a).zfill(3) for a in range(len(all_image_paths))]
    test_ids = [a for a in all_ids if not (a in val_ids + train_ids)]

    def make_dict(id):
        paths1 = [file for file in all_image_paths
                  if os.path.basename(file).split('.nii.gz')[0].split('_')[1] in id]
        paths2 = [file for file in mask_paths
                  if os.path.basename(file).split('.nii.gz')[0].split('_')[1] in id]
        if paths2:
            files_dict = [{"image": image_name, "label": label_name} for
                          image_name, label_name in zip(paths1, paths2)]
        else:
            files_dict = [{"image": image_name} for
                          image_name in paths1]

        return files_dict

    train_files = make_dict(train_ids)[:4]
    val_files = make_dict(val_ids)[:4]
    test_files = make_dict(test_ids)[:4]

    max_epochs = 2
    image_size = [32]
    batch_size = 2
    val_interval = 2

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Resized(keys=["image", "label"],
                    mode=['trilinear', "nearest"],
                    align_corners=[True, None],
                    spatial_size=[256] * 3),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            RandAffined(keys=['image', 'label'], prob=0.5, translate_range=10),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=1.0),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=1.0),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Resized(keys=["image", "label"],
                    mode=['trilinear', "nearest"],
                    align_corners=[True, None],
                    spatial_size=[256] * 3),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Resized(keys=["image"],
                    mode=['trilinear'],
                    align_corners=[True],
                    spatial_size=image_size * 3),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image"]),
        ]
    )

    train_dataset = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=8)

    val_dataset = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=1.0,
        num_workers=8)

    test_dataset = CacheDataset(
        data=test_files,
        transform=test_transforms,
        cache_rate=1.0,
        num_workers=8
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            pin_memory=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             pin_memory=True)

    # s = 150
    # import random
    # m = random.randint(0, len(train_files))
    # s = random.randint(100, 200)
    data_example = test_dataset[0]
    ch_in = data_example['image'].shape[0]
    # plt.figure("sanity check")
    # plt.subplot(1, 2, 1)
    # plt.title(f"image")
    # plt.imshow(np.flipud(data_example["image"][0, :, :, s].detach().cpu()), cmap="gray")
    # plt.axis('off')
    # print(f"label shape: {data_example['label'].shape}")
    # plt.subplot(1, 2, 2)
    # plt.title("label")
    # plt.imshow(np.flipud(data_example["label"][0, :, :, s].detach().cpu()), cmap="gray")
    # plt.axis('off')
    # plt.show()
    # plt.close()



if __name__ == '__main__':
    main()
