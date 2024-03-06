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
import os

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
    out_tag = 'left_hemisphere_mask'
    all_image_paths = glob.glob(os.path.join(data_dir, 'ncct', '*'))
    all_image_paths.sort()
    mask_paths = glob.glob(os.path.join(data_dir, 'left_hemisphere_mask', '*'))
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
    test_ids = [a for a in all_ids if not (a in val_ids+train_ids)]

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

    train_files = make_dict(train_ids)
    val_files = make_dict(val_ids)
    test_files = make_dict(test_ids)

    max_epochs = 600
    image_size = [256]
    batch_size = 2
    val_interval = 2

    train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Resized(keys=["image", "label"],
                        mode=['trilinear', "nearest"],
                        align_corners=[True, None],
                        spatial_size=[256]*3),
                NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
                RandAffined(keys=['image', 'label'], prob=0.5, translate_range=10),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
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
                        spatial_size=[256]*3),
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
    test_dataset = CacheDataset(
        data=test_files,
        transform=test_transforms,
        cache_rate=1.0,
        num_workers=8
    )
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             pin_memory=True)
    data_example = test_dataset[0]
    ch_in = data_example['image'].shape[0]
    device = 'cuda'
    channels = (32, 64, 128, 256)

    model = UNet(
        spatial_dims=3,
        in_channels=ch_in,
        out_channels=2,
        channels=channels,
        strides=(2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
        dropout=0.2).to(device)

    model_path = 'best_metric_' + model._get_name() + '_' + str(max_epochs) + '.pth'
    model_name = model._get_name()
    #loss_name = loss_function._get_name()
    pred_dir = os.path.join(directory + 'out_' + out_tag, "pred")
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)


    post_transforms = Compose([
            EnsureTyped(keys=["pred"]),
            Invertd(
                keys=["pred"],
                transform=test_transforms,
                orig_keys=["image"],
                meta_keys=["pred_meta_dict"],
                orig_meta_keys=["image_meta_dict"],
                meta_key_postfix="meta_dict",
                nearest_interp=[False],
                to_tensor=[True],
            ),
            AsDiscreted(keys="pred", argmax=True, to_onehot=2),
            SaveImaged(
                keys="pred",
                meta_keys="pred_meta_dict",
                output_dir=pred_dir,
                output_postfix="seg",
                resample=False,
                separate_folder=False)
    ])

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    loader = LoadImage(image_only=False)
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'

    model.load_state_dict(torch.load(os.path.join(
                        directory, 'out_' + out_tag, model_path)))

    model.eval()

    # results = pd.DataFrame(columns=['id', 'subject', 'dice'])
    # results['id'] = test_ids
    ctp_dl_df['dl_id'] = ctp_dl_df['dl_id'].apply(lambda row: str(row).zfill(3))
    ctp_dl_df.set_index('dl_id', inplace=True)

    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            test_inputs = test_data["image"].to(device)
            test_data["pred"] = model(test_inputs)

            test_data = [post_transforms(i) for i in decollate_batch(test_data)]

            test_output, test_image = from_engine(["pred", "image"])(test_data)

            #original_image = loader(test_data[0]["image_meta_dict"]["filename_or_obj"])
            #original_image = original_image[0]  # image data
            #original_image = original_image[:, :, :, 0]
            #prediction = test_output[0][1].detach().numpy()
            #name = os.path.basename(
            #    test_data[0]["image_meta_dict"]["filename_or_obj"]).split('.nii.gz')[0].split('_')[1]

if __name__ == '__main__':
    main()
