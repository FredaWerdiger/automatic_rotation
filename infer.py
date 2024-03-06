# following tutorial from BRATs segmentation
import os
import shutil
import sys
sys.path.append('/data/gpfs/projects/punim1086/ctp_project/MONAI/')
sys.path.append('../MONAI/')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from monai.data import Dataset, CacheDataset, DataLoader, decollate_batch
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.transforms import (
    AsDiscreted,
    Compose,
    Invertd,
    LoadImage,
    LoadImaged,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    EnsureTyped,
    Resized,
    SaveImaged,
)

import SimpleITK as sitk

import torch
import os

def main(input_image, output_mask, hemisphere, model_dir):

    if hemisphere == 'left':
        model_path = os.path.join(model_dir, 'best_metric_UNet_600_left.pth')
    elif hemisphere == 'right':
        model_path = os.path.join(model_dir, 'best_metric_UNet_600_right.pth')

    temp = os.path.split(output_mask)[0]  + '/pred/'
    if not os.path.exists(temp):
        os.makedirs(temp)

    test_files = [{"image": input_image}]

    image_size = [256]

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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
            output_dir=temp,
            output_postfix=hemisphere,
            resample=False,
            separate_folder=False)
    ])

    loader = LoadImage(image_only=False)
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'

    model.load_state_dict(torch.load(model_path))

    model.eval()

    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            test_inputs = test_data["image"].to(device)
            test_data["pred"] = model(test_inputs)

            test_data = [post_transforms(i) for i in decollate_batch(test_data)]

    im = sitk.ReadImage(os.path.join(temp, os.listdir(temp)[0]))
    im = im[:,:,:,1]
    sitk.WriteImage(im, output_mask)
    shutil.rmtree(temp, ignore_errors=True)


if __name__ == '__main__':
    main()
