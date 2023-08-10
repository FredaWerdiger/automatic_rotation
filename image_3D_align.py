import sys
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import os
#
# def rotate_and_resample(im, roll, pitch, yaw, show=False):
#     arr = sitk.GetArrayFromImage(im)
#     z, y, x = np.nonzero(arr)
#     mean_z = np.mean(z)
#     mean_x = np.mean(x)
#     mean_y = np.mean(y)
#
#     # ROTATING
#     rotation_centre = (int(np.ceil(mean_x)), int(np.ceil(mean_y)), int(np.ceil(mean_z)))
#     rotation_centre_im = im.TransformIndexToPhysicalPoint(rotation_centre)
#
#     rigid_euler = sitk.Euler3DTransform()
#     rigid_euler.SetRotation(roll, pitch, yaw)
#
#     rigid_versor = sitk.VersorRigid3DTransform()
#     rigid_versor.SetMatrix(rigid_euler.GetMatrix())
#     rigid_versor.SetCenter(rotation_centre_im)
#
#     # RESAMPLING
#     interpolator = sitk.sitkNearestNeighbor
#     default_value = 0
#     resampled = sitk.Resample(
#         im,
#         im,
#         rigid_versor,
#         interpolator,
#         default_value
#     )
#
#     resampled = sitk.Cast(resampled, sitk.sitkFloat32)
#
#     if show:
#         new_array = sitk.GetArrayFromImage(resampled)
#         fig, ax = plt.subplots(1, 3, figsize=(10, 3))
#         axial_slice_num = int(input('Enter axial slice number you wish to see:'))
#         coronal_slice_num = int(input('Enter coronal slice number:'))
#         sagittal_slice_num = int(input('Enter sagittal slice number:'))
#         ax[0].imshow(new_array[axial_slice_num])
#         ax[0].set_title('xy')
#         ax[1].imshow(new_array[:, coronal_slice_num, :], interpolation='nearest', aspect='auto')
#         ax[1].set_title('xz')
#         ax[2].imshow(new_array[:, :, sagittal_slice_num], interpolation='nearest', aspect='auto')
#         ax[2].set_title('yz')
#         plt.show()
#
#     return resampled


def rotate_and_resample_yaw(im, z_slice, yaw, show=False):
    arr = sitk.GetArrayFromImage(im)
    z, y, x = np.nonzero(arr)
    mean_z = np.mean(z)
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # ROTATING
    rotation_centre = (int(np.ceil(mean_x)), int(np.ceil(mean_y)), int(np.ceil(mean_z)))
    rotation_centre_im = im.TransformIndexToPhysicalPoint(rotation_centre)

    rigid_euler = sitk.Euler3DTransform()
    rigid_euler.SetRotation(0, 0, yaw)

    rigid_versor = sitk.VersorRigid3DTransform()
    rigid_versor.SetMatrix(rigid_euler.GetMatrix())
    rigid_versor.SetCenter(rotation_centre_im)

    # RESAMPLING
    interpolator = sitk.sitkNearestNeighbor
    default_value = 0
    resampled = sitk.Resample(
        im,
        im,
        rigid_versor,
        interpolator,
        default_value
    )

    resampled = sitk.Cast(resampled, sitk.sitkFloat32)

    if show:
        new_array = sitk.GetArrayFromImage(resampled)
        fig = plt.figure()
        plt.imshow(new_array[z_slice])
        plt.show()

    return resampled


def rotate_and_resample_2d(im, z_slice, angle, show=False):
    im = im[:, :, z_slice]
    arr = sitk.GetArrayFromImage(im)
    y, x = np.nonzero(arr)
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # ROTATING
    rotation_centre = (int(np.ceil(mean_x)), int(np.ceil(mean_y)))
    rotation_centre_im = im.TransformIndexToPhysicalPoint(rotation_centre)

    rigid_euler = sitk.Euler2DTransform()
    rigid_euler.SetAngle(angle)
    rigid_euler.SetCenter(rotation_centre_im)

    # rigid_versor = sitk.VersorRigid2DTransform()
    # rigid_versor.SetMatrix(rigid_euler.GetMatrix())
    # rigid_versor.SetCenter(rotation_centre_im)

    # RESAMPLING
    interpolator = sitk.sitkNearestNeighbor
    default_value = 0
    resampled = sitk.Resample(
        im,
        im,
        rigid_euler,
        interpolator,
        default_value
    )

    resampled = sitk.Cast(resampled, sitk.sitkFloat32)

    if show:
        new_array = sitk.GetArrayFromImage(resampled)
        plt.imshow(new_array, cmap='gray')
        # plt.set_title('xy')

        plt.show()
        plt.close()
    return resampled


def show_img(im, cmap):
    new_array = sitk.GetArrayFromImage(im)
    plt.imshow(new_array, cmap=cmap)
    plt.axis('off')
    plt.show()
    plt.close()


def main(im_path, mask_path, show_images=False, save_loc=None):

    # input the image plus its mask
    im = sitk.ReadImage(im_path)
    mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)

    def loss_func(angle, args):
        angle = angle[0]
        z_slice, mask = args
        print('angle: {}'.format(angle))
        mask_rotate = rotate_and_resample_2d(mask, z_slice, angle)
        maxProj_flip = sitk.Flip(
            mask_rotate,
            (True, False),
            flipAboutOrigin=True)
        maxProj_flip.CopyInformation(mask_rotate)
        subtArray = sitk.GetArrayFromImage(mask_rotate - maxProj_flip)
        if show_images:
            show_img(mask_rotate - maxProj_flip, 'Greens')
        pix_count = np.count_nonzero(subtArray)
        print('pixel count: {}'.format(pix_count))
        return pix_count


    # initialize angle
    angle_0 = [0]

    # calculate centre of mass for ls_bin_out to use as optimal slice location
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    centroid = stats.GetCentroid(1)
    z_slice = mask.TransformPhysicalPointToIndex(centroid)[2]

    # optimize
    result = optimize.minimize(
        loss_func,
        angle_0,
        args=[z_slice, mask],
        method='Powell',
        bounds=[(-1, 1)])

    angle_final = result.x

    resampled = rotate_and_resample_yaw(im, z_slice, angle_final[0], show=False)
    if show_images:
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        for ax in axs.ravel():
            ax.axis('off')
        axs[0].imshow(sitk.GetArrayFromImage(im[:, :, z_slice]), cmap='gray')
        axs[0].set_title('Original')
        axs[1].imshow(sitk.GetArrayFromImage(resampled[:, :, z_slice]), cmap='gray')
        axs[1].set_title('Result')
        plt.show()
        plt.close()

    if save_loc is not None:
        sitk.WriteImage(resampled, os.path.join(save_loc, os.path.basename(im_path).split('.nii.gz')[0] + '_aligned.nii.gz'))


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: python image_3D_align.py path/to/image path/to/mask {show_images True or False} {save location}")
    main(*sys.argv[1:])
