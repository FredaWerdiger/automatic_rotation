import sys
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import os


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


def rotate_and_resample_2d(im, z_slice, angle):
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
    centroid_index = mask.TransformPhysicalPointToIndex(centroid)
    z_slice = centroid_index[2]

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
        im_array = np.flipud(sitk.GetArrayFromImage(im[:, :, z_slice]))
        m = np.tan(angle_final[0])
        c = centroid_index[1] + (centroid_index[0]*np.tan(angle_final[0]))
        x = np.linspace(0, im_array.shape[1])
        y = -(m*x) + c
        axs[0].imshow(im_array, cmap='gray')
        axs[0].plot(y, x, 'r')
        m = round(m, 2)
        c = round(c, 2)
        fontdict = {'color':'y'}
        axs[0].text(im_array.shape[1]/5, im_array.shape[1]-30, f"y = {round(m,2)}x + {round(c,2)}", fontdict=fontdict)
        axs[0].set_title('Original')
        axs[1].imshow(np.flipud(sitk.GetArrayFromImage(resampled[:, :, z_slice])), cmap='gray')
        axs[1].set_title('Result')
        plt.show()
        plt.close()

    if save_loc is not None:
        sitk.WriteImage(resampled, os.path.join(save_loc, os.path.basename(im_path).split('.nii.gz')[0] + '_aligned.nii.gz'))

    return angle_final

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: python image_3D_align.py path/to/image path/to/mask {show_images True or False} {save location}")
    main(*sys.argv[1:])
