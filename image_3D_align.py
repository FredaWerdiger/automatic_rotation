import sys
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import os
import glob


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
        plt.imshow(np.flipud(new_array[z_slice]))
        plt.show()
        plt.close()

    return resampled


def rotate_and_resample_pitch(im, z_slice, pitch, return_slice=True, show=False):
    arr = sitk.GetArrayFromImage(im)
    z, y, x = np.nonzero(arr)
    mean_z = np.mean(z)
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # ROTATING
    rotation_centre = (int(np.ceil(mean_x)), int(np.ceil(mean_y)), int(np.ceil(mean_z)))
    rotation_centre_im = im.TransformIndexToPhysicalPoint(rotation_centre)

    rigid_euler = sitk.Euler3DTransform()
    rigid_euler.SetRotation(pitch, 0, 0)

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

    try:
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(resampled)
        centroid = stats.GetCentroid(1)
        centroid_index = resampled.TransformPhysicalPointToIndex(centroid)
        z_slice = centroid_index[2]
    except RuntimeError:
        z_slice = z_slice

    resampled = sitk.Cast(resampled, sitk.sitkFloat32)

    if show:
        new_array = sitk.GetArrayFromImage(resampled)
        plt.figure()
        plt.imshow(np.flipud(new_array[z_slice]))
        plt.show()
        plt.close()

    if return_slice:
        return resampled[:, :, z_slice], z_slice
    else:
        return resampled, z_slice


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
    new_array = np.flipud(sitk.GetArrayFromImage(im))
    plt.imshow(new_array, cmap=cmap)
    plt.axis('off')
    plt.show()
    plt.close()


def decide_direction(mask, z_slice):
    # decide which direction is better to flip
    arr = sitk.GetArrayFromImage(mask[:, :, z_slice])
    y, x = np.nonzero(arr)
    mean_y = np.mean(y)
    mean_x = np.mean(x)
    mask_flip_x = sitk.Flip(
        mask[:, :, z_slice],
        (False, True),
        flipAboutOrigin=True)
    arr = sitk.GetArrayFromImage(mask_flip_x)
    y, _ = np.nonzero(arr)
    mean_y_flip = np.mean(y)
    dif_y = abs(mean_y_flip - mean_y)
    mask_flip_y = sitk.Flip(
        mask[:, :, z_slice],
        (True, False),
        flipAboutOrigin=True)
    arr = sitk.GetArrayFromImage(mask_flip_y)
    _, x = np.nonzero(arr)
    mean_x_flip = np.mean(x)
    dif_x = abs(mean_x_flip - mean_x)
    if dif_x > dif_y:
        return (False, True)
    else:
        return (True, False)


# def main(im_path, mask_path, show_images=False, save_loc=None):
masks = glob.glob('Z:/data_freda/ctp_project/CTP_DL_Data/DATA/ncct_mask/*')
# masks = glob.glob('Z:/data_freda/ctp_project/CTP_DL_Data/DATA/ncct_mistar_mask/*')
nccts = glob.glob('Z:/data_freda/ctp_project/CTP_DL_Data/DATA/ncct/*')

good_ones = [0, 1, 5, 6, 11, 14, 15, 17, 23, 26, 27, 31, 39, 40, 41, 42, 44, 47, 48, 50,
             51, 54, 56, 58, 63, 66, 68, 69, 70, 71, 72, 73, 74, 75, 76, 79, 80, 83, 84, 85, 86,
             88, 89, 91, 95, 99, 101, 102, 103, 105, 110, 111, 112, 114, 116, 127, 130, 137,
             139, 140, 145, 150, 156, 158, 159, 162, 169, 180, 181, 184, 186, 187, 190, 199,
             203, 208, 215, 217, 224]

show_images = False
# for im_path, mask_path in zip(nccts, masks):
for ind in good_ones:
    im_path = nccts[ind]
    mask_path = masks[ind]
    print(os.path.basename(im_path).split('.nii.gz')[0])
    # input the image plus its mask
    im = sitk.ReadImage(im_path)
    mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)


    def loss_func(angle, args):
        # should decide which way to flip based on the shift
        angle = angle[0]
        z_slice, mask, direction = args
        print('angle: {}'.format(angle))
        mask_rotate = rotate_and_resample_2d(mask, z_slice, angle)
        maxProj_flip = sitk.Flip(
            mask_rotate,
            direction,
            flipAboutOrigin=True)
        maxProj_flip.CopyInformation(mask_rotate)
        subtArray = sitk.GetArrayFromImage(mask_rotate - maxProj_flip)
        if show_images:
            show_img(mask_rotate - maxProj_flip, 'Greens')
        pix_count = np.count_nonzero(subtArray)
        print('pixel count: {}'.format(pix_count))
        return pix_count


    def loss_func_gantry(angle, args):
        angle = angle[0]
        z_slice, mask = args
        print('angle: {}'.format(angle))
        mask_rotate, z_slice = rotate_and_resample_pitch(mask, z_slice, angle, return_slice=True, show=False)
        if show_images:
            show_img(mask_rotate, 'Greens')
        mask_rotate = sitk.GetArrayFromImage(mask_rotate)
        # maxProj_flip = sitk.Flip(
        #     mask_rotate,
        #     (True, False),
        #     flipAboutOrigin=True)
        # maxProj_flip.CopyInformation(mask_rotate)
        # subtArray = sitk.GetArrayFromImage(mask_rotate - maxProj_flip)
        pix_count = mask_rotate.size - np.count_nonzero(mask_rotate)
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
    # show_img(im[:,:,z_slice], 'gray')

    gantry_results = optimize.minimize(
        loss_func_gantry,
        angle_0,
        args=[z_slice, mask],
        method='Powell',
        bounds=[(-1, 1)])

    angle_final_gantry = gantry_results.x

    mask_gantry, z_slice = rotate_and_resample_pitch(mask,
                                                     z_slice,
                                                     angle_final_gantry[0],
                                                     return_slice=False,
                                                     show=False)

    im_gantry, _ = rotate_and_resample_pitch(im,
                                             z_slice,
                                             angle_final_gantry[0],
                                             return_slice=False,
                                             show=False)
    # recalculate the centroid now.

    # optimize
    direction = decide_direction(mask, z_slice)
    result = optimize.minimize(
        loss_func,
        angle_0,
        args=[z_slice, mask_gantry, direction],
        method='Powell',
        bounds=[(-1, 1)])

    angle_final = result.x

    resampled = rotate_and_resample_yaw(im_gantry, z_slice, angle_final[0], show=False)
    resampled_mask = rotate_and_resample_yaw(mask_gantry, z_slice, angle_final[0], show=False)

    # get centroid
    # resampled_mask = sitk.Cast(resampled_mask, sitk.sitkUInt8)
    # stats.Execute(resampled_mask[:, :, z_slice])
    # centroid_2d = stats.GetCentroid(1)
    # centroid_index_2d = mask_gantry[:, :, z_slice].TransformPhysicalPointToIndex(centroid_2d)
    mask_array = sitk.GetArrayFromImage(mask)
    z, y, x = np.nonzero(mask_array)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    centroid_index_2d = (int(np.ceil(mean_x)), mask_array.shape[1] - int(np.ceil(mean_y))) # flip in y

    # derive equation of line through hemisphere in cartesian coordinates
    # based on angle results and the centre point (Cx, 512 - Cy)
    m = np.tan(angle_final[0])
    c = (centroid_index_2d[1]) + (centroid_index_2d[0] * np.tan(angle_final[0]))
    x = np.linspace(0, mask_array.shape[1])

    y = -(m*x) + c # -m is there to account for the flip
    x = x[y > 0]
    y = y[y > 0]
    x = x[y < mask_array.shape[1]]
    y = y[y < mask_array.shape[1]]
    # # calculate hemisphere mask for 3d image
    my_func = lambda z, x, y: x - ((c-(mask_array.shape[1]-y)) / m) < 0
    right_hemisphere = np.fromfunction(my_func, mask_array.shape) * mask_array
    right_hemisphere_im = sitk.GetImageFromArray(right_hemisphere)
    right_hemisphere_im.CopyInformation(mask)
    my_func = lambda z, x,y: x - ((c-(mask_array.shape[1]-y)) / m) > 0
    left_hemisphere = np.fromfunction(my_func, mask_array.shape) * mask_array
    left_hemisphere_im = sitk.GetImageFromArray(left_hemisphere)
    left_hemisphere_im.CopyInformation(mask)
    save_loc = 'Z:/data_freda/ctp_project/CTP_DL_Data/DATA/'
    if not os.path.exists(os.path.join(save_loc, 'left_hemisphere_mask')):
        os.makedirs(os.path.join(save_loc, 'left_hemisphere_mask'))
    sitk.WriteImage(left_hemisphere_im,
                    os.path.join(save_loc, 'left_hemisphere_mask', 'leftmask_' + os.path.basename(im_path).split('ncct_')[1]))

    if not os.path.exists(os.path.join(save_loc, 'right_hemisphere_mask')):
        os.makedirs(os.path.join(save_loc, 'right_hemisphere_mask'))
    sitk.WriteImage(right_hemisphere_im,
                    os.path.join(save_loc, 'right_hemisphere_mask',
                                 'rightmask_' + os.path.basename(im_path).split('ncct_')[1]))

    im_array = sitk.GetArrayFromImage(im[:, :, z_slice])
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    axs[0].imshow(np.flipud(im_array), cmap='gray')
    axs[0].plot(y,x, 'r')
    for ax in axs.ravel():
        ax.axis('off')
    m = round(m, 2)
    c = round(c, 2)
    fontdict = {'color': 'y'}
    axs[0].text(im_array.shape[1] / 5, im_array.shape[1] - 30, f"y = {round(m, 2)}x + {round(c, 2)}", fontdict=fontdict)
    axs[0].set_title('Original')
    axs[0].plot(centroid_index_2d[1], centroid_index_2d[0], 'r*')
    axs[1].imshow(np.flipud(sitk.GetArrayFromImage(resampled[:, :, z_slice])), cmap='gray')
    axs[1].set_title('Rotated')
    axs[2].imshow(np.flipud(left_hemisphere[z_slice]), cmap='gray')
    axs[2].set_title('Left Hemisphere')
    if not os.path.exists(os.path.join(save_loc, 'ncct_hemisphere_pngs')):
        os.makedirs(os.path.join(save_loc, 'ncct_hemisphere_pngs'))
    plt.savefig(os.path.join(save_loc, 'ncct_hemisphere_pngs', os.path.basename(im_path).split('.nii.gz')[0] + '_autorotate.png'),
                bbox_inches='tight',
                dpi=250)

    plt.close()
    # if save_loc is not None:
    #     sitk.WriteImage(resampled, os.path.join(save_loc, os.path.basename(im_path).split('.nii.gz')[0] + '_aligned.nii.gz'))
    #
    # return angle_final

# if __name__ == '__main__':
#
#     if len(sys.argv) < 2:
#         print("Usage: python image_3D_align.py path/to/image path/to/mask {show_images True or False} {save location}")
#     main(*sys.argv[1:])
