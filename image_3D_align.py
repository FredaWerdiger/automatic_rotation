import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


def rotate_and_resample(im, roll, pitch, yaw, show=False):
    arr = sitk.GetArrayFromImage(im)
    z, y, x = np.nonzero(arr)
    mean_z = np.mean(z)
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # ROTATING
    rotation_centre = (int(np.ceil(mean_x)), int(np.ceil(mean_y)), int(np.ceil(mean_z)))
    rotation_centre_im = im.TransformIndexToPhysicalPoint(rotation_centre)

    rigid_euler = sitk.Euler3DTransform()
    rigid_euler.SetRotation(roll, pitch, yaw)

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
        fig, ax = plt.subplots(1, 3, figsize=(10, 3))
        axial_slice_num = int(input('Enter axial slice number you wish to see:'))
        coronal_slice_num = int(input('Enter coronal slice number:'))
        sagittal_slice_num = int(input('Enter sagittal slice number:'))
        ax[0].imshow(new_array[axial_slice_num])
        ax[0].set_title('xy')
        ax[1].imshow(new_array[:, coronal_slice_num, :], interpolation='nearest', aspect='auto')
        ax[1].set_title('xz')
        ax[2].imshow(new_array[:, :, sagittal_slice_num], interpolation='nearest', aspect='auto')
        ax[2].set_title('yz')
        plt.show()

    return resampled


def main():

    # input the image plus its mask
    im = sitk.ReadImage("../out_files/fixed_brain.nii.gz")
    mask = sitk.ReadImage("../out_files/ls_bin_out.nii.gz")

    # loss function for z-angle (YAW)
    def loss_func(angle, args):
        pitch = angle[0]  # dont know why it passes as an array
        roll = angle[1]
        yaw = angle[2]
        mask, z_slice, im = args
        print('angle: {}'.format(angle))
        mask_rotate = rotate_and_resample(mask, pitch, roll, yaw)
        im_rotate = rotate_and_resample(im, pitch, roll, yaw)
        im_slice = im_rotate[:, :, z_slice]
        maxProj_mask_rotate = sitk.Cast(im_slice, 2) * sitk.Cast(mask_rotate[:, :, z_slice], 2)  # adjust
        maxProj_flip = sitk.Flip(
            maxProj_mask_rotate,
            (True, False, False),
            flipAboutOrigin=True)
        maxProj_flip.CopyInformation(maxProj_mask_rotate)
        #    pix_count = np.count_nonzero(sitk.GetArrayFromImage(maxProj_mask_rotate - maxProj_flip))
        subtArray = sitk.GetArrayFromImage(maxProj_mask_rotate - maxProj_flip)
        pix_count = np.count_nonzero(subtArray > 50)
        print('pixel count: {}'.format(pix_count))
        return pix_count

    # initialize angle
    angle_0 = [0, 0, 0]

    # calculate centre of mass for ls_bin_out to use as optimal slice location
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    centroid = stats.GetCentroid(1)
    z_slice = mask.TransformPhysicalPointToIndex(centroid)[2]

    # optimize
    result = optimize.minimize(
        loss_func,
        angle_0,
        args=[im, z_slice, mask],
        method='Powell',
        bounds=[(-1, 1), (-1, 1), (-1, 1)])

    angle_final = result.x

    resampled = rotate_and_resample(im, 0, 0, angle_final[0])

    sitk.WriteImage(resampled, '../out_files/automatic_rotation.nii.gz')


if __name__ == '__main__':
    main()
