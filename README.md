# Automation alignment of 3D brain images
This code automatically aligns brain CT images in the axial plane so that the right and left hemispheres are on opposite sides of the image.

This way it can be used to make hemispheric measurements.
This code was optimized for CT images but will work for MR. All that is required is a brain mask. The brain mask is what is used to do the rotation. The result is simply applied to the original image at the end.

## Usage
python image_3D_align.py {path/to/image} {path/to/mask} {show_images True or False} {save location}
