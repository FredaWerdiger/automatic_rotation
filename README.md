# Automation alignment of 3D brain images
This code automatically aligns brain CT images in the axial plane so that the right and left hemispheres are on opposite sides of the image.

This way it can be used to make hemispheric measurements.
This code was optimized for CT images but will work for MR. All that is required is a brain mask. The brain mask is what is used to do the rotation. The result is simply applied to the original image at the end.

## Usage
python image_3D_align.py {path/to/image} {path/to/mask} {show_images True or False} {save location}
## output
The file will save a png image with the Line of Best Symmetry drawn, as well as the hemisphere masks.

The hemisphere masks outputs were used to train the Deep learning Solution.

## Deep Learning solutions for project 

The output of the brute force code was used to train deep learning models. Those models were trained on the ATLAS Mean Baseline brain-extracted images
For this you must uses
## train
The code is included for posterity but the trained model is available. You can use this code if you would like to retrain models on different sorts of images, i.e. skull images or MR images.
## infer
For inference, run through one image at a time. Simple denote the path to the image and the pull path name of the output image. The output image will be saved as a single channel NIFTI.
### usage
infer.py {mean baseline brain image full path} {full path to output image e.g. /path/to/image/image.nii.gz} {left or right} {absolute path to this code} 

### Dependencies
- MONAI version: 0.10.dev2237
- Numpy version: 1.21.2
- Pytorch version: 1.10.2