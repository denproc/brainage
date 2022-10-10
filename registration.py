import os
import glob
import SimpleITK as sitk
from src.groupwise import groupwise_registration

config = {
    'input_folder': './data/input/segmented',
    'output_folder': './data/output/registration',
    'version': 50,
    'rigid_param': {'learn_rate': 1,
                    'min_step': 0.01,
                    'max_iter': 50,
                    'shrink': [1],
                    'smooth': [0]},
    'affine_param': {'learn_rate': .05,
                     'min_step': 0.001,
                     'max_iter': 50,
                     'shrink': [1],
                     'smooth': [0]},
    'nonlin_param': {'cpn': 5,
                     'learn_rate': 1,
                     'min_step': 0.1,
                     'max_iter': 10,
                     'shrink': [8, 4, 2],
                     'smooth': [3, 2, 1],
                     'scale_factors': [1, 2, 3]},
    'iter_types': ['Rigid'] * 2 + ['Affine'] * 2 + ['NonLinear'] * 1
}

if __name__ == '__main__':

    os.makedirs(os.path.join(config['output_folder'], f'{config["version"]}'))

    input_images = []
    for file_name in sorted(glob.glob(os.path.join(config['input_folder'], '*img.nii.gz'))):
        input_images.append(sitk.ReadImage(file_name, sitk.sitkFloat32))

    avg_image = None
    init_transf = None

    trans, averages = groupwise_registration(input_images,
                                             config['iter_types'],
                                             config['rigid_param'],
                                             config['affine_param'],
                                             config['nonlin_param'],
                                             init_transformations=None,
                                             average_images=None)

    ##### Save the average images to disk
    for i, img in enumerate(averages):
        sitk.WriteImage(img, os.path.join(config['output_folder'],
                                          f'{config["version"]}', f'nl001_average_{i}.nii.gz'))

        # Save the transformations to disk
    for i, transform in enumerate(trans):
        sitk.WriteTransform(transform, os.path.join(config['output_folder'],
                                                    f'{config["version"]}', f'nl001_transformation_{i}.tfm'))
