import glob
import SimpleITK as sitk
import os
from src.groupwise import reverse_registration

config = {
    'version': 50,
    'unsegmented_input_folder': './data/input/unsegmented',
    'prior_folder': './data/output/priors',
    'output_folder': './data/output/registration_prior',
    'rigid_param': {'learn_rate': 0.1,
                    'min_step': 0.0001,
                    'max_iter': 50,
                    'shrink': [1],
                    'smooth': [0]},
    'affine_param': {'learn_rate': 0.05,
                     'min_step': 0.00001,
                     'max_iter': 50,
                     'shrink': [1],
                     'smooth': [0]},
    'nonlin_param': {'cpn': 5,
                     'learn_rate': 1,
                     'min_step': 0.1,
                     'max_iter': 10,
                     'shrink': [2, 1],
                     'smooth': [1, 0],
                     'scale_factors': [1, 2]},
    'iter_types': ['Rigid'] * 1 + ['Affine'] * 1 + ['NonLinear'] * 1
}

if __name__ == '__main__':

    os.makedirs(os.path.join(config['output_folder'], f'{config["version"]}'))

    template = sitk.ReadImage(os.path.join(config['prior_folder'], f'{config["version"]}', 'template.nii.gz'))
    csf_prior = sitk.ReadImage(os.path.join(config['prior_folder'], f'{config["version"]}', 'csf.nii.gz'))
    gm_prior = sitk.ReadImage(os.path.join(config['prior_folder'], f'{config["version"]}', 'gm.nii.gz'))
    wm_prior = sitk.ReadImage(os.path.join(config['prior_folder'], f'{config["version"]}', 'wm.nii.gz'))

    unsegmented_images = []
    for file_name in sorted(glob.glob(os.path.join(config['unsegmented_input_folder'], '*.nii.gz'))):
        unsegmented_images.append(sitk.ReadImage(file_name, sitk.sitkFloat32))

    transformation_list = []

    for i, current_image in enumerate(unsegmented_images):
        trans, stages = reverse_registration(current_image,
                                             template,
                                             config['iter_types'],
                                             config['rigid_param'],
                                             config['affine_param'],
                                             config['nonlin_param'],
                                             init_transformation=None)
        transformation_list.append(trans)
        sitk.WriteTransform(trans, os.path.join(config['output_folder'],
                                                f'{config["version"]}', f'transformation_{i}.tfm'))
