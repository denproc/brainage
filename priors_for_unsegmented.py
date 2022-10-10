import glob
import SimpleITK as sitk
import os
from src.groupwise import resample_image

config = {
    'version': 50,
    'unsegmented_input_folder': './data/input/unsegmented',
    'transformation_input_folder': './data/output/registration_prior',
    'prior_input_folder': './data/output/priors',
    'output_folder': './data/output/segmentation_priors',
}

if __name__ == '__main__':

    os.makedirs(os.path.join(config['output_folder'], f'{config["version"]}'))

    unsegmented_images = []
    for file_name in sorted(glob.glob(os.path.join(config['unsegmented_input_folder'], '*.nii.gz'))):
        unsegmented_images.append(sitk.ReadImage(file_name, sitk.sitkFloat32))

    transformations = []
    for i in range(len(unsegmented_images)):
        transformations.append(sitk.ReadTransform(os.path.join(config['transformation_input_folder'],
                                                               f'{config["version"]}',
                                                               f'transformation_{i}.tfm')))

    template = sitk.ReadImage(os.path.join(config['prior_input_folder'], f'{config["version"]}', 'template.nii.gz'))
    csf_prior = sitk.ReadImage(os.path.join(config['prior_input_folder'], f'{config["version"]}', 'csf.nii.gz'))
    gm_prior = sitk.ReadImage(os.path.join(config['prior_input_folder'], f'{config["version"]}', 'gm.nii.gz'))
    wm_prior = sitk.ReadImage(os.path.join(config['prior_input_folder'], f'{config["version"]}', 'wm.nii.gz'))

    for i, (current_image, trans) in enumerate(zip(unsegmented_images, transformations)):
        template_resampled = resample_image(current_image, template, trans)
        csf_resampled = resample_image(current_image, csf_prior, trans)
        gm_resampled = resample_image(current_image, gm_prior, trans)
        wm_resampled = resample_image(current_image, wm_prior, trans)
        # save
        path = os.path.join(config['output_folder'], f'{config["version"]}', f'{i:d}')
        if not os.path.exists(path):
            os.makedirs(path)
        sitk.WriteImage(template_resampled, os.path.join(path, 'template.nii.gz'))
        sitk.WriteImage(csf_resampled, os.path.join(path, 'csf.nii.gz'))
        sitk.WriteImage(gm_resampled, os.path.join(path, 'gm.nii.gz'))
        sitk.WriteImage(wm_resampled, os.path.join(path, 'wm.nii.gz'))
