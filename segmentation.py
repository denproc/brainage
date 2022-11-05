import glob
import SimpleITK as sitk
import os
from src.gmm import GMM
import numpy as np

config = {
    'version': 50,
    'input_folder': './data/input/unsegmented',
    'segmentation_prior_input_folder': './data/output/segmentation_priors',
    'output_folder': './data/output/segmentations'
}

if __name__ == '__main__':

    os.makedirs(os.path.join(config['output_folder'], f'{config["version"]}'))

    priors = {'csf': None, 'gm': None, 'wm': None, 'bg': None}
    segmentation_keys = ['csf', 'gm', 'wm']

    for index, file_name in enumerate(sorted(glob.glob(os.path.join(config['input_folder'], '*.nii.gz')))):
        unsegmented_image = sitk.ReadImage(file_name, sitk.sitkFloat32)
        for key in segmentation_keys:
            file_name_prior = os.path.join(config['segmentation_prior_input_folder'],
                                           f'{config["version"]}', f'{index}', f'{key}.nii.gz')
            priors[key] = sitk.GetArrayFromImage(sitk.ReadImage(file_name_prior, sitk.sitkFloat32))
        priors['bg'] = 1 - (priors['csf'] + priors['gm'] + priors['wm'])

        image = sitk.GetArrayFromImage(unsegmented_image)
        masked_img = image * (1 - priors['bg'] > 1e-3)
        masked_prior = {}
        for key in segmentation_keys:
            masked_prior[key] = priors[key] * (1 - priors['bg'] > 1e-3)

        masked_prior = np.stack([masked_prior['csf'], masked_prior['gm'], masked_prior['wm']], -1)

        model_mrf = GMM(n_components=3, max_iter=25, tol=1e-3, prior=True, mrf=0.5)
        losses, probability_map, segmentation = model_mrf.fit_predict(masked_img, masked_prior)

        segmentation_img = sitk.GetImageFromArray(segmentation)
        segmentation_img.CopyInformation(unsegmented_image)

        sitk.WriteImage(segmentation_img, os.path.join(config['output_folder'],
                                                       f'{config["version"]}', f'{index}.nii.gz'))
