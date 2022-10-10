import SimpleITK as sitk
import glob
import os
from src.groupwise import resample_image

config = {
    'version': 50,
    'segmentation_input_folder': './data/input/segmented',
    'registration_input_folder': './data/output/registration',
    'output_folder': './data/output/priors'
}


# def adjust_template(template):
#    pre_euler = sitk.Euler3DTransform()
#    centre = template.TransformContinuousIndexToPhysicalPoint(np.array(template.GetSize()) / 2.)
#    pre_euler.SetCenter(centre)
#    pre_euler.SetRotation(np.pi, 0, 0)
#    res = sitk.Resample(template, pre_euler)
#    new_directions = np.array(res.GetDirection()) * np.array([1, 1, 1, -1, -1, -1, -1, -1, -1])
#    res.SetDirection(new_directions)
#    size = np.array(res.GetSize())
#    new_origin = np.array(res.GetOrigin()) - np.array([0, size[-1], size[-2]])
#    res.SetOrigin(new_origin)
#    return res

def prior(segmentation, values, thr=0.5):
    csf = (values['csf'] - thr <= segmentation) * (segmentation < values['csf'] + thr)  # 1
    gm = (values['gm'] - thr <= segmentation) * (segmentation < values['gm'] + thr)  # 2
    wm = (values['wm'] - thr <= segmentation) * (segmentation < values['wm'] + thr)  # 3
    return csf, gm, wm


if __name__ == '__main__':

    os.makedirs(os.path.join(config['output_folder'], f'{config["version"]}'))

    segmentations = []
    for file_name in sorted(glob.glob(os.path.join(config['segmentation_input_folder'], '*seg.nii.gz'))):
        segmentations.append(sitk.ReadImage(file_name, sitk.sitkFloat32))

    avg_image = sitk.ReadImage(os.path.join(config['registration_input_folder'],
                                            f'{config["version"]}', 'average_4.nii.gz'))

    transformations = []
    for i in range(10):
        transformations.append(sitk.ReadTransform(os.path.join(config['registration_input_folder'],
                                                               f'{config["version"]}',
                                                               f'transformation_{i}.tfm')))

    # Apply transforms to segmentations
    transformed_segmentation = []
    for i, current_segmentation in enumerate(segmentations):
        tmp_seg = resample_image(avg_image, segmentations[i], transformations[i])
        transformed_segmentation.append(tmp_seg)

    segmentation_values = {'csf': 1, 'gm': 2, 'wm': 3}

    size = transformed_segmentation[0].GetSize()
    num = len(transformed_segmentation)
    priors = {'csf': sitk.Image(size, sitk.sitkUInt8),
              'gm': sitk.Image(size, sitk.sitkUInt8),
              'wm': sitk.Image(size, sitk.sitkUInt8)}
    priors['csf'].CopyInformation(transformed_segmentation[0])
    priors['gm'].CopyInformation(transformed_segmentation[0])
    priors['wm'].CopyInformation(transformed_segmentation[0])
    for i, current_segmentation in enumerate(transformed_segmentation):
        csf, gm, wm = prior(current_segmentation, segmentation_values)
        priors['csf'] += csf
        priors['gm'] += gm
        priors['wm'] += wm
    priors['csf'] /= num
    priors['gm'] /= num
    priors['wm'] /= num

    # for key in priors.keys():
    #    priors[key] = adjust_template(priors[key])

    for key, val in priors.items():
        sitk.WriteImage(val, os.path.join(config['output_folder'], f'{config["version"]}', f'{key}.nii.gz'))

    # avg_image = adjust_template(avg_image)
    sitk.WriteImage(avg_image, os.path.join(config['output_folder'], f'{config["version"]}', 'template.nii.gz'))
