# The functionality is provided by Dr Marc Modat (@mmodat)
import SimpleITK as sitk
import glob
import logging
import time

logging.basicConfig(level=logging.DEBUG)


def timing_registration():
    """
    Function use to time another function
    """
    def _wrapper(func):
        def _timer(*args, **kargs):
            start = time.time()
            res = func(*args, **kargs)
            end = time.time()
            logging.info('Registration finished in {} min and {} sec'.format(
                int((end-start)/60),
                round((end - start) % 60)
            ))
            return res
        return _timer
    return _wrapper


@timing_registration()
def pairwise_registration(reference,
                          floating,
                          reg_type='Rigid',
                          reg_param=None,
                          init_trans=None):
    """
    The function is used to register two images, one used as reference and the
    other as floating.
    :param reference: reference image as a SimpleITK image
    :param floating: floating image as a SimpleITK image
    :param reg_type: define the registration type: Rigid, Affine or Non-Linear
    :param reg_param: dictionary containing the registration parameters
    :param init_trans: SimpleITK transformation to use for initialisation
    :return: a SimpleITK transformation
    """

    def _debug_info(method):
        logging.debug(
            " Level {} | Param Num {} | iteration {} | Measure = {}".format(
                method.GetCurrentLevel(),
                method.GetInitialTransform().GetNumberOfParameters(),
                method.GetOptimizerIteration(),
                method.GetMetricValue()))

    # Set the default registration parameters if none were provided
    if reg_param is None:
        reg_param = {'cpn': 1,
                     'learn_rate': 1,
                     'min_step': 1,
                     'max_iter': 1,
                     'shrink': [1],
                     'smooth': [],
                     'scale_factors': [1]}
    else:
        if reg_param.get('cpn') is None:
            reg_param['cpn'] = 1
        if reg_param.get('learn_rate') is None:
            reg_param['learn_rate'] = 1
        if reg_param.get('min_step') is None:
            reg_param['min_step'] = 1
        if reg_param.get('max_iter') is None:
            reg_param['max_iter'] = 1
        if reg_param.get('shrink') is None:
            reg_param['shrink'] = [1]
        if reg_param.get('smooth') is None:
            reg_param['smooth'] = [0]
        if reg_param.get('scale_factors') is None:
            reg_param['scale_factors'] = [1]


    # Create a registration object
    reg = sitk.ImageRegistrationMethod()

    # Define the measure of similarity: mean squares - feel free to change
    reg.SetMetricAsMeanSquares()

    # Define a pyramid to use a coarse-to-fine approach - These parameters can
    # be changed to run the registration at lower resolution (and thus faster)
    reg.SetShrinkFactorsPerLevel(shrinkFactors=reg_param['shrink'])
        #shrinkFactors=list([2**l for l in range(
        #    reg_param['pyramid_lvl']-1, -1, -1)]))
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=reg_param['smooth'])
        # smoothingSigmas=list(range(reg_param['pyramid_lvl']-1, -1, -1)))

    # Create a transformation object based on the provided argument
    tx = None
    if reg_type == 'Rigid':
        if floating.GetDimension() == 2:
            tx = sitk.CenteredTransformInitializer(reference, floating,
                                                   sitk.Euler2DTransform())
        elif floating.GetDimension() == 3:
            tx = sitk.CenteredTransformInitializer(reference, floating,
                                                   sitk.Euler3DTransform())
    elif reg_type == 'Affine':
        tx = sitk.AffineTransform(reference.GetDimension())
    elif reg_type == 'NonLinear':
        # Define the mesh size at the lowest resolution
        reg_param['max_iter'] = int(reg_param['max_iter'])
        transform_mesh_size = [reg_param['cpn']] * floating.GetDimension()
        tx = sitk.BSplineTransformInitializer(reference,
                                              transform_mesh_size)
    else:
        raise ValueError('Unexpected registration type is provided')

    # Initialise with the provided transformation
    if init_trans is not None:
        reg.SetMovingInitialTransform(init_trans)

    if reg_type == 'NonLinear':
        # Increase the mesh size by two along each axis per level
        reg.SetInitialTransformAsBSpline(tx, scaleFactors=reg_param['scale_factors'])
    else:
        reg.SetInitialTransform(tx)

    # Set the parameters of the optimiser
    reg.SetOptimizerAsRegularStepGradientDescent(reg_param['learn_rate'],
                                                 reg_param['min_step'],
                                                 reg_param['max_iter'])

    # Define the interpolation technique used for resampling - linear
    # interpolation is used here
    reg.SetInterpolator(sitk.sitkLinear)

    if logging.getLogger().level == logging.DEBUG:
        reg.AddCommand(sitk.sitkIterationEvent,
                       lambda: _debug_info(reg))

    # Run the pairwise registration and returns the obtained transformation
    current_trans = reg.Execute(reference, floating)
    # Compose the initial and recovered transformation if required
    if init_trans is not None:
        comp_trans = sitk.CompositeTransform([init_trans, current_trans])
        comp_trans.FlattenTransform()
        return comp_trans
    return current_trans


def resample_image(reference,
                   floating,
                   transformation=None):
    """
    The function is used to resample one image, defined as floating, into the
    space of another defined a reference, given a provided transformation or an
    identity transformation
    :param reference: reference image as a SimpleITK image
    :param floating: floating image as a SimpleITK image
    :param transformation: transformation as a SimpleITK transform object
    :return: warped floating image as a SimpleITK image
    """
    res = sitk.ResampleImageFilter()
    res.SetReferenceImage(reference)
    res.SetInterpolator(sitk.sitkLinear)
    res.SetDefaultPixelValue(0)
    if transformation is None:
        res.SetTransform(sitk.Transform(reference.GetDimension(),
                                        sitk.sitkIdentity))
    else:
        res.SetTransform(transformation)
    return res.Execute(floating)


def groupwise_registration(images,
                           iter_reg_types,
                           rig_param,
                           aff_param,
                           nlr_param,
                           init_transformations=None,
                           average_images=None):
    """
    The function, given a list of input images, runs a groupwise registration.
    :param images: list of input images, defined a SimpleITK images
    :param iter_reg_types: list containing the registration type to use
    for each iteration. The type is encoded a a string: Rigid, Affine or
    NonLinear
    :param rig_param: Dictionary that contains the parameters for rigid
    registration
    :param aff_param: Dictionary that contains the parameters for affine
    registration
    :param nlr_param: Dictionary that contains the parameters for non-linear
    registration
    :param init_transformations:
    :param average_images:
    :return: a list of every average images and a list of transformations, one
    per input image.
    """
    # Create two filters, one for multiplying images and one for adding images
    multiply_filter = sitk.MultiplyImageFilter()
    addition_filter = sitk.AddImageFilter()

    # Create an empty list, which will contain all average images
    if average_images is None:
        average_images = []
        print('No average image is provided')

    # Create an list to store the latest transformations
    transformations = []

    # Create a list to store the latest initialisation transformation
    # Affine is initialised with rigid and nonlinear with affine
    if init_transformations is None:
        init_transformations = [None] * len(images)
        print('Transformation initialization from scratch')

    # Iterate over the number of groupwise step, as defined as an argument
    for g, reg_type in enumerate(iter_reg_types):
        logging.info('Starting groupwise step {}/{}: {}'
                     .format(g+1, len(iter_reg_types), reg_type))
        # Based on the registration type, extract the appropriate parameter
        # dictionary
        reg_param = {}
        if reg_type == 'Rigid':
            reg_param = rig_param
        if reg_type == 'Affine':
            reg_param = aff_param
        if reg_type == 'NonLinear':
            reg_param = nlr_param

        # Set the image being used as reference image for the current iteration
        if len(average_images) == 0:
            previous = images[0]
        else:
            previous = average_images[-1]

        # Create an empty image to accumulate the results
        average_image = sitk.Image(previous.GetSize(),
                                   sitk.sitkFloat32)
        average_image.CopyInformation(previous)
        average_image = multiply_filter.Execute(average_image, 0)
        # Reset the transformation list so that only the latest transformations
        # are kept
        transformations = []

        # Apply the pairwise registrations and compute the next average image
        for c, current_image in enumerate(images):
            logging.info('Starting {} registration of image {}/{}'
                         .format(reg_type, c+1, len(images)))
            current_trans = pairwise_registration(previous,
                                                  current_image,
                                                  reg_type,
                                                  reg_param,
                                                  init_transformations[c])
            transformations.append(current_trans)
            result = resample_image(previous,
                                    current_image,
                                    current_trans)
            average_image = addition_filter.Execute(average_image, result)
        average_image = multiply_filter.Execute(average_image, 1/len(images))
        average_images.append(average_image)
        if g < len(iter_reg_types) - 1:
            if iter_reg_types[g+1] != reg_type:
                logging.info('Setting/Updating the initialisation transforms')
                init_transformations = [_ for _ in transformations]
    # Returns the list of all transformations, one per image, and the list of
    # average images, one per iteration.
    return transformations, average_images


def reverse_registration(reference,
                         floating,
                         iter_reg_types,
                         rig_param,
                         aff_param,
                         nlr_param,
                         init_transformation=None):
    """
    The function, given an input images, runs a combination of pairwise registrations.
    :param reference: reference input images, defined a SimpleITK images
    :param floating: floating input images, defined a SimpleITK images
    :param iter_reg_types: list containing the registration type to use
    for each iteration. The type is encoded a a string: Rigid, Affine or
    NonLinear
    :param rig_param: Dictionary that contains the parameters for rigid
    registration
    :param aff_param: Dictionary that contains the parameters for affine
    registration
    :param nlr_param: Dictionary that contains the parameters for non-linear
    registration
    :param init_transformations:
    :return: a list of transformations and a list of registered floating images
    per input image.
    """

    # Create an list to store the latest transformations
    transformation = None
    stages = []

    # Create a list to store the latest initialisation transformation
    # Affine is initialised with rigid and nonlinear with affine
    if init_transformation is None:
        print('Transformation initialization from scratch')

    # Iterate over the number of groupwise step, as defined as an argument
    for g, reg_type in enumerate(iter_reg_types):
        print('Starting reverse step {}/{}: {}'
              .format(g + 1, len(iter_reg_types), reg_type))
        # Based on the registration type, extract the appropriate parameter
        # dictionary
        reg_param = {}
        if reg_type == 'Rigid':
            reg_param = rig_param
        if reg_type == 'Affine':
            reg_param = aff_param
        if reg_type == 'NonLinear':
            reg_param = nlr_param

        # Create an empty image to accumulate the results

        transformation = pairwise_registration(reference,
                                               floating,
                                               reg_type,
                                               reg_param,
                                               init_transformation)
        result = resample_image(reference,
                                floating,
                                transformation)
        stages.append(result)

        if g < len(iter_reg_types) - 1:
            if iter_reg_types[g + 1] != reg_type:
                print('Setting/Updating the initialisation transforms')
                init_transformation = transformation
    # Returns the list of all transformations, one per image, and the list of
    # average images, one per iteration.
    return transformation, stages


if __name__ == "__main__":
    # Read all images and store them within a list
    image_list = []
    for n, filename in enumerate(glob.glob('img_*.nii.gz')):
        image_list.append(sitk.ReadImage(filename,
                                         sitk.sitkFloat32))

    # Define the registration parameters for the three different registration
    # types: rigid, affine and non-linear
    rigid_param = {'learn_rate': 1,
                   'min_step': 1,
                   'max_iter': 1,
                   'pyramid_lvl': 1}
    affine_param = {'learn_rate': 1,
                    'min_step': 1,
                    'max_iter': 1,
                    'pyramid_lvl': 1}
    nonlin_param = {'cpn': 1,
                    'learn_rate': 1,
                    'min_step': 1,
                    'max_iter': 1,
                    'pyramid_lvl': 1}
    # Create a list that contains the number of groupwise registration as well
    # the registration type of each step
    iter_types = ['Rigid'] * 1 + ['Affine'] * 1 + ['NonLinear'] * 1
    # Run the overall groupwise registration and returns the average images,
    # one per groupwise iteration step and the final transformations, one per
    # input image
    trans, averages = groupwise_registration(image_list,
                                             iter_types,
                                             rigid_param,
                                             affine_param,
                                             nonlin_param)
    # Save the average images to disk
    for i, img in enumerate(averages):
        sitk.WriteImage(img, 'average_' + str(i) + '.nii.gz')

    # Save the transformations to disk
    for i, transform in enumerate(trans):
        sitk.WriteTransform(transform, 'transformation_' + str(i) + '.tfm')
