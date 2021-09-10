import SimpleITK
import numpy as np
from sklearn.metrics import normalized_mutual_info_score


def translate_to_best_init_position(fixed_image_filename, moving_image_filename, output_image_filename,
                                    mi_sample_percentage=30):
    # pre-processing  - shift a slab to best starting point ########
    smooth_filter = SimpleITK.MedianImageFilter()
    smooth_filter.SetRadius(3)
    fixed_image_o = SimpleITK.ReadImage(fixed_image_filename)
    fixed_image = smooth_filter.Execute(fixed_image_o)

    moving_image_o = SimpleITK.ReadImage(moving_image_filename)
    moving_image = smooth_filter.Execute(moving_image_o)

    # compute size of images (mm)
    fixed_image_size = np.array(fixed_image.GetSize()) * np.array(fixed_image.GetSpacing())
    moving_image_size = np.array(moving_image.GetSize()) * np.array(moving_image.GetSpacing())

    # compare ratio between images
    max_ratio = np.max(fixed_image_size / moving_image_size)

    best_offset = [0.0, 0.0, 0.0]

    best_score = -10000000000

    print ('max_ratio: %s' % max_ratio)
    if max_ratio > 2:
        max_difference = np.max(fixed_image_size - moving_image_size)
        largest_different_axis = np.argmax(fixed_image_size - moving_image_size)

        translation_direction = np.array([0, 0, 0])
        translation_direction[largest_different_axis] = 1

        translation_step_0 = float(moving_image_size[largest_different_axis]) / 2.0
        translation_init_0 = max_difference - translation_step_0 / 2
        translation_init_0 = [-translation_init_0, translation_init_0]

        fixed_image_array = SimpleITK.GetArrayFromImage(fixed_image).ravel()

        for index_search in range(0, 2):
            translation_step = translation_step_0
            translation_init = translation_init_0
            translation_sizes = np.arange(translation_init[0], translation_init[1], translation_step)

            for translation_size in translation_sizes:

                offset = translation_direction * translation_size

                transform = SimpleITK.Transform(3, 10)
                transform.SetParameters([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, offset[0], offset[1], offset[2]])
                transform.SetFixedParameters([0.0, 0.0, 0.0])

                resample = SimpleITK.ResampleImageFilter()
                resample.SetReferenceImage(fixed_image)

                # SimpleITK supports several interpolation options, we go with the simplest that gives 
                # reasonable results.
                resample.SetInterpolator(SimpleITK.sitkLinear)
                resample.SetTransform(transform)
                offset_image = resample.Execute(moving_image)

                offset_image_array = SimpleITK.GetArrayFromImage(offset_image).ravel()

                isImageOneColor = ((np.max(offset_image_array) - np.min(offset_image_array)) < 2)

                if not isImageOneColor:
                    indexes = range(0, len(offset_image_array))
                    step_size = int(round(100 / mi_sample_percentage))

                    fixed_image_array_sample = fixed_image_array[indexes[::step_size]]
                    offset_image_array_sample = offset_image_array[indexes[::step_size]]

                    mi_score = normalized_mutual_info_score(fixed_image_array_sample, offset_image_array_sample)
                else:
                    mi_score = 0

                if mi_score > best_score:
                    best_score = mi_score
                    best_offset = offset

            translation_init_0 = [best_offset[largest_different_axis] - translation_step_0 / 2.0,
                                  best_offset[largest_different_axis] + translation_step_0 / 2.0 + 1]
            translation_step_0 /= 4

        transform = SimpleITK.Transform(3, 10)
        transform.SetParameters(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, best_offset[0], best_offset[1], best_offset[2]])
        transform.SetFixedParameters([0.0, 0.0, 0.0])

        resample = SimpleITK.ResampleImageFilter()
        resample.SetReferenceImage(fixed_image)

        # SimpleITK supports several interpolation options, we go with the simplest that gives reasonable results.
        resample.SetInterpolator(SimpleITK.sitkLinear)
        resample.SetTransform(transform)
        best_image = resample.Execute(moving_image)

        SimpleITK.WriteImage(best_image, output_image_filename + '.nii.gz')

        SimpleITK.WriteTransform(transform, output_image_filename + '.txt')

    return best_score > 0, best_offset
