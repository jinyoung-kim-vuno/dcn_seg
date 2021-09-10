import os

import nibabel as nib
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.measurements import center_of_mass
from utils.mathutils import compute_statistics, convert_arr_lst_to_elem, transformPoint3d
from utils.stltools import decimate_and_smooth, convert_vtkPolyData_to_vtkImageData, erode_and_dilate, measure_spd_stl

import six
from math import ceil, floor
from skimage import measure
from sklearn import preprocessing
from collections import Counter
from scipy.ndimage import label
import vtk
import itk
import heapq

class CropMask(object):
    def __init__(self, mask_path):
        self._mask_path = mask_path
        self._min_roi = None
        self._max_roi = None
        self._shape = None
        self._affine = None
        self._header = None
        self._loaded = False

    @property
    def mask_path(self):
        return self._mask_path

    @property
    def min_roi(self):
        self._ensure_loaded()
        return self._min_roi

    @property
    def max_roi(self):
        self._ensure_loaded()
        return self._max_roi

    @property
    def shape(self):
        self._ensure_loaded()
        return self._shape

    @property
    def affine(self):
        self._ensure_loaded()
        return self._affine

    @property
    def header(self):
        self._ensure_loaded()
        return self._header

    def crop(self, img_data):
        return img_data[self.min_roi[0]:self.max_roi[0],
               self.min_roi[1]:self.max_roi[1],
               self.min_roi[2]:self.max_roi[2]]

    def uncrop(self, cropped_data):
        data = np.zeros(self.shape)
        data[self.min_roi[0]:self.max_roi[0],
        self.min_roi[1]:self.max_roi[1],
        self.min_roi[2]:self.max_roi[2]] = cropped_data
        return data

    def _ensure_loaded(self):
        if not self._loaded:
            import nibabel as nib
            mask = nib.load(self._mask_path)
            mask_data = mask.get_data()
            mask_data_nonzero = mask_data.nonzero()
            self._min_roi = np.min(mask_data_nonzero, 1)
            self._max_roi = np.max(mask_data_nonzero, 1)
            self._shape = mask_data.shape
            self._affine = mask.get_affine()
            self._header = mask.get_header()
            self._loaded = True


def find_crop_mask(roi_mask_file):
    """
    :return: the cropping mask for the patient, or None if it does not (yet) exist
    :returns: CropMask
    """
    return CropMask(roi_mask_file) if os.path.exists(roi_mask_file) else print('No found crop mask')


def _img_data(path):
    return nib.load(path).get_data()


def _partition_mask(paths):
    result = None
    for path in paths:
        data = _img_data(path)
        if result is not None:
            result = result + data
        else:
            result = data
    return result


def crop_image(img_file, crop_mask, cropped_img_file):
    img = nib.load(img_file)
    img_data = img.get_data()
    croped_data = crop_mask.crop(img_data)
    croped_img = nib.Nifti1Image(croped_data, img.get_affine(), img.get_header())
    croped_img.to_filename(cropped_img_file)


def uncrop_image(cropped_img_file, crop_mask, out_file):

    cropped_img = nib.load(cropped_img_file)
    cropped_data = cropped_img.get_data()

    data = crop_mask.uncrop(cropped_data)
    uncropped_img = nib.Nifti1Image(data, crop_mask.affine, crop_mask.header)
    uncropped_img.to_filename(out_file)


def merge_image(img_path_1, img_path_2, out_file):
    """
    Merging two images (assuming binary masks) to one mask
    """
    img1 = nib.load(img_path_1)
    img2 = nib.load(img_path_2)
    data1 = img1.get_data()
    data2 = img2.get_data()
    merged_data = data1.copy()
    merged_data[data2 > 0] = 1
    merged_img = nib.Nifti1Image(merged_data, img1.get_affine(), img1.get_header())
    if not os.path.exists(os.path.dirname(out_file)):
        os.mkdir(os.path.dirname(out_file))
    merged_img.to_filename(out_file)


def compute_crop_mask(image, roi_mask_file, data_file, margin_crop_mask=(5, 5, 5)):

    image_size = image.shape
    roi_mask_vol = np.zeros(image_size)
    min_roi = np.min(image.nonzero(), 1) - np.array(margin_crop_mask)
    min_roi[min_roi < 0] = 0
    max_roi = np.max(image.nonzero(), 1) + np.array(margin_crop_mask)
    for d in range(3):
        max_roi[d] = min(max_roi[d], image_size[d])

    roi_mask_vol[min_roi[0]:max_roi[0], min_roi[1]:max_roi[1], min_roi[2]:max_roi[2]] += 1
    roi_mask_img = nib.Nifti1Image(roi_mask_vol, data_file.get_affine(), data_file.get_header())
    roi_mask_img.to_filename(roi_mask_file)

    return CropMask(roi_mask_file)


def compute_crop_mask_manual(image, roi_mask_file, data_file, min_roi, max_roi):

    image_size = image.shape
    roi_mask_vol = np.zeros(image_size)
    #min_roi = np.min(image.nonzero(), 1) - np.array(margin_crop_mask)
    #min_roi[min_roi < 0] = 0
    #max_roi = np.max(image.nonzero(), 1) + np.array(margin_crop_mask)
    # for d in range(3):
    #     max_roi[d] = min(max_roi[d], image_size[d])

    roi_mask_vol[min_roi[0]:max_roi[0], min_roi[1]:max_roi[1], min_roi[2]:max_roi[2]] += 1
    roi_mask_img = nib.Nifti1Image(roi_mask_vol, data_file.get_affine(), data_file.get_header())
    roi_mask_img.to_filename(roi_mask_file)

    return CropMask(roi_mask_file)


def identify_nifti_orientation(nifti_file):
    """

    :param nifti_file:      either a name of a nifrt file or a preloaded nifti file object
    :return:
    """
    if isinstance(nifti_file, six.string_types):
        nii_file = nib.load(nifti_file)
    else:
        nii_file = nifti_file
    nii_header = nii_file.header

    vox_size = nii_header.get_zooms()
    scale_mat = np.identity(3) * vox_size

    orientation_mat = np.dot(nii_header.get_best_affine()[0:3, 0:3], np.linalg.inv(scale_mat))

    max_idx = [0, 0, 0]
    max_idx[0] = np.where(np.fabs(orientation_mat[:, 0]) == max(np.fabs(orientation_mat[:, 0])))[0][0]
    max_idx[1] = np.where(np.fabs(orientation_mat[:, 1]) == max(np.fabs(orientation_mat[:, 1])))[0][0]
    max_idx[2] = np.where(np.fabs(orientation_mat[:, 2]) == max(np.fabs(orientation_mat[:, 2])))[0][0]

    orient_label = ['', '', '']
    for i in range(3):
        if max_idx[i] == 0:
            if orientation_mat[:, i][0] > 0:
                orient_label[i] = 'L'
            else:
                orient_label[i] = 'R'
        elif max_idx[i] == 1:
            if orientation_mat[:, i][1] > 0:
                orient_label[i] = 'P'
            else:
                orient_label[i] = 'A'
        elif max_idx[i] == 2:
            if orientation_mat[:, i][2] > 0:
                orient_label[i] = 'I'
            else:
                orient_label[i] = 'S'

    orientation = orient_label[0] + orient_label[1] + orient_label[2]

    view = ''
    if max_idx[2] == 2:
        view = 'Axial'
    elif max_idx[2] == 1:
        view = 'Coronal'
    elif max_idx[2] == 0:
        view = 'Sagittal'

    return orientation, view


def compute_side_mask(seg_vol, data_file, is_check_vol_diff=True):
    # revised in 081519
    # consider only two largest segments or a combined segment for computing a side mask
    input_labeled, num_features = label(seg_vol == 1)
    if num_features == 0:
        print('compute_side_mask: There do NOT exist any labels')
        left_mask, right_mask, seg_vol = None, None, None
    else:
        counts = Counter(input_labeled.flatten())
        del counts[0]  # remove background label
        n_vox = np.array(list(counts.values())).flatten()
        largest_vox_lst = heapq.nlargest(2, n_vox)
        seg_vol_refined = np.zeros(seg_vol.shape)

        # if two segmentations are combined and theres is an artifact, remove the artifact for only thalamus (is_check_vol_diff is True)
        if is_check_vol_diff is True:
            if len(largest_vox_lst) == 2 and float(largest_vox_lst[0]/largest_vox_lst[1]) > 2.0:
                largest_vox_lst.remove(largest_vox_lst[1])

        for largest_vox in largest_vox_lst:
            ind_largest_vox = np.where(n_vox == largest_vox)[0]
            ind_largest_vox = ind_largest_vox.tolist()
            label_ind = np.array(list(counts.keys())).flatten()
            seg_vol_refined[input_labeled == convert_arr_lst_to_elem(label_ind[ind_largest_vox[0]])] = 1

        vol_size = seg_vol_refined.shape
        seg_vol_array_ind = np.where(np.array(seg_vol_refined) == 1)
        left_mask, right_mask, mask = np.zeros(vol_size), np.zeros(vol_size), np.zeros(vol_size)
        orient, _ = identify_nifti_orientation(data_file)

        if 'L' in orient:
            idx = orient.index('L')
        elif 'R' in orient:
            idx = orient.index('R')
        else:
            print ('error: no orientation information or incorrect orientation')

        # Find a border index from slice area in sagittal axis (left <-> right)
        # if there are slices with area 0, middle index of the slices will be a border index
        # otherwise, an index where diff(slice_area) change (-) into (+) and its slice area is minimal will be a final border index
        seg_area_sag_slice_lst = []
        for slice_ind in range(min(seg_vol_array_ind[idx]), max(seg_vol_array_ind[idx]) + 1):
            if idx == 0:
                seg_area_sag_slice = len(np.where(seg_vol_refined[slice_ind, :, :] == 1)[0])
            elif idx == 1:
                seg_area_sag_slice = len(np.where(seg_vol_refined[:, slice_ind, :] == 1)[0])
            else:
                seg_area_sag_slice = len(np.where(seg_vol_refined[:, :, slice_ind] == 1)[0])
            seg_area_sag_slice_lst.append(seg_area_sag_slice)

        if len(np.where(np.array(seg_area_sag_slice_lst) == 0)[0]) == 0:
            seg_area_sag_slice_lst_diff = np.diff(np.array(seg_area_sag_slice_lst))
            seg_area_sag_slice_lst_diff_last_del = seg_area_sag_slice_lst_diff[:len(seg_area_sag_slice_lst_diff) - 1]
            seg_area_sag_slice_lst_diff_left_shift = seg_area_sag_slice_lst_diff[1:]
            border_ind_num = len(seg_area_sag_slice_lst_diff_last_del)
            border_ind_lst = []
            for diff1, diff2, border_ind in zip(seg_area_sag_slice_lst_diff_last_del,
                                                seg_area_sag_slice_lst_diff_left_shift,
                                                range(border_ind_num)):
                if diff1 < 0 and diff2 > 0:
                    border_ind_lst.append(border_ind)
            if border_ind_lst != []:
                # find index with min. seg_area_sag_slice out of border indice with diff1 <0 and diff2 >0
                min_seg_area_sag_slice = 1e+6
                border_ind_f = 1e+6
                # remove indice closer to start or end slices
                for border_ind in border_ind_lst:
                    if border_ind < len(seg_area_sag_slice_lst_diff_last_del) / 4 or \
                            border_ind > len(seg_area_sag_slice_lst_diff_last_del) * 3 / 4:
                        border_ind_lst.remove(border_ind)
                for diff_minus_plus_ind in border_ind_lst:
                    seg_area_sag_slice_at_diff_minus_plus_ind = seg_area_sag_slice_lst[diff_minus_plus_ind + 1]
                    if min_seg_area_sag_slice > seg_area_sag_slice_at_diff_minus_plus_ind:
                        min_seg_area_sag_slice = seg_area_sag_slice_at_diff_minus_plus_ind
                        border_ind_f = diff_minus_plus_ind
                seg_vol_array_ind_border = min(seg_vol_array_ind[idx]) + border_ind_f + 1
            else:
                seg_vol_size = seg_vol.shape
                seg_vol_array_ind_border = seg_vol_size[idx] / 2.0
        else:
            zero_middle_ind = floor(len(np.where(np.array(seg_area_sag_slice_lst) == 0)[0]) / 2.0)
            seg_vol_array_ind_border = min(seg_vol_array_ind[idx]) + \
                                       min(np.where(np.array(seg_area_sag_slice_lst) == 0)[0]) + zero_middle_ind

        if idx == 0:
            seg_vol[floor(seg_vol_array_ind_border), :, :] = 0
            mask[0:floor(seg_vol_array_ind_border), :, :] = 1
        elif idx == 1:
            seg_vol[:, floor(seg_vol_array_ind_border), :] = 0
            mask[:, 0:floor(seg_vol_array_ind_border), :] = 1
        else:
            seg_vol[:, :, floor(seg_vol_array_ind_border)] = 0
            mask[:, :, 0:floor(seg_vol_array_ind_border)] = 1

        if 'L' in orient:
            left_mask = mask
            right_mask[left_mask == 0] = 1
        elif 'R' in orient:
            right_mask = mask
            left_mask[right_mask == 0] = 1
        else:
            print ('error: no orientation information or incorrect orientation')

    return left_mask, right_mask, seg_vol


def check_empty_vol(volume):

    is_empty = False
    _, num_features = label(volume == 1)
    if num_features == 0:
        is_empty = True

    return is_empty


def postprocess(volume):

    vol_filled = binary_fill_holes(volume)

    # leave the largest label only
    vol_out = remove_outliers(vol_filled)


    # remove small spots (< 50 voxels) for small targest structures
    #vol_out = remove_smaller_voxels(vol_filled, min_voxels=50)

    return vol_out


def preprocess_test_image(test_img, train_img_lst, num_modality, mean, std, opt):
    if opt == 1 or opt == 3:
        if mean == [] or std == []:
            mean, std = compute_statistics(train_img_lst, num_modality)
        test_vol = standardize_volume(test_img, num_modality, mean, std)
    elif opt == 4:
        ref_num = 0
        if mean == [] or std == []:
            mean, std = compute_statistics(train_img_lst, num_modality)
        ref_training_vol = train_img_lst[ref_num]
        test_vol = normalize_image(test_img, [np.min(ref_training_vol.flatten()),
                                              np.max(ref_training_vol.flatten())])
    elif opt == 5:
        ref_num = 0
        ref_training_vol = train_img_lst[ref_num]
        print(np.shape(test_img))
        print(np.shape(ref_training_vol))
        test_img = normalize_image(test_img, [np.min(ref_training_vol.flatten()),
                                              np.max(ref_training_vol.flatten())])
        test_vol = hist_match(test_img, ref_training_vol)
        print(np.shape(test_vol))
        # save_intermediate_volume(gen_conf, train_conf, test_vol, test_patient_id, [],
        #                          'test_data_normalized_hist_matched')
    else:
        test_vol = test_img

    return test_vol


def normalize_image(img, ranges):
    img_1d_array = img.flatten()
    # # clamping top 1% bottom 1% intensity value
    # top_intensity = np.percentile(img_1d_array, 99)
    # bot_intensity = np.percentile(img_1d_array, 1)
    # img_1d_array[np.where(np.array(img_1d_array) > top_intensity)] = top_intensity
    # img_1d_array[np.where(np.array(img_1d_array) < bot_intensity)] = bot_intensity
    # # rescaling
    normalized_img_1d_array = preprocessing.minmax_scale(np.double(img_1d_array), feature_range=(ranges[0], ranges[1]))
    normalized_img = np.reshape(normalized_img_1d_array, img.shape)

    return normalized_img


def standardize_set(input_data, num_modality, mean, std):
    print("standardizing training data...")
    input_data_tmp = np.copy(input_data)
    for vol_idx in range(len(input_data_tmp)) :
        for modality in range(num_modality) :
            print(input_data_tmp[vol_idx].shape)
            input_data_tmp[vol_idx][0, modality] -= mean[modality]
            input_data_tmp[vol_idx][0, modality] /= std[modality]
    return input_data_tmp


def hist_match_set(input_data, ref_training_vol, num_modality):
    print("histogram matching training data...")
    input_data_tmp = np.copy(input_data)
    for vol_idx in range(len(input_data_tmp)):
        print(input_data_tmp[vol_idx].shape)
        for modality in range(num_modality):
            input_data_tmp[vol_idx][0, modality] = hist_match(input_data_tmp[vol_idx][0, modality], ref_training_vol)
    return input_data_tmp


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    output = interp_t_values[bin_idx].reshape(oldshape)

    # output = normalize_image(interp_t_values[bin_idx].reshape(oldshape), [np.min(template.flatten()),
    #                                                                              np.max(template.flatten())])
    #output = normalize_image(interp_t_values[bin_idx].reshape(oldshape), [0, 2**8])

    return output


def standardize_volume(input_data, num_modality, mean, std):
    print("standardizing test data...")
    print(input_data.shape)
    input_data_tmp = np.copy(input_data)
    for modality in range(num_modality):
        input_data_tmp[0, modality] -= mean[modality]
        input_data_tmp[0, modality] /= std[modality]
    return input_data_tmp


def remove_smaller_voxels(input, min_voxels=50):
    input_labeled, num_features = label(input == 1)
    counts = Counter(input_labeled.flatten())
    del counts[0]  # remove background label

    n_vox = np.array(list(counts.values())).flatten()
    n_large_vox = np.where(n_vox > min_voxels)[0]
    n_large_vox = n_large_vox.tolist()
    output = []
    if n_large_vox:
        label_ind = np.array(list(counts.keys())).flatten()
        label_vox = []
        for k in n_large_vox:
            label_vox.append(convert_arr_lst_to_elem(label_ind[k]))
        output = np.zeros(input.shape)
        for j in label_vox:
            output[input_labeled == j] = 1
    else:
        print('There are no remaining labels after removing smaller voxels than min_vox. '
              'Bypass the original input')
        output = input

    return output


def remove_outliers(input): # leave only the largest label
    input_labeled, num_features = label(input == 1)
    if num_features == 0:
        print('remove_outliers: There do NOT exist any labels')
        output = None
    else:
        counts = Counter(input_labeled.flatten())
        del counts[0]  # remove background label

        n_vox = np.array(list(counts.values())).flatten()
        ind_largest_vox = np.where(n_vox == np.max(n_vox))[0]
        ind_largest_vox = ind_largest_vox.tolist()
        output = np.zeros(input.shape)
        if ind_largest_vox:
            label_ind = np.array(list(counts.keys())).flatten()
            output[input_labeled == convert_arr_lst_to_elem(label_ind[ind_largest_vox[0]])] = 1
        else:
            print('There are NO remaining labels after removing outliers. Bypass the original input')
            output = input

    return output


def __smooth_binary_img(input_filename, output_filename, dim=3, maximumRMSError=0.01, numberOfIterations=10, numberOfLayers=3):

    PixelType = itk.F
    ImageType = itk.Image[PixelType, dim]

    ReaderType = itk.ImageFileReader[ImageType]
    reader = ReaderType.New()
    reader.SetFileName(input_filename)

    AntiAliasFilterType = itk.AntiAliasBinaryImageFilter[ImageType, ImageType]
    antialiasfilter = AntiAliasFilterType.New()
    antialiasfilter.SetInput(reader.GetOutput())
    antialiasfilter.SetMaximumRMSError(maximumRMSError)
    antialiasfilter.SetNumberOfIterations(numberOfIterations)
    antialiasfilter.SetNumberOfLayers(numberOfLayers)

    WriterType = itk.ImageFileWriter[ImageType]
    writer = WriterType.New()
    writer.SetFileName(output_filename)
    writer.SetInput(antialiasfilter.GetOutput())

    writer.Update()


def generate_structures_surface(str_file, threshold):
    """
    :param str_file:
    :param threshold:
    :return: vertices and faces of input structures
    """

    str_file = nib.load(str_file)
    str_image = str_file.get_data()
    str_image = str_image > threshold

    str_vertices, str_faces, normals, values = measure.marching_cubes_lewiner(str_image, 0)

    return str_vertices, str_faces


def apply_image_orientation_to_stl(vertices, input_file):
    orientation_scale_origin_mat = input_file.header.get_best_affine()

    # note that image space (x:1, y:1, z:1) -> nifti space (x:0, y:0, z:0)
    vertices_tr = transformPoint3d(vertices[0] - 1, vertices[1] - 1, vertices[2] - 1,
                                   np.matrix(orientation_scale_origin_mat))

    return vertices_tr


def write_stl(file_path, vertices, faces, mode='bin', bin_header=None):
    """
    A function to write stl files in ascii or binary format, updating normals with vertices and faces
    (created from stlwrite.m and trimesh.io.stl.export_stl)

    :param file_path:   path and filename.stl to be saved
    :param vertices:    triangle points
    :param faces:       faces
    :param mode:        'ascii' or 'bin'. Note that the newer version of VTK (>6.0) cannot read ascii stl

    :param bin_header:  optional data to be used as header data if mode is 'bin'.
                        when not provided then a default header will be created.
                        up to 80 bytes will be used.
    :return:
    """

    if bin_header is not None and mode == 'bin':
        if bin_header.lower()[0:5] == 'solid':
            raise ValueError("header for binary STL can't start with 'solid'")

        bin_header = bin_header.encode()

    facet = np.single(vertices)
    facet = facet[faces, :]

    v1 = facet[:, 1, :] - facet[:, 0, :]
    v2 = facet[:, 2, :] - facet[:, 0, :]
    normals = np.multiply(v1[:, [1, 2, 0]], v2[:, [2, 0, 1]]) - np.multiply(v2[:, [1, 2, 0]], v1[:, [2, 0, 1]])
    denom = np.divide(1, np.sqrt(np.sum(normals ** 2, 1)))
    normals = [normals[i] * denom[i] for i in range(len(denom))]

    open_modes = {'ascii': 'w'}
    with open(file_path, open_modes.get(mode, 'wb')) as fid:
        # move all the data thats going into the STL file into one array
        if mode == 'ascii':
            blob = np.zeros((len(faces), 4, 3))
            blob[:, 0, :] = normals
            blob[:, 1:, :] = facet

            # create a lengthy format string for the data section of the file
            format_string = 'facet normal {} {} {}\nouter loop\n'
            format_string += 'vertex {} {} {}\n' * 3
            format_string += 'endloop\nendfacet\n'
            format_string *= len(faces)

            # concatenate the header, data, and footer
            export = 'solid ascii\n'
            export += format_string.format(*blob.reshape(-1))
            export += 'endsolid'

        else:
            # define a numpy datatype for the header of a binary STL file
            _stl_dtype_header = np.dtype([('header', np.void, 80),
                                          ('face_count', np.int32)])

            # define a numpy datatype for the data section of a binary STL file
            _stl_dtype = np.dtype([('normals', np.float32, (3)),
                                   ('vertices', np.float32, (3, 3)),
                                   ('attributes', np.uint16)])

            header = np.zeros(1, dtype=_stl_dtype_header)
            header['face_count'] = len(faces)
            if bin_header is not None:
                header['header'] = bin_header

            packed = np.zeros(len(faces), dtype=_stl_dtype)
            packed['normals'] = normals
            packed['vertices'] = facet

            export = header.tobytes()
            export += packed.tobytes()

        fid.write(export)


def __smooth_stl(input_stl_filename, output_stl_filename):

    # 1) Decimate the surface to 99% of its original size and interpolate 10 iterations
    # 2) Convert the surface into a vtk image object
    # 3) Erode and dilate the binary image of the STN
    # 4) Extract the surface from the manipulated image of the STN
    # 5) Smooth (reduction of 20%, 80 iterations) the extracted surface and save it to a file
    # 6) Measure and return the distance from original surface to the smoothed one

    reader = vtk.vtkSTLReader()
    reader.SetFileName(input_stl_filename)

    # 1) Decimate the surface to 99% of its original size and interpolate 120 iterations

    target_surface_reduction = 0.01
    number_of_iteration = 10
    smooth_output = decimate_and_smooth(target_surface_reduction, number_of_iteration, reader.GetOutputPort())

    # 2) Convert the surface into a vtk image object
    inval = 255  # value of inside shape color
    outval = 0  # value of outside shape color
    imgstenc, spacing = convert_vtkPolyData_to_vtkImageData(smooth_output, inval, outval)

    # 3) Erode and dilate the binary image of the STN

    kernel_size_mm = 1  # mm
    kernel_size_voxels = int(kernel_size_mm/spacing[0])
    erode_dilate_image = erode_and_dilate(imgstenc, inval, outval, kernel_size_voxels)

    # 4) Extract the surface from the manipulated image of the STN
    contour = vtk.vtkDiscreteMarchingCubes()
    if vtk.VTK_MAJOR_VERSION <= 5:
        contour.SetInput(erode_dilate_image.GetOutput())
    else:
        contour.SetInputConnection(erode_dilate_image.GetOutputPort())

    contour.ComputeNormalsOn()
    contour.SetValue(0, inval)
    contour.Update()

    # 5.1) Smooth (reduction of 20%, 80 iterations) the extracted surface and save it to a file
    target_surface_reduction = 0.2
    number_of_iteration = 80
    smooth_output = decimate_and_smooth(target_surface_reduction, number_of_iteration, contour.GetOutputPort())

    # 5.2) Write the stl file to disk
    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetFileName(output_stl_filename)
    stl_writer.SetInputConnection(smooth_output.GetOutputPort())
    stl_writer.Write()

    # 6) measure and report surface distance
    print('measuring surface distance between smooth and original surfaces to ensure that it didn\'t significantly '
          'changed the original one...')
    mean_surface_dist = measure_spd_stl(input_stl_filename, output_stl_filename)
    print('Done')
    if mean_surface_dist > 0.3:
        print('Surface distance between smooth and original surfaces larger than 0.3mm %s, '
                'consider using the original surface' % str(mean_surface_dist))

    return mean_surface_dist

#
# def hist_match2(source, template):
#
#     oldshape = source.shape
#     nbr_bins=255
#
#     # imhist, bins = np.histogram(source.flatten(), nbr_bins, normed=True)
#     # tinthist, bins = np.histogram(template.flatten(), nbr_bins, normed=True)
#
#     imhist, im_bins = np.histogram(source.flatten(), nbr_bins)
#     tinthist, tint_bins = np.histogram(template.flatten(), nbr_bins)
#
#     cdfsrc = imhist.cumsum().astype(np.float64) #cumulative distribution function
#     cdfsrc /= cdfsrc[-1]
#     #cdfsrc = (255 * cdfsrc / cdfsrc[-1]).astype(np.uint8) #normalize
#
#     cdftint = tinthist.cumsum().astype(np.float64) #cumulative distribution function
#     cdftint /= cdftint[-1]
#     #cdftint = (255 * cdftint / cdftint[-1]).astype(np.uint8) #normalize
#
#     interp_t_values = np.interp(cdfsrc, cdftint, tint_bins[:-1])
#     output = interp_t_values[bin_idx].reshape(oldshape)
#
#     # im2 = np.interp(source.flatten(), tint_bins[:-1], cdfsrc)
#     # im3 = np.interp(im2, cdftint, tint_bins[:-1])
#     #
#     # output = im3.reshape(oldshape)
#
#     return output
#
#
# def hist_match3(source, template):
#
#     oldshape = source.shape
#     nbr_bins=255
#
#     # imhist, bins = np.histogram(source.flatten(), nbr_bins, normed=True)
#     # tinthist, bins = np.histogram(template.flatten(), nbr_bins, normed=True)
#
#     imhist, im_bins = np.histogram(source.flatten(), nbr_bins)
#     tinthist, tint_bins = np.histogram(template.flatten(), nbr_bins)
#
#     cdfsrc = imhist.cumsum().astype(np.float64) #cumulative distribution function
#     cdfsrc /= cdfsrc[-1]
#     #cdfsrc = (255 * cdfsrc / cdfsrc[-1]).astype(np.uint8) #normalize
#
#     cdftint = tinthist.cumsum().astype(np.float64) #cumulative distribution function
#     cdftint /= cdftint[-1]
#     #cdftint = (255 * cdftint / cdftint[-1]).astype(np.uint8) #normalize
#
#     matchfunc = match(cdfsrc, cdftint)
#
#     for i in range(oldshape[0]):
#         for j in range(oldshape[1]):
#             for k in range(oldshape[2]):
#                 val = source[i, j, k]
#                 source[i, j, k] = matchfunc[int(val)]
#
#     # im2 = np.interp(source.flatten(), tint_bins[:-1], cdfsrc)
#     # im3 = np.interp(im2, cdftint, tint_bins[:-1])
#     #
#     output = source
#
#     return output
#
#
# def match(ref, adj):
#     # -- find histogram matching function
#     i = 0
#     j = 0
#     matchtable = np.zeros(256)
#     for i in range(0, 255):
#         for j in range(0, 255):
#             if ref[i] <= adj[j]:
#                 matchtable[i] = j
#                 break
#     matchtable[255] = 255
#     return matchtable