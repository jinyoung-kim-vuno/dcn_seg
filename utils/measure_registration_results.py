import copy
import csv
import itertools
import os

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io
from scipy import ndimage
from skimage import measure
from stl import mesh

from sispipeline.utils.dirservice import *
from sispipeline.utils.logservice import *
from sistools import spc


def _load(path):
    if path.endswith('.mat'):
        mat_data = scipy.io.loadmat(path)
        result = mat_data['temp'].swapaxes(0, 1)
    else:
        result = nib.load(path).get_data()

    return result


def measure_CM_dist(struct1, struct2):
    """
    :param struct1: ndarray of 3D image 1 (the result of nib.get_data())
    :param struct2: ndarray of 3D image 2 (the result of nib.get_data())
    :return: Euclidan distance between center of mass of the structures (in voxels)
    """
    struct1_cm = np.array(ndimage.measurements.center_of_mass(struct1))
    struct2_cm = np.array(ndimage.measurements.center_of_mass(struct2))
    cm_diff = struct1_cm - struct2_cm
    cm_dist = np.linalg.norm(cm_diff)
    return cm_dist, cm_diff


def measure_CM_dist_real_size(struct1, struct2, pixdim):
    """
    :param struct1: ndarray of 3D image 1 (the result of nib.get_data())
    :param struct2: ndarray of 3D image 2 (the result of nib.get_data())
    :param pixdim:
    :return: Euclidan distance between center of mass of the structures (in mm)
    """
    voxel_size = [pixdim[2], pixdim[1], pixdim[3]]

    struct1_cm = np.array(ndimage.measurements.center_of_mass(struct1))
    struct2_cm = np.array(ndimage.measurements.center_of_mass(struct2))

    cm_diff = (struct1_cm - struct2_cm) * voxel_size
    cm_dist = np.linalg.norm(cm_diff)
    return cm_dist, cm_diff


def measure_surface_dist(struct1, struct2):
    """
    :param struct1: ndarray of 3D image 1 (the result of nib.get_data())
    :param struct2: ndarray of 3D image 2 (the result of nib.get_data())
    :return: Averaged Euclidan distance between the structures surface points (in voxels)
    """
    verts1, faces1, normals1, values1 = measure.marching_cubes(struct1, 0.5)
    verts2, faces2, normals2, values2 = measure.marching_cubes(struct2, 0.5)
    min_s_dist_array = np.zeros(verts1.shape[0])
    for ind, surface_point in enumerate(verts1):
        min_s_dist_array[ind] = min(np.linalg.norm((verts2 - surface_point), axis=1))

    mean_surface_dist = np.mean(min_s_dist_array)
    return mean_surface_dist


def measure_surface_dist_stl(struct1, struct2):
    """
    :param struct1: stl filename
    :param struct2: stl filename
    :return: Averaged Euclidan distance between the structures surface points (in voxels)
    """
    mesh1 = mesh.Mesh.from_file(struct1)
    mesh2 = mesh.Mesh.from_file(struct2)
    verts1 = np.concatenate((mesh1.v0, mesh1.v1, mesh1.v2), axis=0)
    verts2 = np.concatenate((mesh2.v0, mesh2.v1, mesh2.v2), axis=0)
    min_s_dist_array = np.zeros(verts1.shape[0])
    for ind, surface_point in enumerate(verts1):
        min_s_dist_array[ind] = min(np.linalg.norm((verts2 - surface_point), axis=1))

    mean_surface_dist = np.mean(min_s_dist_array)
    return mean_surface_dist


def measure_surface_dist_real_size(struct1, struct2, pixdim):
    """
    :param struct1: ndarray of 3D image 1 (the result of nib.get_data())
    :param struct2: ndarray of 3D image 2 (the result of nib.get_data())
    :param pixdim:
    :return: Averaged Euclidan distance between the structures surface points (in mm)
    """
    voxel_size = [pixdim[2], pixdim[1], pixdim[3]]

    verts1, faces1, normals1, values1 = measure.marching_cubes(struct1, 0.5)
    verts2, faces2, normals2, values2 = measure.marching_cubes(struct2, 0.5)
    min_s_dist_array = np.zeros(verts1.shape[0])
    for ind, surface_point in enumerate(verts1):
        min_s_dist_array[ind] = min(np.linalg.norm(((verts2 - surface_point) * voxel_size), axis=1))

    mean_surface_dist = np.mean(min_s_dist_array)
    return mean_surface_dist


def measure_dc(struct1, struct2):
    a = struct1.astype(bool)
    b = struct2.astype(bool)
    # noinspection PyTypeChecker
    dice = float(2.0 * np.sum(a * b)) / (np.sum(a) + np.sum(b))
    # dice_dist = np.sum(a[b==1])*2.0 / (np.sum(a) + np.sum(b))

    return dice


def measure_cdc(struct1, struct2):

    size_of_A_intersect_B = np.sum(struct1 * struct2)
    size_of_A = np.sum(struct1)
    size_of_B = np.sum(struct2)
    if size_of_A_intersect_B > 0:
        c = size_of_A_intersect_B/np.sum(struct1 * np.sign(struct2))
    else:
        c = 1.0
    cdc = (2.0*size_of_A_intersect_B) / (c*size_of_A + size_of_B)

    return cdc


def measure_prior_reg_results(path_to_prior_reg_lst, path_to_most_similar_lst, threshold=0):
    """
    :param path_to_prior_reg_lst: List of paths to the priors registered to the clinical patient space (RN, SN, STN)
    :param path_to_most_similar_lst: List of the structures of most similar brain
    :param threshold:
    :return: Array of the CM distance and mean surface distance of the prior from most similar
    """
    reg_results_cm = np.zeros(len(path_to_prior_reg_lst))
    reg_results_surface = np.zeros(len(path_to_prior_reg_lst))
    cm_diff_result = np.zeros([len(path_to_prior_reg_lst), 3])
    reg_results_dice = np.zeros(len(path_to_prior_reg_lst))
    for ind, prior_path in enumerate(path_to_prior_reg_lst):
        gt_path = path_to_most_similar_lst[ind]
        gt_data = _load(gt_path)
        prior_data = _load(prior_path)

        if threshold > 0:
            gt_data[gt_data < threshold] = 0
            prior_data[prior_data < threshold] = 0

        reg_results_cm[ind], cm_diff_result[ind, :] = measure_CM_dist(prior_data, gt_data)
        reg_results_surface[ind] = measure_surface_dist(prior_data, gt_data)
        reg_results_dice[ind] = measure_dc(prior_data, gt_data)

    return reg_results_cm, cm_diff_result, reg_results_surface, reg_results_dice


def measure_prior_reg_results_real_size(path_to_prior_reg_lst, path_to_most_similar_lst, threshold=0):
    """
    :param path_to_prior_reg_lst: List of paths to the priors registered to the clinical patient space (RN, SN, STN)
    :param path_to_most_similar_lst: List of the structures of most similar brain
    :param threshold:
    :return: Array of the CM distance and mean surface distance of the prior from most similar
    """
    reg_results_cm = np.zeros(len(path_to_prior_reg_lst))
    reg_results_surface = np.zeros(len(path_to_prior_reg_lst))
    cm_diff_result = np.zeros([len(path_to_prior_reg_lst), 3])
    reg_results_dice = np.zeros(len(path_to_prior_reg_lst))
    for ind, prior_path in enumerate(path_to_prior_reg_lst):
        gt_path = path_to_most_similar_lst[ind]
        gt_data = _load(gt_path)
        prior_data = _load(prior_path)

        if threshold > 0:
            gt_data[gt_data < threshold] = 0
            prior_data[prior_data < threshold] = 0
        # pdb.set_trace()
        # if FlowConfig().correct_bias:
        #     prior_data = np.roll(prior_data, -1, axis=2)

        # pixel size from the header information
        gt_img = nib.load(gt_path)
        pixdim = gt_img.header['pixdim']

        reg_results_cm[ind], cm_diff_result[ind, :] = measure_CM_dist_real_size(prior_data, gt_data, pixdim)
        reg_results_surface[ind] = measure_surface_dist_real_size(prior_data, gt_data, pixdim)
        reg_results_dice[ind] = measure_dc(prior_data, gt_data)

    return reg_results_cm, cm_diff_result, reg_results_surface, reg_results_dice


def check_reg_results(context, most_similar_prior, priors_list):
    """

    :param context: 
    :param most_similar_prior:
    :param priors_list: similar priors list
    :return: (1) Data frame of center of mass distance for each prior
             (2) Data frame of average surface points distance for each prior

    """

    patient_reg_results_cm = []
    patient_reg_results_surface = []
    patient_reg_results_dice = []
    path_to_ground_truth_lst = sorted(context.list_files(DSN.PRIOR_REGTO_BET_3D(prior=most_similar_prior,
                                                                                modality=T2_MODALITY,
                                                                                side=ALL,
                                                                                bg_structure=UNMERGED,
                                                                                thr05=True,
                                                                                cropped=True)))

    priors_name_lst = priors_list

    for prior in priors_name_lst:
        path_to_prior_reg_lst = sorted(context.list_files(DSN.PRIOR_REGTO_BET_3D(prior=prior,
                                                                                 modality=T2_MODALITY,
                                                                                 side=ALL,
                                                                                 bg_structure=UNMERGED,
                                                                                 thr05=True,
                                                                                 cropped=True)))
        reg_results_cm, reg_results_cm_diff, reg_results_surface, reg_results_dice = \
            measure_prior_reg_results(path_to_prior_reg_lst, path_to_ground_truth_lst)
        patient_reg_results_cm.append(reg_results_cm)
        patient_reg_results_surface.append(reg_results_surface)
        patient_reg_results_dice.append(reg_results_dice)

    structure_lst = ['RN_L', 'RN_R', 'SN_L', 'SN_R', 'STN_L', 'STN_R']
    patient_reg_results_cm_df = pd.DataFrame(patient_reg_results_cm, index=priors_name_lst, columns=structure_lst)
    patient_reg_results_surface_df = pd.DataFrame(patient_reg_results_surface, index=priors_name_lst,
                                                  columns=structure_lst)
    patient_reg_results_dice_df = pd.DataFrame(patient_reg_results_dice, index=priors_name_lst, columns=structure_lst)

    return patient_reg_results_cm_df, patient_reg_results_surface_df, patient_reg_results_dice_df


def check_priors_cm_dist(context, cur_similarity_list, resolution=0.5, prior_centering='mean', cache=None):

    # A client can provide a cache across multiple calls to avoid recomputing values for the same priors.
    # The client should not modify the contents of the cache nor depend on its content. If no cache was
    # provided, create an empty one to make the logic simpler below. However, the contents of the cache
    # will then not be available to subsequent calls.
    if cache is None:
        cache = {}

    structure_lst = list(itertools.product(BGStructure.unmerged_choices(), Side.choices))
    structure_name_lst = ['{}_{}'.format(bg_structure.name, side.name[:1]) for bg_structure, side in structure_lst]
    structure_count = len(structure_lst)

    good_priors_name_lst = []
    compute_priors_name_lst = []
    compute_priors_files_lst = []
    for prior_name in cur_similarity_list:
        if prior_name in cache:
            sis_log(INFO, SLM.PREDICTION, None, "Cached center of mass measurements available for %s", prior_name)
            good_priors_name_lst.append(prior_name)
            continue

        priors_files = []
        for bg_structure, side in structure_lst:
            pf = context.get_files_path(DSN.PRIOR_REGTO_BET_3D(prior_name, modality=T2_MODALITY, side=side,
                                                               bg_structure=bg_structure, thr05=True, cropped=False))
            if not os.path.exists(pf):
                sis_log(WARNING, SLM.PREDICTION, SLC.WORKFILES_NOT_FOUND, "Missing prior file: %s", pf)
            else:
                priors_files.append(pf)

        if len(priors_files) < structure_count:
            sis_log(WARNING, SLM.PREDICTION, SLC.WORKFILES_NOT_FOUND,
                    "Skipping center of mass measurements for %s due to missing files", prior_name)

        sis_log(INFO, SLM.PREDICTION, None, "Computing center of mass measurements for %s", prior_name)
        good_priors_name_lst.append(prior_name)
        compute_priors_name_lst.append(prior_name)
        compute_priors_files_lst.append(priors_files)

    for compute_prior_name, compute_result in zip(compute_priors_name_lst,
                                                  compute_priors_cm_dist(compute_priors_files_lst)):
        cache[compute_prior_name] = compute_result

    priors_cm_list = np.array([cache[prior] for prior in good_priors_name_lst])

    str_cm_avg = compute_structure_centers(priors_cm_list, prior_centering)
    priors_cm_diff = priors_cm_list - str_cm_avg

    priors_cm_dist = np.zeros([len(good_priors_name_lst), structure_count])
    for i in range(len(good_priors_name_lst)):
        for j in range(structure_count):
            priors_cm_dist[i][j] = np.linalg.norm(priors_cm_diff[i][j]) * resolution

    priors_cm_dist_df = pd.DataFrame(priors_cm_dist, index=good_priors_name_lst, columns=structure_name_lst)

    return priors_cm_dist_df


def center16(a, axis):
    if axis != 0:
        raise ValueError('Expected axis=0')

    mean_a = np.mean(a, axis=0)
    while len(a) > 16:
        norm_a = np.linalg.norm(a - mean_a, axis=1)
        worst_i = np.argmax(norm_a)

        # Rather than recalculating the mean from scratch, we *could* just shift the
        # old mean by 1/N of the point just removed. I think.
        #    mean_a = mean_a - a[worst_i] / len(a)
        #    a = np.delete(a, worst_i, axis=0)

        a = np.delete(a, worst_i, axis=0)
        mean_a = np.mean(a, axis=0)

    return mean_a


def compute_structure_centers(priors_cm_list, prior_centering='mean'):
    """

    :param priors_cm_list: N priors * M structures * D dimensions (typically 16, 6 and 3)
    :type priors_cm_list: numpy.array
    :param prior_centering: method to compute prior centers ('mean' or 'median')
    :return: structure center coordinates - M structures * D dimensions
    """
    n_priors, n_structures, n_dim = priors_cm_list.shape

    result = np.zeros(priors_cm_list[0].shape)
    method = {'mean': np.mean, 'median': np.median, 'center16': center16}[prior_centering]

    for i in range(n_structures):
        result[i] = method(priors_cm_list[:, i, :], axis=0)

    return result


def compute_priors_cm_dist(priors_files_lst):
    """

    :param priors_files_lst: a list of file lists
    :return: a list of center-of-mass lists
    """
    return spc.call('compute_priors_cm_dist.py', priors_files_lst)


def get_most_similar_prior(context):
    similarity_file = context.get_files_path(DSN.PREDICTION_SIMILARITY_SUMMARY)
    with open(similarity_file, 'r') as sim_file:
        reader = csv.reader(sim_file)
        similar_brain_lst = list(reader)
    most_similar_brain = similar_brain_lst[0][0]
    return most_similar_brain


def compute_priors_cm_metrics(priors_cm_dist_df):
    priors_cm_distr = priors_cm_dist_df.mean(axis=1) + 2 * priors_cm_dist_df.std(axis=1)
    return {key: value for key, value in zip(priors_cm_distr.index, priors_cm_distr)}


# noinspection PyUnresolvedReferences
def detect_failed_registrations(patient_reg_results_cm_df, patient_reg_results_surface_df):

    """
    Detect if a a prior have registration error of more than 3mm compare to the most simila brain

    :param patient_reg_results_cm_df: Data frame that contain the priors center of mass distances
    :param patient_reg_results_surface_df: Data frame that contain the priors average surface distances
    :return: list of brains that faild the registration

    """
    RESOLUTION = 0.5  # resolution of the image (for calculating real distance)
    DISTANCE_THRESHOLD = 3  # distance in mm that above it the registration is considered failed

    priors_above_threshold_cm = (RESOLUTION * patient_reg_results_cm_df.mean(axis=1)) > DISTANCE_THRESHOLD
    priors_above_threshold_surf = (RESOLUTION * patient_reg_results_surface_df.mean(axis=1)) > DISTANCE_THRESHOLD
    failed_reg_priors_cm = priors_above_threshold_cm[priors_above_threshold_cm == True].index.tolist()
    failed_reg_priors_surf = priors_above_threshold_surf[priors_above_threshold_surf == True].index.tolist()
    failed_reg_priors = failed_reg_priors_cm + list(set(failed_reg_priors_surf) - set(failed_reg_priors_cm))

    return failed_reg_priors


def check_patient_reg(context, most_similar_prior, priors_list):

    patient_reg_results_cm_df, patient_reg_results_surface_df, patient_reg_results_dice_df = \
        check_reg_results(context, most_similar_prior, priors_list)

    # list of failed registered brain
    failed_reg_priors_name, failed_reg_priors_dist_values = \
        detect_failed_registrations(patient_reg_results_cm_df, patient_reg_results_surface_df)

    return failed_reg_priors_name, failed_reg_priors_dist_values, patient_reg_results_cm_df, \
        patient_reg_results_surface_df
