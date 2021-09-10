
import numpy as np
from scipy  import ndimage
from skimage import measure
from stl import mesh

def computeDice(autoSeg, groundTruth, label_mapper):
    """ Returns
    -------
    DiceArray : floats array

          Dice coefficient as a float on range [0,1].
          Maximum similarity = 1
          No similarity = 0 """

    #n_classes = int(np.max(groundTruth) + 1)

    DiceArray = []
    for key in label_mapper.keys():
        idx_Auto = np.where(autoSeg.flatten() == label_mapper[key])[0]
        idx_GT = np.where(groundTruth.flatten() == label_mapper[key])[0]

        autoArray = np.zeros(autoSeg.size, dtype=np.bool)
        autoArray[idx_Auto] = 1

        gtArray = np.zeros(autoSeg.size, dtype=np.bool)
        gtArray[idx_GT] = 1

        dsc = dice(autoArray, gtArray)

        # dice = np.sum(autoSeg[groundTruth==c_i])*2.0 / (np.sum(autoSeg) + np.sum(groundTruth))
        DiceArray.append(dsc)

    return DiceArray


def dice(im1, im2):
    """
    Computes the Dice coefficient
    ----------
    im1 : boolean array
    im2 : boolean array

    If they are not boolean, they will be converted.

    -------
    It returns the Dice coefficient as a float on the range [0,1].
        1: Perfect overlapping
        0: Not overlapping
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.size != im2.size:
        raise ValueError("Size mismatch between input arrays!!!")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return 1.0

    # Compute Dice
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def measure_cmd(struct1, struct2, pixdim):
    """
    :param struct1: ndarray of 3D image 1 (the result of nib.get_data())
    :param struct2: ndarray of 3D image 2 (the result of nib.get_data())
    :param pixdim: voxel spacing
    :return: average (Euclidan) distance between center of mass of the structures (in mm)
    """
    voxel_size = [pixdim[1], pixdim[0], pixdim[2]]

    struct1_cm = np.array(ndimage.measurements.center_of_mass(struct1))
    struct2_cm = np.array(ndimage.measurements.center_of_mass(struct2))

    cm_diff = (struct1_cm - struct2_cm) * voxel_size
    cm_dist = np.linalg.norm(cm_diff)
    return cm_dist, cm_diff


def measure_msd(struct1, struct2, pixdim):
    """
    :param struct1: ndarray of 3D image 1 (the result of nib.get_data())
    :param struct2: ndarray of 3D image 2 (the result of nib.get_data())
    :param pixdim:
    :return: averaged Euclidan distance between the structures surface points (in mm)
    """
    voxel_size = [pixdim[1], pixdim[0], pixdim[2]]

    verts1, faces1, normals1, values1 = measure.marching_cubes_lewiner(struct1, 0.5)
    verts2, faces2, normals2, values2 = measure.marching_cubes_lewiner(struct2, 0.5)
    min_s_dist_array = np.zeros(verts1.shape[0])
    for ind, surface_point in enumerate(verts1):
        min_s_dist_array[ind] = min(np.linalg.norm(((verts2 - surface_point) * voxel_size), axis=1))

    mean_surface_dist = np.mean(min_s_dist_array)
    return mean_surface_dist


def compute_statistics(input_data, num_modality):
    print("computing mean and std of training data...")
    mean = np.zeros((num_modality, ))
    std = np.zeros((num_modality, ))
    num_input = len(input_data)
    for modality in range(num_modality):
        modality_data = []
        for i in range(num_input):
            modality_data += np.array(input_data[i][0, modality]).flatten().tolist()
        mean[modality] = np.mean(np.array(modality_data))
        std[modality] = np.std(np.array(modality_data))

    print ('mean: ', mean, 'std: ', std)
    return mean, std


def convert_arr_lst_to_elem(arr_lst):
    """
    This function extracts an element from array or combination of array and list
    :param arr_lst: array or combination of array and list
    :return: element
    """

    while type(arr_lst) is list:
        arr_lst = arr_lst[0]

    while type(arr_lst) is np.ndarray:
        arr_lst = arr_lst.tolist()
        while type(arr_lst) is list:
            arr_lst = arr_lst[0]

    return arr_lst


def transformPoint3d(x_array, y_array, z_array, trans):
    if np.array(x_array).size > 1:
        res = np.dot(np.transpose(np.array([x_array, y_array, z_array, np.ones(np.array(x_array).size)])),
                     np.transpose(trans))
    else:
        res = np.dot(np.array([x_array, y_array, z_array, 1]), np.transpose(trans))

    res = np.array(np.transpose(res))

    trans_array = [res[0], res[1], res[2]]

    return trans_array