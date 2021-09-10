import numpy as np

from keras.utils import np_utils
from operator import itemgetter
from .extraction import extract_patches
from .general_utils import pad_both_sides

def split_train_val(train_indexes, validation_split) :
    N = len(train_indexes)
    val_volumes = np.int32(np.ceil(N * validation_split))
    train_volumes = N - val_volumes

    return train_indexes[:train_volumes], train_indexes[train_volumes:]


def build_training_set(gen_conf, train_conf, input_data, labels):
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    bg_discard_percentage = train_conf['bg_discard_percentage']
    dimension = train_conf['dimension']
    extraction_step = train_conf['extraction_step']
    output_shape = train_conf['output_shape']
    patch_shape = train_conf['patch_shape']
    modality = dataset_info['image_modality']
    num_modality = len(modality)
    target = dataset_info['target']
    if type(target) == list:
        num_targets = len(target) + 1  # include bg
    else:
        num_targets = 2

    #label_selector = determine_label_selector(patch_shape, output_shape)
    minimum_non_bg = bg_discard_percentage * np.prod(output_shape)

    data_patch_shape = (num_modality, ) + patch_shape
    data_extraction_step = (num_modality, ) + extraction_step
    output_patch_shape = (np.prod(output_shape), num_targets)

    x = np.zeros((0, ) + data_patch_shape)
    y = np.zeros((0, ) + output_patch_shape)
    for idx in range(len(input_data)):
        y_length = len(y)

        # pad_size = ()
        # for dim in range(dimension) :
        #     pad_size += (patch_shape[dim] // 2, )

        data_pad_size = ()
        for dim in range(dimension) :
            data_pad_size += (patch_shape[dim] // 2, )

        label_pad_size = ()
        for dim in range(dimension) :
            label_pad_size += (output_shape[dim] // 2, )

        print(labels[idx][0,0].shape)
        print(input_data[idx][0].shape)

        label_vol = pad_both_sides(dimension, labels[idx][0,0], label_pad_size)
        input_vol = pad_both_sides(dimension, input_data[idx][0], data_pad_size)

        label_patches = extract_patches(dimension, label_vol, output_shape, extraction_step)
        #label_patches = extract_patches(dimension, label_vol, patch_shape, extraction_step)
        #label_patches = label_patches[tuple(label_selector)]

        sum_axis = (1, 2, 3) if dimension == 3 else (1, 2)

        valid_idxs = np.where(np.sum(label_patches != 0, axis=sum_axis) >= minimum_non_bg)
        label_patches = label_patches[valid_idxs]

        N = len(label_patches)

        x = np.vstack((x, np.zeros((N, ) + data_patch_shape)))
        y = np.vstack((y, np.zeros((N, ) + output_patch_shape)))

        # one-hot encoding (if sparse_categorical_crossentropy is used, don't do this and just leave integer target)
        for i in range(N) :
            tmp = np_utils.to_categorical(label_patches[i].flatten(), num_targets)
            y[i + y_length] = tmp

        del label_patches

        data_train = extract_patches(dimension, input_vol, data_patch_shape, data_extraction_step)
        x[y_length:] = data_train[valid_idxs]

        del data_train

    # debug
    # w = np.zeros((num_classes,))
    # print(y.shape)
    # for index in range(num_classes):
    #     y_true_i_class = np.ndarray.flatten(y[:,:,index])
    #     print(y_true_i_class)
    #     w[index] = np.sum(np.asarray(y_true_i_class == 1, np.int8))
    # print(w)

    return x, y


def build_training_set_4d(gen_conf, train_conf, input_data, labels):
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    bg_discard_percentage = train_conf['bg_discard_percentage']
    dimension = train_conf['dimension']
    extraction_step = train_conf['extraction_step']
    num_classes = gen_conf['num_classes']
    output_shape = train_conf['output_shape']
    patch_shape = train_conf['patch_shape']

    modality = dataset_info['image_modality']
    num_modality = len(modality)

    label_selector = determine_label_selector(patch_shape, output_shape)
    minimum_non_bg = bg_discard_percentage * np.prod(output_shape)

    data_patch_shape = (num_modality,) + patch_shape
    data_extraction_step = (num_modality,) + extraction_step
    # output_patch_shape = (np.prod(output_shape), num_classes)
    output_patch_shape = (num_modality,) + (np.prod(output_shape), num_classes)

    x = np.zeros((0,) + data_patch_shape)
    y = np.zeros((0,) + output_patch_shape)
    for idx in range(len(input_data)):
        y_length = len(y)

        pad_size = ()
        for dim in range(dimension):
            pad_size += (patch_shape[dim] // 2,)

        # print(labels[idx][0,0].shape)
        print(labels[idx][0].shape)
        print(input_data[idx][0].shape)

        # label_vol = pad_both_sides(dimension, labels[idx][0,0], pad_size)
        label_vol = pad_both_sides(dimension, labels[idx][0], pad_size)
        input_vol = pad_both_sides(dimension, input_data[idx][0], pad_size)

        # label_patches = extract_patches(dimension, label_vol, patch_shape, extraction_step)
        label_patches = extract_patches(dimension, label_vol, data_patch_shape, data_extraction_step)
        label_patches = label_patches[label_selector]

        label_patches = np.transpose(label_patches, [1, 0, 2, 3, 4])

        sum_axis = (1, 2, 3) if dimension == 3 else (1, 2)

        # select useful patches based on labels
        valid_idxs_list = []
        n_patches = []
        for t in range(num_modality):
            valid_idxs = np.where(np.sum(label_patches[t] != 0, axis=sum_axis) >= minimum_non_bg)
            valid_idxs_list.append(valid_idxs[0])
            n_patches.append(len(valid_idxs[0]))

        min_n_patches = min(np.array(n_patches))
        min_valid_idxs_list = []
        label_patches_list = []
        for t in range(num_modality):
            min_valid_idxs = np.random.choice(valid_idxs_list[t], min_n_patches, False)
            min_valid_idxs_list.append(min_valid_idxs)
            label_patches_list.append([label_patches[t][i] for i in min_valid_idxs])

        N = min_n_patches
        x = np.vstack((x, np.zeros((N,) + data_patch_shape)))
        y = np.vstack((y, np.zeros((N,) + output_patch_shape)))

        data_train = extract_patches(dimension, input_vol, data_patch_shape, data_extraction_step)
        data_train = np.transpose(data_train, [1, 0, 2, 3, 4])

        for t in range(num_modality):
            for i in range(N):
                tmp = np_utils.to_categorical(label_patches_list[t][i].flatten(), num_classes)
                y[i + y_length][t] = tmp
                x[y_length + i][t] = data_train[t][min_valid_idxs_list[t][i]]

        print(x.shape)
        print(y.shape)

        del label_patches
        del data_train

    return x, y


def build_testing_set(gen_conf, test_conf, input_data) :
    dataset = test_conf['dataset']
    dimension = test_conf['dimension']
    extraction_step = test_conf['extraction_step']
    patch_shape = test_conf['patch_shape']
    dataset_info = gen_conf['dataset_info'][dataset]
    modality = dataset_info['image_modality']
    num_modality = len(modality)

    data_patch_shape = (num_modality, ) + patch_shape
    data_extraction_step = (num_modality, ) + extraction_step

    return extract_patches(dimension, input_data[0], data_patch_shape, data_extraction_step)


def determine_label_selector(patch_shape, output_shape) :
    ndim = len(patch_shape)
    patch_shape_equal_output_shape = patch_shape == output_shape

    slice_none = slice(None)
    if not patch_shape_equal_output_shape :
        return [slice_none] + [slice(output_shape[i], patch_shape[i] - output_shape[i]) for i in range(ndim)]
    else :
        return [slice_none for i in range(ndim)]