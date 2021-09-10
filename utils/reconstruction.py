import itertools
import numpy as np

from .general_utils import pad_both_sides


def reconstruct_volume(gen_conf, test_conf, patches) :
    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    dimension = test_conf['dimension']
    expected_size = dataset_info['size']
    extraction_step = test_conf['extraction_step']
    num_classes = gen_conf['num_classes']
    output_shape = test_conf['output_shape']
    time_series = dataset_info['time_series']

    print(expected_size)
    if dimension == 2 :
        output_shape = (0, ) + output_shape if dimension == 2 else output_shape
        extraction_step = (1, ) + extraction_step if dimension == 2 else extraction_step

    if time_series is False:
        rec_volume = perform_voting(
            dimension, patches, output_shape, expected_size, extraction_step, num_classes)
    else:
        rec_volume = perform_voting_4d(
            dimension, patches, output_shape, expected_size, extraction_step, num_classes)

    return rec_volume


def reconstruct_volume_modified(gen_conf, train_conf, test_conf, patches, num_classes):
    dataset = test_conf['dataset']
    dimension = test_conf['dimension']
    extraction_step = test_conf['extraction_step']
    output_shape = test_conf['output_shape']
    dataset_info = gen_conf['dataset_info'][dataset]
    expected_size = dataset_info['size']
    #num_classes = gen_conf['num_classes']
    activation = train_conf['activation']
    multi_output = gen_conf['multi_output']

    print(expected_size)
    if dimension == 2 :
        output_shape = (0, ) + output_shape if dimension == 2 else output_shape
        extraction_step = (1, ) + extraction_step if dimension == 2 else extraction_step

    if multi_output == 1:
        rec_volume, prob_volume = [], []
        for patch, n_class, activ in zip(patches, num_classes, activation):
            rec_volume_i, prob_volume_i = perform_voting_modified(
                dimension, patch, output_shape, expected_size, extraction_step, n_class, activ)
            rec_volume.append(rec_volume_i)
            prob_volume.append(prob_volume_i)
    else:
        rec_volume, prob_volume = perform_voting_modified(
            dimension, patches, output_shape, expected_size, extraction_step, num_classes, activation)

    return rec_volume, prob_volume


def perform_voting(dimension, patches, output_shape, expected_shape, extraction_step, num_classes) :
    vote_img = np.zeros(expected_shape + (num_classes, ))
    coordinates = generate_indexes(dimension, output_shape, extraction_step, expected_shape)

    if dimension == 2 :
        output_shape = (1, ) + output_shape[1:]

    for count, coord in enumerate(coordinates):
        selection = [slice(coord[i] - output_shape[i], coord[i]) for i in range(len(coord))]
        selection += [slice(None)]
        vote_img[selection] += patches[count]

    rec_volume = np.argmax(vote_img, axis=3)
    #return np.argmax(vote_img[:, :, :, 1:], axis=3) + 1
    return rec_volume


def perform_voting_modified(dimension, patches, output_shape, expected_shape, extraction_step, num_classes, activation) :
    vote_img = np.zeros(expected_shape + (num_classes, ))
    coordinates = generate_indexes(dimension, output_shape, extraction_step, expected_shape)

    if dimension == 2 :
        output_shape = (1, ) + output_shape[1:]

    for count, coord in enumerate(coordinates):
        selection = [slice(coord[i] - output_shape[i], coord[i]) for i in range(len(coord))]
        selection += [slice(None)]
        vote_img[tuple(selection)] += patches[count]

    if activation == 'softmax':
        rec_volume = np.argmax(vote_img, axis=3)
    elif activation == 'sigmoid':
        rec_volume = (vote_img > 0.5).astype(np.uint8)
    else:
        raise NotImplementedError('choose either softmax or sigmoid')

    return rec_volume, vote_img


def perform_voting_4d(dimension, patches, output_shape, expected_shape, extraction_step, num_classes):
    #vote_img = np.zeros(expected_shape + (num_classes, ))
    vote_img = np.zeros((len(patches),) + expected_shape + (num_classes,))
    rec_volume = np.zeros((len(patches),) + expected_shape + (num_classes,))

    coordinates = generate_indexes(dimension, output_shape, extraction_step, expected_shape)

    for t in range(len(patches)):
        if dimension == 2 :
            output_shape = (1, ) + output_shape[1:]

        for count, coord in enumerate(coordinates):
            selection = [slice(coord[i] - output_shape[i], coord[i]) for i in range(len(coord))]
            selection += [slice(None)]
            vote_img[t][selection] += patches[t][count]

        rec_volume[t] = np.argmax(vote_img[t], axis=3)
        #return np.argmax(vote_img[:, :, :, 1:], axis=3) + 1
    return rec_volume


def generate_indexes(dimension, output_shape, extraction_step, expected_shape) :
    ndims = len(output_shape)

    poss_shape = [output_shape[i] + extraction_step[i] * ((expected_shape[i] - output_shape[i]) // extraction_step[i]) for i in range(ndims)]

    if dimension == 2 :
        output_shape = (1, ) + output_shape[1:]

    idxs = [range(output_shape[i], poss_shape[i] + 1, extraction_step[i]) for i in range(ndims)]
    
    return itertools.product(*idxs)