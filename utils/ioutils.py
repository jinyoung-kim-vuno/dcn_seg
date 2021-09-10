
import nibabel as nib
import numpy as np
import os
import h5py
from utils.BrainImage import BrainImage, _remove_ending
import subprocess
from utils.image import find_crop_mask, compute_crop_mask, compute_crop_mask_manual, crop_image, \
    compute_side_mask, postprocess, normalize_image, generate_structures_surface, apply_image_orientation_to_stl, \
    write_stl, __smooth_stl, __smooth_binary_img, check_empty_vol
from utils.callbacks import generate_output_filename


def read_tha_dataset(gen_conf, train_conf, train_patient_lst, test_patient_lst, preprocess_trn, preprocess_tst,
                         file_output_dir):
    dataset = train_conf['dataset']
    num_epochs = train_conf['num_epochs']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    mode = gen_conf['validation_mode']
    img_pattern = dataset_info['image_name_pattern']
    label_pattern = dataset_info['label_pattern']
    modality = dataset_info['image_modality']
    num_modality = len(modality)
    init_mask_pattern = dataset_info['initial_mask_pattern']

    set_new_roi_mask = dataset_info['set_new_roi_mask']
    margin_crop_mask = dataset_info['margin_crop_mask']

    crop_trn_img_pattern = dataset_info['crop_trn_image_name_pattern']
    crop_tst_img_pattern = dataset_info['crop_tst_image_name_pattern']
    crop_trn_label_pattern = dataset_info['crop_trn_label_pattern']
    crop_tst_label_pattern = dataset_info['crop_tst_label_pattern']

    crop_init_mask_pattern = dataset_info['crop_initial_mask_pattern']

    train_roi_mask_pattern = dataset_info['train_roi_mask_pattern']
    test_roi_mask_pattern = dataset_info['test_roi_mask_pattern']

    modality_idx = []
    for m in modality: # the order of modality images should be fixed due to the absolute path
        if m == 'T1':
            modality_idx.append(0)
        if m == 'B0':
            modality_idx.append(1)
        if m == 'FA':
            modality_idx.append(2)

    # load multi-modal images in train_patient_lst and setting roi
    train_img_lst = []
    label_lst = []
    train_fname_lst = []
    label_fname_lst = []
    if train_patient_lst[0] is not None and num_epochs != 0 and mode != '2':
        for train_patient_id in train_patient_lst:

            file_train_patient_dir = os.path.join(file_output_dir, train_patient_id)
            if not os.path.exists(file_train_patient_dir):
                os.makedirs(os.path.join(file_output_dir, train_patient_id))

            train_roi_mask_file = os.path.join(file_train_patient_dir, train_roi_mask_pattern)
            if os.path.exists(train_roi_mask_file) and set_new_roi_mask is True:
                os.remove(train_roi_mask_file)

            # load label images in train_patient_lst and setting roi
            label_vol_lst = []
            label_fname_side_lst = []
            for side in ['left', 'right']:  # left + right
                for ver in ['2','3']:
                    label_filepath = os.path.join(dataset_path, train_patient_id, 'gt', label_pattern.format(side, ver))
                    if os.path.exists(label_filepath):
                        break
                    else:
                        if ver == '3':
                            print('No Found %s ground truth label' % side)
                            exit()
                print(label_filepath)
                label_fname_side_lst.append(label_filepath)
                label_data = read_volume_data(label_filepath)
                label_vol = label_data.get_data()
                if np.size(np.shape(label_vol)) == 4:
                    label_vol_lst.append(label_vol[:, :, :, 0])
                else:
                    label_vol_lst.append(label_vol)
            label_fname_lst.append(label_fname_side_lst)
            label_merge_vol = label_vol_lst[0] + label_vol_lst[1]

            # crop the roi of the image
            train_crop_mask = find_crop_mask(train_roi_mask_file) or compute_crop_mask(label_merge_vol,
                                                                                       train_roi_mask_file,
                                                                                       label_data, margin_crop_mask)
            label_crop_vol = train_crop_mask.crop(label_merge_vol)

            # save the cropped labels (train_roi)
            for side in ['left', 'right']:
                for ver in ['2', '3']:
                    label_filepath = os.path.join(dataset_path, train_patient_id, 'gt', label_pattern.format(side, ver))
                    if os.path.exists(label_filepath):
                        break
                    else:
                        if ver == '3':
                            print('No Found %s ground truth label' % side)
                            exit()
                cropped_trn_label_file = os.path.join(file_output_dir, train_patient_id,
                                                      crop_trn_label_pattern.format(side, ver))
                if (not os.path.exists(cropped_trn_label_file)) or set_new_roi_mask is True:
                    crop_image(label_filepath, train_crop_mask, cropped_trn_label_file)

            label_crop_array = np.zeros((1, 1) + label_crop_vol.shape[0:3])
            label_crop_array[0, 0] = label_crop_vol
            label_lst.append(label_crop_array)

            # load training images in train_patient_lst and setting roi
            train_image_modal_lst= []
            train_fname_modality_lst = []
            for idx in modality_idx: #T1, B0, or FA
                training_img_path = os.path.join(dataset_path, train_patient_id, 'images', img_pattern[idx])
                print(training_img_path)
                train_fname_modality_lst.append(training_img_path)
                training_vol = read_volume(training_img_path)
                if preprocess_trn == 2 or preprocess_trn == 3 or preprocess_trn == 5:
                    training_vol = normalize_image(training_vol, [0, 2 ** 8])
                if np.size(np.shape(training_vol)) == 4:
                    train_crop_vol = train_crop_mask.crop(training_vol[:, :, :, 0])
                else:
                    train_crop_vol = train_crop_mask.crop(training_vol)
                train_image_modal_lst.append(train_crop_vol)
            train_fname_lst.append(train_fname_modality_lst)

            # save the cropped image
            for idx in modality_idx:
                train_img_path = os.path.join(dataset_path, train_patient_id, 'images', img_pattern[idx])
                crop_train_img_path = os.path.join(file_output_dir, train_patient_id, crop_trn_img_pattern[idx])
                if (not os.path.exists(crop_train_img_path)) or set_new_roi_mask is True:
                    crop_image(train_img_path, train_crop_mask, crop_train_img_path)
            train_crop_array = np.zeros((1, num_modality) + train_crop_vol.shape[0:3])
            for idx in range(num_modality):
                train_crop_array[0, idx] = train_image_modal_lst[idx]
            train_img_lst.append(train_crop_array)

    # load multi-modal images in test_patient_lst and setting roi
    test_img_lst = []
    test_fname_lst = []
    if test_patient_lst[0] is not None:
        for test_patient_id in test_patient_lst:

            file_test_patient_dir = os.path.join(file_output_dir, test_patient_id)
            if not os.path.exists(file_test_patient_dir):
                os.makedirs(os.path.join(file_output_dir, test_patient_id))

            test_roi_mask_file = os.path.join(file_test_patient_dir, test_roi_mask_pattern)
            if os.path.exists(test_roi_mask_file) and set_new_roi_mask is True:
                os.remove(test_roi_mask_file)

            # load initial mask (use fusion output or later linearly registered from ref. training image) for setting roi
            init_mask_vol_lst = []
            for side in ['left', 'right']:  # left + right
                init_mask_filepath = os.path.join(dataset_path, test_patient_id, 'fusion', init_mask_pattern.format(side))
                print(init_mask_filepath)
                init_mask_data = read_volume_data(init_mask_filepath)
                init_mask_vol = init_mask_data.get_data()
                if np.size(np.shape(init_mask_vol)) == 4:
                    init_mask_vol_lst.append(init_mask_vol[:, :, :, 0])
                else:
                    init_mask_vol_lst.append(init_mask_vol)
            init_mask_merge_vol = init_mask_vol_lst[0] + init_mask_vol_lst[1]

            # crop the roi of the image
            test_crop_mask = find_crop_mask(test_roi_mask_file) or compute_crop_mask(init_mask_merge_vol,
                                                                                     test_roi_mask_file,
                                                                                     init_mask_data,
                                                                                     margin_crop_mask)

            # save the cropped initial masks
            for side in ['left', 'right']:
                init_mask_filepath = os.path.join(dataset_path, test_patient_id, 'fusion',
                                                  init_mask_pattern.format(side))
                cropped_init_mask_file = os.path.join(file_test_patient_dir,
                                                      crop_init_mask_pattern.format(side))
                if (not os.path.exists(cropped_init_mask_file)) or set_new_roi_mask is True:
                    crop_image(init_mask_filepath, test_crop_mask, cropped_init_mask_file)

            # save the cropped labels (test_roi)
            for side in ['left', 'right']:
                for ver in ['2', '3']:
                    label_filepath = os.path.join(dataset_path, test_patient_id, 'gt', label_pattern.format(side, ver))
                    if os.path.exists(label_filepath):
                        break
                    else:
                        if ver == '3':
                            print ('No Found %s ground truth label' % side)
                            ver = 'unknown'
                cropped_tst_label_file = os.path.join(file_test_patient_dir,
                                                      crop_tst_label_pattern.format(side, ver))
                if (not os.path.exists(cropped_tst_label_file)) or set_new_roi_mask is True:
                    crop_image(label_filepath, test_crop_mask, cropped_tst_label_file)

            # load test images in test_patient_lst and setting roi
            test_image_modal_lst = []
            test_fname_modality_lst = []
            for idx in modality_idx:
                test_img_path = os.path.join(dataset_path, test_patient_id, 'images', img_pattern[idx])
                print(test_img_path)
                test_fname_modality_lst.append(test_img_path)
                test_vol = read_volume(test_img_path)
                if preprocess_tst == 2 or preprocess_tst == 3 or preprocess_tst == 5:
                    test_vol = normalize_image(test_vol, [0, 2 ** 8])
                if np.size(np.shape(test_vol)) == 4:
                    test_crop_vol = test_crop_mask.crop(test_vol[:, :, :, 0])
                else:
                    test_crop_vol = test_crop_mask.crop(test_vol)
                test_image_modal_lst.append(test_crop_vol)
            test_fname_lst.append(test_fname_modality_lst)

            # save the cropped image
            for idx in modality_idx:
                test_img_path = os.path.join(dataset_path, test_patient_id, 'images', img_pattern[idx])
                crop_test_img_path = os.path.join(file_test_patient_dir, crop_tst_img_pattern[idx])
                if (not os.path.exists(crop_test_img_path)) or set_new_roi_mask is True:
                    crop_image(test_img_path, test_crop_mask, crop_test_img_path)
            test_crop_array = np.zeros((1, num_modality) + test_crop_vol.shape[0:3])
            for idx in range(num_modality):
                test_crop_array[0, idx] = test_image_modal_lst[idx]
            test_img_lst.append(test_crop_array)

    return train_img_lst, label_lst, test_img_lst, train_fname_lst, label_fname_lst, test_fname_lst


def read_tha_dataset_unseen(gen_conf, train_conf, test_conf, train_patient_lst, test_patient_lst, preprocess_trn,
                            preprocess_tst, file_output_dir, is_scaling):
    root_path = gen_conf['root_path']
    dataset = train_conf['dataset']
    num_epochs = train_conf['num_epochs']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    path = dataset_info['path']
    mode = gen_conf['validation_mode']
    img_pattern = dataset_info['image_name_pattern']
    img_resampled_name_pattern = dataset_info['image_resampled_name_pattern']

    label_pattern = dataset_info['label_pattern']
    modality = dataset_info['image_modality']
    num_modality = len(modality)
    init_mask_pattern = dataset_info['initial_mask_pattern']
    init_reg_mask_pattern = dataset_info['initial_reg_mask_pattern']

    set_new_roi_mask = dataset_info['set_new_roi_mask']
    margin_crop_mask = dataset_info['margin_crop_mask']

    crop_trn_img_pattern = dataset_info['crop_trn_image_name_pattern']
    crop_tst_img_pattern = dataset_info['crop_tst_image_name_pattern']
    crop_tst_resampled_img_pattern = dataset_info['crop_tst_image_resampled_name_pattern']

    crop_trn_label_pattern = dataset_info['crop_trn_label_pattern']
    crop_tst_label_pattern = dataset_info['crop_tst_label_pattern']

    crop_init_mask_pattern = dataset_info['crop_initial_mask_pattern']
    crop_init_reg_mask_pattern = dataset_info['crop_initial_reg_mask_pattern']
    train_roi_mask_pattern = dataset_info['train_roi_mask_pattern']
    test_roi_mask_pattern = dataset_info['test_roi_mask_pattern']
    is_reg_flag = test_conf['is_reg']
    if is_reg_flag == 1:
        is_reg = True
    else:
        is_reg = False
    roi_pos_tuple = test_conf['roi_pos']
    roi_start, roi_end = [], []
    for i, j in zip(roi_pos_tuple, range(len(roi_pos_tuple))):
        if j % 2 == 0:
            roi_start.append(i)
        else:
            roi_end.append(i)
        roi_pos = [roi_start, roi_end]

    modality_idx = []
    for m in modality: # the order of modality images should be fixed due to the absolute path
        if m == 'T1':
            modality_idx.append(0)
        if m == 'B0':
            modality_idx.append(1)
        if m == 'FA':
            modality_idx.append(2)

    is_res_diff = False # initial setting
    # load multi-modal images in train_patient_lst and setting roi
    train_img_lst = []
    label_lst = []
    train_fname_lst = []
    label_fname_lst = []
    if train_patient_lst[0] is not None and num_epochs != 0 and mode != '2':
        for train_patient_id in train_patient_lst:

            file_train_patient_dir = os.path.join(file_output_dir, train_patient_id)
            if not os.path.exists(file_train_patient_dir):
                os.makedirs(file_train_patient_dir)

            tha_trn_output_dir = os.path.join(file_train_patient_dir, path)
            if not os.path.exists(tha_trn_output_dir):
                os.makedirs(tha_trn_output_dir)

            train_roi_mask_file = os.path.join(tha_trn_output_dir, train_roi_mask_pattern)
            if os.path.exists(train_roi_mask_file) and set_new_roi_mask is True:
                os.remove(train_roi_mask_file)

            # load label images in train_patient_lst and setting roi
            label_vol_lst = []
            label_fname_side_lst = []
            for side in ['left', 'right']:  # left + right
                for ver in ['2','3']:
                    label_filepath = os.path.join(dataset_path, train_patient_id, 'gt', label_pattern.format(side, ver))
                    if os.path.exists(label_filepath):
                        break
                    else:
                        if ver == '3':
                            print('No Found %s ground truth label' % side)
                            exit()
                print(label_filepath)
                label_fname_side_lst.append(label_filepath)
                label_data = read_volume_data(label_filepath)
                label_vol = label_data.get_data()
                if np.size(np.shape(label_vol)) == 4:
                    label_vol_lst.append(label_vol[:, :, :, 0])
                else:
                    label_vol_lst.append(label_vol)
            label_fname_lst.append(label_fname_side_lst)
            label_merge_vol = label_vol_lst[0] + label_vol_lst[1]

            # crop the roi of the image
            train_crop_mask = find_crop_mask(train_roi_mask_file) or compute_crop_mask(label_merge_vol,
                                                                                       train_roi_mask_file,
                                                                                       label_data, margin_crop_mask)
            label_crop_vol = train_crop_mask.crop(label_merge_vol)

            # save the cropped labels (train_roi)
            for side in ['left', 'right']:
                for ver in ['2', '3']:
                    label_filepath = os.path.join(dataset_path, train_patient_id, 'gt', label_pattern.format(side, ver))
                    if os.path.exists(label_filepath):
                        break
                    else:
                        if ver == '3':
                            print('No Found %s ground truth label' % side)
                            exit()
                cropped_trn_label_file = os.path.join(tha_trn_output_dir,
                                                      crop_trn_label_pattern.format(side, ver))
                if (not os.path.exists(cropped_trn_label_file)) or set_new_roi_mask is True:
                    crop_image(label_filepath, train_crop_mask, cropped_trn_label_file)

            label_crop_array = np.zeros((1, 1) + label_crop_vol.shape[0:3])
            label_crop_array[0, 0] = label_crop_vol
            label_lst.append(label_crop_array)

            # load training images in train_patient_lst and setting roi
            train_image_modal_lst= []
            train_fname_modality_lst = []
            for idx in modality_idx: #T1, B0, or FA
                training_img_path = os.path.join(dataset_path, train_patient_id, 'images', img_pattern[idx])
                print(training_img_path)
                train_fname_modality_lst.append(training_img_path)
                training_vol = read_volume(training_img_path)
                if preprocess_trn == 2 or preprocess_trn == 3 or preprocess_trn == 5:
                    training_vol = normalize_image(training_vol, [0, 2 ** 8])
                if np.size(np.shape(training_vol)) == 4:
                    train_crop_vol = train_crop_mask.crop(training_vol[:, :, :, 0])
                else:
                    train_crop_vol = train_crop_mask.crop(training_vol)
                train_image_modal_lst.append(train_crop_vol)
            train_fname_lst.append(train_fname_modality_lst)

            # save the cropped image
            for idx in modality_idx:
                train_img_path = os.path.join(dataset_path, train_patient_id, 'images', img_pattern[idx])
                crop_train_img_path = os.path.join(tha_trn_output_dir, crop_trn_img_pattern[idx])
                if (not os.path.exists(crop_train_img_path)) or set_new_roi_mask is True:
                    crop_image(train_img_path, train_crop_mask, crop_train_img_path)
            train_crop_array = np.zeros((1, num_modality) + train_crop_vol.shape[0:3])
            for idx in range(num_modality):
                train_crop_array[0, idx] = train_image_modal_lst[idx]
            train_img_lst.append(train_crop_array)

    # load multi-modal images in test_patient_lst and setting roi
    test_img_lst = []
    test_fname_lst = []
    is_res_diff_lst = []
    if test_patient_lst[0] is not None:
        for test_patient_id in test_patient_lst:
            is_res_diff = False

            file_test_patient_dir = os.path.join(file_output_dir, test_patient_id)
            if not os.path.exists(file_test_patient_dir):
                os.makedirs(file_test_patient_dir)

            tha_tst_output_dir = os.path.join(file_test_patient_dir, path)
            if not os.path.exists(tha_tst_output_dir):
                os.makedirs(tha_tst_output_dir)

            test_roi_mask_file = os.path.join(tha_tst_output_dir, test_roi_mask_pattern)
            if os.path.exists(test_roi_mask_file) and set_new_roi_mask is True:
                os.remove(test_roi_mask_file)

            test_fname_modality_lst_for_init_reg = []
            for idx in modality_idx:
                test_img_path = os.path.join(root_path, 'datasets', 'tha', test_patient_id, 'images', img_pattern[idx])
                test_fname_modality_lst_for_init_reg.append(test_img_path)
            test_t1_img_path = test_fname_modality_lst_for_init_reg[0]

            # load initial mask (use fusion output or later linearly registered from ref. training image) for setting roi
            init_mask_path = os.path.join(dataset_path, test_patient_id, 'fusion')
            if os.path.exists(init_mask_path):
                init_mask_vol_lst = []
                for side in ['left', 'right']:  # left + right
                    init_mask_filepath = os.path.join(dataset_path, test_patient_id, 'fusion',
                                                      init_mask_pattern.format(side))
                    print(init_mask_filepath)
                    init_mask_data = read_volume_data(init_mask_filepath)
                    init_mask_vol = init_mask_data.get_data()
                    if np.size(np.shape(init_mask_vol)) == 4:
                        init_mask_vol_lst.append(init_mask_vol[:, :, :, 0])
                    else:
                        init_mask_vol_lst.append(init_mask_vol)
                init_mask_merge_vol = init_mask_vol_lst[0] + init_mask_vol_lst[1]

            else: # localize target via initial registration
                label_fname_lst_for_init_reg = []
                train_fname_lst_for_init_reg = []
                for train_patient_id in train_patient_lst:
                    train_fname_modality_lst_for_init_reg = []
                    for idx in modality_idx:  # T1, B0, or FA
                        training_img_path = os.path.join(dataset_path, train_patient_id, 'images', img_pattern[idx])
                        train_fname_modality_lst_for_init_reg.append(training_img_path)
                    train_fname_lst_for_init_reg.append(train_fname_modality_lst_for_init_reg)

                    label_fname_side_lst_for_init_reg = []
                    for side in ['left', 'right']:  # left + right
                        for ver in ['2', '3']:
                            label_filepath = os.path.join(dataset_path, train_patient_id, 'gt',
                                                          label_pattern.format(side, ver))
                            if os.path.exists(label_filepath):
                                break
                            else:
                                if ver == '3':
                                    print('No Found %s ground truth label' % side)
                                    exit()
                        label_fname_side_lst_for_init_reg.append(label_filepath)
                    label_fname_lst_for_init_reg.append(label_fname_side_lst_for_init_reg)
                init_mask_merge_vol, init_mask_data, test_t1_img_path, is_res_diff, roi_pos = \
                    localize_target(file_output_dir,
                                    modality_idx,
                                    train_patient_lst,
                                    train_fname_lst_for_init_reg,
                                    test_patient_id,
                                    test_fname_modality_lst_for_init_reg,
                                    label_fname_lst_for_init_reg,
                                    img_pattern[0],
                                    img_resampled_name_pattern,
                                    init_reg_mask_pattern,
                                    'tha',
                                    is_res_diff,
                                    roi_pos,
                                    tha_tst_output_dir,
                                    is_scaling,
                                    is_reg)

            if init_mask_merge_vol.tolist():
                # crop the roi of the image
                test_crop_mask = find_crop_mask(test_roi_mask_file) or compute_crop_mask(init_mask_merge_vol,
                                                                                         test_roi_mask_file,
                                                                                         init_mask_data,
                                                                                        margin_crop_mask)
            else:
                test_t1_img_data = read_volume_data(test_t1_img_path)
                test_t1_image = BrainImage(test_t1_img_path, None)
                test_vol = test_t1_image.nii_data_normalized(bits=8)
                test_crop_mask = compute_crop_mask_manual(test_vol, test_roi_mask_file, test_t1_img_data,
                                                          roi_pos[0], roi_pos[1]) # PD091 for tha seg

            # save the cropped initial masks
            for side in ['left', 'right']:
                init_mask_path = os.path.join(dataset_path, test_patient_id, 'fusion')
                if os.path.exists(init_mask_path):
                    init_mask_filepath = os.path.join(dataset_path, test_patient_id, 'fusion',
                                                      init_mask_pattern.format(side))
                    if os.path.exists(init_mask_filepath):
                        cropped_init_mask_file = os.path.join(tha_tst_output_dir,
                                                              crop_init_mask_pattern.format(side))
                        if (not os.path.exists(cropped_init_mask_file)) or set_new_roi_mask is True:
                            crop_image(init_mask_filepath, test_crop_mask, cropped_init_mask_file)
                else:
                    init_mask_filepath = os.path.join(tha_tst_output_dir, 'init_reg',
                                                          init_reg_mask_pattern.format(side, 'tha'))
                    if os.path.exists(init_mask_filepath):
                        cropped_init_reg_mask_file = os.path.join(tha_tst_output_dir,
                                                              crop_init_reg_mask_pattern.format(side))
                        if (not os.path.exists(cropped_init_reg_mask_file)) or set_new_roi_mask is True:
                            crop_image(init_mask_filepath, test_crop_mask, cropped_init_reg_mask_file)

            # save the cropped labels (test_roi)
            for side in ['left', 'right']:
                for ver in ['2', '3']:
                    label_filepath = os.path.join(dataset_path, test_patient_id, 'gt', label_pattern.format(side, ver))
                    if os.path.exists(label_filepath) and os.path.exists(init_mask_filepath):
                        cropped_tst_label_file = os.path.join(tha_tst_output_dir,
                                                              crop_tst_label_pattern.format(side, ver))
                        if (not os.path.exists(cropped_tst_label_file)) or set_new_roi_mask is True:
                            crop_image(label_filepath, test_crop_mask, cropped_tst_label_file)
                        break
                    else:
                        if ver == '3':
                            print ('No found %s ground truth label' % side)

            # load test images in test_patient_lst and setting roi
            test_image_modal_lst = []
            test_fname_modality_lst = []
            for idx in modality_idx:
                if is_res_diff is True:
                    test_img_path = os.path.join(tha_tst_output_dir, img_resampled_name_pattern[idx])
                else:
                    test_img_path = os.path.join(root_path, 'datasets', 'tha', test_patient_id, 'images', img_pattern[idx])
                print(test_img_path)
                test_fname_modality_lst.append(test_img_path)
                test_vol = read_volume(test_img_path)
                if preprocess_tst == 2 or preprocess_tst == 3 or preprocess_tst == 5:
                    test_vol = normalize_image(test_vol, [0, 2 ** 8])
                if np.size(np.shape(test_vol)) == 4:
                    test_crop_vol = test_crop_mask.crop(test_vol[:, :, :, 0])
                else:
                    test_crop_vol = test_crop_mask.crop(test_vol)
                test_image_modal_lst.append(test_crop_vol)
            test_fname_lst.append(test_fname_modality_lst)

            # save the cropped image
            for idx in modality_idx:
                if is_res_diff is True:
                    test_img_path = os.path.join(tha_tst_output_dir, img_resampled_name_pattern[idx])
                    crop_test_img_path = os.path.join(tha_tst_output_dir, crop_tst_resampled_img_pattern[idx])
                else:
                    test_img_path = os.path.join(root_path, 'datasets', 'tha', test_patient_id, 'images', img_pattern[idx])
                    crop_test_img_path = os.path.join(tha_tst_output_dir, crop_tst_img_pattern[idx])
                if (not os.path.exists(crop_test_img_path)) or set_new_roi_mask is True:
                    crop_image(test_img_path, test_crop_mask, crop_test_img_path)
            test_crop_array = np.zeros((1, num_modality) + test_crop_vol.shape[0:3])
            for idx in range(num_modality):
                test_crop_array[0, idx] = test_image_modal_lst[idx]
            test_img_lst.append(test_crop_array)

            is_res_diff_lst.append(is_res_diff)

    return train_img_lst, label_lst, test_img_lst, train_fname_lst, label_fname_lst, test_fname_lst, is_res_diff_lst


def read_dentate_interposed_dataset(gen_conf, train_conf, train_patient_lst, test_patient_lst, preprocess_trn,
                                    preprocess_tst, file_output_dir, target):
    dataset = train_conf['dataset']
    num_epochs = train_conf['num_epochs']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    mode = gen_conf['validation_mode']
    img_pattern = dataset_info['image_name_pattern']
    dentate_label_pattern = dataset_info['manual_corrected_dentate_v2_pattern']
    interposed_label_pattern = dataset_info['manual_corrected_interposed_v2_pattern']

    modality = dataset_info['image_modality']
    num_modality = len(modality)
    init_dentate_mask_pattern = dataset_info['initial_mask_pattern']
    init_interposed_mask_pattern_thres = dataset_info['initial_interposed_mask_pattern_thres']
    init_interposed_mask_pattern_mask = dataset_info['initial_interposed_mask_pattern_mask']

    set_new_roi_mask = dataset_info['set_new_roi_mask']
    margin_crop_mask = dataset_info['margin_crop_mask']

    crop_trn_img_pattern = dataset_info['crop_trn_image_name_pattern']
    crop_tst_img_pattern = dataset_info['crop_tst_image_name_pattern']

    crop_trn_dentate_label_pattern = dataset_info['crop_trn_manual_dentate_v2_corrected_pattern']
    crop_tst_dentate_label_pattern = dataset_info['crop_tst_manual_dentate_v2_corrected_pattern']
    crop_trn_interposed_label_pattern = dataset_info['crop_trn_manual_interposed_v2_pattern']
    crop_tst_interposed_label_pattern = dataset_info['crop_tst_manual_interposed_v2_pattern']

    crop_init_dentate_mask_pattern = dataset_info['crop_initial_mask_pattern']
    crop_init_interposed_mask_pattern = dataset_info['crop_initial_interposed_mask_pattern']

    train_roi_mask_pattern = dataset_info['train_roi_mask_pattern']
    test_roi_mask_pattern = dataset_info['test_roi_mask_pattern']

    patient_id = dataset_info['patient_id']

    is_new_trn_label = train_conf['is_new_trn_label']
    if is_new_trn_label in [1, 2, 3]:  # 1: segmentation using a proposed network, 2: suit labels
        new_label_path = train_conf['new_label_path']
        dentate_new_label_pattern = dataset_info['trn_new_label_dentate_pattern']
        interposed_new_label_pattern = dataset_info['trn_new_label_interposed_pattern']
        crop_trn_dentate_new_label_pattern = dataset_info['crop_trn_new_label_dentate_pattern']
        crop_trn_interposed_new_label_pattern = dataset_info['crop_trn_new_label_interposed_pattern']

    modality_idx = []
    for m in modality:
        if m == 'B0':
            modality_idx.append(0)
        if m == 'T1':
            modality_idx.append(1)
        if m == 'LP':
            modality_idx.append(2)
        if m == 'FA':
            modality_idx.append(3)

    #label_mapper = {0: 0, 10: 1, 150: 2}

    if len(target) == 2:
        target = 'both'

    # load multi-modal images in train_patient_lst and setting roi
    train_img_lst = []
    label_lst = []
    train_fname_lst = []
    dentate_label_fname_lst = []
    interposed_label_fname_lst = []
    if train_patient_lst[0] is not None and num_epochs != 0 and mode != '2':
        for train_patient_id in train_patient_lst:

            file_trn_patient_dir = os.path.join(file_output_dir, train_patient_id)
            if not os.path.exists(file_trn_patient_dir):
                os.makedirs(file_trn_patient_dir)

            train_roi_mask_file = os.path.join(file_output_dir, train_roi_mask_pattern.format(train_patient_id))
            if os.path.exists(train_roi_mask_file) and set_new_roi_mask is True:
                os.remove(train_roi_mask_file)

            # load label images in train_patient_lst and setting roi
            dentate_label_vol_lst = []
            dentate_label_fname_side_lst = []
            interposed_label_vol_lst = []
            interposed_label_fname_side_lst = []
            for side in ['left', 'right']:  # left + right

                if target == 'dentate':
                    if is_new_trn_label == 1:
                        dentate_label_filepath = os.path.join(new_label_path, train_patient_id,
                                                              dentate_new_label_pattern.format(side))
                    elif is_new_trn_label == 2:
                        dentate_label_filepath = os.path.join(new_label_path, train_patient_id,
                                                              dentate_new_label_pattern.format(train_patient_id, side))
                    elif is_new_trn_label == 3:
                        if train_patient_id in patient_id:
                            dentate_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                                  dentate_label_pattern.format(train_patient_id, side))
                        else:
                            dentate_label_filepath = os.path.join(new_label_path, train_patient_id,
                                                                  dentate_new_label_pattern.format(side))
                    else:
                        dentate_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                              dentate_label_pattern.format(train_patient_id, side))
                    print(dentate_label_filepath)
                    dentate_label_fname_side_lst.append(dentate_label_filepath)
                    dentate_label_data = read_volume_data(dentate_label_filepath)
                    dentate_label_vol = dentate_label_data.get_data()
                    if np.size(np.shape(dentate_label_vol)) == 4:
                        dentate_label_vol_lst.append(dentate_label_vol[:, :, :, 0])
                    else:
                        dentate_label_vol_lst.append(dentate_label_vol)

                elif target == 'interposed':
                    if is_new_trn_label == 1:
                        interposed_label_filepath = os.path.join(new_label_path, train_patient_id,
                                                                 interposed_new_label_pattern.format(side))
                    elif is_new_trn_label == 2:
                        interposed_label_filepath = os.path.join(new_label_path, train_patient_id,
                                                                 interposed_new_label_pattern.format(train_patient_id,
                                                                                                     side))
                    elif is_new_trn_label == 3:
                        if train_patient_id in patient_id:
                            interposed_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                                     interposed_label_pattern.format(train_patient_id,
                                                                                                     side))
                        else:
                            interposed_label_filepath = os.path.join(new_label_path, train_patient_id,
                                                                     interposed_new_label_pattern.format(side))
                    else:
                        interposed_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                                 interposed_label_pattern.format(train_patient_id,
                                                                                                 side))
                    print(interposed_label_filepath)
                    interposed_label_fname_side_lst.append(interposed_label_filepath)
                    interposed_label_data = read_volume_data(interposed_label_filepath)
                    interposed_label_vol = interposed_label_data.get_data()
                    if np.size(np.shape(interposed_label_vol)) == 4:
                        interposed_label_vol_lst.append(interposed_label_vol[:, :, :, 0])
                    else:
                        interposed_label_vol_lst.append(interposed_label_vol)

                else:
                    if is_new_trn_label == 1:
                        dentate_label_filepath = os.path.join(new_label_path, train_patient_id,
                                                              dentate_new_label_pattern.format(side))
                        interposed_label_filepath = os.path.join(new_label_path, train_patient_id,
                                                                 interposed_new_label_pattern.format(side))
                    elif is_new_trn_label == 2:
                        dentate_label_filepath = os.path.join(new_label_path, train_patient_id,
                                                              dentate_new_label_pattern.format(train_patient_id, side))
                        interposed_label_filepath = os.path.join(new_label_path, train_patient_id,
                                                                 interposed_new_label_pattern.format(train_patient_id,
                                                                                                     side))
                    elif is_new_trn_label == 3:
                        if train_patient_id in patient_id:
                            dentate_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                                  dentate_label_pattern.format(train_patient_id, side))
                            interposed_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                                     interposed_label_pattern.format(train_patient_id,
                                                                                                     side))
                        else:
                            dentate_label_filepath = os.path.join(new_label_path, train_patient_id,
                                                                  dentate_new_label_pattern.format(side))
                            interposed_label_filepath = os.path.join(new_label_path, train_patient_id,
                                                                     interposed_new_label_pattern.format(side))

                    else:
                        dentate_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                              dentate_label_pattern.format(train_patient_id, side))
                        interposed_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                                 interposed_label_pattern.format(train_patient_id,
                                                                                                 side))
                    print(dentate_label_filepath)
                    dentate_label_fname_side_lst.append(dentate_label_filepath)
                    dentate_label_data = read_volume_data(dentate_label_filepath)
                    dentate_label_vol = dentate_label_data.get_data()
                    if np.size(np.shape(dentate_label_vol)) == 4:
                        dentate_label_vol_lst.append(dentate_label_vol[:, :, :, 0])
                    else:
                        dentate_label_vol_lst.append(dentate_label_vol)

                    print(interposed_label_filepath)
                    interposed_label_fname_side_lst.append(interposed_label_filepath)
                    interposed_label_data = read_volume_data(interposed_label_filepath)
                    interposed_label_vol = interposed_label_data.get_data()
                    if np.size(np.shape(interposed_label_vol)) == 4:
                        interposed_label_vol_lst.append(interposed_label_vol[:, :, :, 0])
                    else:
                        interposed_label_vol_lst.append(interposed_label_vol)

            if target == 'dentate':
                dentate_label_fname_lst.append(dentate_label_fname_side_lst)
                dentate_label_merge_vol = dentate_label_vol_lst[0] + dentate_label_vol_lst[1]
                label_merge_vol = dentate_label_merge_vol

                # crop the roi of the image
                train_crop_mask = find_crop_mask(train_roi_mask_file) or compute_crop_mask(label_merge_vol,
                                                                                           train_roi_mask_file,
                                                                                           dentate_label_data,
                                                                                           margin_crop_mask)

            elif target == 'interposed':
                interposed_label_fname_lst.append(interposed_label_fname_side_lst)
                interposed_label_merge_vol = interposed_label_vol_lst[0] + interposed_label_vol_lst[1]
                label_merge_vol = interposed_label_merge_vol

                # crop the roi of the image
                train_crop_mask = find_crop_mask(train_roi_mask_file) or compute_crop_mask(label_merge_vol,
                                                                                           train_roi_mask_file,
                                                                                           interposed_label_data,
                                                                                           margin_crop_mask)
            else:
                dentate_label_fname_lst.append(dentate_label_fname_side_lst)
                dentate_label_merge_vol = dentate_label_vol_lst[0] + dentate_label_vol_lst[1]

                interposed_label_fname_lst.append(interposed_label_fname_side_lst)
                interposed_label_merge_vol = interposed_label_vol_lst[0] + interposed_label_vol_lst[1]

                label_merge_vol = dentate_label_merge_vol + interposed_label_merge_vol
                label_merge_vol[label_merge_vol == np.max(label_merge_vol)] = 1

                # crop the roi of the image
                train_crop_mask = find_crop_mask(train_roi_mask_file) or compute_crop_mask(label_merge_vol,
                                                                                           train_roi_mask_file,
                                                                                           dentate_label_data,
                                                                                           margin_crop_mask)

                # assign integers, later it should be encoded into one-hot in build_training_set (to_categorical)
                dentate_label_merge_vol[dentate_label_merge_vol == np.max(dentate_label_merge_vol)] = 1
                interposed_label_merge_vol[interposed_label_merge_vol == np.max(interposed_label_merge_vol)] = 2
                label_merge_vol = dentate_label_merge_vol + interposed_label_merge_vol
                label_merge_vol[label_merge_vol == np.max(label_merge_vol)] = 2  # assign overlaps to interposed

            label_crop_vol = train_crop_mask.crop(label_merge_vol)
            # for key in label_mapper.keys():
            #     label_crop_vol[label_crop_vol == key] = label_mapper[key]

            # save the cropped labels (train_roi)
            for side, idx in zip(['left', 'right'], [0, 1]):
                if target == 'dentate':
                    dentate_label_filepath = dentate_label_fname_side_lst[idx]
                    if is_new_trn_label == 1:
                        cropped_trn_dentate_label_file = os.path.join(file_trn_patient_dir,
                                                                      crop_trn_dentate_new_label_pattern.format(side))
                    elif is_new_trn_label == 2:
                        cropped_trn_dentate_label_file = os.path.join(file_trn_patient_dir,
                                                                      crop_trn_dentate_new_label_pattern.format(
                                                                          train_patient_id, side))
                    elif is_new_trn_label == 3:
                        if train_patient_id in patient_id:
                            cropped_trn_dentate_label_file = os.path.join(file_trn_patient_dir,
                                                                          crop_trn_dentate_label_pattern.format(
                                                                              train_patient_id, side))
                        else:
                            cropped_trn_dentate_label_file = os.path.join(file_trn_patient_dir,
                                                                          crop_trn_dentate_new_label_pattern.format(
                                                                              side))
                    else:
                        cropped_trn_dentate_label_file = os.path.join(file_trn_patient_dir,
                                                                      crop_trn_dentate_label_pattern.format(
                                                                          train_patient_id, side))
                    if (not os.path.exists(cropped_trn_dentate_label_file)) or set_new_roi_mask is True:
                        crop_image(dentate_label_filepath, train_crop_mask, cropped_trn_dentate_label_file)

                elif target == 'interposed':
                    interposed_label_filepath = interposed_label_fname_side_lst[idx]
                    if is_new_trn_label == 1:
                        cropped_trn_interposed_label_file = os.path.join(file_trn_patient_dir,
                                                                         crop_trn_interposed_new_label_pattern.format(
                                                                             side))
                    elif is_new_trn_label == 2:
                        cropped_trn_interposed_label_file = os.path.join(file_trn_patient_dir,
                                                                         crop_trn_interposed_new_label_pattern.format(
                                                                             train_patient_id, side))
                    elif is_new_trn_label == 3:
                        if train_patient_id in patient_id:
                            cropped_trn_interposed_label_file = os.path.join(file_trn_patient_dir,
                                                                             crop_trn_interposed_label_pattern.format(
                                                                                 train_patient_id, side))
                        else:
                            cropped_trn_interposed_label_file = os.path.join(file_trn_patient_dir,
                                                                             crop_trn_interposed_new_label_pattern.format(
                                                                                 side))
                    else:
                        cropped_trn_interposed_label_file = os.path.join(file_trn_patient_dir,
                                                                         crop_trn_interposed_label_pattern.format(
                                                                             train_patient_id, side))
                    if (not os.path.exists(cropped_trn_interposed_label_file)) or set_new_roi_mask is True:
                        crop_image(interposed_label_filepath, train_crop_mask, cropped_trn_interposed_label_file)

                else:
                    if is_new_trn_label == 1:
                        cropped_trn_dentate_label_file = os.path.join(file_trn_patient_dir,
                                                                      crop_trn_dentate_new_label_pattern.format(side))
                        cropped_trn_interposed_label_file = os.path.join(file_trn_patient_dir,
                                                                         crop_trn_interposed_new_label_pattern.format(
                                                                             side))
                    elif is_new_trn_label == 2:
                        cropped_trn_dentate_label_file = os.path.join(file_trn_patient_dir,
                                                                      crop_trn_dentate_new_label_pattern.format(
                                                                          train_patient_id, side))
                        cropped_trn_interposed_label_file = os.path.join(file_trn_patient_dir,
                                                                         crop_trn_interposed_new_label_pattern.format(
                                                                             train_patient_id, side))
                    elif is_new_trn_label == 3:
                        if train_patient_id in patient_id:
                            cropped_trn_dentate_label_file = os.path.join(file_trn_patient_dir,
                                                                          crop_trn_dentate_label_pattern.format(
                                                                              train_patient_id, side))
                            cropped_trn_interposed_label_file = os.path.join(file_trn_patient_dir,
                                                                             crop_trn_interposed_label_pattern.format(
                                                                                 train_patient_id, side))
                        else:
                            cropped_trn_dentate_label_file = os.path.join(file_trn_patient_dir,
                                                                          crop_trn_dentate_new_label_pattern.format(
                                                                              side))
                            cropped_trn_interposed_label_file = os.path.join(file_trn_patient_dir,
                                                                             crop_trn_interposed_new_label_pattern.format(
                                                                                 side))
                    else:
                        cropped_trn_dentate_label_file = os.path.join(file_trn_patient_dir,
                                                                      crop_trn_dentate_label_pattern.format(
                                                                          train_patient_id, side))
                        cropped_trn_interposed_label_file = os.path.join(file_trn_patient_dir,
                                                                         crop_trn_interposed_label_pattern.format(
                                                                             train_patient_id, side))

                    dentate_label_filepath = dentate_label_fname_side_lst[idx]
                    if (not os.path.exists(cropped_trn_dentate_label_file)) or set_new_roi_mask is True:
                        crop_image(dentate_label_filepath, train_crop_mask, cropped_trn_dentate_label_file)

                    interposed_label_filepath = interposed_label_fname_side_lst[idx]
                    if (not os.path.exists(cropped_trn_interposed_label_file)) or set_new_roi_mask is True:
                        crop_image(interposed_label_filepath, train_crop_mask, cropped_trn_interposed_label_file)

            label_crop_array = np.zeros((1, 1) + label_crop_vol.shape[0:3])
            label_crop_array[0, 0] = label_crop_vol
            label_lst.append(label_crop_array)

            # load training images in train_patient_lst and setting roi
            train_image_modal_lst = []
            train_fname_modality_lst = []
            for idx in modality_idx:
                training_img_path = os.path.join(dataset_path, train_patient_id,
                                                 img_pattern[idx].format(train_patient_id))
                print(training_img_path)
                train_fname_modality_lst.append(training_img_path)
                training_vol = read_volume(training_img_path)
                if preprocess_trn == 2 or preprocess_trn == 3 or preprocess_trn == 5:
                    training_vol = normalize_image(training_vol, [0, 2 ** 8])
                if np.size(np.shape(training_vol)) == 4:
                    train_crop_vol = train_crop_mask.crop(training_vol[:, :, :, 0])
                else:
                    train_crop_vol = train_crop_mask.crop(training_vol)
                train_image_modal_lst.append(train_crop_vol)
            train_fname_lst.append(train_fname_modality_lst)

            # save the cropped image
            for idx in modality_idx:
                train_img_path = train_fname_modality_lst[idx]
                crop_train_img_path = os.path.join(file_output_dir, train_patient_id,
                                                   crop_trn_img_pattern[idx].format(train_patient_id))
                if (not os.path.exists(crop_train_img_path)) or set_new_roi_mask is True:
                    crop_image(train_img_path, train_crop_mask, crop_train_img_path)

            train_crop_array = np.zeros((1, num_modality) + train_crop_vol.shape[0:3])
            for idx in range(num_modality):
                train_crop_array[0, idx] = train_image_modal_lst[idx]
            train_img_lst.append(train_crop_array)

    # load multi-modal images in test_patient_lst and setting roi
    test_img_lst = []
    test_fname_lst = []
    if test_patient_lst[0] is not None:
        for test_patient_id in test_patient_lst:

            file_test_patient_dir = os.path.join(file_output_dir, test_patient_id)
            if not os.path.exists(file_test_patient_dir):
                os.makedirs(file_test_patient_dir)

            test_roi_mask_file = os.path.join(file_output_dir, test_roi_mask_pattern.format(test_patient_id))
            if os.path.exists(test_roi_mask_file) and set_new_roi_mask is True:
                os.remove(test_roi_mask_file)

            # load initial mask (SUIT output or linearly registered from ref. training image) for setting roi
            init_dentate_mask_vol_lst = []
            init_interposed_mask_vol_lst = []

            for side in ['left', 'right']:  # left + right
                if target == 'dentate':
                    init_dentate_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                              init_dentate_mask_pattern.format(test_patient_id, side))
                    print(init_dentate_mask_filepath)
                    init_dentate_mask_data = read_volume_data(init_dentate_mask_filepath)
                    init_dentate_mask_vol = init_dentate_mask_data.get_data()
                    if np.size(np.shape(init_dentate_mask_vol)) == 4:
                        init_dentate_mask_vol_lst.append(init_dentate_mask_vol[:, :, :, 0])
                    else:
                        init_dentate_mask_vol_lst.append(init_dentate_mask_vol)
                    init_mask_data = init_dentate_mask_data

                elif target == 'interposed':
                    init_interposed_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                                 init_interposed_mask_pattern_thres.format(
                                                                     test_patient_id,
                                                                     side))
                    init_interposed_mask_data = read_volume_data(init_interposed_mask_filepath)
                    init_interposed_mask_vol = init_interposed_mask_data.get_data()
                    is_empty_vol = check_empty_vol(init_interposed_mask_vol)
                    is_init_interposed_mask_thres = True
                    if is_empty_vol:
                        print('There is no label in the initial mask in %s' % init_interposed_mask_filepath)
                        init_interposed_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                                     init_interposed_mask_pattern_mask.format(
                                                                         test_patient_id,
                                                                         side))
                        init_interposed_mask_data = read_volume_data(init_interposed_mask_filepath)
                        init_interposed_mask_vol = init_interposed_mask_data.get_data()
                        is_empty_vol = check_empty_vol(init_interposed_mask_vol)
                        is_init_interposed_mask_thres = False
                        if is_empty_vol:
                            print('There is no label in the initial mask in %s' % init_interposed_mask_filepath)
                            exit()

                    print(init_interposed_mask_filepath)

                    if np.size(np.shape(init_interposed_mask_vol)) == 4:
                        init_interposed_mask_vol_lst.append(init_interposed_mask_vol[:, :, :, 0])
                    else:
                        init_interposed_mask_vol_lst.append(init_interposed_mask_vol)
                    init_mask_data = init_interposed_mask_data

                else:
                    init_dentate_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                              init_dentate_mask_pattern.format(test_patient_id, side))
                    print(init_dentate_mask_filepath)
                    init_dentate_mask_data = read_volume_data(init_dentate_mask_filepath)
                    init_dentate_mask_vol = init_dentate_mask_data.get_data()
                    if np.size(np.shape(init_dentate_mask_vol)) == 4:
                        init_dentate_mask_vol_lst.append(init_dentate_mask_vol[:, :, :, 0])
                    else:
                        init_dentate_mask_vol_lst.append(init_dentate_mask_vol)

                    init_interposed_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                                 init_interposed_mask_pattern_thres.format(
                                                                     test_patient_id,
                                                                     side))
                    init_interposed_mask_data = read_volume_data(init_interposed_mask_filepath)
                    init_interposed_mask_vol = init_interposed_mask_data.get_data()
                    is_empty_vol = check_empty_vol(init_interposed_mask_vol)
                    is_init_interposed_mask_thres = True
                    if is_empty_vol:
                        print('There is no label in the initial mask in %s' % init_interposed_mask_filepath)
                        init_interposed_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                                     init_interposed_mask_pattern_mask.format(
                                                                         test_patient_id,
                                                                         side))
                        init_interposed_mask_data = read_volume_data(init_interposed_mask_filepath)
                        init_interposed_mask_vol = init_interposed_mask_data.get_data()
                        is_empty_vol = check_empty_vol(init_interposed_mask_vol)
                        is_init_interposed_mask_thres = False
                        if is_empty_vol:
                            print('There is no label in the initial mask in %s' % init_interposed_mask_filepath)
                            exit()

                    print(init_interposed_mask_filepath)

                    if np.size(np.shape(init_interposed_mask_vol)) == 4:
                        init_interposed_mask_vol_lst.append(init_interposed_mask_vol[:, :, :, 0])
                    else:
                        init_interposed_mask_vol_lst.append(init_interposed_mask_vol)
                    init_mask_data = init_dentate_mask_data

            if target == 'dentate':
                init_dentate_mask_merge_vol = init_dentate_mask_vol_lst[0] + init_dentate_mask_vol_lst[1]
                init_mask_merge_vol = init_dentate_mask_merge_vol
            elif target == 'interposed':
                init_interposed_mask_merge_vol = init_interposed_mask_vol_lst[0] + init_interposed_mask_vol_lst[1]
                init_mask_merge_vol = init_interposed_mask_merge_vol
            else:
                init_dentate_mask_merge_vol = init_dentate_mask_vol_lst[0] + init_dentate_mask_vol_lst[1]
                init_interposed_mask_merge_vol = init_interposed_mask_vol_lst[0] + init_interposed_mask_vol_lst[1]
                init_mask_merge_vol = init_dentate_mask_merge_vol + init_interposed_mask_merge_vol

            # crop the roi of the image
            test_crop_mask = find_crop_mask(test_roi_mask_file) or compute_crop_mask(init_mask_merge_vol,
                                                                                     test_roi_mask_file,
                                                                                     init_mask_data,
                                                                                     margin_crop_mask)
            # save the cropped initial masks
            for side in ['left', 'right']:
                if target == 'dentate':
                    init_dentate_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                              init_dentate_mask_pattern.format(test_patient_id, side))
                    cropped_init_dentate_mask_file = os.path.join(file_test_patient_dir,
                                                                  crop_init_dentate_mask_pattern.format(test_patient_id,
                                                                                                        side))
                    if (not os.path.exists(cropped_init_dentate_mask_file)) or set_new_roi_mask is True:
                        crop_image(init_dentate_mask_filepath, test_crop_mask, cropped_init_dentate_mask_file)

                elif target == 'interposed':
                    if is_init_interposed_mask_thres:
                        init_interposed_mask_pattern = init_interposed_mask_pattern_thres
                    else:
                        init_interposed_mask_pattern = init_interposed_mask_pattern_mask
                    init_interposed_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                                 init_interposed_mask_pattern.format(test_patient_id,
                                                                                                     side))
                    cropped_init_interposed_mask_file = os.path.join(file_test_patient_dir,
                                                                     crop_init_interposed_mask_pattern.format(
                                                                         test_patient_id, side))
                    if (not os.path.exists(cropped_init_interposed_mask_file)) or set_new_roi_mask is True:
                        crop_image(init_interposed_mask_filepath, test_crop_mask, cropped_init_interposed_mask_file)

                else:
                    init_dentate_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                              init_dentate_mask_pattern.format(test_patient_id, side))
                    cropped_init_dentate_mask_file = os.path.join(file_test_patient_dir,
                                                                  crop_init_dentate_mask_pattern.format(test_patient_id,
                                                                                                        side))
                    if (not os.path.exists(cropped_init_dentate_mask_file)) or set_new_roi_mask is True:
                        crop_image(init_dentate_mask_filepath, test_crop_mask, cropped_init_dentate_mask_file)

                    if is_init_interposed_mask_thres:
                        init_interposed_mask_pattern = init_interposed_mask_pattern_thres
                    else:
                        init_interposed_mask_pattern = init_interposed_mask_pattern_mask
                    init_interposed_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                                 init_interposed_mask_pattern.format(test_patient_id,
                                                                                                     side))
                    cropped_init_interposed_mask_file = os.path.join(file_test_patient_dir,
                                                                     crop_init_interposed_mask_pattern.format(
                                                                         test_patient_id, side))
                    if (not os.path.exists(cropped_init_interposed_mask_file)) or set_new_roi_mask is True:
                        crop_image(init_interposed_mask_filepath, test_crop_mask, cropped_init_interposed_mask_file)

            # save the cropped labels (test_roi)
            for side in ['left', 'right']:
                if target == 'dentate':
                    dentate_label_filepath = os.path.join(dataset_path, test_patient_id,
                                                          dentate_label_pattern.format(test_patient_id,
                                                                                       side))
                    if os.path.exists(dentate_label_filepath):
                        cropped_tst_dentate_label_file = os.path.join(file_test_patient_dir,
                                                                      crop_tst_dentate_label_pattern.format(
                                                                          test_patient_id, side))
                        if (not os.path.exists(cropped_tst_dentate_label_file)) or set_new_roi_mask is True:
                            crop_image(dentate_label_filepath, test_crop_mask,
                                       cropped_tst_dentate_label_file)

                elif target == 'interposed':
                    interposed_label_filepath = os.path.join(dataset_path, test_patient_id,
                                                             interposed_label_pattern.format(
                                                                 test_patient_id, side))
                    if os.path.exists(interposed_label_filepath):
                        cropped_tst_interposed_label_file = os.path.join(file_test_patient_dir,
                                                                         crop_tst_interposed_label_pattern.format(
                                                                             test_patient_id, side))
                        if (not os.path.exists(cropped_tst_interposed_label_file)) or set_new_roi_mask is True:
                            crop_image(interposed_label_filepath, test_crop_mask,
                                       cropped_tst_interposed_label_file)
                else:
                    dentate_label_filepath = os.path.join(dataset_path, test_patient_id,
                                                          dentate_label_pattern.format(test_patient_id,
                                                                                       side))
                    if os.path.exists(dentate_label_filepath):
                        cropped_tst_dentate_label_file = os.path.join(file_test_patient_dir,
                                                                      crop_tst_dentate_label_pattern.format(
                                                                          test_patient_id, side))
                        if (not os.path.exists(cropped_tst_dentate_label_file)) or set_new_roi_mask is True:
                            crop_image(dentate_label_filepath, test_crop_mask,
                                       cropped_tst_dentate_label_file)

                    interposed_label_filepath = os.path.join(dataset_path, test_patient_id,
                                                             interposed_label_pattern.format(
                                                                 test_patient_id, side))
                    if os.path.exists(interposed_label_filepath):
                        cropped_tst_interposed_label_file = os.path.join(file_test_patient_dir,
                                                                         crop_tst_interposed_label_pattern.format(
                                                                             test_patient_id, side))
                        if (not os.path.exists(cropped_tst_interposed_label_file)) or set_new_roi_mask is True:
                            crop_image(interposed_label_filepath, test_crop_mask,
                                       cropped_tst_interposed_label_file)

            # load test images in test_patient_lst and setting roi
            test_image_modal_lst = []
            test_fname_modality_lst = []
            for idx in modality_idx:
                test_img_path = os.path.join(dataset_path, test_patient_id, img_pattern[idx].format(test_patient_id))
                print(test_img_path)
                test_fname_modality_lst.append(test_img_path)
                test_vol = read_volume(test_img_path)
                if preprocess_tst == 2 or preprocess_tst == 3 or preprocess_tst == 5:
                    test_vol = normalize_image(test_vol, [0, 2 ** 8])
                if np.size(np.shape(test_vol)) == 4:
                    test_crop_vol = test_crop_mask.crop(test_vol[:, :, :, 0])
                else:
                    test_crop_vol = test_crop_mask.crop(test_vol)
                test_image_modal_lst.append(test_crop_vol)
            test_fname_lst.append(test_fname_modality_lst)

            # save the cropped image
            for idx in modality_idx:
                test_img_path = os.path.join(dataset_path, test_patient_id, img_pattern[idx].format(test_patient_id))
                crop_test_img_path = os.path.join(file_test_patient_dir,
                                                  crop_tst_img_pattern[idx].format(test_patient_id))
                if (not os.path.exists(crop_test_img_path)) or set_new_roi_mask is True:
                    crop_image(test_img_path, test_crop_mask, crop_test_img_path)

            test_crop_array = np.zeros((1, num_modality) + test_crop_vol.shape[0:3])
            for idx in range(num_modality):
                test_crop_array[0, idx] = test_image_modal_lst[idx]
            test_img_lst.append(test_crop_array)

    return train_img_lst, label_lst, test_img_lst, train_fname_lst, dentate_label_fname_lst, test_fname_lst


def read_dentate_interposed_dataset_unseen(gen_conf, train_conf, test_conf, train_patient_lst, test_patient_lst,
                                           preprocess_trn, preprocess_tst, file_output_dir, target, is_scaling):
    root_path = gen_conf['root_path']
    dataset = train_conf['dataset']
    num_epochs = train_conf['num_epochs']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    path = dataset_info['path']
    mode = gen_conf['validation_mode']
    img_pattern = dataset_info['image_name_pattern']
    image_new_name_pattern = dataset_info['image_new_name_pattern']
    img_resampled_name_pattern = dataset_info['image_resampled_name_pattern']

    dentate_label_pattern = dataset_info['manual_corrected_dentate_v2_pattern']
    interposed_label_pattern = dataset_info['manual_corrected_interposed_v2_pattern']
    dentate_interposed_label_pattern = dataset_info['manual_corrected_dentate_interposed_v2_pattern']

    modality = dataset_info['image_modality']
    num_modality = len(modality)

    file_format = dataset_info['format']

    init_dentate_mask_pattern = dataset_info['initial_mask_pattern']
    init_interposed_mask_pattern = dataset_info['initial_interposed_mask_pattern_thres']
    init_reg_mask_pattern = dataset_info['initial_reg_mask_pattern']

    set_new_roi_mask = dataset_info['set_new_roi_mask']
    margin_crop_mask = dataset_info['margin_crop_mask']

    crop_trn_img_pattern = dataset_info['crop_trn_image_name_pattern']
    crop_tst_img_pattern = dataset_info['crop_tst_image_name_pattern']
    crop_tst_resampled_img_pattern = dataset_info['crop_tst_image_resampled_name_pattern']

    crop_trn_dentate_label_pattern = dataset_info['crop_trn_manual_dentate_v2_corrected_pattern']
    crop_tst_dentate_label_pattern = dataset_info['crop_tst_manual_dentate_v2_corrected_pattern']
    crop_trn_interposed_label_pattern = dataset_info['crop_trn_manual_interposed_v2_pattern']
    crop_tst_interposed_label_pattern = dataset_info['crop_tst_manual_interposed_v2_pattern']

    crop_init_dentate_mask_pattern = dataset_info['crop_initial_mask_pattern']
    crop_init_interposed_mask_pattern = dataset_info['crop_initial_interposed_mask_pattern']
    crop_init_reg_mask_pattern = dataset_info['crop_initial_reg_mask_pattern']

    train_roi_mask_pattern = dataset_info['train_roi_mask_pattern']
    test_roi_mask_pattern = dataset_info['test_roi_mask_pattern']
    is_reg_flag = test_conf['is_reg']
    if is_reg_flag == 1:
        is_reg = True
    else:
        is_reg = False
    roi_pos_tuple = test_conf['roi_pos']
    roi_start, roi_end = [], []
    for i, j in zip(roi_pos_tuple, range(len(roi_pos_tuple))):
        if j % 2 == 0:
            roi_start.append(i)
        else:
            roi_end.append(i)
        roi_pos = [roi_start, roi_end]

    modality_idx = []
    for m in modality:
        if m == 'B0':
            modality_idx.append(0)
        if m == 'T1':
            modality_idx.append(1)
        if m == 'LP':
            modality_idx.append(2)
        if m == 'FA':
            modality_idx.append(3)

    #label_mapper = {0: 0, 10: 1, 150: 2}

    is_res_diff = False # initial setting

    if len(target) == 2:
        target = 'both'

    # load multi-modal images in train_patient_lst and setting roi
    train_img_lst = []
    label_lst = []
    train_fname_lst = []
    dentate_label_fname_lst = []
    interposed_label_fname_lst = []
    if train_patient_lst[0] is not None and num_epochs != 0 and mode != '2':
        for train_patient_id in train_patient_lst:

            file_trn_patient_dir = os.path.join(file_output_dir, train_patient_id)
            if not os.path.exists(file_trn_patient_dir):
                os.makedirs(file_trn_patient_dir)

            dcn_trn_output_dir = os.path.join(file_trn_patient_dir, path)
            if not os.path.exists(dcn_trn_output_dir):
                os.makedirs(dcn_trn_output_dir)

            train_roi_mask_file = os.path.join(dcn_trn_output_dir, train_roi_mask_pattern.format(train_patient_id))
            if os.path.exists(train_roi_mask_file) and set_new_roi_mask is True:
                os.remove(train_roi_mask_file)

            # load label images in train_patient_lst and setting roi
            dentate_label_vol_lst = []
            dentate_label_fname_side_lst = []
            interposed_label_vol_lst = []
            interposed_label_fname_side_lst = []
            for side in ['left', 'right']:  # left + right
                if target == 'dentate':
                    dentate_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                          dentate_label_pattern.format(train_patient_id, side))
                    print(dentate_label_filepath)
                    dentate_label_fname_side_lst.append(dentate_label_filepath)
                    dentate_label_data = read_volume_data(dentate_label_filepath)
                    dentate_label_vol = dentate_label_data.get_data()
                    if np.size(np.shape(dentate_label_vol)) == 4:
                        dentate_label_vol_lst.append(dentate_label_vol[:, :, :, 0])
                    else:
                        dentate_label_vol_lst.append(dentate_label_vol)

                elif target == 'interposed':
                    interposed_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                             interposed_label_pattern.format(train_patient_id, side))
                    print(interposed_label_filepath)
                    interposed_label_fname_side_lst.append(interposed_label_filepath)
                    interposed_label_data = read_volume_data(interposed_label_filepath)
                    interposed_label_vol = interposed_label_data.get_data()
                    if np.size(np.shape(interposed_label_vol)) == 4:
                        interposed_label_vol_lst.append(interposed_label_vol[:, :, :, 0])
                    else:
                        interposed_label_vol_lst.append(interposed_label_vol)

                else:
                    dentate_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                          dentate_label_pattern.format(train_patient_id, side))
                    print(dentate_label_filepath)
                    dentate_label_fname_side_lst.append(dentate_label_filepath)
                    dentate_label_data = read_volume_data(dentate_label_filepath)
                    dentate_label_vol = dentate_label_data.get_data()
                    if np.size(np.shape(dentate_label_vol)) == 4:
                        dentate_label_vol_lst.append(dentate_label_vol[:, :, :, 0])
                    else:
                        dentate_label_vol_lst.append(dentate_label_vol)

                    interposed_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                             interposed_label_pattern.format(train_patient_id, side))
                    print(interposed_label_filepath)
                    interposed_label_fname_side_lst.append(interposed_label_filepath)
                    interposed_label_data = read_volume_data(interposed_label_filepath)
                    interposed_label_vol = interposed_label_data.get_data()
                    if np.size(np.shape(interposed_label_vol)) == 4:
                        interposed_label_vol_lst.append(interposed_label_vol[:, :, :, 0])
                    else:
                        interposed_label_vol_lst.append(interposed_label_vol)

            if target == 'dentate':
                dentate_label_fname_lst.append(dentate_label_fname_side_lst)
                dentate_label_merge_vol = dentate_label_vol_lst[0] + dentate_label_vol_lst[1]
                label_merge_vol = dentate_label_merge_vol

                # crop the roi of the image
                train_crop_mask = find_crop_mask(train_roi_mask_file) or compute_crop_mask(label_merge_vol,
                                                                                       train_roi_mask_file,
                                                                                       dentate_label_data,
                                                                                       margin_crop_mask)

            elif target == 'interposed':
                interposed_label_fname_lst.append(interposed_label_fname_side_lst)
                interposed_label_merge_vol = interposed_label_vol_lst[0] + interposed_label_vol_lst[1]
                label_merge_vol = interposed_label_merge_vol

                # crop the roi of the image
                train_crop_mask = find_crop_mask(train_roi_mask_file) or compute_crop_mask(label_merge_vol,
                                                                                       train_roi_mask_file,
                                                                                       interposed_label_data,
                                                                                       margin_crop_mask)
            else:
                dentate_label_fname_lst.append(dentate_label_fname_side_lst)
                dentate_label_merge_vol = dentate_label_vol_lst[0] + dentate_label_vol_lst[1]

                interposed_label_fname_lst.append(interposed_label_fname_side_lst)
                interposed_label_merge_vol = interposed_label_vol_lst[0] + interposed_label_vol_lst[1]

                label_merge_vol = dentate_label_merge_vol + interposed_label_merge_vol
                label_merge_vol[label_merge_vol == np.max(label_merge_vol)] = 1

                # crop the roi of the image
                train_crop_mask = find_crop_mask(train_roi_mask_file) or compute_crop_mask(label_merge_vol,
                                                                                       train_roi_mask_file,
                                                                                       dentate_label_data,
                                                                                       margin_crop_mask)

                # assign integers, later it should be encoded into one-hot in build_training_set (to_categorical)
                dentate_label_merge_vol[dentate_label_merge_vol == np.max(dentate_label_merge_vol)] = 1
                interposed_label_merge_vol[interposed_label_merge_vol == np.max(interposed_label_merge_vol)] = 2
                label_merge_vol = dentate_label_merge_vol + interposed_label_merge_vol
                label_merge_vol[label_merge_vol == np.max(label_merge_vol)] = 2  # assign overlaps to interposed

            label_crop_vol = train_crop_mask.crop(label_merge_vol)
            # for key in label_mapper.keys():
            #     label_crop_vol[label_crop_vol == key] = label_mapper[key]

            # save the cropped labels (train_roi)
            for side, idx in zip(['left', 'right'], [0, 1]):
                if target == 'dentate':
                    dentate_label_filepath = dentate_label_fname_side_lst[idx]
                    cropped_trn_dentate_label_file = os.path.join(dcn_trn_output_dir,
                                                                  crop_trn_dentate_label_pattern.format(train_patient_id, side))
                    if (not os.path.exists(cropped_trn_dentate_label_file)) or set_new_roi_mask is True:
                        crop_image(dentate_label_filepath, train_crop_mask, cropped_trn_dentate_label_file)

                elif target == 'interposed':
                    interposed_label_filepath = interposed_label_fname_side_lst[idx]
                    cropped_trn_interposed_label_file = os.path.join(dcn_trn_output_dir,
                                                                     crop_trn_interposed_label_pattern.format(train_patient_id, side))
                    if (not os.path.exists(cropped_trn_interposed_label_file)) or set_new_roi_mask is True:
                        crop_image(interposed_label_filepath, train_crop_mask, cropped_trn_interposed_label_file)

                else:
                    dentate_label_filepath = dentate_label_fname_side_lst[idx]
                    cropped_trn_dentate_label_file = os.path.join(dcn_trn_output_dir,
                                                                  crop_trn_dentate_label_pattern.format(train_patient_id, side))
                    if (not os.path.exists(cropped_trn_dentate_label_file)) or set_new_roi_mask is True:
                        crop_image(dentate_label_filepath, train_crop_mask, cropped_trn_dentate_label_file)

                    interposed_label_filepath = interposed_label_fname_side_lst[idx]
                    cropped_trn_interposed_label_file = os.path.join(dcn_trn_output_dir,
                                                                     crop_trn_interposed_label_pattern.format(train_patient_id, side))
                    if (not os.path.exists(cropped_trn_interposed_label_file)) or set_new_roi_mask is True:
                        crop_image(interposed_label_filepath, train_crop_mask, cropped_trn_interposed_label_file)

            label_crop_array = np.zeros((1, 1) + label_crop_vol.shape[0:3])
            label_crop_array[0, 0] = label_crop_vol
            label_lst.append(label_crop_array)

            # load training images in train_patient_lst and setting roi
            train_image_modal_lst= []
            train_fname_modality_lst = []
            for idx in modality_idx:
                training_img_path = os.path.join(dataset_path, train_patient_id, img_pattern[idx].format(train_patient_id))
                print(training_img_path)
                train_fname_modality_lst.append(training_img_path)
                training_vol = read_volume(training_img_path)
                if preprocess_trn == 2 or preprocess_trn == 3 or preprocess_trn == 5:
                    training_vol = normalize_image(training_vol, [0, 2 ** 8])
                if np.size(np.shape(training_vol)) == 4:
                    train_crop_vol = train_crop_mask.crop(training_vol[:, :, :, 0])
                else:
                    train_crop_vol = train_crop_mask.crop(training_vol)
                train_image_modal_lst.append(train_crop_vol)
            train_fname_lst.append(train_fname_modality_lst)

            # save the cropped image
            for idx in modality_idx:
                train_img_path = train_fname_modality_lst[idx]
                crop_train_img_path = os.path.join(dcn_trn_output_dir,
                                                   crop_trn_img_pattern[idx].format(train_patient_id))
                if (not os.path.exists(crop_train_img_path)) or set_new_roi_mask is True:
                    crop_image(train_img_path, train_crop_mask, crop_train_img_path)

            train_crop_array = np.zeros((1, num_modality) + train_crop_vol.shape[0:3])
            for idx in range(num_modality):
                train_crop_array[0, idx] = train_image_modal_lst[idx]
            train_img_lst.append(train_crop_array)

    # load multi-modal images in test_patient_lst and setting roi
    test_img_lst = []
    test_fname_lst = []
    is_res_diff_lst = []
    if test_patient_lst[0] is not None:
        for test_patient_id in test_patient_lst:

            is_res_diff = False

            file_test_patient_dir = os.path.join(file_output_dir, test_patient_id)
            if not os.path.exists(file_test_patient_dir):
                os.makedirs(file_test_patient_dir)

            dcn_tst_output_dir = os.path.join(file_test_patient_dir, path)
            if not os.path.exists(dcn_tst_output_dir):
                os.makedirs(dcn_tst_output_dir)

            test_roi_mask_file = os.path.join(file_test_patient_dir, test_roi_mask_pattern.format(path))
            if os.path.exists(test_roi_mask_file) and set_new_roi_mask is True:
                os.remove(test_roi_mask_file)

            test_fname_modality_lst_for_init_reg = []
            for idx in modality_idx:
                test_img_path = ''
                if not os.path.exists(test_img_path):
                    test_img_path = os.path.join(root_path, 'datasets', 'dcn', test_patient_id, 'image',
                                                 image_new_name_pattern[idx])
                print(test_img_path)
                test_fname_modality_lst_for_init_reg.append(test_img_path)
            test_ref_img_path = test_fname_modality_lst_for_init_reg[0]

            # load initial mask (SUIT output or linearly registered from ref. training image) for setting roi
            init_dentate_mask_vol_lst = []
            init_interposed_mask_vol_lst = []
            init_mask_path = ''
            if os.path.exists(init_mask_path):
                for side in ['left', 'right']:  # left + right
                    if target == 'dentate':
                        init_dentate_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                                  init_dentate_mask_pattern.format(test_patient_id, side))
                        print(init_dentate_mask_filepath)
                        init_dentate_mask_data = read_volume_data(init_dentate_mask_filepath)
                        init_dentate_mask_vol = init_dentate_mask_data.get_data()
                        if np.size(np.shape(init_dentate_mask_vol)) == 4:
                            init_dentate_mask_vol_lst.append(init_dentate_mask_vol[:, :, :, 0])
                        else:
                            init_dentate_mask_vol_lst.append(init_dentate_mask_vol)
                        init_mask_data = init_dentate_mask_data

                    elif target == 'interposed':
                        init_interposed_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                                     init_interposed_mask_pattern.format(test_patient_id, side))
                        print(init_interposed_mask_filepath)
                        init_interposed_mask_data = read_volume_data(init_interposed_mask_filepath)
                        init_interposed_mask_vol = init_interposed_mask_data.get_data()
                        if np.size(np.shape(init_interposed_mask_vol)) == 4:
                            init_interposed_mask_vol_lst.append(init_interposed_mask_vol[:, :, :, 0])
                        else:
                            init_interposed_mask_vol_lst.append(init_interposed_mask_vol)
                        init_mask_data = init_interposed_mask_data

                    else:
                        init_dentate_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                                  init_dentate_mask_pattern.format(test_patient_id, side))
                        print(init_dentate_mask_filepath)
                        init_dentate_mask_data = read_volume_data(init_dentate_mask_filepath)
                        init_dentate_mask_vol = init_dentate_mask_data.get_data()
                        if np.size(np.shape(init_dentate_mask_vol)) == 4:
                            init_dentate_mask_vol_lst.append(init_dentate_mask_vol[:, :, :, 0])
                        else:
                            init_dentate_mask_vol_lst.append(init_dentate_mask_vol)

                        init_interposed_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                                     init_interposed_mask_pattern.format(test_patient_id, side))
                        print(init_interposed_mask_filepath)
                        init_interposed_mask_data = read_volume_data(init_interposed_mask_filepath)
                        init_interposed_mask_vol = init_interposed_mask_data.get_data()
                        if np.size(np.shape(init_interposed_mask_vol)) == 4:
                            init_interposed_mask_vol_lst.append(init_interposed_mask_vol[:, :, :, 0])
                        else:
                            init_interposed_mask_vol_lst.append(init_interposed_mask_vol)
                        init_mask_data = init_dentate_mask_data

                if target == 'dentate':
                    init_dentate_mask_merge_vol = init_dentate_mask_vol_lst[0] + init_dentate_mask_vol_lst[1]
                    init_mask_merge_vol = init_dentate_mask_merge_vol
                elif target == 'interposed':
                    init_interposed_mask_merge_vol = init_interposed_mask_vol_lst[0] + init_interposed_mask_vol_lst[1]
                    init_mask_merge_vol = init_interposed_mask_merge_vol
                else:
                    init_dentate_mask_merge_vol = init_dentate_mask_vol_lst[0] + init_dentate_mask_vol_lst[1]
                    init_interposed_mask_merge_vol = init_interposed_mask_vol_lst[0] + init_interposed_mask_vol_lst[1]
                    init_mask_merge_vol = init_dentate_mask_merge_vol + init_interposed_mask_merge_vol

            else:  # localize target via initial registration
                train_fname_lst_for_init_reg = []
                label_fname_lst_for_init_reg = []

                file_test_patient_label_dir = os.path.join(dcn_tst_output_dir, 'training_labels')
                if not os.path.exists(file_test_patient_label_dir):
                    os.makedirs(file_test_patient_label_dir)

                for train_patient_id in train_patient_lst:
                    train_fname_modality_lst_for_init_reg = []
                    for idx in modality_idx:
                        training_img_path = os.path.join(dataset_path, train_patient_id, img_pattern[idx].format(
                            train_patient_id))
                        train_fname_modality_lst_for_init_reg.append(training_img_path)
                    train_fname_lst_for_init_reg.append(train_fname_modality_lst_for_init_reg)
                    train_image_data = read_volume_data(train_fname_modality_lst_for_init_reg[0])

                    dentate_label_fname_side_lst_for_init_reg = []
                    interposed_label_fname_side_lst_for_init_reg = []
                    dentate_interposed_label_fname_side_lst_for_init_reg = []
                    for side in ['left', 'right']:  # left + right
                        if target == 'dentate':
                            dentate_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                                  dentate_label_pattern.format(train_patient_id, side))
                            dentate_label_fname_side_lst_for_init_reg.append(dentate_label_filepath)
                        elif target == 'interposed':
                            interposed_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                                     interposed_label_pattern.format(train_patient_id,
                                                                                                     side))
                            interposed_label_fname_side_lst_for_init_reg.append(interposed_label_filepath)
                        else:
                            dentate_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                                  dentate_label_pattern.format(train_patient_id, side))

                            dentate_label_fname_side_lst_for_init_reg.append(dentate_label_filepath)
                            dentate_label_data = read_volume_data(dentate_label_filepath)
                            dentate_label_vol = dentate_label_data.get_data()
                            if np.size(np.shape(dentate_label_vol)) == 4:
                                _dentate_label_vol = dentate_label_vol[:, :, :, 0]
                            else:
                                _dentate_label_vol = dentate_label_vol

                            interposed_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                                     interposed_label_pattern.format(train_patient_id,
                                                                                                     side))
                            interposed_label_fname_side_lst_for_init_reg.append(interposed_label_filepath)
                            interposed_label_data = read_volume_data(interposed_label_filepath)
                            interposed_label_vol = interposed_label_data.get_data()
                            if np.size(np.shape(interposed_label_vol)) == 4:
                                _interposed_label_vol = interposed_label_vol[:, :, :, 0]
                            else:
                                _interposed_label_vol = interposed_label_vol

                            dentate_interposed_label_filepath = os.path.join(dataset_path, train_patient_id,
                                                                     dentate_interposed_label_pattern.format(train_patient_id,
                                                                                                     side))
                            dentate_interposed_label_fname_side_lst_for_init_reg.append(dentate_interposed_label_filepath)

                            label_merge_vol = _dentate_label_vol + _interposed_label_vol
                            label_merge_vol[label_merge_vol == np.max(label_merge_vol)] = 1

                            __save_volume(label_merge_vol, train_image_data, dentate_interposed_label_filepath,
                                          file_format, is_compressed=True)

                    if target == 'dentate':
                        label_fname_lst_for_init_reg.append(dentate_label_fname_side_lst_for_init_reg)
                    elif target == 'interposed':
                        label_fname_lst_for_init_reg.append(interposed_label_fname_side_lst_for_init_reg)
                    else:
                        label_fname_lst_for_init_reg.append(dentate_interposed_label_fname_side_lst_for_init_reg)

                init_mask_merge_vol, init_mask_data, test_ref_img_path, is_res_diff, roi_pos = \
                    localize_target(file_output_dir,
                                    modality_idx,
                                    train_patient_lst,
                                    train_fname_lst_for_init_reg,
                                    test_patient_id,
                                    test_fname_modality_lst_for_init_reg,
                                    label_fname_lst_for_init_reg,
                                    _remove_ending(img_pattern[0].format(''), '.nii')+'.nii.gz',
                                    img_resampled_name_pattern,
                                    init_reg_mask_pattern,
                                    target,
                                    is_res_diff,
                                    roi_pos,
                                    dcn_tst_output_dir,
                                    is_scaling,
                                    is_reg)

            if init_mask_merge_vol.tolist():
                # crop the roi of the image
                test_crop_mask = find_crop_mask(test_roi_mask_file) or compute_crop_mask(init_mask_merge_vol,
                                                                                         test_roi_mask_file,
                                                                                         init_mask_data,
                                                                                         margin_crop_mask=(5, 5, 10))
            else:
                print (roi_pos)
                test_ref_img_data = read_volume_data(test_ref_img_path)
                test_ref_image = BrainImage(test_ref_img_path, None)
                test_vol = test_ref_image.nii_data_normalized(bits=8)
                test_crop_mask = compute_crop_mask_manual(test_vol, test_roi_mask_file, test_ref_img_data,
                                                          roi_pos[0], roi_pos[1])

            # save the cropped initial masks
            for side in ['left', 'right']:
                if os.path.exists(init_mask_path):
                    if target == 'dentate':
                        init_dentate_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                                  init_dentate_mask_pattern.format(test_patient_id, side))
                        cropped_init_dentate_mask_file = os.path.join(dcn_tst_output_dir,
                                                                      crop_init_dentate_mask_pattern.format(test_patient_id, side))
                        if (not os.path.exists(cropped_init_dentate_mask_file)) or set_new_roi_mask is True:
                            crop_image(init_dentate_mask_filepath, test_crop_mask, cropped_init_dentate_mask_file)

                    elif target == 'interposed':
                        init_interposed_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                                     init_interposed_mask_pattern.format(test_patient_id, side))
                        cropped_init_interposed_mask_file = os.path.join(dcn_tst_output_dir,
                                                                         crop_init_interposed_mask_pattern.format(test_patient_id, side))
                        if (not os.path.exists(cropped_init_interposed_mask_file)) or set_new_roi_mask is True:
                            crop_image(init_interposed_mask_filepath, test_crop_mask, cropped_init_interposed_mask_file)

                    else:
                        init_dentate_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                                  init_dentate_mask_pattern.format(test_patient_id, side))
                        cropped_init_dentate_mask_file = os.path.join(dcn_tst_output_dir,
                                                                      crop_init_dentate_mask_pattern.format(test_patient_id, side))
                        if (not os.path.exists(cropped_init_dentate_mask_file)) or set_new_roi_mask is True:
                            crop_image(init_dentate_mask_filepath, test_crop_mask, cropped_init_dentate_mask_file)

                        init_interposed_mask_filepath = os.path.join(dataset_path, test_patient_id,
                                                                     init_interposed_mask_pattern.format(test_patient_id, side))
                        cropped_init_interposed_mask_file = os.path.join(dcn_tst_output_dir,
                                                                         crop_init_interposed_mask_pattern.format(test_patient_id, side))
                        if (not os.path.exists(cropped_init_interposed_mask_file)) or set_new_roi_mask is True:
                            crop_image(init_interposed_mask_filepath, test_crop_mask, cropped_init_interposed_mask_file)

                else:
                    init_mask_filepath = os.path.join(dcn_tst_output_dir, 'init_reg',
                                                      init_reg_mask_pattern.format(side, target))
                    if os.path.exists(init_mask_filepath):
                        cropped_init_reg_mask_file = os.path.join(dcn_tst_output_dir,
                                                                  crop_init_reg_mask_pattern.format(side, target))
                        if (not os.path.exists(cropped_init_reg_mask_file)) or set_new_roi_mask is True:
                            crop_image(init_mask_filepath, test_crop_mask, cropped_init_reg_mask_file)

            # save the cropped labels (test_roi)
            for side in ['left', 'right']:
                if target == 'dentate':
                    dentate_label_filepath = os.path.join(dataset_path, test_patient_id,
                                                          dentate_label_pattern.format(test_patient_id,
                                                                                       side))
                    if os.path.exists(dentate_label_filepath):
                        cropped_tst_dentate_label_file = os.path.join(dcn_tst_output_dir,
                                                                      crop_tst_dentate_label_pattern.format(
                                                                          test_patient_id, side))
                        if (not os.path.exists(cropped_tst_dentate_label_file)) or set_new_roi_mask is True:
                            crop_image(dentate_label_filepath, test_crop_mask,
                                       cropped_tst_dentate_label_file)

                elif target == 'interposed':
                    interposed_label_filepath = os.path.join(dataset_path, test_patient_id,
                                                             interposed_label_pattern.format(
                                                                 test_patient_id, side))
                    if os.path.exists(interposed_label_filepath):
                        cropped_tst_interposed_label_file = os.path.join(dcn_tst_output_dir,
                                                                         crop_tst_interposed_label_pattern.format(
                                                                             test_patient_id, side))
                        if (not os.path.exists(cropped_tst_interposed_label_file)) or set_new_roi_mask is True:
                            crop_image(interposed_label_filepath, test_crop_mask,
                                       cropped_tst_interposed_label_file)
                else:
                    dentate_label_filepath = os.path.join(dataset_path, test_patient_id,
                                                          dentate_label_pattern.format(test_patient_id,
                                                                                       side))
                    if os.path.exists(dentate_label_filepath):
                        cropped_tst_dentate_label_file = os.path.join(dcn_tst_output_dir,
                                                                      crop_tst_dentate_label_pattern.format(
                                                                          test_patient_id, side))
                        if (not os.path.exists(cropped_tst_dentate_label_file)) or set_new_roi_mask is True:
                            crop_image(dentate_label_filepath, test_crop_mask,
                                       cropped_tst_dentate_label_file)

                    interposed_label_filepath = os.path.join(dataset_path, test_patient_id,
                                                             interposed_label_pattern.format(
                                                                 test_patient_id, side))
                    if os.path.exists(interposed_label_filepath):
                        cropped_tst_interposed_label_file = os.path.join(dcn_tst_output_dir,
                                                                         crop_tst_interposed_label_pattern.format(
                                                                             test_patient_id, side))
                        if (not os.path.exists(cropped_tst_interposed_label_file)) or set_new_roi_mask is True:
                            crop_image(interposed_label_filepath, test_crop_mask,
                                       cropped_tst_interposed_label_file)

            # load test images in test_patient_lst and setting roi
            test_image_modal_lst = []
            test_fname_modality_lst = []
            for idx in modality_idx:
                if is_res_diff is True:
                    test_img_path = os.path.join(dcn_tst_output_dir,
                                                 img_resampled_name_pattern[idx].format(test_patient_id))
                else:
                    test_img_path = ''
                    if not os.path.exists(test_img_path):
                        test_img_path = os.path.join(root_path, 'datasets', 'dcn', test_patient_id, 'image',
                                                     image_new_name_pattern[idx])
                print(test_img_path)

                test_fname_modality_lst.append(test_img_path)
                test_vol = read_volume(test_img_path)
                if preprocess_tst == 2 or preprocess_tst == 3 or preprocess_tst == 5:
                    test_vol = normalize_image(test_vol, [0, 2 ** 8])
                if np.size(np.shape(test_vol)) == 4:
                    test_crop_vol = test_crop_mask.crop(test_vol[:, :, :, 0])
                else:
                    test_crop_vol = test_crop_mask.crop(test_vol)
                test_image_modal_lst.append(test_crop_vol)
            test_fname_lst.append(test_fname_modality_lst)

            # save the cropped image
            for idx in modality_idx:
                if is_res_diff is True:
                    test_img_path = os.path.join(dcn_tst_output_dir,
                                                 img_resampled_name_pattern[idx].format(test_patient_id))
                    crop_test_img_path = os.path.join(dcn_tst_output_dir,
                                                      crop_tst_resampled_img_pattern[idx].format(test_patient_id))
                else:
                    test_img_path = ''
                    if not os.path.exists(test_img_path):
                        test_img_path = os.path.join(root_path, 'datasets', 'dcn', test_patient_id, 'image',
                                                     image_new_name_pattern[idx])
                    crop_test_img_path = os.path.join(dcn_tst_output_dir,
                                                      crop_tst_img_pattern[idx].format(test_patient_id))

                if (not os.path.exists(crop_test_img_path)) or set_new_roi_mask is True:
                    crop_image(test_img_path, test_crop_mask, crop_test_img_path)

            test_crop_array = np.zeros((1, num_modality) + test_crop_vol.shape[0:3])
            for idx in range(num_modality):
                test_crop_array[0, idx] = test_image_modal_lst[idx]
            test_img_lst.append(test_crop_array)

            is_res_diff_lst.append(is_res_diff)

    return train_img_lst, label_lst, test_img_lst, train_fname_lst, dentate_label_fname_lst, test_fname_lst, is_res_diff_lst


def save_volume_tha(gen_conf, train_conf, test_conf, volume, prob_volume, test_fname, test_patient_id, seg_label,
                    file_output_dir):
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    num_classes = gen_conf['num_classes']
    mode = gen_conf['validation_mode']
    path = dataset_info['path']
    test_roi_mask_pattern = dataset_info['test_roi_mask_pattern']
    crop_tst_img_pattern = dataset_info['crop_tst_image_name_pattern']
    threshold = test_conf['threshold']
    approach = train_conf['approach']
    loss = train_conf['loss']
    dimension = test_conf['dimension']
    file_format = dataset_info['format']

    modality = dataset_info['image_modality']
    modality_idx = []
    for m in modality: # the order of modality images should be fixed due to the absolute path
        if m == 'T1':
            modality_idx.append(0)
        if m == 'B0':
            modality_idx.append(1)
        if m == 'FA':
            modality_idx.append(2)

    file_test_patient_dir = os.path.join(file_output_dir, test_patient_id)
    seg_output_dir = os.path.join(file_test_patient_dir, path)
    if not os.path.exists(seg_output_dir):
        os.makedirs(seg_output_dir)
    prob_map_output_dir = os.path.join(seg_output_dir, 'prob_map_crop')
    if not os.path.exists(prob_map_output_dir):
        os.makedirs(prob_map_output_dir)
    non_smoothed_output_dir = os.path.join(seg_output_dir, 'non_smoothed')
    if not os.path.exists(non_smoothed_output_dir):
        os.makedirs(non_smoothed_output_dir)
    non_smoothed_crop_output_dir = os.path.join(seg_output_dir, 'non_smoothed', 'crop')
    if not os.path.exists(non_smoothed_crop_output_dir):
        os.makedirs(non_smoothed_crop_output_dir)
    smoothed_output_dir = os.path.join(seg_output_dir, 'smoothed')
    if not os.path.exists(smoothed_output_dir):
        os.makedirs(smoothed_output_dir)
    smoothed_norm_output_dir = os.path.join(seg_output_dir, 'smoothed_norm')
    if not os.path.exists(smoothed_norm_output_dir):
        os.makedirs(smoothed_norm_output_dir)
    smoothed_norm_thres_output_dir = os.path.join(seg_output_dir, 'smoothed_norm_thres')
    if not os.path.exists(smoothed_norm_thres_output_dir):
        os.makedirs(smoothed_norm_thres_output_dir)
    smoothed_norm_thres_crop_output_dir = os.path.join(seg_output_dir, 'smoothed_norm_thres', 'crop')
    if not os.path.exists(smoothed_norm_thres_crop_output_dir):
        os.makedirs(smoothed_norm_thres_crop_output_dir)

    is_stl_out = False

    if is_stl_out:
        stl_output_dir = os.path.join(seg_output_dir, 'stl_out')
        if not os.path.exists(stl_output_dir):
            os.makedirs(os.path.join(seg_output_dir, 'stl_out'))

    crop_test_img_path = os.path.join(file_test_patient_dir, crop_tst_img_pattern[modality_idx[0]])
    crop_test_data = read_volume_data(crop_test_img_path)

    test_roi_mask_file = os.path.join(file_test_patient_dir, test_roi_mask_pattern)
    test_crop_mask = find_crop_mask(test_roi_mask_file)
    image_data = read_volume_data(test_fname[0])

    #threshold for overlaped patch images (prob. vol) within the segmented volume
    prob_vol_norm = np.zeros(prob_volume.shape)
    for i in range(num_classes):
        prob_vol_norm[:, :, :, i] = normalize_image(prob_volume[:, :, :, i], [0, 1])
    volume = np.multiply(volume, prob_vol_norm[:, :, :, num_classes-1])
    volume_thr = volume > threshold

    # save probability map for background and foreground
    idx = 0
    class_name_lst = ['bg', 'fg']
    for class_name in class_name_lst:
        prob_map_crop_filename = class_name + '_' + seg_label + '_prob_map_crop_' + approach + '_' + loss + '_.' + \
                                 file_format
        prob_map_crop_out_filepath = os.path.join(prob_map_output_dir, prob_map_crop_filename)
        print(prob_map_crop_out_filepath)
        __save_volume(prob_vol_norm[:, :, :, idx], crop_test_data, prob_map_crop_out_filepath, file_format,
                      is_compressed=False)
        idx += 1

    # split side (in LPI)
    left_mask, right_mask, volume_thr = compute_side_mask(volume_thr, crop_test_data, is_check_vol_diff=True)

    for mask, side in zip([left_mask, right_mask], ['left', 'right']):
        if volume_thr is None:
            failed_cases_filepath = os.path.join(file_output_dir, 'failed_cases.txt')
            with open(failed_cases_filepath, 'a') as f:
                f.write(test_patient_id + '_' + seg_label + '_' + side + '\n')
            continue

        # split the volume into left/right side
        vol = np.multiply(mask, volume_thr)

        # postprocessing
        vol_refined = postprocess(vol)
        if vol_refined is None:
            failed_cases_filepath = os.path.join(file_output_dir, 'failed_cases.txt')
            with open(failed_cases_filepath, 'a') as f:
                f.write(test_patient_id + '_' + seg_label + '_' + side + '\n')
            continue
        # save the cropped/refined result
        nii_crop_filename = side + '_' + seg_label + '_seg_crop_' + approach + '_' + loss + '.' + file_format
        nii_crop_out_filepath = os.path.join(non_smoothed_crop_output_dir, nii_crop_filename)
        print(nii_crop_out_filepath)
        __save_volume(vol_refined, crop_test_data, nii_crop_out_filepath, file_format, is_compressed=True)

        # uncrop segmentation only (left and right)
        vol_uncrop = test_crop_mask.uncrop(vol_refined)

        # save the uncropped result
        nii_uncrop_filename = side + '_' + seg_label + '_seg_' + approach + '_' + loss + '.' + file_format
        nii_uncrop_out_filepath = os.path.join(non_smoothed_output_dir, nii_uncrop_filename)
        print(nii_uncrop_out_filepath)
        __save_volume(vol_uncrop, image_data, nii_uncrop_out_filepath, file_format, is_compressed = True)

        # smoothing (before surface extraction)
        nii_smooth_filename = side + '_' + seg_label + '_seg_smooth_' + approach + '_' + loss + '.' + file_format
        nii_smooth_out_filepath = os.path.join(smoothed_output_dir, nii_smooth_filename)
        print(nii_smooth_out_filepath)
        __smooth_binary_img(nii_uncrop_out_filepath, nii_smooth_out_filepath, dim=dimension, maximumRMSError = 0.01,
                            numberOfIterations = 10, numberOfLayers = 3)

        # normalization
        nii_smooth_image = BrainImage(nii_smooth_out_filepath, None)
        vol_smooth_norm = nii_smooth_image.nii_data_normalized(bits=0)
        nii_smooth_norm_filename = side + '_' + seg_label + '_seg_smooth_norm_' + approach + '_' + loss + '.' + \
                                   file_format
        nii_smooth_norm_out_filepath = os.path.join(smoothed_norm_output_dir, nii_smooth_norm_filename)
        print(nii_smooth_norm_out_filepath)
        __save_volume(vol_smooth_norm, image_data, nii_smooth_norm_out_filepath, file_format, is_compressed=False)

        # threshold
        vol_smooth_norm_thres = vol_smooth_norm > 0.4
        nii_smooth_norm_thres_filename = side + '_' + seg_label + '_seg_smooth_norm_thres_' + approach + '_' + loss + '.' + \
                                         file_format
        nii_smooth_norm_thres_out_filepath = os.path.join(smoothed_norm_thres_output_dir, nii_smooth_norm_thres_filename)
        print(nii_smooth_norm_thres_out_filepath)
        __save_volume(vol_smooth_norm_thres, image_data, nii_smooth_norm_thres_out_filepath, file_format,
                      is_compressed=True)

        # crop threhold image for measure
        nii_smooth_norm_thres_crop_filename = side + '_' + seg_label + '_seg_smooth_norm_thres_crop_' + approach + '_' + \
                                              loss + '.' +  file_format
        nii_smooth_norm_thres_crop_out_filepath = os.path.join(smoothed_norm_thres_crop_output_dir,
                                                               nii_smooth_norm_thres_crop_filename)
        print(nii_smooth_norm_thres_crop_out_filepath)
        crop_image(nii_smooth_norm_thres_out_filepath, test_crop_mask, nii_smooth_norm_thres_crop_out_filepath)

        if is_stl_out:
            # save stl
            stl_filename = side + '_' + seg_label + '_seg' + '.stl'
            stl_out_filepath= os.path.join(stl_output_dir, stl_filename)
            print(stl_out_filepath)
            __create_stl(nii_uncrop_out_filepath, stl_out_filepath)

            # smooth stl
            stl_smooth_filename = side + '_' + seg_label + '_seg_smooth.stl'
            stl_smooth_out_filepath= os.path.join(stl_output_dir, stl_smooth_filename)
            print(stl_smooth_out_filepath)
            __smooth_stl(stl_out_filepath, stl_smooth_out_filepath)


def save_volume_tha_unseen(gen_conf, train_conf, test_conf, volume, prob_volume, test_fname, test_patient_id, seg_label,
                    file_output_dir, is_res_diff):
    root_path = gen_conf['root_path']
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = gen_conf['dataset_path']
    num_classes = gen_conf['num_classes']
    mode = gen_conf['validation_mode']
    path = dataset_info['path']
    test_img_name_pattern = dataset_info['image_name_pattern']
    test_roi_mask_pattern = dataset_info['test_roi_mask_pattern']
    crop_tst_img_pattern = dataset_info['crop_tst_image_name_pattern']
    crop_tst_resampled_img_pattern = dataset_info['crop_tst_image_resampled_name_pattern']

    threshold = test_conf['threshold']

    approach = train_conf['approach']
    loss = train_conf['loss']
    dimension = test_conf['dimension']
    file_format = dataset_info['format']

    modality = dataset_info['image_modality']
    modality_idx = []
    for m in modality: # the order of modality images should be fixed due to the absolute path
        if m == 'T1':
            modality_idx.append(0)
        if m == 'B0':
            modality_idx.append(1)
        if m == 'FA':
            modality_idx.append(2)

    file_test_patient_dir = os.path.join(file_output_dir, test_patient_id)
    tha_output_dir = os.path.join(file_test_patient_dir, path)
    if not os.path.exists(tha_output_dir):
        os.makedirs(tha_output_dir)
    tha_seg_output_dir = os.path.join(tha_output_dir, 'seg')
    if not os.path.exists(tha_seg_output_dir):
        os.makedirs(tha_seg_output_dir)
    prob_map_output_dir = os.path.join(tha_seg_output_dir, 'prob_map_crop')
    if not os.path.exists(prob_map_output_dir):
        os.makedirs(prob_map_output_dir)
    non_smoothed_output_dir = os.path.join(tha_seg_output_dir, 'non_smoothed')
    if not os.path.exists(non_smoothed_output_dir):
        os.makedirs(non_smoothed_output_dir)
    non_smoothed_crop_output_dir = os.path.join(tha_seg_output_dir, 'non_smoothed', 'crop')
    if not os.path.exists(non_smoothed_crop_output_dir):
        os.makedirs(non_smoothed_crop_output_dir)
    smoothed_output_dir = os.path.join(tha_seg_output_dir, 'smoothed')
    if not os.path.exists(smoothed_output_dir):
        os.makedirs(smoothed_output_dir)
    smoothed_norm_output_dir = os.path.join(tha_seg_output_dir, 'smoothed_norm')
    if not os.path.exists(smoothed_norm_output_dir):
        os.makedirs(smoothed_norm_output_dir)
    smoothed_norm_thres_output_dir = os.path.join(tha_seg_output_dir, 'smoothed_norm_thres')
    if not os.path.exists(smoothed_norm_thres_output_dir):
        os.makedirs(smoothed_norm_thres_output_dir)
    smoothed_norm_thres_crop_output_dir = os.path.join(tha_seg_output_dir, 'smoothed_norm_thres', 'crop')
    if not os.path.exists(smoothed_norm_thres_crop_output_dir):
        os.makedirs(smoothed_norm_thres_crop_output_dir)

    is_stl_out = False

    if is_stl_out:
        stl_output_dir = os.path.join(tha_seg_output_dir, 'stl_out')
        if not os.path.exists(stl_output_dir):
            os.makedirs(os.path.join(tha_seg_output_dir, 'stl_out'))

    if is_res_diff is True:
        crop_test_img_path = os.path.join(tha_output_dir, crop_tst_resampled_img_pattern[modality_idx[0]])
    else:
        crop_test_img_path = os.path.join(tha_output_dir, crop_tst_img_pattern[modality_idx[0]])
    crop_test_data = read_volume_data(crop_test_img_path)
    test_roi_mask_file = os.path.join(tha_output_dir, test_roi_mask_pattern)
    test_crop_mask = find_crop_mask(test_roi_mask_file)
    image_data = read_volume_data(test_fname[0])

    if is_res_diff is True:
        resampled_label = '_resampled.'
        test_img_path = os.path.join(root_path, 'datasets', 'tha', test_patient_id, 'images', test_img_name_pattern[0])
        test_image = BrainImage(test_img_path, None)
        test_file = test_image.nii_file
        vox_size = test_file.header.get_zooms()
        image_vol = test_image.nii_data
        org_shape = np.shape(image_vol)
        thres = 0.7
    else:
        resampled_label = '.'

    #threshold for overlaped patch images (prob. vol) within the segmented volume
    prob_vol_norm = np.zeros(prob_volume.shape)
    for i in range(num_classes):
        prob_vol_norm[:, :, :, i] = normalize_image(prob_volume[:, :, :, i], [0, 1])
    volume = np.multiply(volume, prob_vol_norm[:, :, :, num_classes-1])
    volume_thr = volume > threshold

    # save probability map for background and foreground
    idx = 0
    class_name_lst = ['bg', 'fg']
    for class_name in class_name_lst:
        prob_map_crop_filename = class_name + '_' + seg_label + '_prob_map_crop_' + approach + '_' + loss + \
                                 resampled_label + file_format
        prob_map_crop_out_filepath = os.path.join(prob_map_output_dir, prob_map_crop_filename)
        print(prob_map_crop_out_filepath)
        __save_volume(prob_vol_norm[:, :, :, idx], crop_test_data, prob_map_crop_out_filepath, file_format,
                      is_compressed=False)
        idx += 1


    # split side (in LPI)
    left_mask, right_mask, volume_thr = compute_side_mask(volume_thr, crop_test_data, is_check_vol_diff=True)

    for mask, side in zip([left_mask, right_mask], ['left', 'right']):
        if volume_thr is None:
            failed_cases_filepath = os.path.join(file_output_dir, 'failed_cases.txt')
            with open(failed_cases_filepath, 'a') as f:
                f.write(test_patient_id + '_' + seg_label + '_' + side + '\n')
            continue

        # split the volume into left/right side
        vol = np.multiply(mask, volume_thr)

        # postprocessing
        vol_refined = postprocess(vol)
        if vol_refined is None:
            failed_cases_filepath = os.path.join(file_output_dir, 'failed_cases.txt')
            with open(failed_cases_filepath, 'a') as f:
                f.write(test_patient_id + '_' + seg_label + '_' + side + '\n')
            continue
        # save the cropped/refined result
        nii_crop_filename = side + '_' + seg_label + '_seg_crop_' + approach + '_' + loss + resampled_label + \
                            file_format
        nii_crop_out_filepath = os.path.join(non_smoothed_crop_output_dir, nii_crop_filename)
        print(nii_crop_out_filepath)
        __save_volume(vol_refined, crop_test_data, nii_crop_out_filepath, file_format, is_compressed=True)

        # uncrop segmentation only (left and right)
        vol_uncrop = test_crop_mask.uncrop(vol_refined)

        # save the uncropped result
        nii_uncrop_filename = side + '_' + seg_label + '_seg_' + approach + '_' + loss + resampled_label + file_format
        nii_uncrop_out_filepath = os.path.join(non_smoothed_output_dir, nii_uncrop_filename)
        print(nii_uncrop_out_filepath)
        __save_volume(vol_uncrop, image_data, nii_uncrop_out_filepath, file_format, is_compressed = True)

        if is_res_diff is True:
            # resampling back to the original resolution
            res_label = '_org_res.'
            nii_uncrop_org_res_filename = side + '_' + seg_label + '_seg_' + approach + '_' + loss + res_label + \
                                          file_format
            nii_uncrop_org_res_out_filepath = os.path.join(non_smoothed_output_dir, nii_uncrop_org_res_filename)

            print('resampling %s back to %s' % (nii_uncrop_out_filepath, str(vox_size)))
            seg_resample_img = BrainImage(nii_uncrop_out_filepath, None)
            seg_resample_img.ResampleTo(nii_uncrop_org_res_out_filepath, new_affine=test_file.affine,
                                        new_shape=org_shape[:3], new_voxel_size=vox_size, is_ant_resample=False)
            seg_org_res_img_data = read_volume_data(nii_uncrop_org_res_out_filepath)
            seg_org_res_vol = seg_org_res_img_data.get_data()
            seg_org_res_vol_thres = normalize_image(seg_org_res_vol, [0, 1]) > thres
            seg_org_res_vol_thres_refined = postprocess(seg_org_res_vol_thres)
            __save_volume(seg_org_res_vol_thres_refined, seg_org_res_img_data, nii_uncrop_org_res_out_filepath,
                          file_format, is_compressed=True)
            nii_uncrop_out_filepath = nii_uncrop_org_res_out_filepath
        else:
            seg_org_res_img_data = image_data
            res_label = '.'

        # smoothing (before surface extraction)
        nii_smooth_filename = side + '_' + seg_label + '_seg_smooth_' + approach + '_' + loss + res_label + \
                              file_format
        nii_smooth_out_filepath = os.path.join(smoothed_output_dir, nii_smooth_filename)
        print(nii_smooth_out_filepath)
        __smooth_binary_img(nii_uncrop_out_filepath, nii_smooth_out_filepath, dim=dimension, maximumRMSError = 0.01,
                            numberOfIterations = 10, numberOfLayers = 3)

        # normalization
        nii_smooth_image = BrainImage(nii_smooth_out_filepath, None)
        vol_smooth_norm = nii_smooth_image.nii_data_normalized(bits=0)
        nii_smooth_norm_filename = side + '_' + seg_label + '_seg_smooth_norm_' + approach + '_' + loss + res_label + \
                                   file_format
        nii_smooth_norm_out_filepath = os.path.join(smoothed_norm_output_dir, nii_smooth_norm_filename)
        print(nii_smooth_norm_out_filepath)
        __save_volume(vol_smooth_norm, seg_org_res_img_data, nii_smooth_norm_out_filepath, file_format, is_compressed=False)

        # threshold
        vol_smooth_norm_thres = vol_smooth_norm > 0.4
        nii_smooth_norm_thres_filename = side + '_' + seg_label + '_seg_smooth_norm_thres_' + approach + '_' + loss + \
                                         res_label + file_format
        nii_smooth_norm_thres_out_filepath = os.path.join(smoothed_norm_thres_output_dir, nii_smooth_norm_thres_filename)
        print(nii_smooth_norm_thres_out_filepath)
        __save_volume(vol_smooth_norm_thres, seg_org_res_img_data, nii_smooth_norm_thres_out_filepath, file_format,
                      is_compressed = True)

        # crop threhold image for measure
        nii_smooth_norm_thres_crop_filename = side + '_' + seg_label + '_seg_smooth_norm_thres_crop_' + approach + \
                                              '_' + loss + res_label + file_format
        nii_smooth_norm_thres_crop_out_filepath = os.path.join(smoothed_norm_thres_crop_output_dir,
                                                               nii_smooth_norm_thres_crop_filename)
        print(nii_smooth_norm_thres_crop_out_filepath)
        crop_image(nii_smooth_norm_thres_out_filepath, test_crop_mask, nii_smooth_norm_thres_crop_out_filepath)

        if is_stl_out:
            # save stl
            stl_filename = side + '_' + seg_label + '_seg' + '.stl'
            stl_out_filepath= os.path.join(stl_output_dir, stl_filename)
            print(stl_out_filepath)
            __create_stl(nii_uncrop_out_filepath, stl_out_filepath)

            # smooth stl
            stl_smooth_filename = side + '_' + seg_label + '_seg_smooth.stl'
            stl_smooth_out_filepath= os.path.join(stl_output_dir, stl_smooth_filename)
            print(stl_smooth_out_filepath)
            __smooth_stl(stl_out_filepath, stl_smooth_out_filepath)


def save_volume_dentate_interposed(gen_conf, train_conf, test_conf, volume, prob_volume, test_fname, test_patient_id,
                                   file_output_dir, target):
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    path = dataset_info['path']
    test_roi_mask_pattern = dataset_info['test_roi_mask_pattern']
    crop_tst_img_pattern = dataset_info['crop_tst_image_name_pattern']
    threshold = test_conf['threshold']
    multi_output = gen_conf['multi_output']
    num_classes = gen_conf['num_classes']
    mode = gen_conf['validation_mode']
    approach = train_conf['approach']
    loss = train_conf['loss']
    activation = train_conf['activation']
    exclusive_train = train_conf['exclusive_train']
    exclude_label_num = dataset_info['exclude_label_num']

    dimension = test_conf['dimension']
    file_format = dataset_info['format']

    modality = dataset_info['image_modality']
    modality_idx = []
    for m in modality: # the order of modality images should be fixed due to the absolute path
        if m == 'B0':
            modality_idx.append(0)
        if m == 'T1':
            modality_idx.append(1)
        if m == 'LP':
            modality_idx.append(2)
        if m == 'FA':
            modality_idx.append(2)

    if len(target) == 2:
        target = 'both'

    file_test_patient_dir = os.path.join(file_output_dir, test_patient_id)
    seg_output_dir = os.path.join(file_test_patient_dir, path)
    if not os.path.exists(seg_output_dir):
        os.makedirs(seg_output_dir)
    prob_map_output_dir = os.path.join(seg_output_dir, 'prob_map_crop')
    if not os.path.exists(prob_map_output_dir):
        os.makedirs(prob_map_output_dir)
    non_smoothed_output_dir = os.path.join(seg_output_dir, 'non_smoothed')
    if not os.path.exists(non_smoothed_output_dir):
        os.makedirs(non_smoothed_output_dir)
    non_smoothed_crop_output_dir = os.path.join(seg_output_dir, 'non_smoothed', 'crop')
    if not os.path.exists(non_smoothed_crop_output_dir):
        os.makedirs(non_smoothed_crop_output_dir)
    smoothed_output_dir = os.path.join(seg_output_dir, 'smoothed')
    if not os.path.exists(smoothed_output_dir):
        os.makedirs(smoothed_output_dir)
    smoothed_norm_output_dir = os.path.join(seg_output_dir, 'smoothed_norm')
    if not os.path.exists(smoothed_norm_output_dir):
        os.makedirs(smoothed_norm_output_dir)
    smoothed_norm_thres_output_dir = os.path.join(seg_output_dir, 'smoothed_norm_thres')
    if not os.path.exists(smoothed_norm_thres_output_dir):
        os.makedirs(smoothed_norm_thres_output_dir)
    smoothed_norm_thres_crop_output_dir = os.path.join(seg_output_dir, 'smoothed_norm_thres', 'crop')
    if not os.path.exists(smoothed_norm_thres_crop_output_dir):
        os.makedirs(smoothed_norm_thres_crop_output_dir)

    is_stl_out = False

    if is_stl_out:
        stl_output_dir = os.path.join(seg_output_dir, 'stl_out')
        if not os.path.exists(stl_output_dir):
            os.makedirs(os.path.join(seg_output_dir, 'stl_out'))

    crop_test_img_path = os.path.join(seg_output_dir, crop_tst_img_pattern[modality_idx[0]].format(test_patient_id))
    crop_test_data = read_volume_data(crop_test_img_path)

    test_roi_mask_file = os.path.join(file_output_dir, test_roi_mask_pattern.format(test_patient_id))
    test_crop_mask = find_crop_mask(test_roi_mask_file)
    image_data = read_volume_data(test_fname[0])

    # save probability map for background and foreground
    if target == 'dentate':
        class_name_lst = ['bg', 'dentate']
        loss_ch = [loss, loss]
    elif target == 'interposed':
        class_name_lst = ['bg', 'interposed']
        loss_ch = [loss, loss]
    else:
        if multi_output == 1:
            class_name_lst = ['bg_dentate', 'bg_interposed', 'dentate', 'interposed']
            loss_ch = [loss[0], loss[1], loss[0], loss[1]]
            output_ch = [0, 1, 0, 1]
            idx_lst = [0, 0, num_classes[0]-1, num_classes[1]-1]
        else:
            class_name_lst = ['bg', 'dentate', 'interposed']
            loss_ch = [loss, loss, loss]

    if exclusive_train == 1:
        class_name_lst.remove(class_name_lst[exclude_label_num])

    if multi_output == 1:
        for class_name, l_ch, o_ch, idx in zip(class_name_lst, loss_ch, output_ch, idx_lst):
            prob_map_crop_filename = class_name + '_' + '_prob_map_crop_' + approach + '_' + l_ch + '.' + \
                                     file_format
            prob_map_crop_out_filepath = os.path.join(prob_map_output_dir, prob_map_crop_filename)
            print(prob_map_crop_out_filepath)
            __save_volume(prob_volume[o_ch][:, :, :, idx], crop_test_data, prob_map_crop_out_filepath, file_format,
                          is_compressed=False)
    else:
        idx = 0
        for class_name, l_ch in zip(class_name_lst, loss_ch):
            prob_map_crop_filename = class_name + '_' + '_prob_map_crop_' + approach + '_' + l_ch + '.' + \
                                     file_format
            prob_map_crop_out_filepath = os.path.join(prob_map_output_dir, prob_map_crop_filename)
            print(prob_map_crop_out_filepath)
            __save_volume(prob_volume[:, :, :, idx], crop_test_data, prob_map_crop_out_filepath, file_format,
                          is_compressed=False)
            idx += 1

    if target == 'dentate':
        key_lst = [1]
        seg_label_lst = ['dentate']
        if multi_output == 1:
            activation_ch = [activation[0]]
        else:
            activation_ch = [activation]

    elif target == 'interposed':
        key_lst = [1]
        seg_label_lst = ['interposed']
        if multi_output == 1:
            activation_ch = [activation[1]]
        else:
            activation_ch = [activation]
    else:
        seg_label_lst = ['dentate','interposed']
        if multi_output == 1:
            key_lst = [2, 3]
            activation_ch = [activation[0], activation[1]]
        else:
            key_lst = [1, 2]
            activation_ch = [activation, activation]

    if exclusive_train == 1:
        print('exclude_label_num: %s' % exclude_label_num)
        num_classes_org = gen_conf['num_classes']
        print('num_classes_org: %s' % num_classes_org)
        num_classes = num_classes_org - 1
        volume_org_shape = (volume.shape[0], volume.shape[1], volume.shape[2], num_classes_org)
        volume_org = np.zeros(volume_org_shape)
        label_idx = []
        for i in range(num_classes_org):
            label_idx.append(i)
        label_idx.remove(exclude_label_num)
        for i, j in zip(range(num_classes), label_idx):
            volume_org[:, :, :, j] = volume[:, :, :, i]
        volume_org[:, :, :, exclude_label_num] = (np.sum(volume_org, axis=3) == 0).astype(np.uint8)
        volume = volume_org

    for a_ch, seg_label, key in zip(activation_ch, seg_label_lst, key_lst):
        if a_ch == 'softmax':
            if multi_output == 1:
                vol_temp = np.zeros(volume[output_ch[key]].shape)
                vol_temp[volume[output_ch[key]] == num_classes[output_ch[key]]-1] = 1
                print('threshold: %s' % threshold)
                prob_vol_norm = normalize_image(prob_volume[output_ch[key]][:, :, :, num_classes[output_ch[key]]-1],
                                                [0, 1])
                volume_prob = np.multiply(vol_temp, prob_vol_norm)
                volume_f = volume_prob > threshold
            else:
                vol_temp = np.zeros(volume.shape)
                vol_temp[volume == key] = 1
                print('threshold: %s' % threshold)
                prob_vol_norm = normalize_image(prob_volume[:, :, :, key], [0, 1])
                volume_prob = np.multiply(vol_temp, prob_vol_norm)
                volume_f = volume_prob > threshold
        elif a_ch == 'sigmoid':
            if multi_output == 1:
                volume_f = volume[output_ch[key]][:, :, :, num_classes[output_ch[key]]-1]
            else:
                volume_f = volume[:, :, :, key]

        # split side (in LPI)
        left_mask, right_mask, volume_f = compute_side_mask(volume_f, crop_test_data, is_check_vol_diff=False)

        for mask, side in zip([left_mask, right_mask], ['left', 'right']):
            if volume_f is None:
                failed_cases_filepath = os.path.join(file_output_dir, 'failed_cases.txt')
                with open(failed_cases_filepath, 'a') as f:
                    f.write(test_patient_id + '_' + seg_label + '_' + side + '\n')
                continue

            # split the volume into left/right side
            vol = np.multiply(mask, volume_f)

            # postprocessing
            vol_refined = postprocess(vol)
            if vol_refined is None:
                failed_cases_filepath = os.path.join(file_output_dir, 'failed_cases.txt')
                with open(failed_cases_filepath, 'a') as f:
                    f.write(test_patient_id + '_' + seg_label + '_' + side + '\n')
                continue
            # save the cropped/refined result
            nii_crop_filename = side + '_' + seg_label + '_seg_crop_' + approach + '_' + loss_ch[key] + '.' + file_format
            nii_crop_out_filepath = os.path.join(non_smoothed_crop_output_dir, nii_crop_filename)
            print(nii_crop_out_filepath)
            __save_volume(vol_refined, crop_test_data, nii_crop_out_filepath, file_format, is_compressed=True)

            # uncrop segmentation only (left and right)
            vol_uncrop = test_crop_mask.uncrop(vol_refined)

            # save the uncropped result
            nii_uncrop_filename = side + '_' + seg_label + '_seg_' + approach + '_' + loss_ch[key] + '.' + file_format
            nii_uncrop_out_filepath = os.path.join(non_smoothed_output_dir, nii_uncrop_filename)
            print(nii_uncrop_out_filepath)
            __save_volume(vol_uncrop, image_data, nii_uncrop_out_filepath, file_format, is_compressed = True)

            # smoothing (before surface extraction)
            nii_smooth_filename = side + '_' + seg_label + '_seg_smooth_' + approach + '_' + loss_ch[key] + '.' + file_format
            nii_smooth_out_filepath = os.path.join(smoothed_output_dir, nii_smooth_filename)
            print(nii_smooth_out_filepath)
            __smooth_binary_img(nii_uncrop_out_filepath, nii_smooth_out_filepath, dim=dimension, maximumRMSError = 0.01,
                                numberOfIterations = 10, numberOfLayers = 3)

            # normalization
            nii_smooth_image = BrainImage(nii_smooth_out_filepath, None)
            vol_smooth_norm = nii_smooth_image.nii_data_normalized(bits=0)
            nii_smooth_norm_filename = side + '_' + seg_label + '_seg_smooth_norm_' + approach + '_' + loss_ch[key] + '.' + \
                                       file_format
            nii_smooth_norm_out_filepath = os.path.join(smoothed_norm_output_dir, nii_smooth_norm_filename)
            print(nii_smooth_norm_out_filepath)
            __save_volume(vol_smooth_norm, image_data, nii_smooth_norm_out_filepath, file_format, is_compressed=False)

            # threshold
            vol_smooth_norm_thres = vol_smooth_norm > 0.6
            nii_smooth_norm_thres_filename = side + '_' + seg_label + '_seg_smooth_norm_thres_' + approach + '_' + loss_ch[key] + '.' + \
                                             file_format
            nii_smooth_norm_thres_out_filepath = os.path.join(smoothed_norm_thres_output_dir, nii_smooth_norm_thres_filename)
            print(nii_smooth_norm_thres_out_filepath)
            __save_volume(vol_smooth_norm_thres, image_data, nii_smooth_norm_thres_out_filepath, file_format,
                          is_compressed=True)

            # crop threhold image for measure
            nii_smooth_norm_thres_crop_filename = side + '_' + seg_label + '_seg_smooth_norm_thres_crop_' + approach + '_' + \
                                                  loss_ch[key] + '.' +  file_format
            nii_smooth_norm_thres_crop_out_filepath = os.path.join(smoothed_norm_thres_crop_output_dir,
                                                                   nii_smooth_norm_thres_crop_filename)
            print(nii_smooth_norm_thres_crop_out_filepath)
            crop_image(nii_smooth_norm_thres_out_filepath, test_crop_mask, nii_smooth_norm_thres_crop_out_filepath)

            if is_stl_out:
                # save stl
                stl_filename = side + '_' + seg_label + '_seg' + '.stl'
                stl_out_filepath= os.path.join(stl_output_dir, stl_filename)
                print(stl_out_filepath)
                __create_stl(nii_uncrop_out_filepath, stl_out_filepath)

                # smooth stl
                stl_smooth_filename = side + '_' + seg_label + '_seg_smooth.stl'
                stl_smooth_out_filepath= os.path.join(stl_output_dir, stl_smooth_filename)
                print(stl_smooth_out_filepath)
                __smooth_stl(stl_out_filepath, stl_smooth_out_filepath)


def save_volume_dentate_interposed_unseen(gen_conf, train_conf, test_conf, volume, prob_volume, test_fname, test_patient_id,
                                   file_output_dir, target, is_res_diff):
    root_path = gen_conf['root_path']
    dataset_path = gen_conf['dataset_path']
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    path = dataset_info['path']
    test_roi_mask_pattern = dataset_info['test_roi_mask_pattern']
    crop_tst_img_pattern = dataset_info['crop_tst_image_name_pattern']
    crop_tst_resampled_img_pattern = dataset_info['crop_tst_image_resampled_name_pattern']

    threshold = test_conf['threshold']
    multi_output = gen_conf['multi_output']
    num_classes = gen_conf['num_classes']
    mode = gen_conf['validation_mode']
    approach = train_conf['approach']
    loss = train_conf['loss']
    activation = train_conf['activation']
    file_format = dataset_info['format']
    dimension = test_conf['dimension']

    exclusive_train = train_conf['exclusive_train']
    exclude_label_num = dataset_info['exclude_label_num']

    modality = dataset_info['image_modality']
    modality_idx = []

    test_img_pattern = dataset_info['image_name_pattern']
    test_image_new_name_pattern = dataset_info['image_new_name_pattern']

    for m in modality: # the order of modality images should be fixed due to the absolute path
        if m == 'B0':
            modality_idx.append(0)
        if m == 'T1':
            modality_idx.append(1)
        if m == 'LP':
            modality_idx.append(2)
        if m == 'FA':
            modality_idx.append(3)

    if len(target) == 2:
        target = 'both'

    file_test_patient_dir = os.path.join(file_output_dir, test_patient_id)
    dcn_output_dir = os.path.join(file_test_patient_dir, path)
    if not os.path.exists(dcn_output_dir):
        os.makedirs(dcn_output_dir)
    dcn_seg_output_dir = os.path.join(dcn_output_dir, 'seg')
    if not os.path.exists(dcn_seg_output_dir):
        os.makedirs(dcn_seg_output_dir)
    prob_map_output_dir = os.path.join(dcn_seg_output_dir, 'prob_map_crop')
    if not os.path.exists(prob_map_output_dir):
        os.makedirs(prob_map_output_dir)
    non_smoothed_output_dir = os.path.join(dcn_seg_output_dir, 'non_smoothed')
    if not os.path.exists(non_smoothed_output_dir):
        os.makedirs(non_smoothed_output_dir)
    non_smoothed_crop_output_dir = os.path.join(dcn_seg_output_dir, 'non_smoothed', 'crop')
    if not os.path.exists(non_smoothed_crop_output_dir):
        os.makedirs(non_smoothed_crop_output_dir)
    smoothed_output_dir = os.path.join(dcn_seg_output_dir, 'smoothed')
    if not os.path.exists(smoothed_output_dir):
        os.makedirs(smoothed_output_dir)
    smoothed_norm_output_dir = os.path.join(dcn_seg_output_dir, 'smoothed_norm')
    if not os.path.exists(smoothed_norm_output_dir):
        os.makedirs(smoothed_norm_output_dir)
    smoothed_norm_thres_output_dir = os.path.join(dcn_seg_output_dir, 'smoothed_norm_thres')
    if not os.path.exists(smoothed_norm_thres_output_dir):
        os.makedirs(smoothed_norm_thres_output_dir)
    smoothed_norm_thres_crop_output_dir = os.path.join(dcn_seg_output_dir, 'smoothed_norm_thres', 'crop')
    if not os.path.exists(smoothed_norm_thres_crop_output_dir):
        os.makedirs(smoothed_norm_thres_crop_output_dir)

    is_stl_out = False

    if is_stl_out:
        stl_output_dir = os.path.join(dcn_seg_output_dir, 'stl_out')
        if not os.path.exists(stl_output_dir):
            os.makedirs(os.path.join(dcn_seg_output_dir, 'stl_out'))

    if is_res_diff is True:
        crop_test_img_path = os.path.join(dcn_output_dir, crop_tst_resampled_img_pattern[modality_idx[0]].format(test_patient_id))
    else:
        crop_test_img_path = os.path.join(dcn_output_dir, crop_tst_img_pattern[modality_idx[0]].format(test_patient_id))

    crop_test_data = read_volume_data(crop_test_img_path)

    test_roi_mask_file = os.path.join(file_test_patient_dir, test_roi_mask_pattern.format(path))
    test_crop_mask = find_crop_mask(test_roi_mask_file)
    image_data = read_volume_data(test_fname[0])

    if is_res_diff is True:
        resampled_label = '_resampled.'
        test_img_path = os.path.join(dataset_path, test_patient_id,
                                     test_img_pattern[modality_idx[0]].format(test_patient_id))
        if not os.path.exists(test_img_path):
            test_img_path = os.path.join(root_path, 'datasets', 'dcn', test_patient_id, 'image',
                                         test_image_new_name_pattern[modality_idx[0]])
        test_image = BrainImage(test_img_path, None)
        test_file = test_image.nii_file
        vox_size = test_file.header.get_zooms()
        image_vol = test_image.nii_data
        org_shape = np.shape(image_vol)
        thres = 0.7
    else:
        resampled_label = '.'

    # save probability map for background and foreground
    if target == 'dentate':
        class_name_lst = ['bg', 'dentate']
        loss_ch = [loss, loss]
    elif target == 'interposed':
        class_name_lst = ['bg', 'interposed']
        loss_ch = [loss, loss]
    else:
        if multi_output == 1:
            class_name_lst = ['bg_dentate', 'bg_interposed', 'dentate', 'interposed']
            loss_ch = [loss[0], loss[1], loss[0], loss[1]]
            output_ch = [0, 1, 0, 1]
            idx_lst = [0, 0, num_classes[0]-1, num_classes[1]-1]
        else:
            class_name_lst = ['bg', 'dentate', 'interposed']
            loss_ch = [loss, loss, loss]

    if exclusive_train == 1:
        class_name_lst.remove(class_name_lst[exclude_label_num])

    if multi_output == 1:
        for class_name, l_ch, o_ch, idx in zip(class_name_lst, loss_ch, output_ch, idx_lst):
            prob_map_crop_filename = class_name + '_' + '_prob_map_crop_' + approach + '_' + l_ch + '.' + \
                                     file_format
            prob_map_crop_out_filepath = os.path.join(prob_map_output_dir, prob_map_crop_filename)
            print(prob_map_crop_out_filepath)
            __save_volume(prob_volume[o_ch][:, :, :, idx], crop_test_data, prob_map_crop_out_filepath, file_format,
                          is_compressed=False)
    else:
        idx = 0
        for class_name, l_ch in zip(class_name_lst, loss_ch):
            prob_map_crop_filename = class_name + '_' + '_prob_map_crop_' + approach + '_' + l_ch + '.' + \
                                     file_format
            prob_map_crop_out_filepath = os.path.join(prob_map_output_dir, prob_map_crop_filename)
            print(prob_map_crop_out_filepath)
            __save_volume(prob_volume[:, :, :, idx], crop_test_data, prob_map_crop_out_filepath, file_format,
                          is_compressed=False)
            idx += 1

    if target == 'dentate':
        key_lst = [1]
        seg_label_lst = ['dentate']
        if multi_output == 1:
            activation_ch = [activation[0]]
        else:
            activation_ch = [activation]

    elif target == 'interposed':
        key_lst = [1]
        seg_label_lst = ['interposed']
        if multi_output == 1:
            activation_ch = [activation[1]]
        else:
            activation_ch = [activation]
    else:
        seg_label_lst = ['dentate','interposed']
        if multi_output == 1:
            key_lst = [2, 3]
            activation_ch = [activation[0], activation[1]]
        else:
            key_lst = [1, 2]
            activation_ch = [activation, activation]

    if exclusive_train == 1:
        print('exclude_label_num: %s' % exclude_label_num)
        num_classes_org = gen_conf['num_classes']
        print('num_classes_org: %s' % num_classes_org)
        num_classes = num_classes_org - 1
        volume_org_shape = (volume.shape[0], volume.shape[1], volume.shape[2], num_classes_org)
        volume_org = np.zeros(volume_org_shape)
        label_idx = []
        for i in range(num_classes_org):
            label_idx.append(i)
        label_idx.remove(exclude_label_num)
        for i, j in zip(range(num_classes), label_idx):
            volume_org[:, :, :, j] = volume[:, :, :, i]
        volume_org[:, :, :, exclude_label_num] = (np.sum(volume_org, axis=3) == 0).astype(np.uint8)
        volume = volume_org

    for a_ch, seg_label, key in zip(activation_ch, seg_label_lst, key_lst):
        if a_ch == 'softmax':
            if multi_output == 1:
                vol_temp = np.zeros(volume[output_ch[key]].shape)
                vol_temp[volume[output_ch[key]] == num_classes[output_ch[key]]-1] = 1
                print('threshold: %s' % threshold)
                prob_vol_norm = normalize_image(prob_volume[output_ch[key]][:, :, :, num_classes[output_ch[key]]-1],
                                                [0, 1])
                volume_prob = np.multiply(vol_temp, prob_vol_norm)
                volume_f = volume_prob > threshold
            else:
                vol_temp = np.zeros(volume.shape)
                vol_temp[volume == key] = 1
                print('threshold: %s' % threshold)
                prob_vol_norm = normalize_image(prob_volume[:, :, :, key], [0, 1])
                volume_prob = np.multiply(vol_temp, prob_vol_norm)
                volume_f = volume_prob > threshold
        elif a_ch == 'sigmoid':
            if multi_output == 1:
                volume_f = volume[output_ch[key]][:, :, :, num_classes[output_ch[key]]-1]
            else:
                volume_f = volume[:, :, :, key]

        # split side (in LPI)
        left_mask, right_mask, volume_f = compute_side_mask(volume_f, crop_test_data, is_check_vol_diff=False)

        for mask, side in zip([left_mask, right_mask], ['left', 'right']):
            if volume_f is None:
                failed_cases_filepath = os.path.join(file_output_dir, 'failed_cases.txt')
                with open(failed_cases_filepath, 'a') as f:
                    f.write(test_patient_id + '_' + seg_label + '_' + side + '\n')
                continue

            # split the volume into left/right side
            vol = np.multiply(mask, volume_f)

            # postprocessing
            vol_refined = postprocess(vol)
            if vol_refined is None:
                failed_cases_filepath = os.path.join(file_output_dir, 'failed_cases.txt')
                with open(failed_cases_filepath, 'a') as f:
                    f.write(test_patient_id + '_' + seg_label + '_' + side + '\n')
                continue
            # save the cropped/refined result
            nii_crop_filename = side + '_' + seg_label + '_seg_crop_' + approach + '_' + loss_ch[key] + resampled_label + \
                                file_format
            nii_crop_out_filepath = os.path.join(non_smoothed_crop_output_dir, nii_crop_filename)
            print(nii_crop_out_filepath)
            __save_volume(vol_refined, crop_test_data, nii_crop_out_filepath, file_format, is_compressed=True)

            # uncrop segmentation only (left and right)
            vol_uncrop = test_crop_mask.uncrop(vol_refined)

            # save the uncropped result
            nii_uncrop_filename = side + '_' + seg_label + '_seg_' + approach + '_' + loss_ch[key] + resampled_label + \
                                  file_format
            nii_uncrop_out_filepath = os.path.join(non_smoothed_output_dir, nii_uncrop_filename)
            print(nii_uncrop_out_filepath)
            __save_volume(vol_uncrop, image_data, nii_uncrop_out_filepath, file_format, is_compressed = True)

            if is_res_diff is True:
                # resampling back to the original resolution
                res_label = '_org_res.'
                nii_uncrop_org_res_filename = side + '_' + seg_label + '_seg_' + approach + '_' + loss_ch[key] + res_label + \
                                              file_format
                nii_uncrop_org_res_out_filepath = os.path.join(non_smoothed_output_dir, nii_uncrop_org_res_filename)

                print('resampling %s back to %s' % (nii_uncrop_out_filepath, str(vox_size)))
                seg_resample_img = BrainImage(nii_uncrop_out_filepath, None)
                seg_resample_img.ResampleTo(nii_uncrop_org_res_out_filepath, new_affine=test_file.affine,
                                            new_shape=org_shape[:3], new_voxel_size=vox_size, is_ant_resample=False)
                seg_org_res_img_data = read_volume_data(nii_uncrop_org_res_out_filepath)
                seg_org_res_vol = seg_org_res_img_data.get_data()
                seg_org_res_vol_thres = normalize_image(seg_org_res_vol, [0, 1]) > thres
                seg_org_res_vol_thres_refined = postprocess(seg_org_res_vol_thres)
                __save_volume(seg_org_res_vol_thres_refined, seg_org_res_img_data, nii_uncrop_org_res_out_filepath,
                              file_format, is_compressed=True)
                nii_uncrop_out_filepath = nii_uncrop_org_res_out_filepath
            else:
                seg_org_res_img_data = image_data
                res_label = '.'

            # smoothing (before surface extraction)
            nii_smooth_filename = side + '_' + seg_label + '_seg_smooth_' + approach + '_' + loss_ch[key] + res_label + \
                                  file_format
            nii_smooth_out_filepath = os.path.join(smoothed_output_dir, nii_smooth_filename)
            print(nii_smooth_out_filepath)
            __smooth_binary_img(nii_uncrop_out_filepath, nii_smooth_out_filepath, dim=dimension, maximumRMSError = 0.01,
                                numberOfIterations = 10, numberOfLayers = 3)

            # normalization
            nii_smooth_image = BrainImage(nii_smooth_out_filepath, None)
            vol_smooth_norm = nii_smooth_image.nii_data_normalized(bits=0)
            nii_smooth_norm_filename = side + '_' + seg_label + '_seg_smooth_norm_' + approach + '_' + loss_ch[key] + '.' + \
                                       file_format
            nii_smooth_norm_out_filepath = os.path.join(smoothed_norm_output_dir, nii_smooth_norm_filename)
            print(nii_smooth_norm_out_filepath)
            __save_volume(vol_smooth_norm, seg_org_res_img_data, nii_smooth_norm_out_filepath, file_format,
                          is_compressed=False)

            # threshold
            vol_smooth_norm_thres = vol_smooth_norm > 0.6
            nii_smooth_norm_thres_filename = side + '_' + seg_label + '_seg_smooth_norm_thres_' + approach + '_' + loss_ch[key] + '.' + \
                                             file_format
            nii_smooth_norm_thres_out_filepath = os.path.join(smoothed_norm_thres_output_dir, nii_smooth_norm_thres_filename)
            print(nii_smooth_norm_thres_out_filepath)
            __save_volume(vol_smooth_norm_thres, seg_org_res_img_data, nii_smooth_norm_thres_out_filepath, file_format,
                          is_compressed=True)

            # crop threhold image for measure
            nii_smooth_norm_thres_crop_filename = side + '_' + seg_label + '_seg_smooth_norm_thres_crop_' + approach + '_' + \
                                                  loss_ch[key] + '.' + file_format
            nii_smooth_norm_thres_crop_out_filepath = os.path.join(smoothed_norm_thres_crop_output_dir,
                                                                   nii_smooth_norm_thres_crop_filename)
            print(nii_smooth_norm_thres_crop_out_filepath)
            crop_image(nii_smooth_norm_thres_out_filepath, test_crop_mask, nii_smooth_norm_thres_crop_out_filepath)

            if is_stl_out:
                # save stl
                stl_filename = side + '_' + seg_label + '_seg' + '.stl'
                stl_out_filepath = os.path.join(stl_output_dir, stl_filename)
                print(stl_out_filepath)
                __create_stl(nii_uncrop_out_filepath, stl_out_filepath)

                # smooth stl
                stl_smooth_filename = side + '_' + seg_label + '_seg_smooth.stl'
                stl_smooth_out_filepath= os.path.join(stl_output_dir, stl_smooth_filename)
                print(stl_smooth_out_filepath)
                __smooth_stl(stl_out_filepath, stl_smooth_out_filepath)


def save_intermediate_volume(gen_conf, train_conf, volume, case_idx, filename_ext, label) :
    dataset = train_conf['dataset']
    root_path = gen_conf['root_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = root_path + gen_conf['dataset_path']
    results_path = root_path + gen_conf['results_path']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    folder_names = dataset_info['folder_names']

    print(volume.shape)
    volume_tmp = np.zeros(volume.shape + (1, ))
    volume_tmp[:, :, :, 0] = volume
    volume = volume_tmp
    print(volume.shape)

    # it should be 3T test image (not 7T label)
    if not filename_ext:
        data_filename = dataset_path + path + pattern[0].format(folder_names, case_idx)
    else:
        file_dir = dataset_path + path + folder_names
        file_path = file_dir + '/' + filename_ext
        data_filename = file_path

    print(data_filename)
    image_data = read_volume_data(data_filename)

    if not filename_ext:
        if not os.path.exists(results_path + path + folder_names):
            os.makedirs(os.path.join(results_path, path, folder_names))
        out_filename = results_path + path + pattern[1].format(folder_names, str(case_idx) + '_' + label)
    else:
        if not os.path.exists(results_path + path + folder_names[1]):
            os.makedirs(os.path.join(results_path, path, folder_names[1]))
        file_output_dir = results_path + path + folder_names[1]
        file_name, ext = os.path.splitext(filename_ext)
        out_filename = file_output_dir + '/' + file_name + '_' + label + ext

    print(out_filename)
    __save_volume(volume, image_data, out_filename, dataset_info['format'], is_compressed = False)


def __save_volume(volume, image_data, filename, format, is_compressed) :
    img = None
    #max_bit = np.ceil(np.log2(np.max(volume.flatten())))
    if format in ['nii', 'nii.gz'] :
        if is_compressed:
            img = nib.Nifti1Image(volume.astype('uint8'), image_data.affine)
        else:
            img = nib.Nifti1Image(volume, image_data.affine)
    elif format == 'analyze':
        if is_compressed:
            # labels were assigned between 0 and 255 (8bit)
            img = nib.analyze.AnalyzeImage(volume.astype('uint8'), image_data.affine)
        else:
            img = nib.analyze.AnalyzeImage(volume, image_data.affine)
        #img = nib.analyze.AnalyzeImage(volume.astype('uint' + max_bit), image_data.affine)
        #img.set_data_dtype('uint8')
    nib.save(img, filename)


def __create_stl(str_file, str_file_out_path):
    nii_file = nib.load(str_file)
    structures_v, structures_f = generate_structures_surface(str_file, threshold=0.1)
    structures_v_tr = apply_image_orientation_to_stl(np.transpose(structures_v) + 1, nii_file)
    structures_v_tr = np.transpose(structures_v_tr)
    write_stl(str_file_out_path, structures_v_tr, structures_f)


def save_patches(gen_conf, train_conf, patch_data, case_name):

    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    folder_names = dataset_info['folder_names']
    root_path = gen_conf['root_path']
    patches_path = gen_conf['patches_path']
    patch_shape = train_conf['patch_shape']
    extraction_step = train_conf['extraction_step']
    num_classes = gen_conf['num_classes']
    multi_output = gen_conf['multi_output']
    approach = train_conf['approach']
    data_augment = train_conf['data_augment']
    mode = gen_conf['validation_mode']
    preprocess_trn = train_conf['preprocess']
    loss = train_conf['loss']
    dimension = train_conf['dimension']

    if data_augment == 1:
        data_augment_label = 'mixup'
    elif data_augment == 2:
        data_augment_label = 'datagen'
    elif data_augment == 3:
        data_augment_label = 'mixup+datagen'
    else:
        data_augment_label = ''

    data_patches_path = root_path + patches_path + '/' + dataset + '/' + folder_names[0]
    if not os.path.exists(data_patches_path):
        os.makedirs(os.path.join(root_path, patches_path, dataset, folder_names[0]))

    if multi_output == 1:
        loss = loss[0] + '_' + loss[1]
    patches_filename = generate_output_filename(root_path + patches_path, dataset + '/' + folder_names[0],
                                                'mode_' + mode, case_name, approach, loss, 'dim_' + str(dimension),
                                                'n_classes_' + str(num_classes), str(patch_shape), str(extraction_step),
                                                data_augment_label, 'preproc_trn_opt_' + str(preprocess_trn) +
                                                '_training_samples', 'hdf5')

    print('Saving training samples (patches)...')
    with h5py.File(patches_filename, 'w') as f:
        _ = f.create_dataset("x_train", data=patch_data[0])
        _ = f.create_dataset("y_train", data=patch_data[1])
        _ = f.create_dataset("x_val", data=patch_data[2])
        _ = f.create_dataset("y_val", data=patch_data[3])


def read_patches(gen_conf, train_conf, case_name):
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    folder_names = dataset_info['folder_names']
    root_path = gen_conf['root_path']
    patches_path = gen_conf['patches_path']
    patch_shape = train_conf['patch_shape']
    extraction_step = train_conf['extraction_step']
    num_classes = gen_conf['num_classes']
    multi_output = gen_conf['multi_output']
    approach = train_conf['approach']
    data_augment = train_conf['data_augment']
    mode = gen_conf['validation_mode']
    preprocess_trn = train_conf['preprocess']
    loss = train_conf['loss']
    dimension = train_conf['dimension']

    if data_augment == 1:
        data_augment_label = 'mixup'
    elif data_augment == 2:
        data_augment_label = 'datagen'
    elif data_augment == 3:
        data_augment_label = 'mixup+datagen'
    else:
        data_augment_label = ''

    data_patches_path = root_path + patches_path + '/' + dataset + '/' + folder_names[0]
    if not os.path.exists(data_patches_path):
        os.makedirs(os.path.join(root_path, patches_path, dataset, folder_names[0]))

    if multi_output == 1:
        loss = loss[0] + '_' + loss[1]
    patches_filename = generate_output_filename(root_path + patches_path, dataset + '/' + folder_names[0], 'mode_'+
                                                mode, case_name, approach, loss, 'dim_' + str(dimension), 'n_classes_'
                                                + str(num_classes), str(patch_shape), str(extraction_step),
                                                data_augment_label, 'preproc_trn_opt_' + str(preprocess_trn) +
                                                '_training_samples', 'hdf5')

    patches = []
    if os.path.isfile(patches_filename):
        print('Training samples (patches) already exist!')
        print('Loading saved training samples (patches)...')
        with h5py.File(patches_filename, 'r') as f:
            f_x_train = f['x_train']
            f_y_train = f['y_train']
            f_x_val = f['x_val']
            f_y_val = f['y_val']
            patches = [f_x_train[:], f_y_train[:], f_x_val[:], f_y_val[:]]

    return patches


def read_volume(filename):
    return read_volume_data(filename).get_data()


def read_volume_data(filename):
    return nib.load(filename)


def localize_target(file_output_dir, modality_idx, train_patient_lst, train_fname_lst, test_patient_id,
                    test_fname_modality_lst, label_fname_lst, img_pattern, img_resampled_name_pattern,
                    init_reg_mask_pattern, target, is_res_diff, roi_pos, output_dir,
                    is_scaling=False, is_reg=True):

    # composite registration step (from one of training T1 MRI to test T1 MRI)
    # for initial localization to set the roi
    # load T1 data from test set

    # is_reg: True for PD088, PD090, False for PD091

    test_ref_img_path = test_fname_modality_lst[0]
    test_ref_image = BrainImage(test_ref_img_path, None)
    test_vol = test_ref_image.nii_data_normalized(bits=8)
    if np.size(np.shape(test_vol)) == 4:
        test_vol = test_vol[:, :, :, 0]
    mean_tst_vol = np.mean(test_vol.flatten())
    test_vol_size = np.array(test_vol.shape)

    # load T1 data from training set
    mean_trn_vol = []
    train_vol_size_lst = []
    trn_vox_size_lst = []
    for train_fname in train_fname_lst:
        training_img_path = train_fname[0]
        training_img = BrainImage(training_img_path, None)
        train_vol = training_img.nii_data_normalized(bits=8)
        train_vol_size = np.array(train_vol.shape)
        trn_vox_size = training_img.nii_file.header.get_zooms()
        mean_trn_vol.append(np.mean(train_vol.flatten()))
        train_vol_size_lst.append(train_vol_size)
        trn_vox_size_lst.append(trn_vox_size)
    abs_dist_mean = abs(np.array(mean_trn_vol - mean_tst_vol))
    print(test_vol_size)
    print(train_vol_size_lst)
    trn_sel_ind = np.where(abs_dist_mean == np.min(abs_dist_mean))[0][0]

    trn_sel_img_path = train_fname_lst[trn_sel_ind][0]
    trn_sel_img = BrainImage(trn_sel_img_path, None)

    # scaling of trn_sel_img to the size of test data
    scaling_ratio = np.divide(np.array(test_vol_size), np.array(train_vol_size_lst[trn_sel_ind]))
    if ((np.linalg.norm(scaling_ratio) < 2 / 3 or np.linalg.norm(scaling_ratio) > 1.5)) and is_scaling:
        print('scaling_ratio_btw_trn_img_and test_img: %s' % scaling_ratio)
        print('scaling selected training data (%s) to the size of test data (%s) with scaling ratio (norm)(%s)' %
              (train_patient_lst[trn_sel_ind], test_patient_id, np.linalg.norm(scaling_ratio)))
        trn_sel_img_scaling_path = os.path.join(output_dir, train_patient_lst[trn_sel_ind] +
                                                '_7T_T1_brain_scaled.nii.gz')
        trn_sel_img_scaled_vol = trn_sel_img.rescale(scaling_ratio)
        trn_sel_img_scaled_out = nib.Nifti1Image(trn_sel_img_scaled_vol, trn_sel_img.nii_file.affine,
                                                 trn_sel_img.nii_file.header)
        trn_sel_img_scaled_out.to_filename(trn_sel_img_scaling_path)
        trn_sel_img = BrainImage(trn_sel_img_scaling_path, None)

    trn_vox_size_avg = np.mean(trn_vox_size_lst, 0)
    tst_vox_size = test_ref_image.nii_file.header.get_zooms()
    if np.size(tst_vox_size) == 4:
        tst_vox_size = tst_vox_size[:3]
    print(trn_vox_size_avg)
    print(tst_vox_size)
    vox_resampled = []
    for trn_vox_avg, tst_vox in zip(trn_vox_size_avg, tst_vox_size):
        if tst_vox * 0.7 > trn_vox_avg or tst_vox * 2.0 < trn_vox_avg:
            vox_resampled.append(trn_vox_avg)
            is_res_diff = True
        else:
            vox_resampled.append(tst_vox)
    if is_res_diff is True:
        for idx in modality_idx:
            test_img_resampled_path = os.path.join(output_dir,
                                                   img_resampled_name_pattern[idx].format(test_patient_id))
            if not os.path.exists(test_img_resampled_path):
                test_img_path = test_fname_modality_lst[idx]
                print('resampling %s (%s) to %s' % (test_img_path, str(tst_vox_size), str(vox_resampled)))
                test_image = BrainImage(test_img_path, None)
                test_image.ResampleTo(test_img_resampled_path, new_affine=None, new_shape=None,
                                      new_voxel_size=[vox_resampled[0], vox_resampled[1], vox_resampled[2]],
                                      is_ant_resample=False)

        test_ref_img_path = os.path.join(output_dir,
                                         img_resampled_name_pattern[0].format(test_patient_id))
        test_ref_image = BrainImage(test_ref_img_path, None)

        sel_trn_reg2_tst_image_name = train_patient_lst[trn_sel_ind] + '_reg2_' + test_patient_id + '_' + \
                                      _remove_ending(img_resampled_name_pattern[0].format(test_patient_id), '.nii.gz')
        roi_pos_scaled = np.multiply(roi_pos, [tst_vox_size[0] / vox_resampled[0], tst_vox_size[1] / vox_resampled[1],
                                        tst_vox_size[2] / vox_resampled[2]])

        roi_pos = []
        for roi_pos_scaled_elem in roi_pos_scaled:
            roi_pos_scaled_int = [int(i) for i in roi_pos_scaled_elem]
            roi_pos.append(roi_pos_scaled_int)

    else:
        sel_trn_reg2_tst_image_name = train_patient_lst[trn_sel_ind] + '_reg2_' + test_patient_id + '_' + \
                                      _remove_ending(img_pattern, '.nii.gz')

    transform_name = sel_trn_reg2_tst_image_name + '_' + 'composite_rigid_affine.txt'

    if is_reg:
        init_reg_dir = os.path.join(output_dir, 'init_reg')
        if not os.path.exists(init_reg_dir):
            os.makedirs(init_reg_dir)
        composite_transform_path = os.path.join(init_reg_dir, transform_name)
        if not os.path.exists(composite_transform_path):
            # registration
            print("composite registration of a reference image of %s to a test image of %s" %
                  (train_patient_lst[trn_sel_ind], test_patient_id))

            trn_sel_img.composite_rigid_affine_registration_v2(init_reg_dir,
                                                               img_pattern,
                                                               train_patient_lst[trn_sel_ind],
                                                               test_patient_id,
                                                               test_ref_image,
                                                               sel_trn_reg2_tst_image_name,
                                                               do_translation_first=True)

    # load label images in train_patient_lst and setting roi
    init_mask_vol_lst = []
    init_mask_filepath_lst = []
    for side, idx in zip(['left', 'right'], [0, 1]):  # left + right
        if is_reg:
            label_filepath = label_fname_lst[trn_sel_ind][idx]
            print(label_filepath)
             # scaling of training labels
            if (np.linalg.norm(scaling_ratio) < 2 / 3 or np.linalg.norm(scaling_ratio) > 1.5) and is_scaling:
                label_img = BrainImage(label_filepath, None)
                label_img_scaled_vol = label_img.rescale(scaling_ratio)
                label_filepath = os.path.join(output_dir, 'training_labels',
                                              train_patient_lst[trn_sel_ind] + '_' + target + '_' + side +
                                              '_label_scaled.nii.gz')
                label_img_scaled_out = nib.Nifti1Image(label_img_scaled_vol, label_img.nii_file.affine,
                                                       label_img.nii_file.header)
                label_img_scaled_out.to_filename(label_filepath)

            init_mask_filepath = os.path.join(init_reg_dir, init_reg_mask_pattern.format(side, target))
            if not os.path.exists(init_mask_filepath):
                cmd = "Apply_Transform.sh %s %s %s %s" % (label_filepath, test_ref_img_path,
                                                          init_mask_filepath, composite_transform_path)
                try:
                    subprocess.check_call(['bash ./scripts/' + cmd], shell=True)
                except subprocess.CalledProcessError as e:
                    print(e.returncode)
                    print(e.output)

            if os.path.exists(init_mask_filepath):
                print(init_mask_filepath)
                init_mask_data = read_volume_data(init_mask_filepath)
                init_mask_vol = init_mask_data.get_data()
                if np.size(np.shape(init_mask_vol)) == 4:
                    init_mask_vol_lst.append(init_mask_vol[:, :, :, 0])
                else:
                    init_mask_vol_lst.append(init_mask_vol)
            init_mask_filepath_lst.append(init_mask_filepath)
        else:
            init_mask_filepath_lst.append('')

    if os.path.exists(init_mask_filepath_lst[0]) and os.path.exists(init_mask_filepath_lst[1]):
        init_mask_merge_vol = init_mask_vol_lst[0] + init_mask_vol_lst[1]
        init_mask_data = read_volume_data(init_mask_filepath_lst[0])
    else:
        init_mask_merge_vol = np.array([])
        init_mask_data = np.array([])

    return init_mask_merge_vol, init_mask_data, test_ref_img_path, is_res_diff, roi_pos