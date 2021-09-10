
import numpy as np
import os.path
from sklearn.model_selection import LeaveOneOut, KFold, StratifiedKFold, RepeatedKFold
from architectures.arch_creator import generate_model
from utils.callbacks import generate_callbacks, generate_output_filename
from utils.ioutils import read_volume, read_volume_data, save_patches, read_patches, read_dentate_interposed_dataset, \
    save_volume_dentate_interposed, read_tha_dataset, read_tha_dataset_unseen, save_volume_tha, save_volume_tha_unseen, \
    read_dentate_interposed_dataset_unseen, save_volume_dentate_interposed_unseen, save_intermediate_volume
from utils.image import preprocess_test_image, hist_match, normalize_image, standardize_set, hist_match_set, \
    standardize_volume, find_crop_mask, crop_image
from utils.reconstruction import reconstruct_volume, reconstruct_volume_modified
from utils.training_testing_utils import split_train_val, build_training_set, build_testing_set, build_training_set_4d
from utils.general_utils import pad_both_sides
from utils.mixup_generator import MixupGenerator
from utils.mathutils import compute_statistics, computeDice, dice, measure_cmd, measure_msd
from keras.preprocessing.image import ImageDataGenerator
from importance_sampling.training import ImportanceTraining
import glob


def run_evaluation_in_dataset(gen_conf, train_conf, test_conf, args):

    train_conf['approach'] = args.approach
    train_conf['dataset'] = args.dataset
    train_conf['data_augment'] = int(args.data_augment)
    train_conf['attention_loss'] = int(args.attention_loss)
    train_conf['overlap_penalty_loss'] = int(args.overlap_penalty_loss)
    loss = args.loss.split(',')
    if len(loss) == 1:
        train_conf['loss'] = loss[0]
    else:
        train_conf['loss'] = loss
    train_conf['exclusive_train'] = int(args.exclusive_train)
    train_conf['metric'] = args.metric
    train_conf['lamda'] = tuple([float(i) for i in args.lamda.split(',')])
    activation = args.activation.split(',')
    if len(activation) == 1:
        train_conf['activation'] = activation[0]
    else:
        train_conf['activation'] = activation
    train_conf['preprocess'] = int(args.preprocess_trn)
    train_conf['num_k_fold'] = int(args.num_k_fold)
    train_conf['batch_size'] = int(args.batch_size)
    train_conf['num_epochs'] = int(args.num_epochs)
    train_conf['patience'] = int(args.patience)
    train_conf['optimizer'] = args.optimizer
    train_conf['initial_lr'] = float(args.initial_lr)
    train_conf['is_set_random_seed'] = int(args.is_set_random_seed)
    if args.random_seed_num != 'None':
        train_conf['random_seed_num'] = int(args.random_seed_num)
    else:
        train_conf['random_seed_num'] = None
    train_conf['bg_discard_percentage'] = float(args.bg_discard_percentage)
    train_conf['importance_sampling'] = int(args.importance_spl)
    train_conf['oversampling'] = int(args.oversampling)
    train_conf['patch_shape'] = tuple([int(i) for i in args.trn_patch_size.split(',')])
    train_conf['output_shape'] = tuple([int(i) for i in args.trn_output_size.split(',')])
    train_conf['extraction_step'] = tuple([int(i) for i in args.trn_step_size.split(',')])
    train_conf['continue_tr'] = int(args.continue_tr)
    train_conf['is_new_trn_label'] = int(args.is_new_trn_label)
    train_conf['new_label_path'] = args.new_label_path
    test_conf['preprocess'] = int(args.preprocess_tst)
    test_conf['patch_shape'] = tuple([int(i) for i in args.tst_patch_size.split(',')])
    test_conf['output_shape'] = tuple([int(i) for i in args.tst_output_size.split(',')])
    test_conf['extraction_step'] = tuple([int(i) for i in args.tst_step_size.split(',')])
    test_conf['threshold'] = float(args.threshold)
    test_conf['is_measure'] = int(args.is_measure)
    test_conf['is_unseen_case'] = int(args.is_unseen_case)
    test_conf['trn_set_num'] = int(args.trn_set_num)
    test_conf['is_reg'] = int(args.is_reg)
    test_conf['roi_pos'] = tuple([int(i) for i in args.roi_pos.split(',')])
    test_conf['test_subject_id'] = [i for i in args.test_subject_id.split(',')]
    test_conf['dataset'] = args.dataset
    gen_conf['args'] = args
    gen_conf['validation_mode'] = args.mode
    num_classes = tuple([int(i) for i in args.num_classes.split(',')])
    if len(num_classes) == 1:
        gen_conf['num_classes'] = num_classes[0]
    else:
        gen_conf['num_classes'] = num_classes
    gen_conf['multi_output'] = int(args.multi_output)
    output_name = args.output_name.split(',')
    if len(num_classes) == 1:
        gen_conf['output_name'] = output_name[0]
    else:
        gen_conf['output_name'] = output_name
    gen_conf['root_path'] = args.root_path
    gen_conf['dataset_path'] = args.dataset_path
    gen_conf['dataset_info'][train_conf['dataset']]['folder_names'] = args.folder_names.split(',')
    gen_conf['dataset_info'][train_conf['dataset']]['margin_crop_mask'] = \
        tuple([int(i) for i in args.crop_margin.split(',')])
    gen_conf['dataset_info'][train_conf['dataset']]['image_modality'] = args.image_modality.split(',')
    target = args.target.split(',')
    if len(target) == 1:
        gen_conf['dataset_info'][train_conf['dataset']]['target'] = target[0]
    else:
        gen_conf['dataset_info'][train_conf['dataset']]['target'] = target
    exclude_label_num = []
    for i in args.exclude_label_num.split(','):
        if not i == '':
            exclude_label_num.append(int(i))
        else:
            exclude_label_num.append([])
    if len(exclude_label_num) == 1:
        gen_conf['dataset_info'][train_conf['dataset']]['exclude_label_num'] = exclude_label_num[0]
    else:
        gen_conf['dataset_info'][train_conf['dataset']]['exclude_label_num'] = exclude_label_num

    if train_conf['dataset'] in ['dcn_seg_dl']:
        return evaluate_dentate_interposed_seg(gen_conf, train_conf, test_conf)
    elif train_conf['dataset'] in ['tha_seg_dl']:
        return evaluate_tha_seg(gen_conf, train_conf, test_conf)
    else:
        assert True, "error: invalid dataset"

def evaluate_tha_seg(gen_conf, train_conf, test_conf):
    args = gen_conf['args']
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    root_path = gen_conf['root_path']
    results_path = gen_conf['results_path']
    mode = gen_conf['validation_mode']
    approach = train_conf['approach']
    loss = train_conf['loss']
    preprocess_trn = train_conf['preprocess']
    num_k_fold = train_conf['num_k_fold']
    num_epochs = train_conf['num_epochs']
    num_retrain_init = train_conf['num_retrain']
    preprocess_tst = test_conf['preprocess']
    is_unseen_case = test_conf['is_unseen_case']
    is_measure = test_conf['is_measure']
    trn_set_num = test_conf['trn_set_num']
    tst_set = test_conf['test_subject_id']
    folder_names = dataset_info['folder_names']
    modality = dataset_info['image_modality']
    patient_id = dataset_info['patient_id']
    target = dataset_info['target']

    num_data = len(patient_id)
    num_modality = len(modality)

    if mode is not '2':
        file_output_dir = os.path.join(root_path, results_path, dataset, folder_names[0])
    else:
        file_output_dir = os.path.join(root_path, results_path, 'seg_test', folder_names[0])
    if not os.path.exists(file_output_dir):
        os.makedirs(file_output_dir)

    if mode is '0': # k-fold cross validation
        kfold = KFold(num_k_fold, True, 1)  # StratifiedKFold, RepeatedKFold
        train_test_lst = kfold.split(range(num_data))
    elif mode is '1': # designated training and test set (if test set is None, training only)
        # patient id
        train_patient_lst = [['ET018', 'ET019', 'ET020', 'ET021', 'ET028', 'ET030', 'MS001', 'P030', 'P040', 'PD050',
                              'PD061', 'PD074', 'PD081', 'PD085', 'SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP104',
                              'SLEEP105', 'SLEEP106', 'SLEEP107', 'SLEEP108', 'SLEEP109', 'SLEEP110', 'SLEEP112',
                              'SLEEP113', 'SLEEP114', 'SLEEP115', 'SLEEP116', 'SLEEP117', 'SLEEP118', 'SLEEP119',
                              'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP127',
                              'SLEEP128', 'SLEEP130', 'SLEEP131', 'SLEEP133'],
                             ['ET018', 'ET019', 'ET020', 'ET021', 'ET028', 'ET030', 'MS001', 'P030', 'P040', 'PD050',
                              'PD061', 'PD074', 'PD081', 'PD085', 'SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP104',
                              'SLEEP105', 'SLEEP106', 'SLEEP107', 'SLEEP108', 'SLEEP109', 'SLEEP110', 'SLEEP112',
                              'SLEEP113', 'SLEEP114', 'SLEEP115', 'SLEEP116', 'SLEEP117', 'SLEEP118', 'SLEEP119',
                              'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP127',
                              'SLEEP128', 'SLEEP130', 'SLEEP131', 'SLEEP133']]
        test_patient_lst = [[None], [None]]
        train_test_lst = zip(train_patient_lst, test_patient_lst)  # set test_patient_lst as None for only training
    elif mode is '2': # test only # combine to mode 1
        # patient id

        if trn_set_num == 1:
            # 1st fold training data list
            train_patient_lst = [['ET018', 'ET019', 'ET028', 'ET030', 'MS001', 'P030', 'P040', 'PD050', 'PD061', 'PD074',
                                  'PD081', 'PD085', 'SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP105', 'SLEEP106', 'SLEEP107',
                                  'SLEEP108', 'SLEEP110', 'SLEEP113', 'SLEEP114', 'SLEEP115', 'SLEEP116', 'SLEEP117',
                                  'SLEEP119', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP126', 'SLEEP127', 'SLEEP128',
                                  'SLEEP131', 'SLEEP133']]
        elif trn_set_num == 2:
            # 2nd fold training data list
            train_patient_lst = [['ET018', 'ET019', 'ET020', 'ET021', 'ET030', 'MS001', 'P030', 'P040', 'PD050', 'PD061',
                                  'PD074', 'PD081', 'PD085', 'SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP105',
                                  'SLEEP107', 'SLEEP109', 'SLEEP110', 'SLEEP112', 'SLEEP113', 'SLEEP114', 'SLEEP115',
                                  'SLEEP116', 'SLEEP118', 'SLEEP120', 'SLEEP125', 'SLEEP126', 'SLEEP127', 'SLEEP130',
                                  'SLEEP131', 'SLEEP133']]
        elif trn_set_num == 3:
            # 3rd fold training data list
            train_patient_lst = [['ET018', 'ET019', 'ET020', 'ET021', 'ET028', 'ET030', 'MS001', 'P030', 'P040', 'PD050',
                                  'PD074', 'PD081', 'PD085', 'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP106', 'SLEEP108',
                                  'SLEEP109', 'SLEEP112', 'SLEEP113', 'SLEEP116', 'SLEEP117', 'SLEEP118', 'SLEEP119',
                                  'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP128',
                                  'SLEEP130', 'SLEEP131']]
        elif trn_set_num == 4:
            # 4th fold training data list
            train_patient_lst = [['ET018', 'ET020', 'ET021', 'ET028', 'ET030', 'P040', 'PD050', 'PD061', 'PD074', 'PD081',
                                  'SLEEP101', 'SLEEP102', 'SLEEP104', 'SLEEP105', 'SLEEP106', 'SLEEP107', 'SLEEP108',
                                  'SLEEP109', 'SLEEP110', 'SLEEP112', 'SLEEP114', 'SLEEP115', 'SLEEP117', 'SLEEP118',
                                  'SLEEP119', 'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP125', 'SLEEP126',
                                  'SLEEP127', 'SLEEP128', 'SLEEP130', 'SLEEP133']]
        else:
            # 5th fold training data list
            train_patient_lst = [['ET019', 'ET020', 'ET021', 'ET028', 'MS001', 'P030', 'PD061', 'PD085', 'SLEEP101',
                                  'SLEEP103', 'SLEEP104', 'SLEEP105', 'SLEEP106', 'SLEEP107', 'SLEEP108', 'SLEEP109',
                                  'SLEEP110', 'SLEEP112', 'SLEEP113', 'SLEEP114', 'SLEEP115', 'SLEEP116', 'SLEEP117',
                                  'SLEEP118', 'SLEEP119', 'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP125',
                                  'SLEEP127', 'SLEEP128', 'SLEEP130', 'SLEEP131', 'SLEEP133']]

        test_patient_lst = [tst_set]
        train_test_lst = zip(train_patient_lst, test_patient_lst)
    else:
        raise NotImplementedError('mode ' + mode + 'does not exist')

    k = 0
    for train_idx, test_idx in train_test_lst:
        if mode is '0': # k-fold cross validation
            train_patient_lst = [patient_id[i] for i in train_idx]
            test_patient_lst = [patient_id[i] for i in test_idx]

            if k in []:
                k += 1
                continue

            # set training set and test set to skip
            skip_train_lst = ['']
            for skip_patient_id in skip_train_lst:
                if skip_patient_id in train_patient_lst:
                    train_patient_lst.remove(skip_patient_id)

            skip_test_lst = ['']
            for skip_patient_id in skip_test_lst:
                if skip_patient_id in test_patient_lst:
                    test_patient_lst.remove(skip_patient_id)

            if len(test_patient_lst) == 0:
                k += 1
                continue
        else:

            if k in []:
                k += 1
                continue

            if mode is '2':
                k = trn_set_num - 1

            train_patient_lst = train_idx
            test_patient_lst = test_idx

        # prepare for the files to write
        f = prepare_files_to_write(train_patient_lst, test_patient_lst, file_output_dir, approach, loss, k, mode, args)

        # load training images, labels, and test images on the roi
        if is_unseen_case == 0:
            train_img_lst, label_lst, test_img_lst, train_fname_lst, label_fname_lst, test_fname_lst  = \
                read_tha_dataset(gen_conf, train_conf, train_patient_lst, test_patient_lst, preprocess_trn,
                                     preprocess_tst, file_output_dir)
            is_res_diff_lst = []
            for t in test_img_lst:
                is_res_diff_lst.append(None)
        else:
            is_scaling = False
            train_img_lst, label_lst, test_img_lst, train_fname_lst, label_fname_lst, test_fname_lst, is_res_diff_lst = \
                read_tha_dataset_unseen(gen_conf, train_conf, test_conf, train_patient_lst, test_patient_lst,
                                        preprocess_trn, preprocess_tst, file_output_dir, is_scaling)

        # train the model
        if mode is not '2':
            cur_epochs = -1
            metric_best = 0
            metric_thres = 0.024 # 0.976 for acc (# 0.975 for 32 patch size; retrain when it was early stopped), 0.024 for loss
            num_retrain = 0
            lim_epochs = 7

            while cur_epochs <= lim_epochs and num_epochs > cur_epochs and metric_best <= metric_thres:
                # (num_epochs > cur_epochs) and current epoch is less than 8 (originally 6)(by val_acc or val_acc_dc)
                if cur_epochs != -1:
                    num_retrain += 1
                    train_conf['num_retrain'] = num_retrain
                    train_conf['continue_tr'] = 1
                    lim_epochs = 5
                else:
                    train_conf['num_retrain'] = num_retrain_init
                    if train_conf['continue_tr'] == 1:
                        print('continuing training with the trained model...')
                model, cur_epochs, metric_best, mean, std = train_model(gen_conf, train_conf, train_img_lst, label_lst,
                                                                   train_fname_lst,label_fname_lst, k+1)
                print ('current epoch (total # of epochs): ' + str(cur_epochs) + '(' + str(num_epochs) + ')')

            if mode is '1' and test_patient_lst[0] is None: # training only
                k += 1
                continue
        else: # test only
            # read the trained model
            case_name = k + 1 # or a designated case: train_cases = [0,1], case_name = train_cases[k]
            model = read_model(gen_conf, train_conf, case_name) # model filename should have 'mode_2_'
            mean = []
            std = []

        # predict the test set
        for test_img, test_patient_id, test_fname, is_res_diff in zip(test_img_lst, test_patient_lst, test_fname_lst,
                                                                      is_res_diff_lst):
            print('#' + str(k + 1) + ': processing test_patient_id - ' + test_patient_id)

            # preprocess test image
            test_vol = preprocess_test_image(test_img, train_img_lst, num_modality, mean, std, preprocess_tst)

            # inference from the learned model
            rec_vol_crop, prob_vol_crop, test_patches = inference(gen_conf, train_conf, test_conf, test_vol, model)

            # uncrop and save the segmentation result

            if is_unseen_case == 0:
                save_volume_tha(gen_conf, train_conf, test_conf, rec_vol_crop, prob_vol_crop, test_fname,
                                test_patient_id, target, file_output_dir)
                # compute DC
                if is_measure == 1:
                    _ = measure_thalamus(gen_conf, train_conf, test_conf, test_patient_id, target, k, mode)
            else:
                save_volume_tha_unseen(gen_conf, train_conf, test_conf, rec_vol_crop, prob_vol_crop, test_fname,
                                   test_patient_id, target, file_output_dir, is_res_diff)

            # processing smaller size
            # save_volume_tha_v2(gen_conf, train_conf, test_conf, rec_vol_crop, prob_vol_crop, test_fname,
            #                     test_patient_id, target[0], file_output_dir)

            del test_patches

        k += 1
        f.close()

    return True


def evaluate_dentate_interposed_seg(gen_conf, train_conf, test_conf):
    args = gen_conf['args']
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    root_path = gen_conf['root_path']
    results_path = gen_conf['results_path']
    mode = gen_conf['validation_mode']
    multi_output = gen_conf['multi_output']
    approach = train_conf['approach']
    loss = train_conf['loss']
    preprocess_trn = train_conf['preprocess']
    num_k_fold = train_conf['num_k_fold']
    num_epochs = train_conf['num_epochs']
    num_retrain_init = train_conf['num_retrain']
    preprocess_tst = test_conf['preprocess']
    is_unseen_case = test_conf['is_unseen_case']
    is_measure = test_conf['is_measure']
    trn_set_num = test_conf['trn_set_num']
    tst_set = test_conf['test_subject_id']

    target = dataset_info['target']
    folder_names = dataset_info['folder_names']
    modality = dataset_info['image_modality']
    patient_id = dataset_info['patient_id']

    is_new_trn_label = train_conf['is_new_trn_label']

    num_data = len(patient_id)
    num_modality = len(modality)

    if mode is not '2':
        file_output_dir = os.path.join(root_path, results_path, dataset, folder_names[0])
    else:
        file_output_dir = os.path.join(root_path, results_path, 'seg_test', folder_names[0])
    if not os.path.exists(file_output_dir):
        os.makedirs(file_output_dir)

    if mode is '0': # k-fold cross validation
        kfold = KFold(num_k_fold, True, 1)  # StratifiedKFold, RepeatedKFold
        train_test_lst = kfold.split(range(num_data))
    elif mode is '1': # designated training and test set (if test set is None, training only)
        # patient id
        # train_patient_lst = [['PD081', 'SLEEP101', 'SLEEP105', 'SLEEP108', 'SLEEP109', 'SLEEP110', 'SLEEP113',
        #                       'SLEEP114', 'SLEEP115', 'SLEEP117', 'SLEEP118', 'SLEEP130', 'SLEEP131', 'SLEEP133',
        #                       'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP107', 'SLEEP112', 'SLEEP116', 'SLEEP119',
        #                       'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP127',
        #                       'SLEEP128']]
        # train_patient_lst = [['SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP107', 'SLEEP112', 'SLEEP116', 'SLEEP119',
        #                       'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP127',
        #                       'SLEEP128']]

        # selected 31 cases out of 42 cases (for pre-training)
        train_patient_lst = [['C052', 'C053', 'C056', 'C057', 'C060', 'P042', 'P043', 'PD055', 'PD060', 'PD061',
                             'PD062', 'PD063', 'PD064', 'PD069', 'PD074', 'PD078', 'PD079', 'PD080', 'SLEEP106',
                             'SLEEP134', 'SLEEP135', 'SLEEP136', 'SLEEP137', 'SLEEP138', 'SLEEP139', 'SLEEP140',
                             'SLEEP141', 'SLEEP142', 'SLEEP143', 'SLEEP144', 'SLEEP145']]

        #selected 32 cases out of 42 cases (for pre-training)
        # train_patient_lst = [['C052', 'C053', 'C056', 'C057', 'C058', 'C060', 'P042', 'P043', 'PD055', 'PD060', 'PD061',
        #                      'PD062', 'PD063', 'PD064', 'PD069', 'PD074', 'PD078', 'PD079', 'PD080', 'SLEEP106',
        #                      'SLEEP134', 'SLEEP135', 'SLEEP136', 'SLEEP137', 'SLEEP138', 'SLEEP139', 'SLEEP140',
        #                      'SLEEP141', 'SLEEP142', 'SLEEP143', 'SLEEP144', 'SLEEP145']]

        # original 42 cases
        # test_patient_lst = [['C050', 'C052', 'C053', 'C056', 'C057', 'C058', 'C059', 'C060', 'P030', 'P032', 'P035',
        #                       'P038', 'P040', 'P042', 'P043', 'P045', 'PD053', 'PD055', 'PD060', 'PD061', 'PD062',
        #                       'PD063','PD064', 'PD065', 'PD069', 'PD074', 'PD078', 'PD079', 'PD080', 'SLEEP106',
        #                       'SLEEP134', 'SLEEP135', 'SLEEP136', 'SLEEP137', 'SLEEP138', 'SLEEP139', 'SLEEP140',
        #                       'SLEEP141', 'SLEEP142', 'SLEEP143', 'SLEEP144', 'SLEEP145']]

        # 29 test cases
        test_patient_lst = [['PD081', 'SLEEP101', 'SLEEP105', 'SLEEP108', 'SLEEP109', 'SLEEP110', 'SLEEP113',
                              'SLEEP114', 'SLEEP115', 'SLEEP117', 'SLEEP118', 'SLEEP130', 'SLEEP131', 'SLEEP133',
                              'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP107', 'SLEEP112', 'SLEEP116', 'SLEEP119',
                              'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP127',
                              'SLEEP128']]

        train_test_lst = zip(train_patient_lst, test_patient_lst)  # set test_patient_lst as None for only training
    elif mode is '2': # test only # combine to mode 1
        # patient id
        if trn_set_num == 1:
            #1st fold training data list
            train_patient_lst = [['PD081', 'SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP105', 'SLEEP107',
                                  'SLEEP108', 'SLEEP109', 'SLEEP110', 'SLEEP112', 'SLEEP113', 'SLEEP114', 'SLEEP115',
                                  'SLEEP117', 'SLEEP118', 'SLEEP119', 'SLEEP125', 'SLEEP126', 'SLEEP127', 'SLEEP130',
                                  'SLEEP131', 'SLEEP133']]
        elif trn_set_num == 2:
            # 2nd fold training data list
            train_patient_lst = [['PD081', 'SLEEP101', 'SLEEP105', 'SLEEP107', 'SLEEP108', 'SLEEP109', 'SLEEP110',
                                  'SLEEP113', 'SLEEP114', 'SLEEP115', 'SLEEP116', 'SLEEP117', 'SLEEP118', 'SLEEP119',
                                  'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP127', 'SLEEP128', 'SLEEP130',
                                  'SLEEP131', 'SLEEP133']]
        elif trn_set_num == 3:
            # 3rd fold training data list
            train_patient_lst = [['PD081', 'SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP105', 'SLEEP109',
                                  'SLEEP110', 'SLEEP112', 'SLEEP113', 'SLEEP114', 'SLEEP116', 'SLEEP117', 'SLEEP118',
                                  'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP128',
                                  'SLEEP131', 'SLEEP133']]
        elif trn_set_num == 4:
            # 4th fold training data list
            train_patient_lst = [['SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP105', 'SLEEP107', 'SLEEP108', 'SLEEP109',
                                  'SLEEP110', 'SLEEP112', 'SLEEP113', 'SLEEP114', 'SLEEP115', 'SLEEP116', 'SLEEP119',
                                  'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP127',
                                  'SLEEP128', 'SLEEP130']]
        elif trn_set_num == 5:
            # 5th fold training data list
            train_patient_lst = [['PD081', 'SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP107', 'SLEEP108',
                                  'SLEEP112', 'SLEEP115', 'SLEEP116', 'SLEEP117', 'SLEEP118', 'SLEEP119', 'SLEEP120',
                                  'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP127', 'SLEEP128',
                                  'SLEEP130', 'SLEEP131', 'SLEEP133']]
        elif trn_set_num == 6:
            # 29 patients with manual labels
            train_patient_lst = [['PD081', 'SLEEP101', 'SLEEP105', 'SLEEP108', 'SLEEP109', 'SLEEP110', 'SLEEP113',
                                 'SLEEP114', 'SLEEP115', 'SLEEP117', 'SLEEP118', 'SLEEP130', 'SLEEP131', 'SLEEP133',
                                 'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP107', 'SLEEP112', 'SLEEP116', 'SLEEP119',
                                 'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP127',
                                 'SLEEP128']]
        else:
            raise NotImplementedError('trn_set_num ' + trn_set_num + 'does not exist')

        test_patient_lst = [tst_set]
        train_test_lst = zip(train_patient_lst, test_patient_lst)
    else:
        raise NotImplementedError('mode ' + mode + 'does not exist')

    k = 0
    for train_idx, test_idx in train_test_lst:

        if mode is '0': # k-fold cross validation
            if is_new_trn_label == 3:
                new_trn_label = ['C052', 'C053', 'C056', 'C057', 'C060', 'P042', 'P043', 'PD055', 'PD060', 'PD061',
                                 'PD062', 'PD063', 'PD064', 'PD069', 'PD074', 'PD078', 'PD079', 'PD080', 'SLEEP106',
                                 'SLEEP134', 'SLEEP135', 'SLEEP136', 'SLEEP137', 'SLEEP138', 'SLEEP139', 'SLEEP140',
                                 'SLEEP141', 'SLEEP142', 'SLEEP143', 'SLEEP144', 'SLEEP145']
                train_patient_lst = [patient_id[i] for i in train_idx] + new_trn_label
            else:
                train_patient_lst = [patient_id[i] for i in train_idx]
            test_patient_lst = [patient_id[i] for i in test_idx]

            if k in []:
                k += 1
                continue

            # set training set and test set to skip
            skip_train_lst = [''] #['C052', 'C053', 'C056', 'PD062', 'PD063', 'PD078', 'PD079', 'PD080', 'PD081']
            for skip_patient_id in skip_train_lst:
                if skip_patient_id in train_patient_lst:
                    train_patient_lst.remove(skip_patient_id)

            skip_test_lst = ['']
            for skip_patient_id in skip_test_lst:
                if skip_patient_id in test_patient_lst:
                    test_patient_lst.remove(skip_patient_id)

            if len(test_patient_lst) == 0:
                k += 1
                continue
        else:

            if k in []:
                k += 1
                continue

            if mode is '2':
                k = trn_set_num - 1

            train_patient_lst = train_idx
            test_patient_lst = test_idx

        # prepare for the files to write
        f = prepare_files_to_write_dcn(train_patient_lst, test_patient_lst, file_output_dir, approach, loss, k, mode,
                                       args, target, multi_output)

        # load training images, labels, and test images on the roi
        if is_unseen_case == 0:
            train_img_lst, label_lst, test_img_lst, train_fname_lst, label_fname_lst, test_fname_lst  = \
                read_dentate_interposed_dataset(gen_conf, train_conf, train_patient_lst, test_patient_lst, preprocess_trn,
                                     preprocess_tst, file_output_dir, target)
            is_res_diff_lst = []
            for t in test_img_lst:
                is_res_diff_lst.append(None)
        else:
            is_scaling = False
            train_img_lst, label_lst, test_img_lst, train_fname_lst, label_fname_lst, test_fname_lst, is_res_diff_lst  = \
                read_dentate_interposed_dataset_unseen(gen_conf, train_conf, test_conf, train_patient_lst,
                                                       test_patient_lst, preprocess_trn, preprocess_tst,
                                                       file_output_dir, target, is_scaling)

        # train the model
        if mode is not '2':
            cur_epochs = -1
            metric_best = 0
            metric_thres = 0.997  # 0.992 for 16 patch size # 0.997 for 32 patch size; retrain when it was early stopped # 0.024 for loss
            num_retrain = 0
            lim_epochs = 7

            while cur_epochs <= lim_epochs and num_epochs > cur_epochs and metric_best <= metric_thres:
                # (num_epochs > cur_epochs) and current epoch is less than 8 (originally 6) (by val_acc or val_acc_dc)
                if cur_epochs != -1:
                    num_retrain += 1
                    train_conf['num_retrain'] = num_retrain
                    train_conf['continue_tr'] = 1
                    lim_epochs = 5
                else:
                    train_conf['num_retrain'] = num_retrain_init
                    if train_conf['continue_tr'] == 1:
                        print('continuing training with the trained model...')
                model, cur_epochs, metric_best, mean, std = train_model(gen_conf, train_conf, train_img_lst, label_lst,
                                                                   train_fname_lst, label_fname_lst, k+1)
                print('current epoch (total # of epochs): ' + str(cur_epochs) + '(' + str(num_epochs) + ')')

            if mode is '1' and test_patient_lst[0] is None: # training only
                k += 1
                continue
        else: # test only
            # read the trained model
            case_name = k + 1 # or a designated case: train_cases = [0,1], case_name = train_cases[k]
            model = read_model(gen_conf, train_conf, case_name) # model filename should have 'mode_2_'
            mean = []
            std = []

        # predict the test set
        for test_img, test_patient_id, test_fname, is_res_diff in zip(test_img_lst, test_patient_lst, test_fname_lst,
                                                                      is_res_diff_lst):
            print('#' + str(k + 1) + ': processing test_patient_id - ' + test_patient_id)

            # preprocess test image
            test_vol = preprocess_test_image(test_img, train_img_lst, num_modality, mean, std, preprocess_tst)

            # inference from the learned model
            rec_vol_crop, prob_vol_crop, test_patches = inference(gen_conf, train_conf, test_conf, test_vol, model)

            # uncrop and save the segmentation result
            if is_unseen_case == 0:
                save_volume_dentate_interposed(gen_conf, train_conf, test_conf, rec_vol_crop, prob_vol_crop, test_fname,
                                    test_patient_id, file_output_dir, target)
                # compute DC
                if is_measure == 1:
                    _ = measure_dentate_interposed(gen_conf, train_conf, test_conf, test_patient_id, k, mode, target)

            else:
                save_volume_dentate_interposed_unseen(gen_conf, train_conf, test_conf, rec_vol_crop, prob_vol_crop, test_fname,
                                    test_patient_id, file_output_dir, target, is_res_diff)

            del test_patches

        k += 1
        f.close()

    return True


def prepare_files_to_write(train_patient_lst, test_patient_lst, file_output_dir, approach, loss, k, mode, args):

    measure_pkl_filepath = file_output_dir + '/' + 'mode_' + mode + '_#' + str(k + 1) + '_' + 'measurement_' + \
                               approach + '_' + loss + '.pkl'

    if os.path.exists(measure_pkl_filepath):
        os.remove(measure_pkl_filepath)

    measure_pkl_filepath_staple = file_output_dir + '/' + 'mode_' + mode + '_#' + str(k + 1) + '_' + 'measurement_' + \
                                  'staple' + '_' + loss + '.pkl'
    if os.path.exists(measure_pkl_filepath_staple):
        os.remove(measure_pkl_filepath_staple)

    k_fold_mode_filepath = file_output_dir + '/' + 'mode_' + mode + '_#'+ str(k + 1) + '_' + \
                           'training_test_patient_id_' + approach + '_' + loss + '.txt'
    if os.path.exists(k_fold_mode_filepath):
        os.remove(k_fold_mode_filepath)

    k_fold_patient_list = '#' + str(k + 1) + '\ntrain sets: %s \ntest sets: %s \nparameter sets: %s' \
                          % (train_patient_lst, test_patient_lst, args)
    print(k_fold_patient_list)

    with open(k_fold_mode_filepath, 'a') as f:
        f.write('\n' + k_fold_patient_list)

    failed_cases_filepath = os.path.join(file_output_dir, 'failed_cases.txt')
    if os.path.exists(failed_cases_filepath):
        os.remove(failed_cases_filepath)

    return f



def prepare_files_to_write_dcn(train_patient_lst, test_patient_lst, file_output_dir, approach, loss, k, mode, args,
                               target, multi_output):

    if multi_output == 1:
        loss=loss[0]

    for seg_label in target:
        measure_pkl_filepath = file_output_dir + '/' + 'mode_' + mode + '_#' + str(k + 1) + '_' + 'measurement_' + \
                               approach + '_' + loss + '_' + seg_label + '_seg.pkl'

        if os.path.exists(measure_pkl_filepath):
            os.remove(measure_pkl_filepath)

    k_fold_mode_filepath = file_output_dir + '/' + 'mode_' + mode + '_#'+ str(k + 1) + '_' + \
                           'training_test_patient_id_' + approach + '_' + loss + '.txt'
    if os.path.exists(k_fold_mode_filepath):
        os.remove(k_fold_mode_filepath)

    k_fold_patient_list = '#' + str(k + 1) + '\ntrain sets: %s \ntest sets: %s \nparameter sets: %s' \
                          % (train_patient_lst, test_patient_lst, args)
    print(k_fold_patient_list)

    with open(k_fold_mode_filepath, 'a') as f:
        f.write('\n' + k_fold_patient_list)

    failed_cases_filepath = os.path.join(file_output_dir, 'failed_cases.txt')
    if os.path.exists(failed_cases_filepath):
        os.remove(failed_cases_filepath)

    return f


def train_model(
    gen_conf, train_conf, input_data, labels, data_filename_ext_list, label_filename_ext_list, case_name):
    dataset_info = gen_conf['dataset_info'][train_conf['dataset']]
    time_series = dataset_info['time_series']
    folder_names = dataset_info['folder_names']
    preprocess = train_conf['preprocess']
    use_saved_patches = train_conf['use_saved_patches']
    num_epochs = train_conf['num_epochs']
    modality = dataset_info['image_modality']
    num_modality = len(modality)

    mean = []
    std = []
    cur_epochs = 0
    metric_best = 0
    if num_epochs > 0:

        train_index, val_index = split_train_val(range(len(input_data)), train_conf['validation_split'])

        if preprocess == 1 or preprocess == 3:
            mean, std = compute_statistics(input_data, num_modality)
            input_data = standardize_set(input_data, num_modality, mean, std)
        elif preprocess == 4:
            ref_num = 0
            mean, std = compute_statistics(input_data, num_modality)
            ref_training_vol = input_data[ref_num]
            input_data = hist_match_set(input_data, ref_training_vol, num_modality)
            input_data = standardize_set(input_data, num_modality, mean, std)
            for idx in range(len(input_data)):
                save_intermediate_volume(gen_conf, train_conf, input_data[idx], idx + 1, [],
                                         'train_data_hist_matched_standardized')
        elif preprocess == 5:
            ref_num = 0
            ref_training_vol = input_data[ref_num]
            input_data[0] = hist_match_set(input_data, ref_training_vol, num_modality)
            for idx in range(len(input_data)):
                save_intermediate_volume(gen_conf, train_conf, input_data[idx], idx + 1, [],
                                         'train_data_hist_matched')

        train_img_list = [input_data[i] for i in train_index]
        train_label_list = [labels[i] for i in train_index]
        val_img_list = [input_data[i] for i in val_index]
        val_label_list = [labels[i] for i in val_index]

        if use_saved_patches is True:
            train_data = read_patches(gen_conf, train_conf, case_name)
        else:
            train_data = []

        if train_data == []:
            print('Building training samples (patches)...')
            if time_series is False:
                x_train, y_train = build_training_set(
                    gen_conf, train_conf, train_img_list, train_label_list)
                x_val, y_val = build_training_set(
                    gen_conf, train_conf, val_img_list, val_label_list)
            else: # if data is time series
                x_train, y_train = build_training_set_4d(
                    gen_conf, train_conf, train_img_list, train_label_list)
                x_val, y_val = build_training_set_4d(
                    gen_conf, train_conf, val_img_list, val_label_list)

            train_data = [x_train, y_train, x_val, y_val]
            if use_saved_patches is True:
                save_patches(gen_conf, train_conf, train_data, case_name)
                print('Saved training samples (patches)')
        else:
            x_train = train_data[0]
            y_train = train_data[1]
            x_val = train_data[2]
            y_val = train_data[3]
            print('Loaded training samples (patches')

        callbacks = generate_callbacks(gen_conf, train_conf, case_name)
        cur_epochs, metric_best = __train_model(gen_conf, train_conf, x_train, y_train, x_val, y_val, callbacks)

    model = read_model(gen_conf, train_conf, case_name)

    return model, cur_epochs, metric_best, mean, std


def __train_model(gen_conf, train_conf, x_train, y_train, x_val, y_val, callbacks):
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    num_classes = gen_conf['num_classes']
    multi_output = gen_conf['multi_output']
    output_name = gen_conf['output_name']
    data_augment = train_conf['data_augment']
    is_continue = train_conf['continue_tr']
    is_shuffle = train_conf['shuffle']
    metric_opt = train_conf['metric']
    batch_size = train_conf['batch_size'] # default
    importance_spl = train_conf['importance_sampling']
    is_oversampling = train_conf['oversampling']
    attention_loss = train_conf['attention_loss']
    overlap_penalty_loss = train_conf['overlap_penalty_loss']
    exclusive_train = train_conf['exclusive_train']
    exclude_label_num = dataset_info['exclude_label_num']

    if exclusive_train == 1 and multi_output == 0:
        y_train = np.delete(y_train, exclude_label_num[0], 2)
        y_val = np.delete(y_val, exclude_label_num[0], 2)
    elif exclusive_train == 0 and multi_output == 1:
        # fill interposed labels and remove interposed labels for output 1
        y_train_1 = np.delete(y_train, exclude_label_num[0], 2)
        y_val_1 = np.delete(y_val, exclude_label_num[0], 2)

        if exclude_label_num[0] == 2:
            y_train_bg_interposed_one_hot = y_train[:, :, 0] + y_train[:, :, 2]
            y_train_bg_interposed_one_hot[y_train_bg_interposed_one_hot == np.max(y_train_bg_interposed_one_hot)] = 1
            y_train_1[:, :, 0] = y_train_bg_interposed_one_hot

            y_val_bg_interposed_one_hot = y_val[:, :, 0] + y_val[:, :, 2]
            y_val_bg_interposed_one_hot[y_val_bg_interposed_one_hot == np.max(y_val_bg_interposed_one_hot)] = 1
            y_val_1[:, :, 0] = y_val_bg_interposed_one_hot

        # fill interposed labels and remove interposed labels for output 2
        y_train_2 = np.delete(y_train, exclude_label_num[1], 2)
        y_val_2 = np.delete(y_val, exclude_label_num[1], 2)

        if exclude_label_num[1] == 1:
            y_train_bg_dentate_one_hot = y_train[:, :, 0] + y_train[:, :, 1]
            y_train_bg_dentate_one_hot[y_train_bg_dentate_one_hot == np.max(y_train_bg_dentate_one_hot)] = 1
            y_train_2[:, :, 0] = y_train_bg_dentate_one_hot

            y_val_bg_dentate_one_hot = y_val[:, :, 0] + y_val[:, :, 1]
            y_val_bg_dentate_one_hot[y_val_bg_dentate_one_hot == np.max(y_val_bg_dentate_one_hot)] = 1
            y_val_2[:, :, 0] = y_val_bg_dentate_one_hot

        if attention_loss == 1:
            # only labels: dentate + interposed
            y_train_3 = np.delete(y_train, 2, 2)
            y_val_3 = np.delete(y_val, 2, 2)

            y_train_dentate_interposed_one_hot = y_train[:, :, 1] + y_train[:, :, 2]
            y_train_dentate_interposed_one_hot[y_train_dentate_interposed_one_hot ==
                                               np.max(y_train_dentate_interposed_one_hot)] = 1
            y_train_3[:, :, 1] = y_train_dentate_interposed_one_hot

            y_val_dentate_interposed_one_hot = y_val[:, :, 1] + y_val[:, :, 2]
            y_val_dentate_interposed_one_hot[y_val_dentate_interposed_one_hot ==
                                               np.max(y_val_dentate_interposed_one_hot)] = 1
            y_val_3[:, :, 1] = y_val_dentate_interposed_one_hot

            if overlap_penalty_loss == 1:
                y_train_bg_one = y_train[:, :, 0]
                y_train_bg_one[y_train_bg_one == 0] = 1

                y_val_bg_one = y_val[:, :, 0]
                y_val_bg_one[y_val_bg_one == 0] = 1

                print (y_train_bg_one.shape)
                print(y_val_bg_one.shape)

                y_train = {
                    output_name[0]: y_train_1,
                    output_name[1]: y_train_2,
                    'attention_maps': y_train_3,
                    'overlap_dentate_interposed': y_train_bg_one
                }
                y_val = {
                    output_name[0]: y_val_1,
                    output_name[1]: y_val_2,
                    'attention_maps': y_val_3,
                    'overlap_dentate_interposed': y_val_bg_one
                }
            else:
                y_train = {
                    output_name[0]: y_train_1,
                    output_name[1]: y_train_2,
                    'attention_maps': y_train_3
                }
                y_val = {
                    output_name[0]: y_val_1,
                    output_name[1]: y_val_2,
                    'attention_maps': y_val_3
                }
        else:
            if overlap_penalty_loss == 1:
                y_train_bg_one = y_train[:, :, 0]
                y_train_bg_one[y_train_bg_one == 0] = 1

                y_val_bg_one = y_val[:, :, 0]
                y_val_bg_one[y_val_bg_one == 0] = 1

                print (y_train_bg_one.shape)
                print(y_val_bg_one.shape)

                y_train = {
                    output_name[0]: y_train_1,
                    output_name[1]: y_train_2,
                    'overlap_dentate_interposed': y_train_bg_one
                }
                y_val = {
                    output_name[0]: y_val_1,
                    output_name[1]: y_val_2,
                    'overlap_dentate_interposed': y_val_bg_one
                }
            else:
                y_train = {
                    output_name[0]: y_train_1,
                    output_name[1]: y_train_2
                }
                y_val = {
                    output_name[0]: y_val_1,
                    output_name[1]: y_val_2
                }
    elif exclusive_train == 1 and multi_output == 1:
        print('In multi_output option, exclusive_train should be off')
        exit()
    else:
        if attention_loss == 1:
            # only labels: dentate + interposed
            y_train_3 = np.delete(y_train, 2, 2)
            y_val_3 = np.delete(y_val, 2, 2)

            y_train_dentate_interposed_one_hot = y_train[:, :, 1] + y_train[:, :, 2]
            y_train_dentate_interposed_one_hot[y_train_dentate_interposed_one_hot ==
                                               np.max(y_train_dentate_interposed_one_hot)] = 1
            y_train_3[:, :, 1] = y_train_dentate_interposed_one_hot

            y_val_dentate_interposed_one_hot = y_val[:, :, 1] + y_val[:, :, 2]
            y_val_dentate_interposed_one_hot[y_val_dentate_interposed_one_hot ==
                                             np.max(y_val_dentate_interposed_one_hot)] = 1
            y_val_3[:, :, 1] = y_val_dentate_interposed_one_hot

            y_train = {
                output_name: y_train,
                'attention_maps': y_train_3
            }
            y_val = {
                output_name: y_val,
                'attention_maps': y_val_3
            }

    print('Generating a model to be trained...')
    model = generate_model(gen_conf, train_conf)

    if is_continue == 1:
        model_filename = load_saved_model_filename(gen_conf, train_conf, 'pre_trained')
        if os.path.isfile(model_filename):
            #model = load_model(model_filename)
            print('Loading saved weights from a trained model (%s)' % model_filename)
            # If `by_name` is True, weights are loaded into layers only if they share the same name.
            # This is useful for fine-tuning or transfer-learning models where some of the layers have changed.
            #model.load_weights(model_filename, by_name=True)
            model.load_weights(model_filename)
        else:
            print('No Found a trained model in the path (%s). Newly starting training...' % model_filename)

    # computation to informative/important samples (by sampling mini-batches from a distribution other than uniform)
    # thus accelerating the convergence
    if importance_spl == 1:
        model = ImportanceTraining(model)

    if data_augment == 1: # mixup
        training_generator = MixupGenerator(x_train, y_train, batch_size, alpha=0.2)()
        model_fit = model.fit_generator(generator=training_generator,
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=train_conf['num_epochs'],
                            validation_data=(x_val, y_val),
                            verbose=train_conf['verbose'],
                            callbacks=callbacks, shuffle=is_shuffle)

    elif data_augment == 2: #datagen in keras
        datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            channel_shift_range=0.1,
            horizontal_flip=True)

        model_fit = model.fit_generator(datagen.flow(x_train, y_train, batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=train_conf['num_epochs'],
                            validation_data=(x_val, y_val),
                            verbose=train_conf['verbose'],
                            callbacks=callbacks, shuffle=is_shuffle)

    elif data_augment == 3: #mixup + datagen
        datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            channel_shift_range=0.1,
            horizontal_flip=True)

        training_generator = MixupGenerator(x_train, y_train, batch_size, alpha=0.2, datagen=datagen)()
        model_fit = model.fit_generator(generator=training_generator,
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=train_conf['num_epochs'],
                            validation_data=(x_val, y_val),
                            verbose=train_conf['verbose'],
                            callbacks=callbacks, shuffle=is_shuffle)

    else: # no data augmentation
        if is_oversampling == 1:
            from imblearn.keras import BalancedBatchGenerator
            from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE
            patch_shape = train_conf['patch_shape']
            #num_classes = gen_conf['num_classes']
            num_modality = len(gen_conf['dataset_info'][train_conf['dataset']]['image_modality'])

            print(x_train.shape)
            print(y_train.shape)

            sm = SMOTE()
            x_train_sampled, y_train_sampled = [], []
            x_train = x_train.reshape(x_train.shape[0], np.prod(patch_shape), num_modality)
            for i in range(x_train.shape[0]):
                x_train_tr, y_train_tr = sm.fit_resample(x_train[i], y_train[i])
                x_train_sampled.append(x_train_tr)
                y_train_sampled.append(y_train_tr)
                print(x_train_tr.shape)
                print(y_train_tr.shape)

            model_fit = model.fit(
                x_train_sampled, y_train_sampled, batch_size=batch_size,
                epochs=train_conf['num_epochs'],
                validation_data=(x_val, y_val),
                verbose=train_conf['verbose'],
                callbacks=callbacks, shuffle=is_shuffle)

        else:
            model_fit = model.fit(
                x_train, y_train, batch_size=batch_size,
                epochs=train_conf['num_epochs'],
                validation_data=(x_val, y_val),
                verbose=train_conf['verbose'],
                callbacks=callbacks, shuffle=is_shuffle)

    cur_epochs = len(model_fit.history['loss'])
    metric_best = None
    if multi_output == 1:
        if metric_opt in ['acc', 'acc_dc', 'loss']:
            metric_monitor = 'val_' + output_name[0] + '_' + metric_opt
        elif metric_opt == 'loss_total':
            metric_monitor = 'val_loss'
        else:
            print('unknown metric for early stopping')
            metric_monitor = None
        metric = model_fit.history[metric_monitor]
        if metric_opt in ['loss', 'loss_total']:
            metric_best = np.min(metric)
        else:
            metric_best = np.max(metric)
    else:
        if attention_loss == 1:
            if metric_opt in ['acc', 'acc_dc', 'loss']:
                metric_monitor = 'val_' + output_name + '_' + metric_opt
            elif metric_opt == 'loss_total':
                metric_monitor = 'val_loss'
            else:
                print('unknown metric for early stopping')
                metric_monitor = None
            metric = model_fit.history[metric_monitor]
            if metric_opt == ['loss', 'loss_total']:
                metric_best = np.min(metric)
            else:
                metric_best = np.max(metric)
        else:
            metric = model_fit.history['val_' + metric_opt]
            if metric_opt == 'loss':
                metric_best = np.min(metric)
            else:
                metric_best = np.max(metric)

    return cur_epochs, metric_best


def load_saved_model_filename(gen_conf, train_conf, case_name):
    root_path = gen_conf['root_path']
    model_path = gen_conf['model_path']
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    num_classes = gen_conf['num_classes']
    folder_names = dataset_info['folder_names']
    mode = gen_conf['validation_mode']
    multi_output = gen_conf['multi_output']
    approach = train_conf['approach']
    loss = train_conf['loss']
    dimension = train_conf['dimension']
    patch_shape = train_conf['patch_shape']
    extraction_step = train_conf['extraction_step']
    data_augment = train_conf['data_augment']
    preprocess_trn = train_conf['preprocess']

    if data_augment == 1:
        data_augment_label = 'mixup'
    elif data_augment == 2:
        data_augment_label = 'datagen'
    elif data_augment == 3:
        data_augment_label = 'mixup+datagen'
    else:
        data_augment_label = ''

    if mode is not '2':
        model_dir = root_path + model_path
        sub_dir = dataset + '/' + folder_names[0]
    else:
        model_dir = os.path.join(os.path.dirname(__file__), '../trained_models')
        sub_dir = dataset
    if multi_output == 1:
        loss = loss[0] + '_' + loss[1]
    model_filename = generate_output_filename(model_dir, sub_dir, 'mode_'+ mode, case_name, approach, loss,
                                              'dim_' + str(dimension), 'n_classes_' + str(num_classes),
                                              str(patch_shape), str(extraction_step), data_augment_label,
                                              'preproc_trn_opt_' + str(preprocess_trn), 'h5')

    return model_filename


def read_model(gen_conf, train_conf, case_name):

    model = generate_model(gen_conf, train_conf)
    model_filename = load_saved_model_filename(gen_conf, train_conf, case_name)
    model.load_weights(model_filename)

    return model


def test_model_modified(gen_conf, train_conf, test_conf, x_test, model):
    exclusive_train = train_conf['exclusive_train']
    attention_loss = train_conf['attention_loss']
    num_classes = gen_conf['num_classes']
    multi_output = gen_conf['multi_output']
    output_shape = test_conf['output_shape']

    if exclusive_train == 1:
        num_classes -= 1

    if multi_output == 1:
        pred = model.predict(x_test, verbose=1)
        pred_multi = []
        for pred_i, n_class in zip(pred[:len(num_classes)], num_classes):
            pred_multi.append(pred_i.reshape((len(pred_i),) + output_shape + (n_class,)))
        pred_recon = reconstruct_volume_modified(gen_conf, train_conf, test_conf, pred_multi, num_classes)
    else:
        if attention_loss == 1:
            pred = model.predict(x_test, verbose=1)
            pred = pred[0].reshape((len(pred[0]),) + output_shape + (num_classes,))
            pred_recon = reconstruct_volume_modified(gen_conf, train_conf, test_conf, pred, num_classes)
        else:
            pred = model.predict(x_test, verbose=1)
            pred = pred.reshape((len(pred),) + output_shape + (num_classes,))
            pred_recon = reconstruct_volume_modified(gen_conf, train_conf, test_conf, pred, num_classes)
    return pred_recon


def inference(gen_conf, train_conf, test_conf, test_vol, trained_model):
    dataset = test_conf['dataset']
    dimension = test_conf['dimension']
    patch_shape = test_conf['patch_shape']
    output_shape = test_conf['output_shape']
    extraction_step = test_conf['extraction_step']
    multi_output = gen_conf['multi_output']

    test_vol_org_size = test_vol[0].shape
    print(test_vol_org_size)

    pad_size_total = np.zeros(np.shape(output_shape)).astype(int)
    for dim in range(dimension):
        if test_vol_org_size[dim+1] < output_shape[dim]:
            pad_size_total[dim] = np.ceil((output_shape[dim] - test_vol_org_size[dim+1]) / 2)

    # add zeros with same size to both side if patch_shape != output_shape_real:
    if np.sum(pad_size_total) != 0:
        pad_size = ()
        for dim in range(dimension):
            pad_size += (pad_size_total[dim],)
        test_vol_pad_org = pad_both_sides(dimension, test_vol[0], pad_size)
    else:
        test_vol_pad_org = test_vol[0]
    print(test_vol_pad_org.shape)

    # To avoid empty regions (which is not processed) around the edge of input (prob) image
    test_vol_pad_org_size = test_vol_pad_org.shape
    extra_pad_size = ()
    extra_pad_value = np.zeros(np.shape(output_shape)).astype(int)
    for dim in range(dimension):
        extra_pad_value[dim] = np.ceil((output_shape[dim] + extraction_step[dim] *
                          np.ceil((test_vol_pad_org_size[dim + 1] - output_shape[dim]) / extraction_step[dim]) -
                          test_vol_pad_org_size[dim + 1]) / 2)
        extra_pad_size += (extra_pad_value[dim], )
        pad_size_total[dim] += extra_pad_value[dim]
    test_vol_pad_extra = pad_both_sides(dimension, test_vol_pad_org, extra_pad_size)
    print(test_vol_pad_extra.shape)

    # only for building test patches
    tst_data_pad_size = ()
    for dim in range(dimension):
        tst_data_pad_size += ((patch_shape[dim] - output_shape[dim]) // 2, )
    test_vol_pad = pad_both_sides(dimension, test_vol_pad_extra, tst_data_pad_size)
    test_vol_size = test_vol_pad.shape
    print(test_vol_size)

    test_vol_crop_array = np.zeros((1, test_vol_pad.shape[0]) + test_vol_pad.shape[1:4])
    test_vol_crop_array[0] = test_vol_pad
    print(test_vol_crop_array[0].shape)

    x_test = build_testing_set(gen_conf, test_conf, test_vol_crop_array)

    gen_conf['dataset_info'][dataset]['size'] = test_vol_pad_extra.shape[1:4]  # output image size
    rec_vol, prob_vol = test_model_modified(gen_conf, train_conf, test_conf, x_test, trained_model)
    if multi_output == 1:
        for r in rec_vol:
            print(r.shape)
        for p in prob_vol:
            print(p.shape)
    else:
        print(rec_vol.shape)
        print(prob_vol.shape)

    # re-crop zero-padded vol
    if np.sum(pad_size_total) != 0:
        start_ind = np.zeros(dimension).astype(int)
        end_ind = np.zeros(dimension).astype(int)
        for dim in range(dimension):
            if pad_size_total[dim] != 0:
                start_ind[dim] = pad_size_total[dim]
                end_ind[dim] = pad_size_total[dim] + test_vol_org_size[dim + 1]
            else:
                start_ind[dim] = 0
                end_ind[dim] = test_vol_org_size[dim + 1]

        if multi_output == 1:
            rec_vol_crop, prob_vol_crop = [], []
            for rec, prob in zip(rec_vol, prob_vol):
                rec_vol_crop.append(rec[start_ind[0]:end_ind[0], start_ind[1]:end_ind[1], start_ind[2]:end_ind[2]])
                prob_vol_crop.append(prob[start_ind[0]:end_ind[0], start_ind[1]:end_ind[1], start_ind[2]:end_ind[2]])
        else:
            rec_vol_crop = rec_vol[start_ind[0]:end_ind[0], start_ind[1]:end_ind[1], start_ind[2]:end_ind[2]]
            prob_vol_crop = prob_vol[start_ind[0]:end_ind[0], start_ind[1]:end_ind[1], start_ind[2]:end_ind[2]]

    else:
        rec_vol_crop = rec_vol
        prob_vol_crop = prob_vol
    if multi_output == 1:
        for rc in rec_vol_crop:
            print(rc.shape)
        for pc in prob_vol_crop:
            print(pc.shape)
    else:
        print(rec_vol_crop.shape)
        print(prob_vol_crop.shape)

    return rec_vol_crop, prob_vol_crop, x_test


def measure_thalamus(gen_conf, train_conf, test_conf, idx, seg_label, k, mode):

    import pandas as pd
    import pickle

    dataset = train_conf['dataset']
    approach = train_conf['approach']
    loss = train_conf['loss']
    root_path = gen_conf['root_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = gen_conf['dataset_path']
    results_path = gen_conf['results_path']
    path = dataset_info['path']
    folder_names= dataset_info['folder_names']
    data_augment = train_conf['data_augment']
    file_format = dataset_info['format']

    test_patient_id = idx
    file_output_dir = os.path.join(root_path, results_path, dataset, folder_names[0])
    test_patient_dir = os.path.join(file_output_dir, test_patient_id)
    seg_output_dir = os.path.join(test_patient_dir, path)

    crop_tst_label_pattern = dataset_info['crop_tst_label_pattern']
    staple_pattern = dataset_info['staple_pattern']
    crop_staple_pattern = dataset_info['crop_staple_pattern']
    test_roi_mask_pattern = dataset_info['test_roi_mask_pattern']
    test_roi_mask_file = os.path.join(test_patient_dir, test_roi_mask_pattern)
    test_crop_mask = find_crop_mask(test_roi_mask_file)

    cmd, msd, dc, vol = [], [], [], []
    cmd_staple, msd_staple, dc_staple, vol_staple = [], [], [], []

    # measure on the ROI
    for side in ['left', 'right']:
        label_file_lst = []
        for ver in ['2', '3']:
            label_file = os.path.join(test_patient_dir, crop_tst_label_pattern.format(side, ver))
            label_file_lst.append(label_file)
            if os.path.exists(label_file):
                break
        if os.path.exists(label_file):
            label_data = read_volume_data(label_file)
            label_image = label_data.get_data()
            if np.size(np.shape(label_image)) == 4:
                label_image_f = label_image[:, :, :, 0]
            else:
                label_image_f = label_image
            label_image_vox_size = label_data.header.get_zooms()
            label_vol = len(np.where(label_image_f == 1)[0]) * label_image_vox_size[0] * label_image_vox_size[1] * \
                        label_image_vox_size[2]
        else:
            print('No found: %s or %s' % (label_file[0], label_file[1]))
            label_image_vox_size = None
            label_vol = None
            label_image_f = None

        # read cropped smoothen threhold image for measure
        seg_filename = side + '_' + seg_label + '_seg_crop_' + approach + '_' + loss + '.' + \
                       file_format # measure original one (not smoothed/normalized/threshold)
        non_smoothed_crop_output_dir = os.path.join(seg_output_dir, 'non_smoothed', 'crop')
        seg_out_file = os.path.join(non_smoothed_crop_output_dir, seg_filename)

        if os.path.exists(seg_out_file):
            seg_data = read_volume_data(seg_out_file)
            seg_image = seg_data.get_data()
            if np.size(np.shape(seg_image)) == 4:
                seg_image_f = seg_image[:, :, :, 0]
            else:
                seg_image_f = seg_image
            seg_image_vox_size = seg_data.header.get_zooms()
            seg_vol = len(np.where(seg_image_f == 1)[0]) * seg_image_vox_size[0] * seg_image_vox_size[1] * \
                      seg_image_vox_size[2]
        else:
            print('No found: ' + seg_out_file)
            seg_image_vox_size = None
            seg_vol = None
            seg_image_f = None

        if label_image_vox_size is None or seg_image_vox_size is None:
            cmd.append(None)
            msd.append(None)
            dc.append(None)
        else:
            if label_image_f.shape != seg_image_f.shape:
                print('image size mismatches: manual - ' + str(label_image_f.shape), ', seg - ' +
                      str(seg_image_f.shape))
                cmd.append(None)
                msd.append(None)
                dc.append(None)
            else:
                cm_dist, _ = measure_cmd(label_image_f, seg_image_f, label_image_vox_size)
                cmd.append(cm_dist)
                msd.append(measure_msd(label_image_f, seg_image_f, label_image_vox_size))
                dc.append(dice(label_image_f, seg_image_f))
        vol.append([seg_vol, label_vol])

        staple_file = os.path.join(dataset_path, test_patient_id, 'fusion', staple_pattern.format(side))
        crop_staple_pattern_file = os.path.join(test_patient_dir, crop_staple_pattern.format(side))
        if (not os.path.exists(crop_staple_pattern_file)):
            crop_image(staple_file, test_crop_mask, crop_staple_pattern_file)

        if os.path.exists(crop_staple_pattern_file):
            staple_data = read_volume_data(crop_staple_pattern_file)
            staple_image = staple_data.get_data()
            if np.size(np.shape(staple_image)) == 4:
                staple_image_f = staple_image[:, :, :, 0]
            else:
                staple_image_f = staple_image
            staple_image_vox_size = staple_data.header.get_zooms()
            staple_vol = len(np.where(staple_image_f == 1)[0]) * staple_image_vox_size[0] * staple_image_vox_size[1] * \
                      staple_image_vox_size[2]
        else:
            print('No found: ' + crop_staple_pattern_file)
            staple_image_vox_size = None
            staple_vol = None
            staple_image_f = None

        if label_image_vox_size is None or staple_image_vox_size is None:
            cmd_staple.append(None)
            msd_staple.append(None)
            dc_staple.append(None)
        else:
            if label_image_f.shape != staple_image_f.shape:
                print('image size mismatches: manual - ' + str(label_image_f.shape), ', seg - ' +
                      str(staple_image_f.shape))
                cmd_staple.append(None)
                msd_staple.append(None)
                dc_staple.append(None)
            else:
                cm_dist_staple, _ = measure_cmd(label_image_f, staple_image_f, label_image_vox_size)
                cmd_staple.append(cm_dist_staple)
                msd_staple.append(measure_msd(label_image_f, staple_image_f, label_image_vox_size))
                dc_staple.append(dice(label_image_f, staple_image_f))
        vol_staple.append([staple_vol, label_vol])


    # Save segmentation results in dataframe as pkl
    file_output_dir = os.path.join(root_path, results_path, dataset, folder_names[0])
    measure_pkl_filename = 'mode_' + mode + '_#' + str(k + 1) + '_' + 'measurement_' + \
                           approach + '_' + loss + '.pkl'
    measure_pkl_filepath = os.path.join(file_output_dir, measure_pkl_filename)
    patient_results = {'CMD_L': [cmd[0]], 'CMD_R': [cmd[1]],
                       'MSD_L': [msd[0]], 'MSD_R': [msd[1]],
                       'DC_L': [dc[0]], 'DC_R': [dc[1]],
                       'VOL_L': [vol[0][0]], 'VOL_R': [vol[1][0]],
                       'VOL_manual_L': [vol[0][1]], 'VOL_manual_R': [vol[1][1]]}
    columns_name_lst = ['CMD_L', 'CMD_R', 'MSD_L', 'MSD_R', 'DC_L', 'DC_R', 'VOL_L', 'VOL_R',
                        'VOL_manual_L', 'VOL_manual_R']
    patient_results_df = pd.DataFrame(patient_results, columns=columns_name_lst)
    patient_results_df.insert(0, 'patient_id', [test_patient_id])

    #staple results
    file_output_dir = os.path.join(root_path, results_path, dataset, folder_names[0])
    measure_pkl_filename_staple = 'mode_' + mode + '_#' + str(k + 1) + '_' + 'measurement_' + 'staple' + '_' + \
                                  loss + '.pkl'
    measure_pkl_filepath_staple = os.path.join(file_output_dir, measure_pkl_filename_staple)
    patient_staple_results = {'CMD_L': [cmd_staple[0]], 'CMD_R': [cmd_staple[1]],
                       'MSD_L': [msd_staple[0]], 'MSD_R': [msd_staple[1]],
                       'DC_L': [dc_staple[0]], 'DC_R': [dc_staple[1]],
                       'VOL_L': [vol_staple[0][0]], 'VOL_R': [vol_staple[1][0]],
                       'VOL_manual_L': [vol_staple[0][1]], 'VOL_manual_R': [vol_staple[1][1]]}
    columns_name_lst = ['CMD_L', 'CMD_R', 'MSD_L', 'MSD_R', 'DC_L', 'DC_R', 'VOL_L', 'VOL_R',
                        'VOL_manual_L', 'VOL_manual_R']
    patient_staple_results_df = pd.DataFrame(patient_staple_results, columns=columns_name_lst)
    patient_staple_results_df.insert(0, 'patient_id', [test_patient_id])

    if os.path.exists(measure_pkl_filepath):
        with open(measure_pkl_filepath, 'rb') as handle:
            patient_results_df_total = pickle.load(handle)
            patient_results_df_total = pd.concat([patient_results_df_total, patient_results_df])
    else:
        patient_results_df_total = patient_results_df

    patient_results_df_total.to_pickle(measure_pkl_filepath)
    print(patient_results_df_total)
    print(patient_results_df_total.describe(percentiles=[0.25, 0.5, 0.75, 0.9]))

    if os.path.exists(measure_pkl_filepath_staple):
        with open(measure_pkl_filepath_staple, 'rb') as handle:
            patient_staple_results_df_total = pickle.load(handle)
            patient_staple_results_df_total = pd.concat([patient_staple_results_df_total, patient_staple_results_df])
    else:
        patient_staple_results_df_total = patient_staple_results_df

    patient_staple_results_df_total.to_pickle(measure_pkl_filepath_staple)
    print(patient_staple_results_df_total)
    print(patient_staple_results_df_total.describe(percentiles=[0.25, 0.5, 0.75, 0.9]))

    return patient_results_df_total, patient_staple_results_df_total


def measure_dentate_interposed(gen_conf, train_conf, test_conf, idx, k, mode, target):

    import pandas as pd
    import pickle

    dataset = train_conf['dataset']
    approach = train_conf['approach']
    loss = train_conf['loss']
    root_path = gen_conf['root_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    results_path = gen_conf['results_path']
    num_classes = gen_conf['num_classes']
    multi_output = gen_conf['multi_output']
    path = dataset_info['path']
    folder_names= dataset_info['folder_names']
    data_augment = train_conf['data_augment']
    file_format = dataset_info['format']

    test_patient_id = idx
    dentate_label_pattern = dataset_info['crop_tst_manual_dentate_v2_corrected_pattern']
    interposed_label_pattern = dataset_info['crop_tst_manual_interposed_v2_pattern']

    file_output_dir = os.path.join(root_path, results_path, dataset, folder_names[0])
    test_patient_dir = os.path.join(file_output_dir, test_patient_id)
    seg_output_dir = os.path.join(test_patient_dir, path)

    if len(target) == 2:
        target = 'both'

    if target == 'dentate':
        label_pattern_lst = [dentate_label_pattern]
        seg_label_lst = ['dentate_seg']
        if multi_output == 1:
            loss_ch = [loss[0]]
            num_classes_ch = [num_classes[0]]
        else:
            loss_ch = [loss]
            num_classes_ch = [num_classes]
    elif target == 'interposed':
        label_pattern_lst = [interposed_label_pattern]
        seg_label_lst = ['interposed_seg']
        if multi_output == 1:
            loss_ch = [loss[1]]
            num_classes_ch = [num_classes[1]]
        else:
            loss_ch = [loss]
            num_classes_ch = [num_classes]
    else:
        label_pattern_lst = [dentate_label_pattern, interposed_label_pattern]
        seg_label_lst = ['dentate_seg', 'interposed_seg']
        if multi_output == 1:
            loss_ch = [loss[0], loss[1]]
            num_classes_ch = [num_classes[0], num_classes[1]]
        else:
            loss_ch = [loss, loss]
            num_classes_ch = [num_classes, num_classes]

    for label_pattern, seg_label, l_ch, n_ch in zip(label_pattern_lst, seg_label_lst, loss_ch, num_classes_ch):
        cmd, msd, dc, vol = [], [], [], []
        for side in ['left', 'right']:
            label_file = os.path.join(test_patient_dir, label_pattern.format(test_patient_id, side))
            seg_filename = side + '_' + seg_label + '_crop_' + approach + '_' + l_ch + '.' + \
                           file_format # measure original one (not smoothed/normalized/threshold)
            non_smoothed_crop_output_dir = os.path.join(seg_output_dir, 'non_smoothed', 'crop')
            seg_out_file = os.path.join(non_smoothed_crop_output_dir, seg_filename)

            if os.path.exists(label_file):
                label_data = read_volume_data(label_file)
                label_image = label_data.get_data()
                if np.size(np.shape(label_image)) == 4:
                    label_image_f = label_image[:, :, :, 0]
                else:
                    label_image_f = label_image
                label_image_vox_size = label_data.header.get_zooms()
                label_vol = len(np.where(label_image_f == 1)[0]) * label_image_vox_size[0] * label_image_vox_size[1] * \
                            label_image_vox_size[2]
            else:
                print('No found: ' + label_file)
                label_image_vox_size = None
                label_vol = None
                label_image_f = None

            if os.path.exists(seg_out_file):
                seg_data = read_volume_data(seg_out_file)
                seg_image = seg_data.get_data()
                if np.size(np.shape(seg_image)) == 4:
                    seg_image_f = seg_image[:, :, :, 0]
                else:
                    seg_image_f = seg_image
                seg_image_vox_size = seg_data.header.get_zooms()
                seg_vol = len(np.where(seg_image_f == 1)[0]) * seg_image_vox_size[0] * seg_image_vox_size[1] * \
                          seg_image_vox_size[2]
            else:
                print('No found: ' + seg_out_file)
                seg_image_vox_size = None
                seg_vol = None
                seg_image_f = None

            if label_image_vox_size is None or seg_image_vox_size is None or seg_vol == 0:
                cmd.append(None)
                msd.append(None)
                dc.append(None)
            else:
                if label_image_vox_size[0:3] != seg_image_vox_size[0:3]:
                    print('voxel size mismatches: manual - ' + str(label_image_vox_size), ', seg - ' +
                          str(seg_image_vox_size))
                    cmd.append(None)
                    msd.append(None)
                    dc.append(None)
                else:
                    cm_dist, _ = measure_cmd(label_image_f, seg_image_f, label_image_vox_size)
                    cmd.append(cm_dist)
                    msd.append(measure_msd(label_image_f, seg_image_f, label_image_vox_size))
                    dc.append(dice(label_image_f, seg_image_f))
            vol.append([seg_vol, label_vol])

        # Save results in dataframe as pkl
        measure_pkl_filename = 'mode_' + mode + '_#' + str(k + 1) + '_' + 'measurement_' + \
                               approach + '_' + l_ch + '_n_classes_' + str(n_ch) + '_' + seg_label + '.pkl'
        measure_pkl_filepath = os.path.join(file_output_dir, measure_pkl_filename)
        patient_results = {'CMD_L': [cmd[0]], 'CMD_R': [cmd[1]],
                           'MSD_L': [msd[0]], 'MSD_R': [msd[1]],
                           'DC_L': [dc[0]], 'DC_R': [dc[1]],
                           'VOL_L': [vol[0][0]], 'VOL_R': [vol[1][0]],
                           'VOL_manual_L': [vol[0][1]], 'VOL_manual_R': [vol[1][1]]}
        columns_name_lst = ['CMD_L', 'CMD_R', 'MSD_L', 'MSD_R', 'DC_L', 'DC_R', 'VOL_L', 'VOL_R',
                            'VOL_manual_L', 'VOL_manual_R']
        patient_results_df = pd.DataFrame(patient_results, columns=columns_name_lst)
        patient_results_df.insert(0, 'patient_id', [test_patient_id])
        patient_results_df.insert(0, 'structure', [seg_label])

        if os.path.exists(measure_pkl_filepath):
            with open(measure_pkl_filepath, 'rb') as handle:
                patient_results_df_total = pickle.load(handle)
                patient_results_df_total = pd.concat([patient_results_df_total, patient_results_df])
        else:
            patient_results_df_total = patient_results_df

        patient_results_df_total.to_pickle(measure_pkl_filepath)
        print(patient_results_df_total)
        print(patient_results_df_total.describe(percentiles=[0.25,0.5,0.75,0.9]))

    return patient_results_df_total