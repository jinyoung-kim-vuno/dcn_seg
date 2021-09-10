general_config = {
    'args': None,
    'validation_mode': 0, # validation mode (k-fold or loo, training only, testing only)
    'multi_output' : 0,
    'output_name' : 'dentate_seg',
    'num_classes' : 2, # 4 for brain tissue, 3 for cbct,
    'root_path' : '/home/asaf/jinyoung/projects/', # @kesem
    #'root_path' : '/content/g_drive/', # colab root path
    #'dataset_path': '/home/shira-raid1/DBS_data/Cerebellum_Tools/',
    'dataset_path': '/home/asaf/jinyoung/projects/datasets/thalamus/',
    'log_path' : 'log',
    'model_path' : 'models',
    'results_path' : 'results/',
    'patches_path' : 'patches/',
    'dataset_info' : {
        'dcn_seg_dl': {
            'format': 'nii.gz',
            'time_series': False,
            'size': [],
            'target': ['dentate', 'interposed'], #['dentate'], # ['interposed']
            'exclude_label_num': (2),
            'image_modality': ['B0'],
            'image_name_pattern': ['{}_B0_LPI.nii', '{}_T1_LPI.nii',
                                   'monogenic_signal/{}_B0_LP_corrected.nii',
                                   'monogenic_signal/{}_B0_FA_corrected.nii'],
            'image_new_name_pattern': ['B0_image.nii.gz', 'T1_image.nii.gz', 'B0_LP_image.nii', 'B0_FA_image.nii'],
            'image_resampled_name_pattern': ['{}_B0_LPI_resampled.nii.gz', '{}_T1_LPI_resampled.nii.gz',
                                             '{}_B0_LP_corrected_resampled.nii.gz', '{}_B0_FA_corrected_resampled.nii.gz'],

            'manual_corrected_pattern': 'hc_{}_{}_dentate_corrected.nii.gz',
            'manual_corrected_dentate_v2_pattern': 'hc_{}_{}_dentate_corrected_v2.nii.gz',
            'manual_corrected_interposed_v2_pattern': 'hc_{}_{}_interposed_v2.nii.gz',
            'manual_corrected_dentate_interposed_v2_pattern': 'hc_{}_{}_dentate_interposed_merged_v2.nii.gz',

            'trn_new_label_dentate_pattern': 'segmentation/{}_dentate_seg_fc_densenet_dilated_tversky_focal_multiclass.nii.gz',
            'trn_new_label_interposed_pattern': 'segmentation/{}_interposed_seg_fc_densenet_dilated_tversky_focal_multiclass.nii.gz',

            'initial_mask_pattern': 'DCN_masks/thresh35/{}_{}_dentate_final.nii',
            'initial_interposed_mask_pattern_thres': 'DCN_masks/thresh35/{}_{}_interposed_thresh35.nii',
            'initial_interposed_mask_pattern_mask': 'DCN_masks/{}_{}_interposed_mask.nii',
            'initial_reg_mask_pattern': 'ini_reg_{}_{}.nii.gz',

            'suit_dentate_mask_pattern': 'DCN_masks/{}_{}_dentate_mask.nii',
            'suit_interposed_mask_pattern': 'DCN_masks/{}_{}_interposed_mask.nii',
            'suit_fastigial_mask_pattern': 'DCN_masks/{}_{}_fastigial_mask.nii',

            'set_new_roi_mask': True, #True,#False,
            'margin_crop_mask': (5, 5, 5), # (10,10,5) for thalamus #(5, 5, 5) for dentate, interposed
            'crop_trn_image_name_pattern': ['{}_B0_LPI_trn_crop.nii', '{}_T1_LPI_trn_crop.nii',
                                                '{}_B0_LP_corrected_trn_crop.nii',
                                                '{}_B0_FA_corrected_trn_crop.nii'],
            'crop_tst_image_name_pattern': ['{}_B0_LPI_tst_crop.nii', '{}_T1_LPI_tst_crop.nii',
                                                '{}_B0_LP_corrected_tst_crop.nii',
                                                '{}_B0_FA_corrected_tst_crop.nii'],
            'crop_tst_image_resampled_name_pattern': ['{}_B0_LPI_tst_resampled_crop.nii', '{}_T1_LPI_tst_resampled_crop.nii',
                                            '{}_B0_LP_corrected_tst_resampled_crop.nii',
                                            '{}_B0_FA_corrected_tst_resampled_crop.nii'],

            'crop_trn_manual_corrected_pattern': 'hc_{}_{}_dentate_corrected_trn_crop.nii.gz',
            'crop_tst_manual_corrected_pattern': 'hc_{}_{}_dentate_corrected_tst_crop.nii.gz',

            'crop_trn_manual_dentate_v2_corrected_pattern': 'hc_{}_{}_dentate_corrected_v2_trn_crop.nii.gz',
            'crop_tst_manual_dentate_v2_corrected_pattern': 'hc_{}_{}_dentate_corrected_v2_tst_crop.nii.gz',
            'crop_trn_manual_interposed_v2_pattern': 'hc_{}_{}_interposed_v2_trn_crop.nii.gz',
            'crop_tst_manual_interposed_v2_pattern': 'hc_{}_{}_interposed_v2_tst_crop.nii.gz',

            'crop_trn_new_label_dentate_pattern': '{}_dentate_seg_fc_densenet_dilated_tversky_focal_multiclass_crop.nii.gz',
            'crop_trn_new_label_interposed_pattern': '{}_interposed_seg_fc_densenet_dilated_tversky_focal_multiclass_crop.nii.gz',

            'crop_suit_dentate_mask_pattern': '{}_{}_dentate_mask_trn_crop.nii',

            'crop_initial_mask_pattern': '{}_{}_dentate_final_tst_crop.nii',
            'crop_initial_interposed_mask_pattern': '{}_{}_interposed_final_tst_crop.nii',
            'crop_initial_reg_mask_pattern': 'ini_reg_{}_{}_tst_crop.nii.gz',

            'crop_suit_interposed_mask_pattern': '{}_{}_interposed_mask_tst_crop.nii',
            'crop_suit_fastigial_mask_pattern': '{}_{}_fastigial_mask_tst_crop.nii',

            'train_roi_mask_pattern': '{}/train_roi_mask.nii.gz',
            'test_roi_mask_pattern': '{}/test_roi_mask.nii.gz',
            'patient_id': ['PD081', 'SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP104', 'SLEEP105', 'SLEEP107',
                           'SLEEP108', 'SLEEP109', 'SLEEP110', 'SLEEP112', 'SLEEP113', 'SLEEP114', 'SLEEP115',
                           'SLEEP116','SLEEP117', 'SLEEP118', 'SLEEP119', 'SLEEP120', 'SLEEP122', 'SLEEP123',
                           'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP127', 'SLEEP128', 'SLEEP130', 'SLEEP131',
                           'SLEEP133'],
            'path': 'dcn',
            'folder_names': ['dcn_baseline_test_b0']
        },
        'tha_seg_dl': {
            'format': 'nii.gz',
            'time_series': False,
            'size': [],
            'target': ['tha'],
            'exclude_label_num': (),
            'image_modality': ['T1','B0','FA'],
            'image_name_pattern': ['7T_T1_brain.nii.gz', 'registered_B0.nii.gz',
                                   'registered_FA.nii.gz'],
            'image_resampled_name_pattern': ['7T_T1_brain_resampled.nii.gz', 'registered_B0_resampled.nii.gz',
                                   'registered_FA_resampled.nii.gz'],

            'label_pattern': 'fused_{}_thalamus_final_GM_v{}.nii.gz',
            'initial_mask_pattern': 'fused_{}_thalamus_final_GM.nii.gz', # use initially corrected output temporally
            'initial_reg_mask_pattern': 'ini_reg_{}_{}.nii.gz',

            'staple_pattern': 'fused_{}_thalamus_inverted.nii.gz', # updated on 3/28/19 (previously fused_{}_thalamus_final_GM.nii.gz was used)
            'crop_staple_pattern': 'fused_{}_thalamus_inverted_tst_crop.nii.gz', # updated on 3/28/19 (previously fused_{}_thalamus_final_GM_tst_crop.nii.gz was used)

            'set_new_roi_mask': True,  # True,#False,
            'margin_crop_mask': (9, 9, 9), # for thalamus #(5, 5, 5) for dentate, interposed
            'crop_trn_image_name_pattern': ['7T_T1_brain_trn_crop.nii.gz', 'registered_B0_trn_crop.nii.gz',
                                   'registered_FA_trn_crop.nii.gz'],
            'crop_tst_image_name_pattern': ['7T_T1_brain_tst_crop.nii.gz', 'registered_B0_tst_crop.nii.gz',
                                   'registered_FA_tst_crop.nii.gz'],

            'crop_tst_image_resampled_name_pattern': ['7T_T1_brain_tst_resampled_crop.nii.gz',
                                                      'registered_B0_tst_resampled_crop.nii.gz',
                                                      'registered_FA_tst_resampled_crop.nii.gz'],

            'crop_trn_label_pattern': 'fused_{}_thalamus_final_GM_v{}_trn_crop.nii.gz',
            'crop_tst_label_pattern': 'fused_{}_thalamus_final_GM_v{}_tst_crop.nii.gz',

            'crop_trn_label_downsampled_pattern': 'fused_thalamus_final_GM_v{}_trn_crop_downsampled.nii.gz',
            'crop_tst_label_downsampled_pattern': 'fused_thalamus_final_GM_v{}_tst_crop_downsampled.nii.gz',

            'crop_initial_mask_pattern': 'fused_{}_thalamus_final_GM_tst_crop.nii.gz',
            'crop_initial_reg_mask_pattern': 'ini_reg_{}_thalamus_tst_crop.nii.gz',

            'train_roi_mask_pattern': 'train_roi_mask.nii.gz',
            'test_roi_mask_pattern': 'test_roi_mask.nii.gz',
            'patient_id': ['ET018', 'ET019', 'ET020', 'ET021', 'ET028', 'ET030', 'MS001', 'P030', 'P040', 'PD050',
                           'PD061', 'PD074', 'PD081', 'PD085', 'SLEEP101', 'SLEEP102', 'SLEEP103', 'SLEEP104',
                           'SLEEP105', 'SLEEP106', 'SLEEP107', 'SLEEP108', 'SLEEP109', 'SLEEP110', 'SLEEP112',
                           'SLEEP113', 'SLEEP114', 'SLEEP115', 'SLEEP116', 'SLEEP117', 'SLEEP118', 'SLEEP119',
                           'SLEEP120', 'SLEEP122', 'SLEEP123', 'SLEEP124', 'SLEEP125', 'SLEEP126', 'SLEEP127',
                           'SLEEP128', 'SLEEP130', 'SLEEP131', 'SLEEP133'],
            'path': 'thalamus',
            'folder_names': ['tha_baseline_test_t1_b0_fa']
        }
    }
}

training_config = {
    'exclusive_train' : 0,
    'activation' : 'softmax', #sigmoid: independent multiple training, # softmax: dependent single label training
    'approach' : 'fc_densenet_ms', #'approach' : 'cc_3d_fcn', unet, livianet, uresnet, fc_densenet, deepmedic, wnet, fc_capsnet, attention_unet, attention_se_fcn, fc_rna, pr_fb_net
    'dataset' : 'tha_seg_dl', #'dcn_seg_dl' #'3T7T', # 3T7T, 3T7T_real, 3T7T_total, 3T+7T, CBCT16, CBCT57, CT30 for training
    'data_augment': 0, # 0: offdcn_seg_dl, 1: mixup, 2. datagen, 3: mixup + datagen
    'dimension' : 3,
    'extraction_step' : (5, 5, 5), # for thalamus #(5, 5, 5) for dentate, #(2, 2, 2) for interposed
    'attention_loss' : 0,
    'overlap_penalty_loss' : 0,
    'loss' : 'tversky_focal_multiclass',  # Dice_Dense_NS (dice_dense_nosquare), Dice(dice), GDSC (generalised_dice_loss),
    # WGDL (generalised_wasserstein_dice_loss; currently not working), Tversky(tversky), CrossEntropy_Dense (cross_entropy_dense),
    # CrossEntropy (cross_entropy), SensSpec (sensitivity_specificity_loss), msd
                        # old version: weighted_categorical_crossentropy, categorical_crossentropy, dc, tversky
    'metric' : 'loss', # metric for early stopping: acc, acc_dc, loss
    'lamda' : (1.0, 1.0, 0.1, 0.05),
    'batch_size' : 8,
    'num_epochs' : 50,
    'num_retrain' : 0,
    'optimizer' : 'Adam', #Adam, Adamax, Nadam, SGD
    'initial_lr' : '0.001',
    'is_set_random_seed' : '0',
    'random_seed_num' : '1',
    'output_shape' : (32, 32, 32), #for thalamus, dentate, interposed (48, 48, 48), #(32, 32, 32), #(32,32,32), #(9, 9, 9), #(32, 32, 32), #(9, 9, 9),
    'patch_shape' : (32, 32, 32), #for thalamus, dentate, interposed (48, 48, 48), #(32, 32, 32), #(9, 9, 9), #(32, 32, 32),
    'bg_discard_percentage' : 0, # 0 for dentate,interposed,thalamus segmentation, 0.2 for brain tissue segmentation
    'patience' : 20, #1
    'num_k_fold': 5,
    'validation_split' : 0.20,
    'shuffle' : True, # Default
    'verbose' : 1,
    'preprocess' : 2,   #0: no preprocessing, 1: standardization, 2: normalization,
    # 3: normalization + standardization, 4: histogram matching (or normalization) to one training ref + standardization,
    # 5: normalization + histogram matching
    'importance_sampling' : 0,
    'oversampling' : 0,
    'use_saved_patches' : False,
    'is_new_trn_label': 0,
    'new_label_path': '/home/asaf/jinyoung/projects/results/dcn_seg_dl/dentate_interposed_dilated_fc_densenet_dual_output_only_dentate_only_interposed_softmax_attn_loss_0_1_overlap_loss_updated_0_5_train_29_test_32',
    'continue_tr' : 0,
}

test_config = {
    'dataset' : 'tha_seg_dl', #'ADNI', # dcn_seg_dl # tha_seg_dl
    'test_subject_id': [],
    'dimension' : 3,
    'extraction_step' : (9, 9, 9), # for thalamus #(5, 5, 5) for dentate, #(2, 2, 2) for interposed
    'output_shape' : (32, 32, 32), # for thalamus, dentate, interposed #(48, 48, 48), #(32, 32, 32), #(9, 9, 9),(32, 32, 32)
    'patch_shape' : (32, 32, 32), # for thalamus, dentate, interposed #(48, 48, 48), #(32, 32, 32),
    'threshold': 0, # 0.1 for thalamus, 0 for dentate
    'verbose' : 1,
    'preprocess' : 2,   #0: no preprocessing, 1: standardization, 2: normalization,
    # 3: normalization + standardization, 4: histogram matching (or normalization) to one training ref + standardization,
    # 5: normalization + histogram matching
    'is_measure': 1,
    'is_unseen_case' : 0,
    'trn_set_num': 5,
    'is_reg': 1,
    'roi_pos': (0,0,0,0,0,0),
}