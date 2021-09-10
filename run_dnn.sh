#!/usr/bin/env bash

export SRC_DIR="/home/asaf/jinyoung/projects/datasets/data_temp"
export ROOT_DIR="/home/asaf/jinyoung/projects" # set your root path here

export DATA_DIR="${ROOT_DIR}/datasets"
export TARGET_DIR="${DATA_DIR}/$1"
if [ ! -d "${DATA_DIR}" ]
then
    mkdir "${DATA_DIR}"
fi
if [ ! -d "${TARGET_DIR}" ]
then
    mkdir "${TARGET_DIR}"
fi

export DATE=$(date +'%m%d%y')

IFS=','
read -a strarr <<< "$2"
for val in "${strarr[@]}";
do
    printf "$val\n"
    export SUBJECT_DIR="${TARGET_DIR}/$val"
    if [ ! -d "${SUBJECT_DIR}" ]
    then
        mkdir "${SUBJECT_DIR}"
    fi
    if [ "$1" == dcn ]
    then
        if [ ! -d "${SUBJECT_DIR}/image" ]
        then
            mkdir "${SUBJECT_DIR}/image"
        fi
        #rsync -av "${SRC_DIR}/$val/Dentate/eddy_wrapped_B0_image_brain.nii.gz" "${SUBJECT_DIR}/image/B0_image.nii.gz"
        rsync -av "${SRC_DIR}/$val/Dentate/eddy_wrapped_B0_image_brain_Regto__NA_MI.nii.gz" "${SUBJECT_DIR}/image/B0_image.nii.gz"
        #rsync -av "${SRC_DIR}/$val/Dentate/eddy_wrapped_B0_image.nii.gz" "${SUBJECT_DIR}/image/B0_image.nii.gz"
    elif [ "$1" == tha ]
    then
        if [ ! -d "${SUBJECT_DIR}/images" ]
        then
            mkdir "${SUBJECT_DIR}/images"
        fi
        rsync -av "${SRC_DIR}/$val/Thalamus/T1w_brain_restore.nii.gz" "${SUBJECT_DIR}/images/7T_T1_brain.nii.gz"
        ##rsync -av "${SRC_DIR}/$val/Thalamus/eddy_wrapped_B0_image_brain_Regto_7T_NA_MI.nii.gz" "${SUBJECT_DIR}/images/registered_B0.nii.gz"
        rsync -av "${SRC_DIR}/$val/Thalamus/eddy_wrapped_B0_image_brain_Regto__NA_MI.nii.gz" "${SUBJECT_DIR}/images/registered_B0.nii.gz"
        ##rsync -av "${SRC_DIR}/$val/Thalamus/dtiwls_54dir_FA_Regto_7T_NA_MI.nii.gz" "${SUBJECT_DIR}/images/registered_FA.nii.gz"
        rsync -av "${SRC_DIR}/$val/Thalamus/dtiwls_54dir_FA_Regto__NA_MI.nii.gz" "${SUBJECT_DIR}/images/registered_FA.nii.gz"

#        rsync -av "${SRC_DIR}/$val/Thalamus/T1w_brain_restore.nii.gz" "${SUBJECT_DIR}/images/7T_T1_brain.nii.gz"
#        rsync -av "${SRC_DIR}/$val/Thalamus/eddy_wrapped_B0_image_brain_Regto_7T_T1_MI.nii.gz" "${SUBJECT_DIR}/images/registered_B0.nii.gz"
#        rsync -av "${SRC_DIR}/$val/Thalamus/dtiwls_54dir_FA_Regto_7T_T1_MI.nii.gz" "${SUBJECT_DIR}/images/registered_FA.nii.gz"

#        rsync -av "${SRC_DIR}/$val/Thalamus/T1w_brain_restore.nii.gz" "${SUBJECT_DIR}/images/7T_T1_brain.nii.gz"
#        rsync -av "${SRC_DIR}/$val/Thalamus/eddy_wrapped_B0_image_brain_Regto__NA_MI.nii.gz" "${SUBJECT_DIR}/images/registered_B0.nii.gz"
#        rsync -av "${SRC_DIR}/$val/Thalamus/dtiwls_54dir_FA_Regto__NA_MI.nii.gz" "${SUBJECT_DIR}/images/registered_FA.nii.gz"
    fi
done
IFS=''

if [ "$1" == dcn ]
then
    # parameters for dentate and interposed nuclei (unseen case)
    trn_set_num='6'
    test_subject_id=$2
    mode='2' # 0. training + testing (n-fold cross-validation), 1. training + testing (for designated cases) , 2. testing
    gpu_id='-2' # 0: gpu id 0, -2: cpu only
    num_classes='2,2' # 2 # 3
    multi_output='1'
    output_name='dentate_seg,interposed_seg'
    attention_loss='1'
    overlap_penalty_loss='1'
    loss='tversky_focal_multiclass,tversky_focal_multiclass' #implemented: dc, tversky, tversky_focal, tversky_focal_multiclass, dice_focal, focal  #niftynet built-in: 'Tversky' #Dice #Dice_Dense_NS # sparse_categorical_crossentropy for integer target, categorical_crossentropy for one-hot encoded input
    approach='fc_densenet_dilated' #'pr_fb_net' #unet #livianet # fc_densenet_dilated # deepmedic
    dataset='dcn_seg_dl' #'tha_seg_dl' #dcn_seg_dl
    preprocess_trn='2'
    preprocess_tst='2'
    num_k_fold='5' #
    batch_size='8'
    num_epochs='50' #'20'
    patience='20' #'2'
    optimizer='Adam'
    initial_lr='0.001'
    is_set_random_seed='0'
    random_seed_num='None'
    metric='loss_total' #loss #acc #acc_dc
    lamda='1.0,1.0,0.1,0.5' # weight for losses - dentate/interposed/attention/overlap penalty
    exclusive_train='0' #if exclusive_train is '1', during training, remove designated label
    exclude_label_num='2,1' #exclude_label_num: 0. bg, 1.dentate, 2. interposed
    activation='softmax,softmax' # last layer activation - sigmoid: independent multi-label training, softmax: dependent single label training
    target='dentate,interposed'  #'tha' #dentate # dentate,interpose
    image_modality='B0' #'B0,T1,LP,FA' #'B0' #'B0,T1,LP,FA'
    trn_patch_size='32,32,32' #'32,32,32'
    trn_output_size='32,32,32'
    trn_step_size='5,5,5' #
    tst_patch_size='32,32,32' #'32,32,32'
    tst_output_size='32,32,32'
    tst_step_size='5,5,5'
    crop_margin='5,5,5'
    bg_discard_percentage='0.0'
    threshold='0'
    continue_tr='0'
    is_unseen_case='1'
    is_reg='0' # 0. manual roi setting, 1. ROI localization using registration
    roi_pos='34,78,15,50,30,48' # set ROI position (start_x (R), end_x (L), start_y (A), end_y (P), start_z (S), end_z (I)) if is_reg==0 (localization by manually setting ROI)'
    is_measure='0'
    is_new_trn_label='0' # 0: 29 manual labels, 1: 31 segmentation using a proposed network, 2: suit labels, 3: 29 manual labels + 31 seg (self-training?)
    new_label_path='None'
    folder_names=${DATE}
    root_path=${ROOT_DIR}
    dataset_path='/home/shira-raid1/DBS_data/Cerebellum_Tools/' # don't change this path

    if [ $is_set_random_seed == '1' ]
    then
        export PYTHONHASHSEED=0 # to make python 3 code reproducible by deterministic hashing
    fi

    python run.py --mode $mode --gpu_id $gpu_id --num_classes $num_classes --multi_output $multi_output \
    --attention_loss $attention_loss --overlap_penalty_loss $overlap_penalty_loss --output_name $output_name --loss $loss \
    --exclusive_train $exclusive_train --exclude_label_num $exclude_label_num --approach $approach --dataset $dataset \
    --preprocess_trn $preprocess_trn --preprocess_tst $preprocess_tst --num_k_fold $num_k_fold --batch_size $batch_size \
    --num_epochs $num_epochs --patience $patience --optimizer $optimizer --initial_lr $initial_lr \
    --is_set_random_seed $is_set_random_seed --random_seed_num $random_seed_num --trn_patch_size $trn_patch_size \
    --trn_output_size $trn_output_size --trn_step_size $trn_step_size --tst_patch_size $tst_patch_size \
    --tst_output_size $tst_output_size --tst_step_size $tst_step_size --crop_margin $crop_margin \
    --bg_discard_percentage $bg_discard_percentage --threshold $threshold --metric $metric --lamda $lamda \
    --target $target --activation $activation --image_modality $image_modality --folder_names $folder_names \
    --root_path $root_path --dataset_path $dataset_path --continue_tr $continue_tr --is_unseen_case $is_unseen_case \
    --is_measure $is_measure --trn_set_num $trn_set_num --test_subject_id $test_subject_id \
    --new_label_path $new_label_path --is_reg $is_reg --roi_pos $roi_pos

elif [ "$1" == tha ]
then
    # parameters for thalamus (unseen case)
    trn_set_num='2' # default : 2
    test_subject_id=$2
    mode='2' # 0. training + testing (n-fold cross-validation), 1. training + testing (for designated cases) , 2. testing
    gpu_id='0' # 0: gpu id 0, -2: cpu only
    num_classes='2'
    output_name='tha_seg'
    loss='tversky_focal' #'Tversky' #Dice #Dice_Dense_NS # sparse_categorical_crossentropy for integer target, categorical_crossentropy for one-hot encoded input
    approach='fc_densenet_ms' #'unet' #'pr_fb_net' #unet #livianet, fc_densenet, fc_densenet_ms, fc_densenet_ms_attention
    dataset='tha_seg_dl' #'tha_seg_dl'
    preprocess_trn='2'
    preprocess_tst='2'
    num_k_fold='5'
    batch_size='16' #16, 32
    num_epochs='50'
    patience='10'
    optimizer='Adam'
    initial_lr='0.001'
    is_set_random_seed='0'
    random_seed_num='None'
    metric='loss' #loss #acc #acc_dc
    activation='softmax' # last layer activation - sigmoid: independent multi-label training, softmax: dependent single label training
    target='tha'
    image_modality='T1,B0,FA' #T1
    trn_patch_size='64,64,64' #for livianet 27,27,27 (32,32,32); for deepmedic 48,48,48; for CAFP 64,64,64; for others 32,32,32
    trn_output_size='32,32,32' #for livianet 9,9,9 (14,14,14); for deepmedic 16,16,16; for others 32,32,32
    trn_step_size='9,9,9' #for livianet and deepmedic 5,5,5 (9,9,9); for others 15,15,15
    tst_patch_size='64,64,64' #for livianet 27,27,27 (32,32,32); for deepmedic 48,48,48; for CAFP 64,64,64; for others 32,32,32
    tst_output_size='32,32,32' #for livianet 9,9,9 (14,14,14); for deepmedic 16,16,16; for others 32,32,32
    tst_step_size='9,9,9' #for livianet and deepmedic 5,5,5 (9,9,9); for others 15,15,15
    crop_margin='9,9,9'
    bg_discard_percentage='0.0'
    threshold='0.1'
    continue_tr='0'
    is_unseen_case='1'
    is_reg='1' # 0. manual roi setting, 1. ROI localization using registration
    roi_pos='54,126,78,128,126,168' # set ROI position (start_x (R), end_x (L), start_y (A), end_y (P), start_z (S), end_z (I)) if is_reg==0 (localization by manually setting ROI)'
    is_measure='0'
    folder_names=${DATE}
    root_path=${ROOT_DIR}
    dataset_path='/home/asaf/jinyoung/projects/datasets/thalamus/' # don't change this path

    if [ $is_set_random_seed == '1' ]
    then
        export PYTHONHASHSEED=0 # to make python 3 code reproducible by deterministic hashing
    fi

    python run.py --mode $mode --gpu_id $gpu_id --num_classes $num_classes --output_name $output_name --loss $loss \
    --approach $approach --dataset $dataset --preprocess_trn $preprocess_trn --preprocess_tst $preprocess_tst \
    --num_k_fold $num_k_fold --batch_size $batch_size --num_epochs $num_epochs --patience $patience \
    --optimizer $optimizer --initial_lr $initial_lr --is_set_random_seed $is_set_random_seed \
    --random_seed_num $random_seed_num --trn_patch_size $trn_patch_size --trn_output_size $trn_output_size \
    --trn_step_size $trn_step_size --tst_patch_size $tst_patch_size --tst_output_size $tst_output_size \
    --tst_step_size $tst_step_size --crop_margin $crop_margin --bg_discard_percentage $bg_discard_percentage \
    --threshold $threshold --metric $metric --target $target --activation $activation --image_modality $image_modality \
    --folder_names $folder_names --root_path $root_path --dataset_path $dataset_path --continue_tr $continue_tr \
    --is_unseen_case $is_unseen_case --is_measure $is_measure --trn_set_num $trn_set_num \
    --test_subject_id $test_subject_id --is_reg $is_reg --roi_pos $roi_pos
else
    echo invalid target name. please use dcn or tha.
fi