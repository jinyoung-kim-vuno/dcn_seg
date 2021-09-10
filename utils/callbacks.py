import os
import math
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler

def generate_output_filename(path, dataset, mode, case_name, approach, loss, dimension, num_classes, patch_shape, extraction_step,
                             data_augmentation, preprocess_trn, extension) :
    if type(case_name) == str:
        file_pattern = '{}/{}/{}-{}-{}-{}-{}-{}-{}-{}-{}-{}.{}'
    else:
        file_pattern = '{}/{}/{}-{:02}-{}-{}-{}-{}-{}-{}-{}-{}.{}'
    return file_pattern.format(path, dataset, mode, case_name, approach, loss, dimension, num_classes, patch_shape, extraction_step,
                               data_augmentation, preprocess_trn, extension)

def generate_callbacks(gen_conf, train_conf, case_name) :
    root_path = gen_conf['root_path']
    model_path = gen_conf['model_path']
    log_path = gen_conf['log_path']
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    num_classes = gen_conf['num_classes']
    multi_output = gen_conf['multi_output']
    output_name = gen_conf['output_name']
    folder_names = dataset_info['folder_names']
    mode = gen_conf['validation_mode']
    approach = train_conf['approach']
    loss = train_conf['loss']
    dimension = train_conf['dimension']
    patch_shape = train_conf['patch_shape']
    extraction_step = train_conf['extraction_step']
    data_augment = train_conf['data_augment']
    preprocess_trn = train_conf['preprocess']
    metric_opt = train_conf['metric']
    attention_loss = train_conf['attention_loss']
    optimizer = train_conf['optimizer']
    initial_lr = train_conf['initial_lr']

    if data_augment == 1:
        data_augment_label = 'mixup'
    elif data_augment == 2:
        data_augment_label = 'datagen'
    elif data_augment == 3:
        data_augment_label = 'mixup+datagen'
    else:
        data_augment_label = ''

    if not os.path.exists(root_path + model_path + '/' + dataset + '/' + folder_names[0]):
        os.makedirs(os.path.join(root_path, model_path, dataset, folder_names[0]))

    if not os.path.exists(root_path + log_path + '/' + dataset + '/' + folder_names[0]):
        os.makedirs(os.path.join(root_path, log_path, dataset, folder_names[0]))

    if multi_output == 1:
        loss = loss[0] + '_' + loss[1]
    model_filename = generate_output_filename(root_path + model_path, dataset + '/' + folder_names[0], 'mode_'+ mode,
                                              case_name, approach, loss, 'dim_' + str(dimension), 'n_classes_' +
                                              str(num_classes), str(patch_shape), str(extraction_step),
                                              data_augment_label, 'preproc_trn_opt_' + str(preprocess_trn), 'h5')

    csv_filename = generate_output_filename(root_path + log_path, dataset + '/' + folder_names[0], 'mode_'+ mode,
                                            case_name, approach, loss, 'dim_' + str(dimension), 'n_classes_' +
                                            str(num_classes), str(patch_shape), str(extraction_step),
                                            data_augment_label, 'preproc_trn_opt_' + str(preprocess_trn), 'cvs')

    # EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None)
    if multi_output == 1:
        if metric_opt in ['acc', 'acc_dc', 'loss']:
            metric_monitor = 'val_' + output_name[0] + '_' + metric_opt
        elif metric_opt == 'loss_total':
            metric_monitor = 'val_loss'
        else:
            print('unknown metric for early stopping')
            metric_monitor = None
    else:
        if attention_loss == 1:
            if metric_opt in ['acc', 'acc_dc', 'loss']:
                metric_monitor = 'val_' + output_name + '_' + metric_opt
            elif metric_opt == 'loss_total':
                metric_monitor = 'val_loss'
            else:
                print('unknown metric for early stopping')
                metric_monitor = None
        else:
            if metric_opt in ['acc', 'acc_dc', 'loss']:
                metric_monitor = 'val_' + metric_opt
            else:
                print('unknown metric for early stopping')
                metric_monitor = None

    stopper = EarlyStopping(monitor=metric_monitor, patience=train_conf['patience'])
    #For val_acc, 'mode' should be max, for val_loss this should be min, etc.
    #In auto mode, the direction is automatically inferred from the name of the monitored quantity.
    checkpointer = ModelCheckpoint(filepath=model_filename, monitor=metric_monitor, verbose=0, save_best_only=True,
                                   save_weights_only=True, mode='auto')
    csv_logger = CSVLogger(csv_filename, separator=';')

    if optimizer == 'SGD':
        # learning rate schedule for SGD (Adam, Adagrad, and RMSprop are adaptive learning rate methods and keras provides options)
        decay_method = 'step'
        if decay_method == 'step':
            def decay(epoch): # step_decay
                drop = 0.5
                epochs_drop = 10.0
                lrate = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
                return lrate
        elif decay_method == 'exp':
            def decay(epoch): # exp_decay
                k = 0.1
                lrate = initial_lr * np.exp(-k * t)
                return lrate
        elif decay_method == 'constant':
            def decay(epoch): # constant_decay
                decay_rate = 0.0001
                decay_step = 90
                if epoch % decay_step == 0 and epoch:
                    return initial_lr * decay_rate
                return initial_lr
        lrate = LearningRateScheduler(decay)
        return [stopper, checkpointer, csv_logger, lrate]
    else:
        return [stopper, checkpointer, csv_logger]


