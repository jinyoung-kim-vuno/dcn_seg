import numpy as np
from keras import backend as K

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Reshape, Conv2D, Conv3D, Conv2DTranspose, \
    Conv3DTranspose, UpSampling2D, UpSampling3D, MaxPooling2D, MaxPooling3D, AveragePooling2D, AveragePooling3D, \
    GlobalMaxPooling2D, GlobalMaxPooling3D, GlobalAveragePooling2D, GlobalAveragePooling3D, concatenate, \
    BatchNormalization, Activation, Add, Multiply, Cropping2D, Cropping3D
from keras.regularizers import l2
from keras.layers.core import Permute
from keras.engine.topology import get_source_inputs
from keras_contrib.layers import SubPixelUpscaling
from keras.initializers import he_normal

from utils import loss_functions, metrics, optimizers_builtin
from .squeeze_excitation import spatial_and_channel_squeeze_excite_block2D, spatial_and_channel_squeeze_excite_block3D

K.set_image_dim_ordering('th')

# Ref.
# Jegou et al., CVPRW 17, "The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation"
# Kamnitsas et al., MedIA17, "Efficient multi-scale 3D CNN with fully connected CRF for accurate brain lesion segmentation"


def generate_fc_densenet_ms(gen_conf, train_conf) :

    '''Instantiate the DenseNet FCN architecture.
        Note that when using TensorFlow,
        for best performance you should set
        `image_data_format='channels_last'` in your Keras config
        at ~/.keras/keras.json.
        # Arguments
            nb_dense_block: number of dense blocks to add to end (generally = 3)
            growth_rate: number of filters to add per dense block
            nb_layers_per_block: number of layers in each dense block.
                Can be a positive integer or a list.
                If positive integer, a set number of layers per dense block.
                If list, nb_layer is used as provided. Note that list size must
                be (nb_dense_block + 1)
            reduction: reduction factor of transition blocks.
                Note : reduction value is inverted to compute compression.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            init_conv_filters: number of layers in the initial convolution layer
            include_top: whether to include the fully-connected
                layer at the top of the network.
            weights: one of `None` (random initialization) or
                'cifar10' (pre-training on CIFAR-10)..
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(32, 32, 3)` (with `channels_last` dim ordering)
                or `(3, 32, 32)` (with `channels_first` dim ordering).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 8.
                E.g. `(200, 200, 3)` would be one valid value.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
            activation: Type of activation at the top layer. Can be one of 'softmax'
                or 'sigmoid'. Note that if sigmoid is used, classes must be 1.
            upsampling_type: Can be one of 'deconv', 'upsampling' and
                'subpixel'. Defines type of upsampling algorithm used.
            batchsize: Fixed batch size. This is a temporary requirement for
                computation of output shape in the case of Deconvolution2D layers.
                Parameter will be removed in next iteration of Keras, which infers
                output shape of deconvolution layers automatically.
            early_transition: Start with an extra initial transition down and end with
                an extra transition up to reduce the network size.
            initial_kernel_size: The first Conv2D kernel might vary in size based on the
                application, this parameter makes it configurable.
        # Returns
            A Keras model instance.
    '''

    dataset = train_conf['dataset']
    activation = train_conf['activation']
    dimension = train_conf['dimension']
    num_classes = gen_conf['num_classes']
    modality = gen_conf['dataset_info'][dataset]['image_modality']
    num_modality = len(modality)
    expected_output_shape = train_conf['output_shape']
    patch_shape = train_conf['patch_shape']

    loss_opt = train_conf['loss']
    metric_opt = train_conf['metric']
    optimizer = train_conf['optimizer']
    initial_lr = train_conf['initial_lr']
    is_set_random_seed = train_conf['is_set_random_seed']
    random_seed_num = train_conf['random_seed_num']
    exclusive_train = train_conf['exclusive_train']

    if is_set_random_seed == 1:
        kernel_init = he_normal(seed=random_seed_num)
    else:
        kernel_init = he_normal()
    # #dentate
    # nb_dense_block = 2 # for dentate, 3 #for thalamus
    # nb_layers_per_block = [4,5,6] # for dentate [3,4,5,6] #for thalamus #[3,4,5,6] # 4 #
    # growth_rate = 16 # 8 for thalamus
    #dropout_rate = 0.2

    #thalamus
    nb_dense_block = 3 # for dentate 3 #for thalamus
    nb_layers_per_block = [3,4,5,6] #[3,4,5,6] # for dentate[3,4,5,6] #for thalamus #[3,4,5,6] # 4 #
    growth_rate = 8 # 8 for thalamus
    dropout_rate = 0.2 #None, #0.2

    reduction = 0.0
    weight_decay = 1E-4
    init_conv_filters = 16 #16 #32
    include_top = True
    weights = None
    input_tensor = None
    upsampling_type = 'deconv'
    transition_pooling = 'max'
    early_transition = False

    multi_channel = True #True
    multi_scale_input = True #True
    glam = True #True
    glam_arrange_type = 'two_way_sequential' #'two_way_sequential' #concurrent_scSE
    glam_input_short_cut = True #True #False
    glam_final_conv = False # False: addition # True: applying fully connected conv layer (1x1x1) after concatenation instead of adding up
    glam_position = 'before_shortcut' #before_shortcut #after_shortcut

    if dimension == 2:
        initial_kernel_size = (3, 3)
    else:
        initial_kernel_size = (3, 3, 3)

    if exclusive_train == 1:
        num_classes -= 1
    input_shape = (num_modality, ) + patch_shape
    output_shape = (num_classes, np.prod(expected_output_shape))

    if weights not in {None}:
        raise ValueError('The `weights` argument should be '
                         '`None` (random initialization) as no '
                         'model weights are provided.')

    upsampling_type = upsampling_type.lower()

    if upsampling_type not in ['upsampling', 'deconv', 'subpixel']:
        raise ValueError('Parameter "upsampling_type" must be one of "upsampling", '
                         '"deconv" or "subpixel".')

    if input_shape is None:
        raise ValueError('For fully convolutional models, input shape must be supplied.')

    if type(nb_layers_per_block) is not list and nb_dense_block < 1:
        raise ValueError('Number of dense layers per block must be greater than 1. '
                         'Argument value was %s.' % nb_layers_per_block)

    if activation not in ['softmax', 'sigmoid']:
        raise ValueError('activation must be one of "softmax" or "sigmoid"')

    if activation == 'sigmoid' and num_classes != 1:
        raise ValueError('sigmoid activation can only be used when classes = 1')

    assert dimension in [2, 3]

    model = __generate_fc_densenet_ms(dimension, num_classes, kernel_init, random_seed_num, input_shape, output_shape,
                                      input_tensor, nb_dense_block, nb_layers_per_block, growth_rate, reduction,
                                      dropout_rate, weight_decay, init_conv_filters, include_top,
                                      upsampling_type, activation, early_transition, transition_pooling,
                                      initial_kernel_size, multi_channel, multi_scale_input, glam, glam_arrange_type,
                                      glam_input_short_cut, glam_final_conv, glam_position)

    model.summary()
    model.compile(loss=loss_functions.select(num_classes, loss_opt),
                  optimizer=optimizers_builtin.select(optimizer, initial_lr),
                  metrics=metrics.select(metric_opt))

    return model


def __generate_fc_densenet_ms(dimension, num_classes, kernel_init, random_seed_num, input_shape=None, output_shape=None,
                              input_tensor=None, nb_dense_block=5, nb_layers_per_block=4, growth_rate=12, reduction=0.0,
                              dropout_rate=None, weight_decay=1e-4, init_conv_filters=48, include_top=None,
                              upsampling_type='deconv', activation='softmax', early_transition=False,
                              transition_pooling='max', initial_kernel_size=(3, 3, 3), multi_channel=True,
                              multi_scale_input=True, glam=True, glam_arrange_type='two_way_sequential',
                              glam_input_short_cut=True, glam_final_conv=False, glam_position='before_shortcut'):

    ''' Build the DenseNet-FCN model
    # Arguments
        nb_classes: number of classes
        img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        include_top: flag to include the final Dense layer
        nb_dense_block: number of dense blocks to add to end (generally = 3)
        growth_rate: number of filters to add per dense block
        reduction: reduction factor of transition blocks. Note : reduction value
            is inverted to compute compression
        dropout_rate: dropout rate
        weight_decay: weight decay
        nb_layers_per_block: number of layers in each dense block.
            Can be a positive integer or a list.
            If positive integer, a set number of layers per dense block.
            If list, nb_layer is used as provided. Note that list size must
            be (nb_dense_block + 1)
        upsampling_type: Can be one of 'upsampling', 'deconv' and 'subpixel'. Defines
            type of upsampling algorithm used.
        input_shape: Only used for shape inference in fully convolutional networks.
        activation: Type of activation at the top layer. Can be one of 'softmax' or
            'sigmoid'. Note that if sigmoid is used, classes must be 1.
        early_transition: Start with an extra initial transition down and end with an
            extra transition up to reduce the network size.
        transition_pooling: 'max' for max pooling (default), 'avg' for average pooling,
            None for no pooling. Please note that this default differs from the DenseNet
            paper in accordance with the DenseNetFCN paper.
        initial_kernel_size: The first Conv2D kernel might vary in size based on the
            application, this parameter makes it configurable.
    # Returns
        a keras tensor
    # Raises
        ValueError: in case of invalid argument for `reduction`,
            `nb_dense_block`.
    '''
    with K.name_scope('DenseNetFCN'):

        # Determine proper input shape
        min_size = 2 ** nb_dense_block

        if K.image_data_format() == 'channels_first':

            if dimension == 2:
                if input_shape is not None:
                    if ((input_shape[1] is not None and input_shape[1] < min_size) or
                            (input_shape[2] is not None and input_shape[2] < min_size)):
                        raise ValueError('Input size must be at least ' +
                                         str(min_size) + 'x' + str(min_size) +
                                         ', got `input_shape=' + str(input_shape) + '`')
            else:
                if input_shape is not None:
                    if ((input_shape[1] is not None and input_shape[1] < min_size) or
                            (input_shape[2] is not None and input_shape[2] < min_size) or
                            (input_shape[3] is not None and input_shape[3] < min_size)):
                        raise ValueError('Input size must be at least ' +
                                         str(min_size) + 'x' + str(min_size) +
                                         ', got `input_shape=' + str(input_shape) + '`')
        else:
            if dimension == 2:
                if input_shape is not None:
                    if ((input_shape[0] is not None and input_shape[0] < min_size) or
                            (input_shape[1] is not None and input_shape[1] < min_size)):
                        raise ValueError('Input size must be at least ' +
                                         str(min_size) + 'x' + str(min_size) +
                                         ', got `input_shape=' + str(input_shape) + '`')
            else:
                if input_shape is not None:
                    if ((input_shape[0] is not None and input_shape[0] < min_size) or
                            (input_shape[1] is not None and input_shape[1] < min_size) or
                            (input_shape[2] is not None and input_shape[2] < min_size)):
                        raise ValueError('Input size must be at least ' +
                                         str(min_size) + 'x' + str(min_size) +
                                         ', got `input_shape=' + str(input_shape) + '`')

        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        if concat_axis == 1:  # channels_first dim ordering
            if dimension == 2:
                _, rows, cols = input_shape
            else:
                _, rows, cols, axes = input_shape
        else:
            if dimension == 2:
                rows, cols, _ = input_shape
            else:
                rows, cols, axes, _ = input_shape

        if reduction != 0.0:
            if not (reduction <= 1.0 and reduction > 0.0):
                raise ValueError('`reduction` value must lie between 0.0 and 1.0')

        # layers in each dense block
        if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
            nb_layers = list(nb_layers_per_block)  # Convert tuple to list

            if len(nb_layers) != (nb_dense_block + 1):
                raise ValueError('If `nb_dense_block` is a list, its length must be '
                                 '(`nb_dense_block` + 1)')

            bottleneck_nb_layers = nb_layers[-1]
            rev_layers = nb_layers[::-1]
            nb_layers.extend(rev_layers[1:])
        else:
            bottleneck_nb_layers = nb_layers_per_block
            nb_layers = [nb_layers_per_block] * (2 * nb_dense_block + 1)

        # compute compression factor
        compression = 1.0 - reduction

        if multi_channel:
            # img_input 64x64x64 patches for thalamus, 32x32x32 patches for dentate
            input_crop_size = np.int(img_input._keras_shape[2] / 4)
            normal_res_input = get_cropping_layer(dimension, img_input, crop_size=(input_crop_size, input_crop_size)) # local features
            low_res_input = get_low_res_layer(dimension, img_input, type='average') # global contextual features
        else:
            normal_res_input = img_input
            low_res_input = None

        x_normal, nb_filter, block_prefix, skip_list_normal_res = conv_dense_transition_down(dimension,
                                                                                             normal_res_input,
                                                                                             init_conv_filters,
                                                                                             initial_kernel_size,
                                                                                             kernel_init,
                                                                                             random_seed_num,
                                                                                             weight_decay,
                                                                                             concat_axis,
                                                                                             early_transition,
                                                                                             compression,
                                                                                             transition_pooling,
                                                                                             nb_dense_block,
                                                                                             nb_layers,
                                                                                             growth_rate,
                                                                                             dropout_rate,
                                                                                             multi_scale_input,
                                                                                             glam,
                                                                                             glam_arrange_type,
                                                                                             glam_input_short_cut,
                                                                                             glam_final_conv,
                                                                                             glam_position,
                                                                                             block_prefix_num=0)
        if multi_channel:
            x_low_res, _, _, skip_list_low_res = conv_dense_transition_down(dimension, low_res_input, init_conv_filters,
                                                                            initial_kernel_size, kernel_init,
                                                                            random_seed_num, weight_decay, concat_axis,
                                                                            early_transition, compression,
                                                                            transition_pooling, nb_dense_block,
                                                                            nb_layers, growth_rate, dropout_rate,
                                                                            multi_scale_input, glam, glam_arrange_type,
                                                                            glam_input_short_cut, glam_final_conv,
                                                                            glam_position, block_prefix_num=1)
        else:
            x_low_res = None
            skip_list_low_res = None

        bottleneck_out = bottleneck_block(dimension, x_normal, x_low_res, weight_decay, concat_axis,
                                          bottleneck_nb_layers, nb_filter, kernel_init, random_seed_num, growth_rate,
                                          dropout_rate, block_prefix, multi_channel, glam, glam_arrange_type,
                                          glam_input_short_cut, glam_final_conv, glam_position)

        skip_list = [skip_list_normal_res, skip_list_low_res]
        output_temp = transition_up_dense_conv(dimension, bottleneck_out, weight_decay, concat_axis, early_transition,
                                               nb_dense_block, nb_layers, kernel_init, random_seed_num, growth_rate,
                                               upsampling_type, skip_list, dropout_rate, include_top, num_classes,
                                               multi_channel, glam, glam_arrange_type, glam_input_short_cut,
                                               glam_final_conv, glam_position)

        output = organise_output(output_temp, output_shape, activation)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            input = get_source_inputs(input_tensor)
        else:
            input = img_input

        print(input.shape)
        print(output.shape)

        return Model(inputs=[input], outputs=[output], name='ms_fc_densenet')


def organise_output(input, output_shape, activation) :
    pred = Reshape(output_shape)(input)
    pred = Permute((2, 1))(pred)
    return Activation(activation)(pred)


def conv_dense_transition_down(dimension, img_input, init_conv_filters, initial_kernel_size, kernel_init,
                               random_seed_num, weight_decay, concat_axis, early_transition, compression,
                               transition_pooling, nb_dense_block, nb_layers, growth_rate, dropout_rate,
                               multi_scale_input, glam, glam_arrange_type, glam_input_short_cut, glam_final_conv,
                               glam_position, block_prefix_num):

    # Initial convolution
    if dimension == 2:
        x = Conv2D(init_conv_filters, initial_kernel_size,
                   kernel_initializer=kernel_init, padding='same',
                   name='initial_conv2D_%i' % block_prefix_num, use_bias=False,
                   kernel_regularizer=l2(weight_decay))(img_input)
    else:
        x = Conv3D(init_conv_filters, initial_kernel_size,
                   kernel_initializer=kernel_init, padding='same',
                   name='initial_conv3D_%i' % block_prefix_num, use_bias=False,
                   kernel_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5, name='initial_bn_%i' % block_prefix_num)(x)
    x = Activation('elu')(x)

    nb_filter = init_conv_filters

    skip_list = []

    if early_transition:
        x = __transition_down_block(dimension, x, nb_filter, kernel_init, random_seed_num, compression=compression,
                               weight_decay=weight_decay, dropout_rate=dropout_rate,
                               block_prefix='tr_early_%i' % block_prefix_num,
                               transition_pooling=transition_pooling)

    # Add dense blocks and transition down block
    for block_idx in range(nb_dense_block):
        if glam:
            org_input = x
        else:
            org_input = None

        x, nb_filter, concat_list = __dense_block(dimension, x, nb_layers[block_idx], kernel_init, random_seed_num,
                                                  nb_filter, growth_rate, dropout_rate=dropout_rate,
                                                  weight_decay=weight_decay, return_concat_list=True,
                                                  block_prefix='dense_%i_%i' % (block_idx, block_prefix_num))

        if glam:
            if glam_position == 'before_shortcut':
                l = concatenate(concat_list[1:], axis=concat_axis)
                if dimension == 2:
                    l = spatial_and_channel_squeeze_excite_block2D(l, kernel_init, arrange_type=glam_arrange_type,
                                                                   input_short_cut=glam_input_short_cut,
                                                                   final_conv=glam_final_conv)
                else:
                    l = spatial_and_channel_squeeze_excite_block3D(l, kernel_init, arrange_type=glam_arrange_type,
                                                                   input_short_cut=glam_input_short_cut,
                                                                   final_conv=glam_final_conv)
                x = concatenate([org_input, l], axis=concat_axis)
            else:
                if dimension == 2:
                    x = spatial_and_channel_squeeze_excite_block2D(x, kernel_init, arrange_type=glam_arrange_type,
                                                                   input_short_cut=glam_input_short_cut,
                                                                   final_conv=glam_final_conv)
                else:
                    x = spatial_and_channel_squeeze_excite_block3D(x, kernel_init, arrange_type=glam_arrange_type,
                                                                   input_short_cut=glam_input_short_cut,
                                                                   final_conv=glam_final_conv)

        # Skip connection
        skip_list.append(x)

        # add transition_block
        x = __transition_down_block(dimension, x, nb_filter, kernel_init, random_seed_num, compression=compression,
                                    weight_decay=weight_decay, dropout_rate=dropout_rate,
                                    block_prefix='tr_%i_%i' % (block_idx, block_prefix_num),
                                    transition_pooling=transition_pooling)

        if multi_scale_input and block_idx < nb_dense_block-1:
            img_input = get_low_res_layer(dimension, img_input, type='max')
            if dimension == 2:
                img_input_conv = Conv2D(nb_filter, initial_kernel_size,
                           kernel_initializer=kernel_init, padding='same',
                           name='multi_input_conv2D_%i_%i' % (block_idx, block_prefix_num), use_bias=False,
                           kernel_regularizer=l2(weight_decay))(img_input)
            else:
                img_input_conv = Conv3D(nb_filter, initial_kernel_size,
                           kernel_initializer=kernel_init, padding='same',
                           name='multi_input_conv3D_%i_%i' % (block_idx, block_prefix_num), use_bias=False,
                           kernel_regularizer=l2(weight_decay))(img_input)
            img_input_bn = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                                              name='multi_input_bn_%i_%i' % (block_idx, block_prefix_num))(img_input_conv)
            img_input_activation = Activation('elu')(img_input_bn)
            x = concatenate([x, img_input_activation], axis=concat_axis)
            nb_filter += nb_filter

        # this is calculated inside transition_down_block
        nb_filter = int(nb_filter * compression)

    # The last dense_block does not have a transition_down_block
    # return the concatenated feature maps without the concatenation of the input
    block_prefix = 'dense_%i_%i' % (nb_dense_block, block_prefix_num)

    skip_list = skip_list[::-1]  # reverse the skip list

    return x, nb_filter, block_prefix, skip_list


def bottleneck_block(dimension, x_local, x_global, weight_decay, concat_axis, bottleneck_nb_layers, nb_filter,
                     kernel_init, random_seed_num, growth_rate, dropout_rate, block_prefix, multi_channel, glam,
                     glam_arrange_type, glam_input_short_cut, glam_final_conv, glam_position):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    #input_channels = x_local._keras_shape[channel_axis]

    if multi_channel:
        db_input = __transition_block(dimension, x_local, x_global, nb_filter, kernel_init, random_seed_num,
                                      weight_decay=weight_decay, dropout_rate=dropout_rate,
                                      block_prefix='bottleneck_input')
        bottleneck = True
    else:
        db_input = x_local
        bottleneck = False

    _, _, concat_list = __dense_block(dimension, db_input, bottleneck_nb_layers, kernel_init, random_seed_num, nb_filter,
                                      growth_rate, bottleneck=bottleneck, dropout_rate=dropout_rate,
                                      weight_decay=weight_decay, return_concat_list=True, block_prefix=block_prefix)

    l = concatenate(concat_list[1:], axis=concat_axis)
    if glam:
        if dimension == 2:
            l = spatial_and_channel_squeeze_excite_block2D(l, kernel_init, arrange_type=glam_arrange_type,
                                                           input_short_cut=glam_input_short_cut,
                                                           final_conv=glam_final_conv)
        else:
            l = spatial_and_channel_squeeze_excite_block3D(l, kernel_init, arrange_type=glam_arrange_type,
                                                           input_short_cut=glam_input_short_cut,
                                                           final_conv=glam_final_conv)

    return l

def transition_up_dense_conv(dimension, l, weight_decay, concat_axis, early_transition, nb_dense_block,
                             nb_layers, kernel_init, random_seed_num, growth_rate, upsampling_type, skip_list,
                             dropout_rate, include_top, num_classes, multi_channel, glam, glam_arrange_type,
                             glam_input_short_cut, glam_final_conv, glam_position):

    # Add dense blocks and transition up block
    for block_idx in range(nb_dense_block):
        n_filters_keep = growth_rate * nb_layers[nb_dense_block + block_idx]

        # upsampling block must upsample only the feature maps (concat_list[1:]),
        # not the concatenation of the input with the feature maps (concat_list[0].
        # if block_idx == 0:
        #     l = concatenate(concat_list[1:], axis=concat_axis)

        t = __transition_up_block(dimension, l, kernel_init, nb_filters=n_filters_keep, type=upsampling_type,
                                  weight_decay=weight_decay, block_prefix='tr_up_%i' % block_idx, stride=2)

        if multi_channel:
            skip_connection = __transition_block(dimension, skip_list[0][block_idx], skip_list[1][block_idx],
                                                 n_filters_keep, kernel_init, random_seed_num, weight_decay=weight_decay,
                                                 dropout_rate=dropout_rate, block_prefix='skip_tr_%i' % block_idx)
            db_input = concatenate([t, skip_connection], axis=concat_axis)
            bottleneck = True
        else:
            db_input = concatenate([t, skip_list[0][block_idx]], axis=concat_axis)
            bottleneck = False

        # Dont allow the feature map size to grow in upsampling dense blocks
        block_layer_index = nb_dense_block + 1 + block_idx
        block_prefix = 'dense_%i' % (block_layer_index)
        x_up, nb_filter, concat_list = __dense_block(dimension,
                                                     db_input,
                                                     nb_layers[block_layer_index],
                                                     kernel_init,
                                                     random_seed_num,
                                                     nb_filter=growth_rate,
                                                     growth_rate=growth_rate,
                                                     bottleneck=bottleneck,
                                                     dropout_rate=dropout_rate,
                                                     weight_decay=weight_decay,
                                                     return_concat_list=True,
                                                     grow_nb_filters=False,
                                                     block_prefix=block_prefix)

        l = concatenate(concat_list[1:], axis=concat_axis)
        if glam:
            if block_idx < nb_dense_block - 1:
                if dimension == 2:
                    l = spatial_and_channel_squeeze_excite_block2D(l, kernel_init, arrange_type=glam_arrange_type,
                                                                   input_short_cut=glam_input_short_cut,
                                                                   final_conv=glam_final_conv)
                else:
                    l = spatial_and_channel_squeeze_excite_block3D(l, kernel_init, arrange_type=glam_arrange_type,
                                                                   input_short_cut=glam_input_short_cut,
                                                                   final_conv=glam_final_conv)

            else:
                if glam_position == 'before_shortcut':
                    if dimension == 2:
                        l = spatial_and_channel_squeeze_excite_block2D(l, kernel_init, arrange_type=glam_arrange_type,
                                                                       input_short_cut=glam_input_short_cut,
                                                                       final_conv=glam_final_conv)
                    else:
                        l = spatial_and_channel_squeeze_excite_block3D(l, kernel_init, arrange_type=glam_arrange_type,
                                                                       input_short_cut=glam_input_short_cut,
                                                                       final_conv=glam_final_conv)
                    x_up = concatenate([db_input, l], axis=concat_axis)
                else:
                    if dimension == 2:
                        x_up = spatial_and_channel_squeeze_excite_block2D(x_up, kernel_init,
                                                                          arrange_type=glam_arrange_type,
                                                                          input_short_cut=glam_input_short_cut,
                                                                          final_conv=glam_final_conv)
                    else:
                        x_up = spatial_and_channel_squeeze_excite_block3D(x_up, kernel_init,
                                                                          arrange_type=glam_arrange_type,
                                                                          input_short_cut=glam_input_short_cut,
                                                                          final_conv=glam_final_conv)

    if early_transition:
        x_up = __transition_up_block(dimension, x_up, kernel_init, nb_filters=nb_filter, type=upsampling_type,
                                     weight_decay=weight_decay, block_prefix='tr_up_early', stride=2)
    if include_top:
        if dimension == 2:
            x = Conv2D(num_classes, (1, 1), activation='linear', padding='same', kernel_initializer=kernel_init,
                       use_bias=False)(x_up)
        else:
            x = Conv3D(num_classes, (1, 1, 1), activation='linear', padding='same', kernel_initializer=kernel_init,
                       use_bias=False)(x_up)
        output = x
    else:
        output = x_up

    return output


def __conv_block(dimension, ip, nb_filter, kernel_init, random_seed_num, bottleneck=False, dropout_rate=None,
                 weight_decay=1e-4, block_prefix=None):
    '''
    Adds a convolution layer (with batch normalization and elu),
    and optionally a bottleneck layer.
    # Arguments
        ip: Input tensor
        nb_filter: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        bottleneck: if True, adds a bottleneck convolution block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        block_prefix: str, for unique layer naming
     # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    # Returns
        output tensor of block
    '''
    with K.name_scope('ConvBlock'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                               name=name_or_none(block_prefix, '_bn'))(ip)
        x = Activation('elu')(x)

        if bottleneck:
            inter_channel = nb_filter * 4

            if dimension == 2:
                x = Conv2D(inter_channel, (1, 1), kernel_initializer=kernel_init,
                           padding='same', use_bias=False,
                           kernel_regularizer=l2(weight_decay),
                           name=name_or_none(block_prefix, '_bottleneck_conv2D'))(x)
            else:
                x = Conv3D(inter_channel, (1, 1, 1), kernel_initializer=kernel_init,
                           padding='same', use_bias=False,
                           kernel_regularizer=l2(weight_decay),
                           name=name_or_none(block_prefix, '_bottleneck_conv3D'))(x)
            x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                                   name=name_or_none(block_prefix, '_bottleneck_bn'))(x)
            x = Activation('elu')(x)

        if dimension == 2:
            x = Conv2D(nb_filter, (3, 3), kernel_initializer=kernel_init, padding='same',
                       use_bias=False, name=name_or_none(block_prefix, '_conv2D'))(x)
        else:
            x = Conv3D(nb_filter, (3, 3, 3), kernel_initializer=kernel_init, padding='same',
                       use_bias=False, name=name_or_none(block_prefix, '_conv3D'))(x)
        if dropout_rate:
            x = Dropout(dropout_rate, seed=random_seed_num)(x)

    return x


def __dense_block(dimension, x, nb_layers, kernel_init, random_seed_num, nb_filter, growth_rate, bottleneck=False,
                  dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True,
                  return_concat_list=False, block_prefix=None):
    '''
    Build a dense_block where the output of each conv_block is fed
    to subsequent ones
    # Arguments
        x: input keras tensor
        nb_layers: the number of conv_blocks to append to the model
        nb_filter: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        growth_rate: growth rate of the dense block
        bottleneck: if True, adds a bottleneck convolution block to
            each conv_block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: if True, allows number of filters to grow
        return_concat_list: set to True to return the list of
            feature maps along with the actual output
        block_prefix: str, for block unique naming
    # Return
        If return_concat_list is True, returns a list of the output
        keras tensor, the number of filters and a list of all the
        dense blocks added to the keras tensor
        If return_concat_list is False, returns a list of the output
        keras tensor and the number of filters
    '''
    with K.name_scope('DenseBlock'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x_list = [x]

        for i in range(nb_layers):
            cb = __conv_block(dimension, x, growth_rate, kernel_init, random_seed_num, bottleneck, dropout_rate,
                              weight_decay, block_prefix=name_or_none(block_prefix, '_%i' % i))
            x_list.append(cb)

            x = concatenate([x, cb], axis=concat_axis)

            if grow_nb_filters:
                nb_filter += growth_rate

        if return_concat_list:
            return x, nb_filter, x_list
        else:
            return x, nb_filter


def __transition_down_block(dimension, ip, nb_filter, kernel_init, random_seed_num,  compression=1.0, weight_decay=1e-4,
                            dropout_rate=None, block_prefix=None, transition_pooling='max'):
    '''
    Adds a pointwise convolution layer (with batch normalization and elu),
    and an average pooling layer. The number of output convolution filters
    can be reduced by appropriately reducing the compression parameter.
    # Arguments
        ip: input keras tensor
        nb_filter: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        compression: calculated as 1 - reduction. Reduces the number
            of feature maps in the transition block.
        weight_decay: weight decay factor
        block_prefix: str, for block unique naming
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, nb_filter * compression, rows / 2, cols / 2)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows / 2, cols / 2, nb_filter * compression)`
        if data_format='channels_last'.
    # Returns
        a keras tensor
    '''
    with K.name_scope('Transition'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                               name=name_or_none(block_prefix, '_bn'))(ip)
        x = Activation('elu')(x)

        if dimension == 2:
            x = Conv2D(int(nb_filter * compression), (1, 1), kernel_initializer=kernel_init,
                       padding='same', use_bias=False, kernel_regularizer=l2(weight_decay),
                       name=name_or_none(block_prefix, '_conv2D'))(x)
            if dropout_rate:
                x = Dropout(dropout_rate, seed=random_seed_num)(x)
            if transition_pooling == 'avg':
                x = AveragePooling2D((2, 2), strides=(2, 2))(x)
            elif transition_pooling == 'max':
                x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        else:
            x = Conv3D(int(nb_filter * compression), (1, 1, 1), kernel_initializer=kernel_init,
                       padding='same', use_bias=False, kernel_regularizer=l2(weight_decay),
                       name=name_or_none(block_prefix, '_conv3D'))(x)
            if dropout_rate:
                x = Dropout(dropout_rate, seed=random_seed_num)(x)
            if transition_pooling == 'avg':
                x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(x)
            elif transition_pooling == 'max':
                x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(x)

        return x


def __transition_up_block(dimension, ip, kernel_init, nb_filters, type='deconv', weight_decay=1E-4,
                          block_prefix=None, stride=2):
    '''Adds an upsampling block. Upsampling operation relies on the the type parameter.
    # Arguments
        ip: input keras tensor
        nb_filters: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        type: can be 'upsampling', 'subpixel', 'deconv'. Determines
            type of upsampling performed
        weight_decay: weight decay factor
        block_prefix: str, for block unique naming
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, nb_filter, rows * 2, cols * 2)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows * 2, cols * 2, nb_filter)` if data_format='channels_last'.
    # Returns
        a keras tensor
    '''
    with K.name_scope('TransitionUp'):
        strides = (stride, stride) if dimension == 2 else (stride, stride, stride)

        if dimension == 2:
            if type == 'upsampling':
                x = UpSampling2D(name=name_or_none(block_prefix, '_upsampling'))(ip)
            elif type == 'subpixel':
                x = Conv2D(nb_filters, (3, 3), activation='elu', padding='same',
                           kernel_regularizer=l2(weight_decay), use_bias=False,
                           kernel_initializer=kernel_init,
                           name=name_or_none(block_prefix, '_conv2D'))(ip)
                x = SubPixelUpscaling(scale_factor=2,
                                      name=name_or_none(block_prefix, '_subpixel'))(x)
                x = Conv2D(nb_filters, (3, 3), activation='elu', padding='same',
                           kernel_regularizer=l2(weight_decay), use_bias=False,
                           kernel_initializer=kernel_init,
                           name=name_or_none(block_prefix, '_conv2D'))(x)
            else:
                x = Conv2DTranspose(nb_filters, (3, 3), activation='elu', padding='same',
                                    strides=strides, kernel_initializer=kernel_init,
                                    kernel_regularizer=l2(weight_decay),
                                    name=name_or_none(block_prefix, '_conv2DT'))(ip)

        else:
            if type == 'upsampling':
                x = UpSampling3D(name=name_or_none(block_prefix, '_upsampling'))(ip)
            elif type == 'subpixel':
                x = Conv3D(nb_filters, (3, 3, 3), activation='elu', padding='same',
                           kernel_regularizer=l2(weight_decay), use_bias=False,
                           kernel_initializer=kernel_init,
                           name=name_or_none(block_prefix, '_conv3D'))(ip)
                x = SubPixelUpscaling(scale_factor=2,
                                      name=name_or_none(block_prefix, '_subpixel'))(x)
                x = Conv3D(nb_filters, (3, 3, 3), activation='elu', padding='same',
                           kernel_regularizer=l2(weight_decay), use_bias=False,
                           kernel_initializer=kernel_init,
                           name=name_or_none(block_prefix, '_conv3D'))(x)
            else:
                x = Conv3DTranspose(nb_filters, (3, 3, 3), activation='elu', padding='same',
                                    strides=strides, kernel_initializer=kernel_init,
                                    kernel_regularizer=l2(weight_decay),
                                    name=name_or_none(block_prefix, '_conv3DT'))(ip)

        return x


def __transition_block(dimension, normal, low_res, nb_filter, kernel_init, random_seed_num, weight_decay=1e-4,
                       dropout_rate=None, block_prefix=None):

    with K.name_scope('Transition'):
        #channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        sub_sample_factor = (2, 2) if dimension == 2 else (2, 2, 2)
        #strides = (2, 2) if dimension == 2 else (2, 2, 2)
        # x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
        #                        name=name_or_none(block_prefix, '_bn'))(ip)
        # x = Activation('elu')(x)

        low_res_crop_size = np.int((low_res._keras_shape[2] * sub_sample_factor[0] - low_res._keras_shape[2]) / 2)

        if dimension == 2:
            normal = Conv2D(nb_filter, (1, 1), kernel_initializer=kernel_init,
                       padding='same', use_bias=False, kernel_regularizer=l2(weight_decay),
                       name=name_or_none(block_prefix, '_conv2D_normal'))(normal)
            low_res = Conv2D(nb_filter, (1, 1), kernel_initializer=kernel_init,
                       padding='same', use_bias=False, kernel_regularizer=l2(weight_decay),
                       name=name_or_none(block_prefix, '_conv2D_low_res'))(low_res)
            low_res_up = UpSampling2D(size=sub_sample_factor, interpolation='bilinear')(low_res)
            low_res_up_crop = get_cropping_layer(dimension, low_res_up, crop_size=(low_res_crop_size, low_res_crop_size))

            # low_res_up = Conv2DTranspose(nb_filter, (1, 1), activation='elu', padding='same',
            #                     strides=strides, kernel_initializer=kernel_init,
            #                     kernel_regularizer=l2(weight_decay),
            #                     name=name_or_none(block_prefix, '_conv2DT'))(low_res)
            # if dropout_rate:
            #     x = Dropout(dropout_rate, seed=random_seed_num)(x)
        else:
            normal = Conv3D(nb_filter, (1, 1, 1), kernel_initializer=kernel_init,
                       padding='same', use_bias=False, kernel_regularizer=l2(weight_decay),
                       name=name_or_none(block_prefix, '_conv3D_normal'))(normal)
            low_res = Conv3D(nb_filter, (1, 1, 1), kernel_initializer=kernel_init,
                       padding='same', use_bias=False, kernel_regularizer=l2(weight_decay),
                       name=name_or_none(block_prefix, '_conv3D_low_res'))(low_res)
            low_res_up = UpSampling3D(size=sub_sample_factor)(low_res)
            low_res_up_crop = get_cropping_layer(dimension, low_res_up, crop_size=(low_res_crop_size, low_res_crop_size))

            # low_res_up = Conv3DTranspose(nb_filter, (1, 1, 1), activation='elu', padding='same',
            #                     strides=strides, kernel_initializer=kernel_init,
            #                     kernel_regularizer=l2(weight_decay),
            #                     name=name_or_none(block_prefix, '_conv3DT'))(low_res)


        x = Add()([normal, low_res_up_crop])

            # if dropout_rate:
            #     x = Dropout(dropout_rate, seed=random_seed_num)(x)
        # x = BatchNormalization(axis=channel_axis, epsilon=1.1e-5,
        #                        name=name_or_none(block_prefix, '_bn'))(x)
        # x = Activation('elu')(x)

        return x


def get_cropping_layer(dimension, input, crop_size=(16, 16)) :
    cropping_param = (crop_size, crop_size) if dimension == 2 else (crop_size, crop_size, crop_size)

    if dimension == 2 :
        return Cropping2D(cropping=cropping_param)(input)
    else :
        return Cropping3D(cropping=cropping_param)(input)


def get_low_res_layer(dimension, input, type='max'):
    if dimension == 2 :
        if type == 'average':
            return AveragePooling2D()(input)
        else:
            return MaxPooling2D()(input)
    else :
        if type == 'average':
            return AveragePooling3D()(input)
        else:
            return MaxPooling3D()(input)


def name_or_none(prefix, name):
    return prefix + name if (prefix is not None and name is not None) else None

