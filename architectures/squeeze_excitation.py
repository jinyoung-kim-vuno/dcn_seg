from keras.layers import GlobalAveragePooling2D, GlobalAveragePooling3D, Reshape, Dense, multiply, \
    Permute, Conv2D, Conv3D, add, concatenate, BatchNormalization
from keras import backend as K

K.set_image_dim_ordering('th')

def squeeze_excite_block2D(input, kernel_init, ratio=16):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer=kernel_init, use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer=kernel_init, use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def spatial_squeeze_channel_excite_block2D(input, kernel_init, ratio=2):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer=kernel_init, use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer=kernel_init, use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def channel_squeeze_spatial_excite_block2D(input, kernel_init, ratio=2):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]

    se = Conv2D(filters // ratio, (1, 1), activation='elu', kernel_initializer=kernel_init, use_bias=False)(init)
    se = Conv2D(filters // ratio, (3, 3), dilation_rate=(4, 4), kernel_initializer=kernel_init, padding='same',
                use_bias=False)(se)
    se = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer=kernel_init, use_bias=False)(se)

    x = multiply([init, se])

    return x


def channel_squeeze_spatial_excite_block2D_orginal(input, kernel_init):
    se = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer=kernel_init, use_bias=False)(input)
    x = multiply([input, se])

    return x


def spatial_and_channel_squeeze_excite_block2D(input, kernel_init, arrange_type='two_way_sequential', input_short_cut=True,
                                               final_conv=False):

    # ref. BMVC 2018, BAM: Bottleneck Attention Module
    #      ECCV 2018, CBAM: Convolutional Block Attention Module
    #      MICCAI 2018, Concurrent Spatial and Channel 'Squeeze & Excitation' in Fully Convolutional Networks
    #      ICLR 2019, Learning what and where to attend

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = input._keras_shape[channel_axis]

    if arrange_type == 'sequential':
        # channel_excitation (attention) first
        ssce_out = spatial_squeeze_channel_excite_block2D(input, kernel_init, ratio=2)
        csse_out = channel_squeeze_spatial_excite_block2D(ssce_out, kernel_init, ratio=2)

        if input_short_cut:
            if final_conv:
                concat = concatenate([input, csse_out], axis=channel_axis)
                output = Conv2D(filters, (1, 1), kernel_initializer=kernel_init, padding='same',
                                use_bias=False)(concat)
            else:
                output = add([input, csse_out])
        else:
            output = csse_out
    elif arrange_type == 'two_way_sequential':
        # channel_excitation (attention) first
        ssce_out1 = spatial_squeeze_channel_excite_block2D(input, kernel_init, ratio=2)
        csse_out1 = channel_squeeze_spatial_excite_block2D(ssce_out1, kernel_init, ratio=2)

        # spatial_excitation (attention) first
        csse_out2 = channel_squeeze_spatial_excite_block2D(input, kernel_init, ratio=2)
        ssce_out2 = spatial_squeeze_channel_excite_block2D(csse_out2, kernel_init, ratio=2)

        if input_short_cut:
            if final_conv:
                concat = concatenate([input, csse_out1, ssce_out2], axis=channel_axis)
                output = Conv2D(filters, (1, 1), kernel_initializer=kernel_init, padding='same',
                                use_bias=False)(concat)
            else:
                output = add([input, csse_out1, ssce_out2])
        else:
            if final_conv:
                concat = concatenate([csse_out1, ssce_out2], axis=channel_axis)
                output = Conv2D(filters, (1, 1), kernel_initializer=kernel_init, padding='same',
                                use_bias=False)(concat)
            else:
                output = add([csse_out1, ssce_out2])
    elif arrange_type == 'concurrent_scSE':
        ssce_out = spatial_squeeze_channel_excite_block2D(input, kernel_init, ratio=2)
        csse_out = channel_squeeze_spatial_excite_block2D_orginal(input, kernel_init)

        output = add([ssce_out, csse_out])
    else:
        ssce_out = spatial_squeeze_channel_excite_block2D(input, kernel_init, ratio=2)
        csse_out = channel_squeeze_spatial_excite_block2D(input, kernel_init, ratio=2)

        if input_short_cut:
            if final_conv:
                concat = concatenate([input, ssce_out, csse_out], axis=channel_axis)
                output = Conv2D(filters, (1, 1), kernel_initializer=kernel_init, padding='same',
                                use_bias=False)(concat)
            else:
                output = add([input, ssce_out, csse_out])
        else:
            if final_conv:
                concat = concatenate([ssce_out, csse_out], axis=channel_axis)
                output = Conv2D(filters, (1, 1), kernel_initializer=kernel_init, padding='same',
                                use_bias=False)(concat)
            else:
                output = add([ssce_out, csse_out])

    return output


def squeeze_excite_block3D(input, kernel_init, ratio=16):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    init = input
    print(input.shape)

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, 1, filters)

    se = GlobalAveragePooling3D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='elu', kernel_initializer=kernel_init, use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer=kernel_init, use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((4, 1, 2, 3))(se)

    x = multiply([init, se])
    return x


def spatial_squeeze_channel_excite_block3D(input, kernel_init, ratio=2):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, 1, filters)

    se = GlobalAveragePooling3D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='elu', kernel_initializer=kernel_init, use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer=kernel_init, use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((4, 1, 2, 3))(se)

    x = multiply([init, se])
    return x


def channel_squeeze_spatial_excite_block3D(input, kernel_init, ratio):

    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]

    se = Conv3D(filters // ratio, (1, 1, 1), activation='elu', kernel_initializer=kernel_init, use_bias=False)(init)
    se = Conv3D(filters // ratio, (3, 3, 3), dilation_rate=(4, 4, 4), kernel_initializer=kernel_init, padding='same',
                use_bias=False)(se)
    se = Conv3D(1, (1, 1, 1), activation='sigmoid', kernel_initializer=kernel_init, use_bias=False)(se)

    x = multiply([init, se])

    return x


def channel_squeeze_spatial_excite_block3D_orginal(input, kernel_init):
    se = Conv3D(1, (1, 1, 1), activation='sigmoid', kernel_initializer=kernel_init, use_bias=False)(input)
    x = multiply([input, se])

    return x


def spatial_and_channel_squeeze_excite_block3D(input, kernel_init, arrange_type='two_way_sequential', input_short_cut=True,
                                               final_conv=False):

    # ref. BMVC 2018, BAM: Bottleneck Attention Module
    #      ECCV 2018, CBAM: Convolutional Block Attention Module
    #      MICCAI 2018, Concurrent Spatial and Channel 'Squeeze & Excitation' in Fully Convolutional Networks
    #      ICLR 2019, Learning what and where to attend

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = input._keras_shape[channel_axis]

    if arrange_type == 'sequential':
        # channel_excitation (attention) first
        ssce_out = spatial_squeeze_channel_excite_block3D(input, kernel_init, ratio=2)
        csse_out = channel_squeeze_spatial_excite_block3D(ssce_out, kernel_init, ratio=2)

        if input_short_cut:
            if final_conv:
                concat = concatenate([input, csse_out], axis=channel_axis)
                output = Conv3D(filters, (1, 1, 1), kernel_initializer=kernel_init, padding='same',
                                use_bias=False)(concat)
            else:
                output = add([input, csse_out])
        else:
            output = csse_out
    elif arrange_type == 'two_way_sequential':
        # channel_excitation (attention) first
        ssce_out1 = spatial_squeeze_channel_excite_block3D(input, kernel_init, ratio=2)
        csse_out1 = channel_squeeze_spatial_excite_block3D(ssce_out1, kernel_init, ratio=2)

        # spatial_excitation (attention) first
        csse_out2 = channel_squeeze_spatial_excite_block3D(input, kernel_init, ratio=2)
        ssce_out2 = spatial_squeeze_channel_excite_block3D(csse_out2, kernel_init, ratio=2)

        if input_short_cut:
            if final_conv:
                concat = concatenate([input, csse_out1, ssce_out2], axis=channel_axis)
                output = Conv3D(filters, (1, 1, 1), kernel_initializer=kernel_init, padding='same',
                           use_bias=False)(concat)
            else:
                output = add([input, csse_out1, ssce_out2])
        else:
            if final_conv:
                concat = concatenate([csse_out1, ssce_out2], axis=channel_axis)
                output = Conv3D(filters, (1, 1, 1), kernel_initializer=kernel_init, padding='same',
                           use_bias=False)(concat)
            else:
                output = add([csse_out1, ssce_out2])
    elif arrange_type == 'concurrent_scSE':
        ssce_out = spatial_squeeze_channel_excite_block3D(input, kernel_init, ratio=2)
        csse_out = channel_squeeze_spatial_excite_block3D_orginal(input, kernel_init)

        output = add([ssce_out, csse_out])
    else:
        ssce_out = spatial_squeeze_channel_excite_block3D(input, kernel_init, ratio=2)
        csse_out = channel_squeeze_spatial_excite_block3D(input, kernel_init, ratio=2)

        if input_short_cut:
            if final_conv:
                concat = concatenate([input, ssce_out, csse_out], axis=channel_axis)
                output = Conv3D(filters, (1, 1, 1), kernel_initializer=kernel_init, padding='same',
                                use_bias=False)(concat)
            else:
                output = add([input, ssce_out, csse_out])
        else:
            if final_conv:
                concat = concatenate([ssce_out, csse_out], axis=channel_axis)
                output = Conv3D(filters, (1, 1, 1), kernel_initializer=kernel_init, padding='same',
                                use_bias=False)(concat)
            else:
                output = add([ssce_out, csse_out])

    return output