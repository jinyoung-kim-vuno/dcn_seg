
import numpy as np
from math import floor
from keras import backend as K
from keras.layers import Activation, Add, Multiply, UpSampling2D, UpSampling3D, Lambda, Concatenate, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv3D
from keras.regularizers import l2
from keras.initializers import he_normal
import tensorflow as tf

K.set_image_dim_ordering('th')


def grid_attention(dimension, fused_f, gate, d_lst, scale, nb_filter, kernel_init, weight_decay=1e-4):

    #kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)
    kernel_size = (1, 1) if dimension == 2 else (1, 1, 1)
    sub_sample_factor = (2, 2) if dimension == 2 else (2, 2, 2)

    #fused_f_size = np.int(fused_f._keras_shape[2])
    #d = max(1, floor(fused_f_size / kernel_size[0]))
    #d = 1

    #d = 8 / (pow(2, scale - 1))
    d = d_lst[scale-1]
    dilation_rate = (d, d) if dimension == 2 else (d, d, d)

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    if dimension == 2:
        fused_f = Conv3D(nb_filter, (1, 1), kernel_initializer=kernel_init, padding='same', use_bias=False,
                         kernel_regularizer=l2(weight_decay))(fused_f)
        gate = Conv3D(nb_filter, (1, 1), kernel_initializer=kernel_init, padding='same', use_bias=False,
                         kernel_regularizer=l2(weight_decay))(gate)
        gate_up = UpSampling2D(size=sub_sample_factor)(gate)
        fused_gate = Add()([fused_f, gate_up])
        channel_wise_avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=channel_axis, keepdims=True))(fused_gate)
        print(channel_wise_avg_pool._keras_shape)
        channel_wise_max_pool = Lambda(lambda x: tf.reduce_max(x, axis=channel_axis, keepdims=True))(fused_gate)
        print(channel_wise_max_pool._keras_shape)
        channel_wise_squeeze = Conv2D(1, (1, 1), kernel_initializer=kernel_init, use_bias=False)(fused_gate)
        print(channel_wise_squeeze._keras_shape)
        concate_avg_max_squeeze = Concatenate(axis=channel_axis)([channel_wise_avg_pool, channel_wise_max_pool,
                                                          channel_wise_squeeze])
        #bn = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(concate_avg_max_squeeze)
        f = Activation('elu')(concate_avg_max_squeeze)
        sigmoid_in = Conv2D(1, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='same',
                            kernel_initializer=kernel_init)(f)

    elif dimension == 3:
        fused_f = Conv3D(nb_filter, (1, 1, 1), kernel_initializer=kernel_init, padding='same', use_bias=False,
                         kernel_regularizer=l2(weight_decay))(fused_f)
        gate = Conv3D(nb_filter, (1, 1, 1), kernel_initializer=kernel_init, padding='same', use_bias=False,
                         kernel_regularizer=l2(weight_decay))(gate)
        gate_up = UpSampling3D(size=sub_sample_factor)(gate)
        fused_gate = Add()([fused_f, gate_up])
        channel_wise_avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=channel_axis, keepdims=True))(fused_gate)
        print(channel_wise_avg_pool._keras_shape)
        channel_wise_max_pool = Lambda(lambda x: tf.reduce_max(x, axis=channel_axis, keepdims=True))(fused_gate)
        print(channel_wise_max_pool._keras_shape)
        channel_wise_squeeze = Conv3D(1, (1, 1, 1), kernel_initializer=kernel_init, use_bias=False)(fused_gate)
        print(channel_wise_squeeze._keras_shape)
        concate_avg_max_squeeze = Concatenate(axis=channel_axis)([channel_wise_avg_pool, channel_wise_max_pool,
                                                          channel_wise_squeeze])
        #bn = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(concate_avg_max_squeeze)
        f = Activation('elu')(concate_avg_max_squeeze)
        sigmoid_in = Conv3D(1, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='same',
                            kernel_initializer=kernel_init)(f)

    else:
        raise NotImplemented

    compatibility_score = Activation('sigmoid')(sigmoid_in)
    attention_map = Multiply()([compatibility_score, fused_f])

    return attention_map, compatibility_score


def get_gate_signal(dimension, input, num_filters, kernel_init):
    x = None
    kernel_size = (1, 1) if dimension == 2 else (1, 1, 1)
    if dimension == 2 :
        x = Conv2D(num_filters, kernel_size=kernel_size, padding='same', kernel_initializer=kernel_init)(input)
        x = BatchNormalization(axis=1)(x)
        x = Activation('relu')(x)

    else :
        x = Conv3D(num_filters, kernel_size=kernel_size, padding='same', kernel_initializer=kernel_init)(input)
        x = BatchNormalization(axis=1)(x)
        x = Activation('relu')(x)

    return x