from keras import backend as K
from keras.layers import Activation, Add, Multiply, UpSampling2D, UpSampling3D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.convolutional import Conv3D, Conv3DTranspose
from keras.layers.normalization import BatchNormalization

K.set_image_dim_ordering('th')


def grid_attention(dimension, input, gate, in_channels, out_channels, mode, kernel_init):

    kernel_size = (1, 1) if dimension == 2 else (1, 1, 1)
    sub_sample_factor = (2, 2) if dimension == 2 else (2, 2, 2)
    sub_sample_kernel_size = sub_sample_factor

    # Define the operation
    if mode == 'concatenation':
        return _concatenation(dimension, input, gate, in_channels, out_channels, kernel_size, sub_sample_factor,
                              sub_sample_kernel_size, 'relu', kernel_init)
    elif mode == 'concatenation_debug':
        return _concatenation(dimension, input, gate, in_channels, out_channels, kernel_size, sub_sample_factor,
                              sub_sample_kernel_size, 'softplus', kernel_init)
    elif mode == 'concatenation_residual':
        return _concatenation_residual(dimension, input, gate, in_channels, out_channels, kernel_size,
                                       sub_sample_factor, sub_sample_kernel_size, 'relu', kernel_init)
    else:
        raise NotImplementedError('Unknown operation function.')


def _concatenation(dimension, input, gate, in_channels, out_channels, kernel_size, sub_sample_factor,
                   sub_sample_kernel_size, activation, kernel_init):

    if dimension == 2:
        # Theta^T * x_ij + Phi^T * gating_signal + bias
        theta_x = Conv2D(out_channels, kernel_size=sub_sample_kernel_size, strides=sub_sample_factor,
                         padding='same', kernel_initializer=kernel_init)(input)
        phi_g = Conv2D(out_channels, kernel_size=kernel_size, strides=kernel_size, padding='same',
                       kernel_initializer=kernel_init)(gate)

        f = Activation(activation)(Add()([theta_x, phi_g]))
        psi_f = Conv2D(1, kernel_size=kernel_size, strides=kernel_size, padding='same',
                       kernel_initializer=kernel_init)(f)
        sigm_psi_f = Activation('sigmoid')(psi_f)

        # upsample the attentions and multiply
        #sigm_psi_f_up = Conv2DTranspose(1, kernel_size=kernel_size, strides=sub_sample_factor)(sigm_psi_f)
        sigm_psi_f_up = UpSampling2D(size=sub_sample_factor, interpolation='bilinear')(sigm_psi_f)

        y = Multiply()([sigm_psi_f_up, input])
        #print('y: ', y)

        # worked here

        # Output transform
        w = Conv2D(in_channels, kernel_size=kernel_size, padding='same', kernel_initializer=kernel_init)(y)
        w = BatchNormalization(axis=1)(w)
        #print(w)

    elif dimension == 3:
        # Theta^T * x_ij + Phi^T * gating_signal + bias
        theta_x = Conv3D(out_channels, kernel_size=sub_sample_kernel_size, strides=sub_sample_factor,
                         padding='same', kernel_initializer=kernel_init)(input)
        phi_g = Conv3D(out_channels, kernel_size=kernel_size, strides=kernel_size, padding='same',
                       kernel_initializer=kernel_init)(gate)

        f = Activation(activation)(Add()([theta_x, phi_g]))
        psi_f = Conv3D(1, kernel_size=kernel_size, strides=kernel_size, padding='same', kernel_initializer=kernel_init)(f)
        sigm_psi_f = Activation('sigmoid')(psi_f)

        # upsample the attentions and multiply
        #sigm_psi_f_up = Conv3DTranspose(1, kernel_size=kernel_size, strides=sub_sample_factor)(sigm_psi_f)
        sigm_psi_f_up = UpSampling3D(size=sub_sample_factor)(sigm_psi_f)

        y = Multiply()([sigm_psi_f_up, input])
        #print('y: ', y)

        # worked here

        # Output transform
        w = Conv3D(in_channels, kernel_size=kernel_size, padding='same', kernel_initializer=kernel_init)(y)
        w = BatchNormalization(axis=1)(w)
        #print(w)

    else:
        raise NotImplemented

    return w, sigm_psi_f_up


def grid_attention_up(dimension, input, gate, in_channels, out_channels, mode, kernel_init):

    kernel_size = (1, 1) if dimension == 2 else (1, 1, 1)
    sub_sample_factor = (2, 2) if dimension == 2 else (2, 2, 2)
    sub_sample_kernel_size = sub_sample_factor

    # Define the operation
    if mode == 'concatenation':
        return _concatenation_up(dimension, input, gate, in_channels, out_channels, kernel_size, sub_sample_factor,
                              sub_sample_kernel_size, 'relu', kernel_init)
    else:
        raise NotImplementedError('Unknown operation function.')


def _concatenation_up(dimension, input, gate, in_channels, out_channels, kernel_size, sub_sample_factor,
                   sub_sample_kernel_size, activation, kernel_init):

    if dimension == 2:
        # Theta^T * x_ij + Phi^T * gating_signal + bias
        theta_x = Conv2DTranspose(out_channels, kernel_size=sub_sample_kernel_size, strides=sub_sample_factor,
                                  kernel_initializer=kernel_init)(input)
        phi_g = Conv2D(out_channels, kernel_size=kernel_size, strides=kernel_size, padding='same',
                       kernel_initializer=kernel_init)(gate)
        f = Activation(activation)(Add()([theta_x, phi_g]))

        psi_f = Conv2D(1, kernel_size=kernel_size, strides=kernel_size, padding='same', kernel_initializer=kernel_init)(f)
        sigm_psi_f = Activation('sigmoid')(psi_f)

        # upsample the attentions and multiply
        sigm_psi_f_up = Conv2DTranspose(1, kernel_size=kernel_size, strides=sub_sample_factor,
                                        kernel_initializer=kernel_init)(sigm_psi_f)

        y = Multiply()([sigm_psi_f_up, input])
        #print('y: ', y)

        # worked here

        # Output transform
        w = Conv2D(in_channels, kernel_size=kernel_size, padding='same', kernel_initializer=kernel_init)(y)
        w = BatchNormalization(axis=1)(w)
        #print(w)

    elif dimension == 3:
        # Theta^T * x_ij + Phi^T * gating_signal + bias
        theta_x = Conv3DTranspose(out_channels, kernel_size=sub_sample_kernel_size, strides=sub_sample_factor,
                                  kernel_initializer=kernel_init)(input)
        phi_g = Conv3D(out_channels, kernel_size=kernel_size, strides=kernel_size, padding='same',
                       kernel_initializer=kernel_init)(gate)
        f = Activation(activation)(Add()([theta_x, phi_g]))

        psi_f = Conv3D(1, kernel_size=kernel_size, strides=kernel_size, padding='same', kernel_initializer=kernel_init)(f)
        sigm_psi_f = Activation('sigmoid')(psi_f)

        # upsample the attentions and multiply
        sigm_psi_f_up = Conv3DTranspose(1, kernel_size=kernel_size, strides=sub_sample_factor, padding='same',
                                        kernel_initializer=kernel_init)(sigm_psi_f)

        y = Multiply()([sigm_psi_f_up, input])
        #print('y: ', y)

        # worked here

        # Output transform
        w = Conv3D(in_channels, kernel_size=kernel_size, padding='same', kernel_initializer=kernel_init)(y)
        w = BatchNormalization(axis=1)(w)
        #print(w)

    else:
        raise NotImplemented

    return w, sigm_psi_f_up


def _concatenation_residual(dimension, input, gate, in_channels, out_channels, kernel_size, sub_sample_factor,
                            sub_sample_kernel_size, activation, kernel_init):

    if dimension == 2:
        # Theta^T * x_ij + Phi^T * gating_signal + bias
        theta_x = Conv2D(out_channels, kernel_size=sub_sample_kernel_size, strides=sub_sample_factor,
                         padding='same', kernel_initializer=kernel_init)(input)
        phi_g = Conv2D(out_channels, kernel_size=kernel_size, strides=kernel_size, padding='same',
                       kernel_initializer=kernel_init)(gate)
        f = Activation(activation)(Add()([theta_x, phi_g]))

        psi_f = Conv2D(1, kernel_size=kernel_size, strides=kernel_size, padding='same',
                       kernel_initializer=kernel_init)(f)
        sigm_psi_f = Activation('softmax')(psi_f)

        # upsample the attentions and multiply
        sigm_psi_f_up = Conv2DTranspose(1, kernel_size=kernel_size, strides=sub_sample_factor,
                                        kernel_initializer=kernel_init)(sigm_psi_f)

        y = Multiply()([sigm_psi_f_up, input])
        #print('y: ', y)

        # worked here

        # Output transform
        w = Conv2D(in_channels, kernel_size=kernel_size, padding='same', kernel_initializer=kernel_init)(y)
        w = BatchNormalization(axis=1)(w)
        #print(w)

    elif dimension == 3:
        # Theta^T * x_ij + Phi^T * gating_signal + bias
        theta_x = Conv3D(out_channels, kernel_size=sub_sample_kernel_size, strides=sub_sample_factor,
                         padding='same', kernel_initializer=kernel_init)(input)
        phi_g = Conv3D(out_channels, kernel_size=kernel_size, strides=kernel_size, padding='same',
                       kernel_initializer=kernel_init)(gate)
        f = Activation(activation)(Add()([theta_x, phi_g]))

        psi_f = Conv3D(1, kernel_size=kernel_size, strides=kernel_size, padding='same',
                       kernel_initializer=kernel_init)(f)
        sigm_psi_f = Activation('softmax')(psi_f)

        # upsample the attentions and multiply
        sigm_psi_f_up = Conv3DTranspose(1, kernel_size=kernel_size, strides=sub_sample_factor,
                                        kernel_initializer=kernel_init)(sigm_psi_f)

        y = Multiply()([sigm_psi_f_up, input])
        #print('y: ', y)

        # worked here

        # Output transform
        w = Conv3D(in_channels, kernel_size=kernel_size, padding='same', kernel_initializer=kernel_init)(y)
        w = BatchNormalization(axis=1)(w)
        #print(w)

    else:
        raise NotImplemented

    return w, sigm_psi_f_up


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