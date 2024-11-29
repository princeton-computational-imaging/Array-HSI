import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model,Sequential

def conv(filters, size, stride, activation, apply_instnorm=True):
    result = Sequential()
    result.add(Conv2D(filters, size, stride, padding='SAME', use_bias=True))
    if apply_instnorm:
        result.add(InstanceNormalization())
    if not activation == None:
        result.add(activation())
    return result

def conv_transp(filters, size, stride, activation, apply_instnorm=True):
    result = Sequential()
    result.add(Conv2DTranspose(filters, size, stride, padding='SAME', use_bias=True))
    if not activation == None:
        result.add(activation())
    return result

# limited by the GPU RAM available to the authors, we had to train the 2 reconstruction heads seperately

def HR_UNet(patch_size, param,nf = 32):

    noisy_img  = Input(shape=[None, None, param.array_num])

    x_00   = conv(nf , 5, 1, LeakyReLU, apply_instnorm=False)(noisy_img)
    x_00   = conv(nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_00)

    x_01 = conv(nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_00)
    x_01 = conv(nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_01)

    x_11   = conv(2*nf , 3, 2, LeakyReLU, apply_instnorm=False)(x_00)
    x_11   = conv(2*nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_11)
    x_11   = conv(2*nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_11)

    x_02 = tf.concat([x_01, conv_transp(nf, 2, 2, ReLU, apply_instnorm=False)(x_11)], axis = -1)
    x_02 = conv(nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_02)
    x_02 = conv(nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_02)

    x_12 = tf.concat([x_11, conv(2*nf , 3, 2, LeakyReLU, apply_instnorm=False)(x_01)], axis = -1)
    x_12 = conv(2*nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_12)
    x_12 = conv(2*nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_12)   

    x_22 = tf.concat([conv(4*nf , 3, 2, LeakyReLU, apply_instnorm=False)(x_11), conv(4*nf , 3, 4, LeakyReLU, apply_instnorm=False)(x_01)], axis = -1)
    x_22   = conv(4*nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_22)
    x_22   = conv(4*nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_22)    

    x_03= tf.concat([x_02, conv_transp(nf, 2, 2, ReLU, apply_instnorm=False)(x_12), conv_transp(nf, 2, 4, ReLU, apply_instnorm=False)(x_22)], axis = -1)
    x_03 = conv(nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_03)
    x_03 = conv(nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_03)

    x_13= tf.concat([x_11, x_12, conv(2*nf , 3, 2, LeakyReLU, apply_instnorm=False)(x_02), conv_transp(2*nf, 2, 2, ReLU, apply_instnorm=False)(x_22)], axis = -1)
    x_13 = conv(2*nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_13)
    x_13 = conv(2*nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_13)

    x_04= tf.concat([x_00, x_03, conv_transp(nf, 2, 2, ReLU, apply_instnorm=False)(x_13)], axis = -1)
    x_04 = conv(nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_04)
    x_04 = conv(nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_04)
    
    output_ms = conv(param.channel_out, 5, 1, ReLU, apply_instnorm=False)(x_04)
    output_rgb = tf.nn.conv2d(output_ms,param.response_weight_rgb,strides=[1,1,1,1],padding = 'SAME')

    return Model(noisy_img, [output_ms, output_rgb],name = "HR_UNet")

def HR_UNet_duotask(patch_size, param,nf = 32):

    noisy_img  = Input(shape=[None, None, param.array_num])

    x_00   = conv(nf , 5, 1, LeakyReLU, apply_instnorm=False)(noisy_img)
    x_00   = conv(nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_00)

    x_01 = conv(nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_00)
    x_01 = conv(nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_01)

    x_11   = conv(2*nf , 3, 2, LeakyReLU, apply_instnorm=False)(x_00)
    x_11   = conv(2*nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_11)
    x_11   = conv(2*nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_11)

    x_02 = tf.concat([x_01, conv_transp(nf, 2, 2, ReLU, apply_instnorm=False)(x_11)], axis = -1)
    x_02 = conv(nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_02)
    x_02 = conv(nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_02)

    x_12 = tf.concat([x_11, conv(2*nf , 3, 2, LeakyReLU, apply_instnorm=False)(x_01)], axis = -1)
    x_12 = conv(2*nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_12)
    x_12 = conv(2*nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_12)   

    x_22 = tf.concat([conv(4*nf , 3, 2, LeakyReLU, apply_instnorm=False)(x_11), conv(4*nf , 3, 4, LeakyReLU, apply_instnorm=False)(x_01)], axis = -1)
    x_22   = conv(4*nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_22)
    x_22   = conv(4*nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_22)    

    x_03= tf.concat([x_02, conv_transp(nf, 2, 2, ReLU, apply_instnorm=False)(x_12), conv_transp(nf, 2, 4, ReLU, apply_instnorm=False)(x_22)], axis = -1)
    x_03 = conv(nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_03)
    x_03 = conv(nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_03)

    x_13= tf.concat([x_11, x_12, conv(2*nf , 3, 2, LeakyReLU, apply_instnorm=False)(x_02), conv_transp(2*nf, 2, 2, ReLU, apply_instnorm=False)(x_22)], axis = -1)
    x_13 = conv(2*nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_13)
    x_13 = conv(2*nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_13)

    x_04= tf.concat([x_00, x_03, conv_transp(nf, 2, 2, ReLU, apply_instnorm=False)(x_13)], axis = -1)
    x_04 = conv(nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_04)
    x_04 = conv(nf , 3, 1, LeakyReLU, apply_instnorm=False)(x_04)
    
    output_ms = conv(param.channel_out, 5, 1, ReLU, apply_instnorm=False)(x_04)
    output_rgb = conv(3, 5, 1, ReLU, apply_instnorm=False)(x_04)

    return Model(noisy_img, [output_ms, output_rgb],name = "HR_UNet_duotask")