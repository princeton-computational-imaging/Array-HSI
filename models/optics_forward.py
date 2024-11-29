import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

import models.optics as optics

def forward_model(patch_size, param):
    input_field1 = Input(shape=[param.wave_resolution[0], 1, param.channel_in],dtype=tf.complex64)
    input_image = Input(shape=[patch_size, patch_size, param.channel_in])
    inputs = [input_field1, input_image]
    input_field2, height_map, height_map_2D = optics.DOE_1D_array(input_field1.shape[1], param.wave_lengths, param.refractive_idcs, max_height = param.max_height, array_num = param.array_num,
                                   height_tolerance = param.height_tolerance, height_map_regularizer =optics.laplace_l1_regularizer(param.hm_reg_scale),
                                   height_map_init_value = param.height_map_init_value, trainable_flag = True, quantization = param.quantization, isfall = False)(input_field1)
    
    # height_map (4, 750, 1, 1)  
    psf_temp = optics.fresnel_Prop_layer_hankel_1D_2step(input_field2,
                                                     param.sensor_distance,
                                                     param.optical_sampling_interval,
                                                     param.image_sampling_interval,
                                                     param.wave_lengths,
                                                     param.output_size)
    # psf_temp: (4, 2048, 1, 31)
    crop_range = np.arange(patch_size) 
    psf_temp = tf.gather(psf_temp,crop_range,axis = 1)
    # psf_temp: (4, 384, 1, 31)
    psf = optics.psf1D_to_2D_half(psf_temp, isDiv = False)
    # psf: (4, 768, 768, 31)
    psf = tf.nn.avg_pool(psf,
                    [1, 2, 2, 1],
                     strides=[1, 2, 2, 1],
                     padding="SAME")
    psf = tf.math.divide(psf,tf.reduce_sum(psf, axis = [1,2], keepdims = True))
    if param.noise_max > 0:
        noise_sigma = tf.random.uniform(minval=param.noise_min, maxval=param.noise_max, shape=[])
    else:
        noise_sigma = 0
    output_image_sensor_org, psfs, GT_img, response_curve = optics.sensor_sample_array(param.q_tensor, param.response_weight_native, noise_sigma = noise_sigma, noise_model = optics.gaussian_noise,trainable_flag = param.train_responsive_curve,use_psf_encoding = True)([psf,input_image])    
    return Model(inputs, [output_image_sensor_org, [psfs, psf_temp], GT_img, response_curve, [height_map, height_map_2D]], name="OpticalForward")

def forward_model_exp(input_image, param):
    # use this forward model for experimentally measured PSFs
    psf = tf.math.divide(param.psfs,tf.reduce_sum(param.psfs, axis = [1,2], keepdims = True))
    input_image = optics.img_psf_conv(input_image, psf, circular=False)
    sensor_img = input_image * tf.transpose(param.response_weight * param.response_weight_native,[3,0,1,2])
    sensor_img = tf.reduce_sum(sensor_img,axis=[3], keepdims=True)
    sensor_img_noisy = optics.simulate_gaussian_poisson_noise(sensor_img, param.gaussian, param.poisson)
    sensor_img_noisy = tf.transpose(sensor_img_noisy,[3,1,2,0])
    return sensor_img_noisy
