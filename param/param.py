import numpy as np
import models.optics as optics
import tensorflow as tf

data_dir = '/n/fs/pci-sharedt/Array_DOE/data/'

# optics.point_source_1D_layer
distance = 1000
wave_resolution = 750,750
wave_lengths = np.arange(429,701,9) *1e-9
optical_sampling_interval = 2e-6 
input_field1 = optics.point_source_1D_layer(distance, wave_resolution, optical_sampling_interval,wave_lengths, isfall = False)

# optics.DOE_1D_layer_array
refractive_idcs = optics.get_refractive_idcs(wave_lengths)  
max_height = 2e-6
array_num = 4
height_tolerance = 20e-9
hm_reg_scale = 0. # play with
height_map_init_value = np.random.rand(array_num,wave_resolution[0],1,1) * 2e-6

# optics.fresnel_Prop_layer_hankel_1D_2step
sensor_distance = 20e-3 
image_sampling_interval = 3.45e-6
output_size = 2048

# sensor_model
noise_min = 0.001
noise_max = 0.01

# training
color_sensor = True
q_tensor_rgb = optics.set_sensor_curve(wave_lengths,color_flag = color_sensor)
q_tensor = optics.set_sensor_curve_RGGB_array(wave_lengths)
response_weight_rgb = tf.convert_to_tensor(q_tensor_rgb,dtype=tf.float32)
response_weight_native = tf.convert_to_tensor(optics.set_sensor_curve_array_native(wave_lengths),dtype=tf.float32)
train_responsive_curve = False
num_depths = 1
channel_in = 31
channel_out = 31
