import numpy as np
import tensorflow as tf
import scipy.io as sio
from matplotlib import pyplot
import json
import os
import argparse

import param
import models.optics as optics
import utils.edof_reader as edof_reader
from train import build_model

class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value

parser = argparse.ArgumentParser()
parser.add_argument('--result_path', type=str, required=True,
                    help='Directory that checkpoints')
parser.add_argument('--ckpt', type=str, default='latest',
                    help='lastest checkpoint or a specific checkpoint')
eval_args = parser.parse_args()

result_path = eval_args.result_path
args = json.load(open(os.path.join(result_path,'args.json'),'r'))
args = AttributeDict(args)
param.noise_min = 0.008
param.noise_max = 0.0081
param.quantization = True
args.param = param

F = build_model(args.forward_model, args)
G = build_model(args.generator, args)

checkpoint = tf.train.Checkpoint(F = F, G = G)
manager = tf.train.CheckpointManager(checkpoint, directory=result_path, max_to_keep=10)
if eval_args.ckpt.lower() == 'latest':
    ckpt = manager.latest_checkpoint
else:
    ckpt = os.path.join(eval_args.result_path, eval_args.ckpt)
status = checkpoint.restore(ckpt).expect_partial()

dataset_image, dataset_length, num_channels = edof_reader.load_CAVE(os.path.join(param.data_dir, 'CAVE'))
file_lists = os.listdir(os.path.join(param.data_dir, 'CAVE'))

if not os.path.exists(os.path.join(result_path, 'output_hs')):
    os.makedirs(os.path.join(result_path, 'output_hs')) 
if not os.path.exists(os.path.join(result_path, 'output_rgb')):
    os.makedirs(os.path.join(result_path, 'output_rgb'))
if not os.path.exists(os.path.join(result_path, 'GT_hs')):
    os.makedirs(os.path.join(result_path, 'GT_hs'))
if not os.path.exists(os.path.join(result_path, 'GT_rgb')):
    os.makedirs(os.path.join(result_path, 'GT_rgb'))   


PSNR = []
PSNR_rgb = []
for i in range(dataset_length):
    x_input = dataset_image[i]
    X_val, _, _ = edof_reader.dataset_preprocess(x_input, patch_size = args.patch_size, num_depths= param.num_depths, is_val=True)
    output_image_sensor_org, [psfs,psf_temps], GT_img, response_curve, [height_map, height_map_2D] = F([args.param.input_field1, X_val], training = False)
    output_image, output_image_rgb = G(output_image_sensor_org, training = False)
    PSNR.append(tf.image.psnr(GT_img, output_image, 1, name=None)[0].numpy())
    GT_img_rgb =  tf.gather(tf.nn.conv2d(GT_img,args.param.response_weight_rgb,strides=[1,1,1,1],padding = 'SAME'), [2,1,0], axis = -1)
    output_image_rgb =  tf.gather(tf.clip_by_value(output_image_rgb,0.,1.), [2,1,0], axis = -1)
    PSNR_rgb.append(tf.image.psnr(GT_img_rgb, output_image_rgb, 1, name=None)[0].numpy())

    file_name = file_lists[i]

    np.save(os.path.join(result_path, 'output_hs', file_name + '.npy'),output_image[0].numpy())
    np.save(os.path.join(result_path, 'output_rgb', file_name + '.npy'),output_image_rgb[0].numpy())
    np.save(os.path.join(result_path, 'GT_hs', file_name + '.npy'),GT_img[0].numpy())
    np.save(os.path.join(result_path, 'GT_rgb', file_name + '.npy'),GT_img_rgb[0].numpy())
    

f = open(os.path.join(result_path, 'PSNR.txt'), "w")
f.write(ckpt + '\n')
f.write("Multispectral PSNR: %f \n" % np.mean(PSNR))
f.write("RGB PSNR: %f" % np.mean(PSNR_rgb))
f.close()

print(ckpt)
print("Multispectral PSNR: %f" % np.mean(PSNR))
print("RGB PSNR: %f" % np.mean(PSNR_rgb))
