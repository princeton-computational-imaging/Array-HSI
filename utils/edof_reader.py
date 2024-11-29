import numpy as np
import tensorflow as tf
import os
import matplotlib.image as mpimg
import scipy.io as sio
import h5py
from glob import glob

def file_match(filetype,root):

    files = []
    pattern   = "*.%s" %filetype
    for dir,_,_ in os.walk(root):
        files.extend(glob(os.path.join(dir,pattern))) 

    return files

def dataset_load(target_dir, filetype, color = True ):
    print("Start data loading.....")
    file_lists = file_match(filetype, target_dir)

    input_image = []
    if filetype == 'mat':
    
        for i in range(len(file_lists)):
            data_temp = sio.loadmat(file_lists[i]) 
            if np.mean(data_temp['ref'].astype(np.float32)) > 1e-5: # two image has mean of e-7
                input_image.append(data_temp['ref'].astype(np.float32))
    else:
        for i in range(len(file_lists)):
            data_temp =  mpimg.imread(file_lists[i])
            if color is False:
                data_temp =  np.mean(data_temp,axis = 2,keepdims = True)
            input_image.append(data_temp.astype(np.float32))

    image_shape = input_image[0].shape
    num_channels = image_shape[2]

    print("Finish loading %d files from %s" %(len(input_image), target_dir))
    return (input_image, len(input_image), num_channels)

def load_CAVE(target_dir, filetype = 'mat'):
    print("Start loading data from %s" % target_dir)
    file_lists = os.listdir(target_dir)

    input_image = []
    for file in file_lists:
        image = []
        for i in range(31):
            data = mpimg.imread(os.path.join(target_dir, file, file, "%s_%02d.png" % (file,i+1)))
            if data.ndim == 2:
                image.append(data[...,None])
            elif data.ndim == 3:
                image.append(np.mean(data, axis = -1, keepdims=True))
            else:
                assert False
        image = np.concatenate(image, axis = -1)
        input_image.append(image)

    image_shape = input_image[0].shape
    num_channels = image_shape[2]

    print("Finish loading %d files from %s" %(len(input_image), target_dir))
    return input_image
    
def load_hsdb(target_dir, filetype = 'mat'):
    print("Start loading data from %s" % target_dir)
    file_lists = file_match(filetype, target_dir)

    input_image = []
    for i in range(len(file_lists)):
        data_temp = sio.loadmat(file_lists[i]) 
        image = data_temp['ref'].astype(np.float32)
        input_image.append(image / np.max(image))

    image_shape = input_image[0].shape
    num_channels = image_shape[2]

    print("Finish loading %d files from %s" %(len(input_image), target_dir))
    return input_image

def load_ICVL(target_dir, filetype = 'mat'):
    print("Start loading data from %s" % target_dir)
    file_lists = file_match(filetype, target_dir)

    input_image = []
    for i in range(len(file_lists)):
        data_temp = h5py.File(file_lists[i], 'r') 
        image = np.flip(np.transpose(np.array(data_temp['rad']),[1,2,0]),0).astype(np.float32)
        input_image.append(image / np.max(image))

    image_shape = input_image[0].shape
    num_channels = image_shape[2]

    print("Finish loading %d files from %s" %(len(input_image), target_dir))
    return input_image

def load_Real(target_dir, filetype = 'mat'):
    print("Start loading data from %s" % target_dir)
    file_lists = file_match(filetype, target_dir)

    input_image = []
    for i in range(len(file_lists)):
        data_temp = sio.loadmat(file_lists[i]) 
        image = data_temp['GT_use'].astype(np.float32)
        input_image.append(image / np.max(image))

    image_shape = input_image[0].shape
    num_channels = image_shape[2]

    print("Finish loading %d files from %s" %(len(input_image), target_dir))
    return input_image   

def load_KAUST(target_dir, filetype = 'h5'):
    print("Start loading data from %s" % target_dir)
    file_lists = file_match(filetype, target_dir)

    input_image = []
    for i in range(len(file_lists)):
        data_temp = h5py.File(file_lists[i], 'r') 
        image = np.rot90(np.transpose(np.array(data_temp['img\\']),[1,2,0])[...,1:-2],-1).astype(np.float32)
        input_image.append(np.clip(image,0,1))

    image_shape = input_image[0].shape
    num_channels = image_shape[2]

    print("Finish loading %d files from %s" %(len(input_image), target_dir))
    return input_image

def measured_dataset(target_dir, files, patch_size):
    input_image = []
    for f in files:
        img = sio.loadmat(os.path.join(target_dir,f))['GT_use']
        GT_img = tf.convert_to_tensor(img, dtype=tf.float32)[None,...]/4095
        # resize
        GT_img = tf.image.central_crop(GT_img, central_fraction=0.3)
        GT_img = tf.image.resize(GT_img, (patch_size, patch_size))
        # flip
        GT_imgs = [GT_img, tf.image.flip_left_right(GT_img)]
        # roll
        for img in GT_imgs:
            for c in range(GT_img.shape[-1]):
                input_image.append(tf.roll(img, shift=c, axis=-1))

    print("Finish loading %d files from %s" %(len(input_image), target_dir))
    return input_image

def dataset_preprocess(image, patch_size, num_depths=4, is_val = False):

    H, W, C = image.shape

    if is_val: 
        image = tf.image.resize_with_crop_or_pad(image, patch_size, patch_size)
        image = image[tf.newaxis,...]
    else:

        # random crop
        image = tf.image.random_crop(image, (patch_size, patch_size, C))
        image = image[tf.newaxis,...]
        
        # random flip
        image = tf.image.random_flip_left_right(image)

        # random roll channels (change color)
        if np.random.uniform() > 0.5:
            c = np.random.randint(0,C)
            image = tf.roll(image, shift=c, axis=-1)

    return image