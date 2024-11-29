import abc
import tensorflow as tf
from tensorflow.keras.layers import *

import numpy as np
from scipy import interpolate
import scipy.io as sio
from scipy import special

import fractions
import poppy
import os
import csv
##############################
# optical materials
##############################

def mid_fiter(img):
    
    M = np.size(img,axis=1)
    out = np.zeros([1,M,1,1])
    for i in range(M):      # 10,20
        if i < M//10:
            kernel_size = 3
        elif i>=M//10 and i < M//20:
            kernel_size = 2
        else:
            kernel_size = 1
        imin = np.maximum(0,i-kernel_size)
        imax = np.minimum(M-1,i+kernel_size)
        out[0,i,0,0] = np.median(img[0,imin:imax+1,0,0])
    return out
    
def generate_C_for_div(height):
    [x, y] = np.mgrid[-height:height,
                 -height:height]
        
    R = np.sqrt(x**2 +y**2).astype(np.int32) 
    R = np.reshape(R,(-1))
            
    A = np.zeros(len(R))
    B = np.ones(len(R))
    C = []
    for i in np.arange(height):
        CC = (np.where(R==i,B,A))
        CC = CC[np.newaxis,:]
        C.append(np.sum(CC, axis = 1))   
    C = np.sum(C,axis = 1)
    C = C[np.newaxis,:,np.newaxis,np.newaxis]
    return C

def get_refractive_idcs_slica(wavelengths):
    _wavelengths = wavelengths * 1e6
    _refractive_idcs = np.sqrt(1 + 0.6961663*_wavelengths**2/(_wavelengths**2-0.0684043**2) +
                                  0.4079426*_wavelengths**2/(_wavelengths**2-0.1162414**2) +
                                  0.8974794*_wavelengths**2/(_wavelengths**2-9.896161**2))
    return _refractive_idcs


def get_refractive_idcs_pdms(wavelengths):
    _wavelengths = wavelengths * 1e6
    _refractive_idcs = np.sqrt(1 + 1.0047*_wavelengths**2/(_wavelengths**2-0.013217))
    return _refractive_idcs

def get_refractive_idcs_NOA61(wavelengths):
    _wavelengths = wavelengths * 1e6
    _refractive_idcs = 1.5375 + 0.00829045/(_wavelengths**2) - 0.000211046/(_wavelengths**4)
    return _refractive_idcs

def get_refractive_idcs(wavelengths):
    _wavelengths = wavelengths * 1e6
    _refractive_idcs = 1.4982 + 0.0053362/(_wavelengths**2) - 0.000058521/(_wavelengths**4)
    return _refractive_idcs

def  get_native_response_curve(wavelengths, name='param/native_response_curve.csv'):
    _wavelengths = wavelengths *1e9
    with open(name, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)
        x = []
        y = []
        # extracting each data row one by one
        for row in csvreader:
            x.append(float(row[0]))
            y.append(float(row[1]))
    f_linear = interpolate.interp1d(np.array(x), np.array(y))
    qutm_efficience = f_linear(_wavelengths)
    return qutm_efficience

def get_respose_curve(wavelengths, name = 'QE_R.txt'):
    _wavelengths = wavelengths *1e9
    tmp = np.loadtxt(name)
    f_linear = interpolate.interp1d(tmp[:,0],tmp[:,1])
    qutm_efficience = f_linear(_wavelengths)
    return qutm_efficience/np.sum(qutm_efficience)

def set_sensor_curve(wave_lengths, color_flag = True):
    
    if color_flag:     
          q_r = get_respose_curve(wavelengths = wave_lengths, name = 'param/QE_R.txt')
          q_g = get_respose_curve(wavelengths = wave_lengths, name = 'param/QE_G.txt')
          q_b = get_respose_curve(wavelengths = wave_lengths, name = 'param/QE_B.txt')

          q_r_tensor = np.reshape(q_r, (1,1,len(q_r),1))
          q_g_tensor = np.reshape(q_g, (1,1,len(q_g),1))
          q_b_tensor = np.reshape(q_b, (1,1,len(q_b),1))

          q_tensor = np.concatenate((q_b_tensor,q_g_tensor, q_r_tensor),axis = 3)
    else:
          q_mono = get_respose_curve(wavelengths = wave_lengths, name = 'param/QE_mono.txt')
          q_tensor = np.reshape(q_mono, (1,1,len(q_mono),1))

    return q_tensor

def set_sensor_curve_array(wave_lengths):
       
    q_r = get_respose_curve(wavelengths = wave_lengths, name = 'param/QE_R.txt')
    q_g = get_respose_curve(wavelengths = wave_lengths, name = 'param/QE_G.txt')
    q_b = get_respose_curve(wavelengths = wave_lengths, name = 'param/QE_B.txt')

    q_r_tensor = np.reshape(q_r, (1,1,len(q_r),1))
    q_g_tensor = np.reshape(q_g, (1,1,len(q_g),1))
    q_b_tensor = np.reshape(q_b, (1,1,len(q_b),1))

    q_tensor = np.concatenate((q_b_tensor,q_g_tensor,q_g_tensor,q_r_tensor),axis = 3)

    return q_tensor

def set_sensor_curve_array_native(wave_lengths):
       
    q = get_native_response_curve(wave_lengths)

    q_tensor = np.reshape(q, (1,1,len(q),1))
    q_tensor = np.concatenate((q_tensor,q_tensor,q_tensor,q_tensor),axis = 3)

    return q_tensor

def set_sensor_curve_RGGB_array(wave_lengths):
       
    q_r = get_respose_curve(wavelengths = wave_lengths, name = 'param/QE_R.txt')
    q_g = get_respose_curve(wavelengths = wave_lengths, name = 'param/QE_G.txt')
    q_b = get_respose_curve(wavelengths = wave_lengths, name = 'param/QE_B.txt')

    q_r_tensor = np.reshape(q_r, (1,1,len(q_r),1))
    q_g_tensor = np.reshape(q_g, (1,1,len(q_g),1))
    q_b_tensor = np.reshape(q_b, (1,1,len(q_b),1))
    
    q_tensor = np.concatenate((q_b_tensor,q_g_tensor,q_g_tensor,q_r_tensor),axis = 3)
    return q_tensor

def set_sensor_curve_gaussian(wave_lengths):
    matrix = np.zeros((31,4))
    # Peak positions (P_i) for each column, assuming they are between 0 and 30
    P = [3, 11, 19, 27]
    sigma = 4
    for i in range(4):
        x = np.arange(31)
        gaussian = np.exp(-(x - P[i])**2 / (2 * sigma**2))
        matrix[:, i] = gaussian / np.max(gaussian)  # Normalize to have values between 0 and 1

    matrix/= np.sum(matrix,0, keepdims=True)
    return matrix[None, None,...]


def myprint(name, data):
    print(name)
    print(data)


##############################
# usefull functions
##############################

def simulate_gaussian_poisson_noise(image, gaussian_sigma, poisson_lam):
    dtype = image.dtype
    # Generate Gaussian noise
    gaussian_noise = gaussian_sigma * tf.random.normal(shape=tf.shape(image), dtype=dtype)

    # Generate Poisson noise
    if poisson_lam > 0:
        poisson_noise = tf.random.poisson([1], image/poisson_lam, dtype=dtype)[0]
        poisson_noise = poisson_noise / tf.reduce_mean(poisson_noise) * tf.reduce_mean(image)
    else:
        poisson_noise = image

    # Add the Gaussian and Poisson noise
    noisy_image = gaussian_noise + poisson_noise

    return tf.clip_by_value(noisy_image,0,1)

def gaussian_noise(image, stddev=0.001):
    dtype = image.dtype
    mean_value = 1.
    return tf.clip_by_value(image + tf.random.normal(image.shape, 0.0, stddev*mean_value, dtype=dtype),0,1)

# def poisson_noise(image, intensity=0.001, lam = 1):
#     dtype = image.dtype
#     return tf.clip_by_value(image + intensity * tf.random.poisson(image.shape, lam, dtype=dtype),0,1)

def random_uniform(image,interval=[0.001,0.02]):
    return image + tf.random.uniform(minval=interval[0], maxval=interval[1], shape=[])

def get_zernike_volume(resolution, n_terms, scale_factor=1e-6):
    zernike_volume = poppy.zernike.zernike_basis(nterms=n_terms, npix=resolution, outside=0.0)
    return zernike_volume * scale_factor

def fspecial(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def zoom(image_batch, zoom_fraction):
    """Get central crop of batch
    """
    images = tf.unstack(image_batch, axis=0)
    crops = []
    for image in images:
        crop = tf.image.central_crop(image, zoom_fraction)
        crops.append(crop)
    return tf.stack(crops, axis=0)


def transp_fft2d(a_tensor, dtype=tf.complex64):
    """Takes images of shape [batch_size, x, y, channels] and transposes them
    correctly for tensorflows fft2d to work.
    """
    # Tensorflow's fft only supports complex64 dtype
    a_tensor = tf.cast(a_tensor, tf.complex64)
    # Tensorflow's FFT operates on the two innermost (last two!) dimensions
    a_tensor_transp = tf.transpose(a_tensor, [0, 3, 1, 2])
    a_fft2d = tf.signal.fft2d(a_tensor_transp)
    a_fft2d = tf.cast(a_fft2d, dtype)
    a_fft2d = tf.transpose(a_fft2d, [0, 2, 3, 1])
    return a_fft2d


def transp_fft1d(a_tensor, dtype=tf.complex64):
    """Takes images of shape [batch_size, x, y, channels] and transposes them
    correctly for tensorflows fft2d to work.
    """
    # Tensorflow's fft only supports complex64 dtype
    a_tensor = tf.cast(a_tensor, tf.complex64)
    # Tensorflow's FFT operates on the two innermost (last two!) dimensions
    a_tensor_transp = tf.transpose(a_tensor, [0, 2, 3, 1])
    a_fft1d = tf.signal.fft(a_tensor_transp)
    a_fft1d = tf.cast(a_fft1d, dtype)
    a_fft1d = tf.transpose(a_fft1d, [0, 3, 1, 2])
    return a_fft1d



def transp_ifft2d(a_tensor, dtype=tf.complex64):
    a_tensor = tf.transpose(a_tensor, [0, 3, 1, 2])
    a_tensor = tf.cast(a_tensor, tf.complex64)
    a_ifft2d_transp = tf.signal.ifft2d(a_tensor)
    # Transpose back to [batch_size, x, y, channels]
    a_ifft2d = tf.transpose(a_ifft2d_transp, [0, 2, 3, 1])
    a_ifft2d = tf.cast(a_ifft2d, dtype)
    return a_ifft2d

def transp_ifft1d(a_tensor, dtype=tf.complex64):
    a_tensor = tf.transpose(a_tensor, [0, 2, 3, 1])
    a_tensor = tf.cast(a_tensor, tf.complex64)
    a_ifft1d_transp = tf.signal.ifft(a_tensor)
    # Transpose back to [batch_size, x, y, channels]
    a_ifft1d = tf.transpose(a_ifft1d_transp, [0, 3, 1, 2])
    a_ifft1d = tf.cast(a_ifft1d, dtype)
    return a_ifft1d




def compl_exp_tf(phase, dtype=tf.complex64, name='complex_exp'):
    """Complex exponent via euler's formula, since Cuda doesn't have a GPU kernel for that.
    Casts to *dtype*.
    """
    return tf.add(tf.cast(tf.cos(phase), dtype=dtype),
                  1.j * tf.cast(tf.sin(phase), dtype=dtype),
                  name=name)


def laplacian_filter_tf(img_batch):
    """Laplacian filter. Also considers diagonals.
    """
    laplacian_filter = tf.constant([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=tf.float32)
    laplacian_filter = tf.reshape(laplacian_filter, [3, 3, 1, 1])

    filter_input = tf.cast(img_batch, tf.float32)
    filtered_batch = tf.nn.convolution(filter_input, filter=laplacian_filter, padding="SAME")
    return filtered_batch

def gd1D_filter_tf(img_batch):
    """1D gradent filter. the shape of img_batch is [1 ,N,1,1]
    """
    gd1D_filter = tf.constant([[1, -1]], dtype=tf.float32)
    gd1D_filter = tf.reshape(gd1D_filter, [2, 1, 1, 1])

    filter_input = tf.cast(img_batch, tf.float32)
    filtered_batch = tf.nn.convolution(filter_input, filter=gd1D_filter, padding="SAME")
    return filtered_batch


def gd1D_l1_regularizer(scale):
    if np.allclose(scale, 0.):
        print("Scale of zero disables the laplace_l1_regularizer.")

    def gd1D_l1(a_tensor):
        with tf.name_scope('laplace_l1_regularizer'):
            gd1D_filtered = gd1D_filter_tf(a_tensor)
            gd1D_filtered = gd1D_filtered[:, 1:-1, :, :]
            attach_summaries("Laplace_filtered", tf.abs(gd1D_filtered), image=True, log_image=True)
            return scale * tf.reduce_mean(tf.abs(gd1D_filtered))

    return gd1D_l1


def generate_dominator_circle(image):
    batch,height,_,channel = image.shape.as_list()
    C_temp = sio.loadmat('C.mat')
    C = C_temp['C'].astype(np.float32)      
    C = np.tile(C,[batch,1,1,channel])
    dominate_temp = tf.constant(C,dtype = tf.float32)
        
    return  dominate_temp * image  

def laplace_l1_regularizer(scale):
    if np.allclose(scale, 0.):
        print("Scale of zero disables the laplace_l1_regularizer.")

    def laplace_l1(a_tensor):
        laplace_filtered = laplacian_filter_tf(a_tensor)
        laplace_filtered = laplace_filtered[:, 1:-1, 1:-1, :]
        attach_summaries("Laplace_filtered", tf.abs(laplace_filtered), image=True, log_image=True)
        return scale * tf.reduce_mean(tf.abs(laplace_filtered))

    return laplace_l1

def sum_regularizer(scale, lens,isfall = False):
    if np.allclose(scale, 0.):
        print("Scale of zero disables the sum_regularizer.")
    def sum_l2(a_tensor):
        with tf.name_scope('sum_l2_regularizer'):
            print(a_tensor.shape)
            if isfall is True:
                _,M,_,_= a_tensor.shape.as_list()               
                gather_range = np.arange((M//2-lens+1),(M//2+lens+2))
            else:
                gather_range = np.arange(lens)
            new_tensor = tf.gather(a_tensor, gather_range, axis =1)
   
            return scale * tf.reduce_mean(1 - tf.div(tf.reduce_sum(new_tensor,axis = 1), tf.reduce_sum(a_tensor,axis = 1)))
        
    return sum_l2

def sum_regularizer_all(scale, lens):
    if np.allclose(scale, 0.):
        print("Scale of zero disables the sum_regularizer.")
    def sum_l2(a_tensor):
        gather_range = np.arange(lens)
        a_tensor1 = generate_dominator_circle(a_tensor)
        new_tensor = tf.gather(a_tensor1, gather_range, axis =1)  
        return scale * tf.reduce_mean(1 - tf.math.divide(tf.reduce_sum(new_tensor,axis = 1), tf.reduce_sum(a_tensor1,axis = 1)))
        
    return sum_l2

def sum_regularizer_norm(scale, lens, isfall = False):
    if np.allclose(scale, 0.):
        print("Scale of zero disables the sum_regularizer.")
    def sum_l2(a_tensor):
        if isfall is True:
            _,M,_,_= a_tensor.shape.as_list()               
            gather_range = np.arange((M//2-lens+1),(M//2+lens+2))
        else:
            gather_range = np.arange(lens)
        new_tensor = tf.gather(a_tensor, gather_range, axis =1)  
        return scale * tf.reduce_mean(1 - tf.reduce_sum(new_tensor,axis = 1,keepdims = True))
        
    return sum_l2




def laplace_l2_regularizer(scale):
    if np.allclose(scale, 0.):
        print("Scale of zero disables the laplace_l1_regularizer.")

    def laplace_l2(a_tensor):
        laplace_filtered = laplacian_filter_tf(a_tensor)
        laplace_filtered = laplace_filtered[:, 1:-1, 1:-1, :]
        attach_summaries("Laplace_filtered", tf.abs(laplace_filtered), image=True, log_image=True)
        return scale * tf.reduce_mean(tf.math.square(laplace_filtered))

    return laplace_l2


def cross_correlation_regularizer(scale):
    if np.allclose(scale,0.):
        print("Scale of zero disables the cross_correlation_regularizer.")

    def cross_correlation(a_tensor):
        _,_,_,M = a_tensor.shape.as_list()
        mask = tf.cast(np.ones([M,M])-np.eye(M),dtype = bool)
        input1 = tf.transpose(a_tensor,[3,1,2,0])    # the shape of the input tensor is 1 * K *K * channels
        input2 = tf.transpose(a_tensor,[1,2,0,3]) 
        output1 = tf.nn.convolution(input1,input2,padding = "SAME")

        output = tf.math.divide(tf.reduce_max(output1,axis = [1,2]), tf.reduce_max(output1))

        output = tf.where(mask, tf.zeros_like(output), output)
        return scale*tf.reduce_mean(output)
    return cross_correlation
             


def phaseshifts_from_height_map(height_map, wave_lengths, refractive_idcs):
    '''Calculates the phase shifts created by a height map with certain
    refractive index for light with specific wave length.
    '''
    # refractive index difference
    delta_N = refractive_idcs - 1.    # shape [1,1,1,n]
    # wave number
    wave_nos = 2. * np.pi / wave_lengths   # shape [1,1,1,n]

    # phase delay indiced by height field
    phi = wave_nos * delta_N * height_map
    phase_shifts = compl_exp_tf(phi, dtype = tf.complex64, name='DOE_phase')
    return phase_shifts


def get_one_phase_shift_thickness(wave_lengths, refractive_index):
    """Calculate the thickness (in meter) of a phaseshift of 2pi.
    """
    # refractive index difference
    delta_N = refractive_index - 1.
    # wave number
    wave_nos = 2. * np.pi / wave_lengths

    two_pi_thickness = (2. * np.pi) / (wave_nos * delta_N)
    return two_pi_thickness


def attach_summaries(name, var, image=False, log_image=False):
    shape_size = var.shape.as_list()
    if image:
        if shape_size[3] >3:
           var = tf.reduce_sum(var, axis = -1, keepdims = True)
        tf.summary.image(name, var, max_outputs=31)
    if log_image and image:
        if shape_size[3] >3:
           var = tf.reduce_sum(var, axis = -1, keepdims = True)
        tf.summary.image(name + '_log', tf.math.log((var + 1e-12)), max_outputs=31)
    tf.summary.scalar(name + '_mean', tf.reduce_mean(var))
    tf.summary.scalar(name + '_max', tf.reduce_max(var))
    tf.summary.scalar(name + '_min', tf.reduce_min(var))
    tf.summary.histogram(name + '_histogram', var)


def fftshift2d_tf(a_tensor):
    input_shape = a_tensor.shape.as_list()

    new_tensor = a_tensor
    for axis in range(1, 3):
        split = (input_shape[axis] + 1) // 2
        mylist = np.concatenate((np.arange(split, input_shape[axis]), np.arange(split)))
        new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor

def fftshift1d_tf(a_tensor):
    input_shape = a_tensor.shape.as_list()

    new_tensor = a_tensor
    axis = 1
    split = (input_shape[axis] + 1) // 2
    mylist = np.concatenate((np.arange(split, input_shape[axis]), np.arange(split)))
    new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor


def ifftshift2d_tf(a_tensor):
    input_shape = a_tensor.shape.as_list()

    new_tensor = a_tensor
    for axis in range(1, 3):
        n = input_shape[axis]
        split = n - (n + 1) // 2
        mylist = np.concatenate((np.arange(split, n), np.arange(split)))
        new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor


def ifftshift1d_tf(a_tensor):
    input_shape = a_tensor.shape.as_list()

    new_tensor = a_tensor
    axis = 1
    n = input_shape[axis]
    split = n - (n + 1) // 2
    mylist = np.concatenate((np.arange(split, n), np.arange(split)))
    new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor



def psf2otf(input_filter, output_size):
    '''Convert 4D tensorflow filter into its FFT.

    :param input_filter: PSF. Shape (num_color_channels, height, width,  num_color_channels)
    :param output_size: Size of the output OTF.
    :return: The otf.
    '''
    # pad out to output_size with zeros
    # circularly shift so center pixel is at 0,0
    _, fh, fw,  _ = input_filter.shape.as_list()

    if output_size[0] != fh:
        pad = (output_size[0] - fh) / 2

        if (output_size[0] - fh) % 2 != 0:
            pad_top = pad_left = int(np.ceil(pad))
            pad_bottom = pad_right = int(np.floor(pad))
        else:
            pad_top = pad_left = int(pad) + 1
            pad_bottom = pad_right = int(pad) - 1

        padded = tf.pad(input_filter, [[0, 0],[pad_top, pad_bottom],
                                       [pad_left, pad_right],  [0, 0]], "CONSTANT")
    else:
        padded = input_filter

    padded = ifftshift2d_tf(padded)

    ## Take FFT
    tmp = tf.transpose(padded, [0, 3, 1, 2])
    tmp = tf.signal.fft2d(tf.complex(tmp, 0.))
    return tf.transpose(tmp, [0, 2, 3, 1])


def next_power_of_two(number):
    closest_pow = np.power(2, np.ceil(np.math.log(number, 2)))
    return closest_pow


def img_psf_conv(img, psf, otf=None, adjoint=False, circular=False):
    '''Performs a convolution of an image and a psf in frequency space.

    :param img: Image tensor.
    :param psf: PSF tensor.
    :param otf: If OTF is already computed, the otf.
    :param adjoint: Whether to perform an adjoint convolution or not.
    :param circular: Whether to perform a circular convolution or not.
    :return: Image convolved with PSF.
    '''
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    psf = tf.convert_to_tensor(psf, dtype=tf.float32)

    img_shape = img.shape.as_list()
    

    if not circular:
        target_side_length = 2 * img_shape[1]

        height_pad = (target_side_length - img_shape[1]) / 2
        width_pad = (target_side_length - img_shape[1]) / 2

        pad_top, pad_bottom = int(np.ceil(height_pad)), int(np.floor(height_pad))
        pad_left, pad_right = int(np.ceil(width_pad)), int(np.floor(width_pad))

        img = tf.pad(img, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], "CONSTANT")
        img_shape = img.shape.as_list()

    img_fft = transp_fft2d(img)

    if otf is None:
        otf = psf2otf(psf, output_size=img_shape[1:3])

    otf = tf.cast(otf, tf.complex64)
    img_fft = tf.cast(img_fft, tf.complex64)

    if adjoint:
        result = transp_ifft2d(img_fft * tf.conj(otf))
    else:
        result = transp_ifft2d(img_fft * otf)

    result = tf.cast(tf.math.real(result), tf.float32)

    if not circular:
        result = result[:, pad_top:-pad_bottom, pad_left:-pad_right, :]

    return result

def fft3d(a_tensor, dtype=tf.complex64):
    # Tensorflow's fft only supports complex64 dtype
    a_tensor = tf.cast(a_tensor, tf.complex64)
    a_fft3d = tf.signal.fft3d(a_tensor)
    a_fft3d = tf.cast(a_fft3d, dtype)
    return a_fft3d
def ifft3d(a_tensor, dtype=tf.complex64):
    a_tensor = tf.cast(a_tensor, tf.complex64)
    a_ifft3d = tf.signal.ifft3d(a_tensor)
    return a_ifft3d
def ifftshift3d_tf(a_tensor):
    input_shape = a_tensor.shape.as_list()
    new_tensor = a_tensor
    for axis in range(1, 4):
        n = input_shape[axis]
        split = n - (n + 1) // 2
        mylist = np.concatenate((np.arange(split, n), np.arange(split)))
        new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor
def psf2otf3d(psf, h, w):
    psf = tf.image.resize_with_crop_or_pad(psf, h, w)
    psf = ifftshift3d_tf(psf)
    otf = tf.signal.fft3d(tf.complex(psf, 0.))
    return tf.cast(otf, tf.complex64)

def img_psf_conv3D(img, psf):
    '''Performs a convolution of an image and a psf in frequency space.

    :param img: Image tensor.
    :param psf: PSF tensor.
    :param otf: If OTF is already computed, the otf.
    :param adjoint: Whether to perform an adjoint convolution or not.
    :param circular: Whether to perform a circular convolution or not.
    :return: Image convolved with PSF.
    '''
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    psf = tf.convert_to_tensor(psf, dtype=tf.float32)

    img_shape = img.shape.as_list()

    img_fft = fft3d(img)

    otf = psf2otf3d(psf, img_shape[1], img_shape[2])

    otf = tf.cast(otf, tf.complex64)
    img_fft = tf.cast(img_fft, tf.complex64)

    result = ifft3d(img_fft * otf)

    result = tf.cast(tf.math.real(result), tf.float32)

    return result

def depth_dep_convolution(img, psfs, disc_depth_map):
    """Convolves an image with different psfs at different depths as determined by a discretized depth map.

    Args:
        img: image with shape (batch_size, height, width, num_img_channels)
        psfs: filters with shape (kernel_height, kernel_width, num_img_channels, num_filters)
        disc_depth_map: Discretized depth map.
        use_fft: Use img_psf_conv or normal conv2d
    """
    # TODO: only convolve with PSFS that are necessary.
    img = tf.cast(img, dtype=tf.float32)

    input_shape = img.shape.as_list()

    zeros_tensor = tf.zeros_like(img, dtype=tf.float32)
    disc_depth_map = tf.tile(tf.cast(disc_depth_map, tf.int16),
                             multiples=[1, 1, 1, input_shape[3]])

    blurred_imgs = []
    for depth_idx, psf in enumerate(psfs):
        psf = tf.cast(psf, dtype=tf.float32)
        condition = tf.equal(disc_depth_map, tf.convert_to_tensor(depth_idx, tf.int16))
        blurred_img = img_psf_conv(img, psf)
        blurred_imgs.append(tf.where(condition,
                                     blurred_img,
                                     zeros_tensor))

    result = tf.reduce_sum(blurred_imgs, axis=0)
    return result



def least_common_multiple(a, b):
    return abs(a * b) / fractions.gcd(a, b) if a and b else 0


def area_downsampling_tf(input_image, downsampling_scale, is1D = False):


    input_image = tf.cast(input_image, tf.float32)

    if is1D is True:
        output_img = tf.nn.avg_pool(input_image,
                                [1, downsampling_scale, 1, 1],
                                 strides=[1, downsampling_scale, 1, 1],
                                 padding="VALID")
    else:
        output_img = tf.nn.avg_pool(input_image,
                                [1, downsampling_scale, downsampling_scale, 1],
                                 strides=[1, downsampling_scale, downsampling_scale, 1],
                                 padding="VALID")
   
    return output_img


def get_intensities(input_field):
    return tf.math.square(tf.abs(input_field), name='intensities')


##################################
# add for 1D optimization
##################################


def psf1D_to_2D_half(psf, isDiv = False):
    
    batch, height, _ , channel = psf.shape.as_list()
    psf = tf.cast(psf,dtype = tf.float32)
    
    psf_zeros = np.zeros(height)
    psf_zeros = psf_zeros[np.newaxis,:,np.newaxis,np.newaxis]
    psf_zeros = np.tile(psf_zeros,[batch,1,1,channel])
    
    
    psf_temp = tf.concat([psf,tf.constant(psf_zeros, dtype = tf.float32)],axis = 1)
    
    [x, y] = np.mgrid[-height:height,
             -height:height]
    
    N, M = x.shape
    R = np.sqrt(x**2 +y**2).astype(np.int32)
    R = np.reshape(R,(-1))
    psf2D = tf.gather(psf_temp, R, axis=1)
    psf2D = tf.reshape(psf2D,(batch,N,M,channel))
    
    if isDiv is True:
        psf2D = tf.math.divide(psf2D, tf.reduce_sum(psf2D,axis = [1,2], keepdims= True)) 

    return psf2D


def psf1D_to_2D_all(psf, C = None, isDiv = False):
    
    batch, height, _ , channel = psf.shape.as_list()
    psf = tf.cast(psf,dtype = tf.float32)
    psf = tf.pad(psf,[[0,0],[height//2,height//2],[0,0],[0,0]])
    psf_temp = ifftshift1d_tf(psf)

    [x, y] = np.mgrid[-height//2:height//2,
             -height//2:height//2]
    
    N, M = x.shape

    R = np.sqrt(x**2 +y**2).astype(np.int32)
    
    R1 = np.reshape(R,(-1))
    RR = np.ones(len(R1)) * (height*1.5)
    R = np.where(R1 == height//2,RR,R1)
    R = np.reshape(R,(-1)).astype(np.int32)
    
    if isDiv is True:
        if C is None:  
            
            A = np.zeros(len(R1))
            B = np.ones(len(R1))
            C = []
            for i in np.arange(np.max(R1)+1):
                CC = (np.where(R1==i,B,A))
                CC = CC[np.newaxis,:]
                C.append(np.sum(CC, axis = 1))   
            C = np.sum(C,axis = 1)
            C = C[np.newaxis,:,np.newaxis,np.newaxis]
            
        C = np.tile(C,[batch,1,1,channel])
        
        dominate_temp = tf.constant(C,dtype = tf.float32)    
        
        dominate = tf.gather(dominate_temp, R1, axis=1)  # this place sholud keep the same with orgin
        dominate = tf.reshape(dominate,(batch,N,M,channel))
    
    
    psf2D = tf.gather(psf_temp, R, axis=1)
    psf2D = tf.reshape(psf2D,(batch,N,M,channel))
    
    if isDiv is True:
        psf2D = tf.math.divide(psf2D,dominate)
        psf2D = tf.math.divide(psf2D, tf.reduce_sum(psf2D, axis=[1,2], keepdims=True), name='psf_depth_idx')
    
    return psf2D

class DOE_1D_array_ablation(Layer):
    def __init__(self,height,                                    
                       wave_lengths,
                       refractive_idcs,
                       max_height = 1.2e-6,
                       array_num = 4,
                       height_tolerance = None,
                       height_map_init_value = None,
                       height_map_regularizer = None,
                       trainable_flag = True,
                       quantization = False,
                       isfall = False):
        super(DOE_1D_array_ablation, self).__init__()
        if isfall is True:
            height_map_shape = [1, height//2 , 1, 1]
        else: 
            height_map_shape = [1, height , 1, 1]   
          
        
        if height_map_init_value is None:
            height_map_init_value = np.ones(shape=height_map_shape, dtype=np.float64) * 1e-4
        self.height_map_sqrt = tf.Variable(name="height_map_sqrt",
                                            shape=height_map_shape,
                                            dtype=tf.float64,
                                            trainable=trainable_flag,
                                            initial_value=height_map_init_value) 
        self.wave_lengths = wave_lengths
        self.refractive_idcs = refractive_idcs
        self.max_height = max_height
        self.height_tolerance = height_tolerance
        self.isfall = isfall
        self.quantization = quantization

        if height_tolerance is not None:
            print("Phase plate with manufacturing tolerance %0.2e" % self.height_tolerance)

    def call(self, input_field):
        if self.isfall is True:
            height_map_sqrt = tf.concat([tf.constant(np.zeros([1,1,1,1]),dtype = tf.float64),self.height_map_sqrt], axis = 1)
            half_index = np.arange(0, height//2+1).astype(np.int32)
            other_half_index = np.fliplr(half_index[None,:])
            full_index = np.concatenate((other_half_index[0,:],half_index[1:]),axis = 0)    

            height_map_sqrt1 = tf.gather(self.height_map_sqrt, full_index, axis=1)

        else:
            height_map_sqrt1 = self.height_map_sqrt
         
         
        height_map_temp = tf.math.square(height_map_sqrt1)

        height_map = tf.math.floormod(height_map_temp,tf.cast(self.max_height,dtype=tf.float64))
        if self.quantization:
            height_map = tf.clip_by_value(tf.math.round(height_map/(self.max_height/16))*(self.max_height/16),0,self.max_height)
        
        if self.isfall is True:
            height_map_all = psf1D_to_2D_all(height_map)

        else:
            height_map_all = psf1D_to_2D_half(height_map)    

        if self.height_tolerance is not None:
            height_map1 = height_map + tf.random.uniform(shape=height_map.shape,
                                                        minval=-self.height_tolerance,
                                                        maxval=self.height_tolerance,
                                                        dtype=height_map.dtype)
        else:
            height_map1 = height_map
        
        phase_shifts = phaseshifts_from_height_map(tf.tile(height_map1, [4, 1, 1, 1]),
                                                   self.wave_lengths,
                                                   self.refractive_idcs)
        return tf.multiply(phase_shifts, input_field), height_map, height_map_all


class DOE_1D_array(Layer):
    def __init__(self,height,                                    
                       wave_lengths,
                       refractive_idcs,
                       max_height = 1.2e-6,
                       array_num = 4,
                       height_tolerance = None,
                       height_map_init_value = None,
                       height_map_regularizer = None,
                       trainable_flag = True,
                       quantization = False,
                       isfall = False):
        super(DOE_1D_array, self).__init__()
        if isfall is True:
            height_map_shape = [array_num, height//2 , 1, 1]
        else: 
            height_map_shape = [array_num, height , 1, 1]   
          
        
        if height_map_init_value is None:
            height_map_init_value = np.ones(shape=height_map_shape, dtype=np.float64) * 1e-4
        self.height_map_sqrt = tf.Variable(name="height_map_sqrt",
                                            shape=height_map_shape,
                                            dtype=tf.float64,
                                            trainable=trainable_flag,
                                            initial_value=height_map_init_value) 
        self.wave_lengths = wave_lengths
        self.refractive_idcs = refractive_idcs
        self.max_height = max_height
        self.height_tolerance = height_tolerance
        self.isfall = isfall
        self.quantization = quantization

        if height_tolerance is not None:
            print("Phase plate with manufacturing tolerance %0.2e" % self.height_tolerance)

    def call(self, input_field):
        if self.isfall is True:
            height_map_sqrt = tf.concat([tf.constant(np.zeros([1,1,1,1]),dtype = tf.float64),self.height_map_sqrt], axis = 1)
            half_index = np.arange(0, height//2+1).astype(np.int32)
            other_half_index = np.fliplr(half_index[None,:])
            full_index = np.concatenate((other_half_index[0,:],half_index[1:]),axis = 0)    

            height_map_sqrt1 = tf.gather(self.height_map_sqrt, full_index, axis=1)

        else:
            height_map_sqrt1 = self.height_map_sqrt
         
         
        height_map_temp = tf.math.square(height_map_sqrt1)

        height_map = tf.math.floormod(height_map_temp,tf.cast(self.max_height,dtype=tf.float64))
        if self.quantization:
            height_map = tf.clip_by_value(tf.math.round(height_map/(self.max_height/16))*(self.max_height/16),0,self.max_height)
        
        if self.isfall is True:
            height_map_all = psf1D_to_2D_all(height_map)

        else:
            height_map_all = psf1D_to_2D_half(height_map)    

        if self.height_tolerance is not None:
            height_map1 = height_map + tf.random.uniform(shape=height_map.shape,
                                                        minval=-self.height_tolerance,
                                                        maxval=self.height_tolerance,
                                                        dtype=height_map.dtype)
        else:
            height_map1 = height_map
        phase_shifts = phaseshifts_from_height_map(height_map1,
                                                   self.wave_lengths,
                                                   self.refractive_idcs)
        return tf.multiply(phase_shifts, input_field), height_map, height_map_all


def pad_zeros_1D_to_2D_double(img1,img2):
   
    _,M,_,_ = img1.shape.as_list()  
    index = np.arange(M,0,-1).astype(np.int32)
    img1_new = tf.gather(img1,index,axis = 1)
    img = tf.concat([img1_new,img2], axis = 1)
    
    img_out = tf.pad(img, [[0, 0], [0, 0], [M-1, M-1], [0, 0]])
    

    return img_out 


def pad_zeros_1D_to_2D(img):
   
    _,M,_,_ = img.shape.as_list()  
    half_index = np.arange(M).astype(np.int32)
    other_half_index = np.fliplr(half_index[None,:])[0,0:-1]
    full_index = np.concatenate((other_half_index, half_index),axis = 0)    
    img_full = tf.gather(img, full_index, axis=1)
    
    img_out = tf.pad(img_full, [[0, 0], [0, 0], [M-1, M-1], [0, 0]])

    return img_out 


def pad_zeros_1D_to_2D_new(img,length):
   
    _,M,_,_ = img.shape.as_list()
    start_index = (M-length)//2
    end_index = start_index + length
    crop_index = np.arange(start_index,end_index) 
    img_crop = tf.gather(img, crop_index, axis=1)
  
    img_out = tf.pad(img_crop, [[0, 0], [0, 0], [length//2, length//2-1], [0, 0]])

    return img_out  


def get_hankel_kernel_1D(input_N,output_N,factor = 1):

    r = np.arange(input_N) + 0.5
    r = r[np.newaxis,:]
    p = np.arange(output_N) / (factor * (2. *input_N-1))
    p = p[:,np.newaxis]
    p[0][0] = 1e-18

    I = np.dot(1. / p, r) * (special.jv(1, 2. * np.pi * np.dot(p,r)))
    I[0,:] = np.pi * r * r
    
    I1 = np.concatenate([np.zeros((output_N,1)),I],axis = 1)
    I2 = I - I1[:,0:-1]
    I2 = np.transpose(I2,[1,0])
    return I2



def get_hankel_kernel_1D_full_wvl(input_N,output_N,wavelengths,factor = 1):
  
    wvl = np.reshape(wavelengths,[-1,1,1])
    W = np.size(wvl,axis = 0)
    factor_temp = wvl / np.nanmin(wvl)
    factor = factor * factor_temp 
    r = np.arange(input_N) + 0.5
    r = r[np.newaxis,:]
    p_temp = np.arange(output_N) / ( (2. *input_N-1))
    p_temp[0] = 1e-18
    p_temp = p_temp[np.newaxis,:,np.newaxis]
    p = p_temp / factor

    I = np.dot(1. / p, r) * (special.jv(1, 2. * np.pi * np.dot(p,r)))
    I[:,0,:] = np.pi * r * r

    I1 = np.concatenate([np.zeros((W,output_N,1)),I],axis = 2)
    I2 = I - I1[:,:,0:-1]
    I2 = np.transpose(I2,[0,2,1])
    return I2




def generate_sphere_1D(distance,
                       wave_resolution,
                       pixel_size,
                       wave_lengths):
    
    x = np.arange(wave_resolution[0]) * pixel_size
    wave_nos = 2.* np.pi / wave_lengths
    curvature = -1 * x**2 / (2. *distance)
    curvature = curvature[np.newaxis, :, np.newaxis, np.newaxis]
    phase_temp = wave_nos * curvature
    phase = tf.constant(phase_temp, dtype = tf.float32)
    sphere_field = compl_exp_tf(phase, dtype=tf.complex64)
    
    return sphere_field


def point_source_1D_layer(distance,
                          wave_resolution,
                          pixel_size,
                          wave_lengths,
                          isfall = False):
    if isfall is True:
        x = np.arange(-wave_resolution[0]//2, wave_resolution[0]//2, 1) * pixel_size 
    else:
        x = np.arange(wave_resolution[0]) * pixel_size 
    square_sum = x**2
    square_sum = square_sum[np.newaxis,:,np.newaxis,np.newaxis]
    
    wave_nos = 2.* np.pi / wave_lengths
    
    curvature = tf.math.sqrt(np.float64(square_sum) + tf.cast(distance,tf.float64)**2)  #distance shape [n,1,1,1]

    phase_temp = wave_nos * curvature
    phase = tf.cast(phase_temp, dtype = tf.float32)
    sphere_field = compl_exp_tf(phase, dtype=tf.complex64)
    
    return sphere_field


def fresnel_Prop_layer_hankel_1D_2step(input_field,
                                       distance,
                                       pixel_size_input,
                                       pixel_size_output,
                                       wave_lengths,
                                       output_size,
                                       I1 = None,
                                       I2 = None):
    
    m = -pixel_size_output / pixel_size_input
    distance1 = distance / (1+m)
    batch, M, _, channel  =  input_field.shape.as_list()
   
    if M < 8000:
        input_field = tf.pad(input_field,[[0,0],[0,8000-M],[0,0],[0,0]])
        
    M = input_field.shape.as_list()[1]     
    x = np.arange(M) * pixel_size_input
    curvature = x**2 / (2. *distance1)
    curvature = curvature[np.newaxis, :, np.newaxis, np.newaxis]
    wave_nos = 2. * np.pi / wave_lengths  # wave_lengths shape = [1,1,1,n]
    phase_temp = wave_nos * curvature
    phase = tf.constant(phase_temp, dtype = tf.float32)
    sphere_field = compl_exp_tf(phase, dtype=tf.complex64)
    
    input_field_new = input_field * sphere_field  # input_field_shape = [batch,M,1,n]
    
    # first step
    p = np.arange(M) / (2. *M-1)
    x1_temp = p * distance1 / pixel_size_input
    x1_temp = x1_temp[np.newaxis,:,np.newaxis,np.newaxis]
    x1 = x1_temp * wave_lengths
    
    curvature1 = x1**2 / (2. *distance1)
    phase_temp = wave_nos * curvature1
    phase1 = tf.constant(phase_temp, dtype = tf.float32)
    sphere_field1 = compl_exp_tf(phase1, dtype=tf.complex64)
    
    ## orgin
    if I1 is None:
        I1 = get_hankel_kernel_1D(M, M)
        I1 = tf.constant(I1, dtype = tf.float32)
    I1 = tf.complex(I1,0.0)
    input_field_new = tf.transpose(input_field_new,[0,3,2,1])
    input_field_new = tf.reshape(input_field_new,(-1,M))
    out_field_mid = tf.matmul(input_field_new, I1)
    out_field_mid = tf.reshape(out_field_mid,(batch,channel,1,M))
    out_field_mid = tf.transpose(out_field_mid,[0,3,2,1])
    
    
    out_field_mid = out_field_mid * sphere_field1
    
    # second step
    curvature2 = x1**2 / (2. * (distance - distance1))
    phase_temp = wave_nos * curvature2
    phase2 = tf.constant(phase_temp, dtype = tf.float32)
    sphere_field2 = compl_exp_tf(phase2, dtype=tf.complex64)
    input_field_new2 = out_field_mid * sphere_field2
    
    
    x2 = np.arange(output_size) * pixel_size_output
    x2 = x2[np.newaxis,:,np.newaxis,np.newaxis]
    curvature3 = x2**2 / (2. *(distance - distance1))
    phase_temp = wave_nos * curvature3
    phase3 = tf.constant(phase_temp, dtype = tf.float32)
    
    sphere_field3 = compl_exp_tf(phase3, dtype=tf.complex64)

    #old
    if I2 is None:
        I2 = get_hankel_kernel_1D(M, output_size)
        I2 = tf.constant(I2, dtype = tf.float32)

    I2 = tf.complex(I2,0.0)
    input_field_new2 = tf.transpose(input_field_new2,[0,3,2,1])
    input_field_new2 = tf.reshape(input_field_new2,(-1,M))
    output = tf.matmul(input_field_new2, I2)
    output = tf.reshape(output,(batch,channel,1,output_size))
    output = tf.transpose(output,[0,3,2,1])
    
    output = output * sphere_field3
    
    psf = get_intensities(output)
    
    r = (np.arange(output_size) + 0.5) **2
    r1 = np.concatenate([np.zeros(1),r[0:-1]], axis = 0)
    r = r - r1
    r = r / r[0]
    r = np.reshape(r,(1,-1,1,1))
    
    r = tf.constant(r,dtype = tf.float32)    # keep energy equal
    return psf
    # return tf.math.divide(psf, tf.reduce_sum(psf, axis=[1], keepdims=True), name='psf') 

class sensor_sample_array(Layer):
    def __init__(self,
                      q_tensor,
                      native_response_weight,
                      noise_sigma = 0.001,
                      noise_model = gaussian_noise,
                      name ='response_curve',
                      trainable_flag = True,
                      use_psf_encoding = False):
        super(sensor_sample_array, self).__init__()
        self.response_weight_sqrt = tf.Variable(name="response_curve_sqrt",
                                      shape = q_tensor.shape,
                                      dtype = tf.float32,
                                      trainable=trainable_flag,
                                      initial_value=np.sqrt(q_tensor)) 
        self.native_response_weight = native_response_weight
        self.use_psf_encoding = use_psf_encoding
        self.noise_sigma = noise_sigma
        self.noise_model = noise_model

    def call(self, inputs):
        psf, input_image = inputs
        GT_img = input_image

        response_weight = tf.math.square(self.response_weight_sqrt, name='response_curve')
        response_curve = response_weight
        response_weight = tf.transpose(response_weight,[3,0,1,2])
        response_weight = tf.math.divide(response_weight,tf.reduce_sum(response_weight,axis=[3],keepdims=True)) 

        if self.use_psf_encoding is True:
            input_image = img_psf_conv(input_image, psf, circular=True)     # shape of input_image [batch,M,M,channel]
         
        sensor_img = input_image * response_weight * tf.transpose(self.native_response_weight,[3,0,1,2])
        sensor_img = tf.reduce_sum(sensor_img,axis=[3], keepdims=True)
        psfs = psf * response_weight
        # psfs = tf.reduce_sum(psfs,axis=[3], keepdims=True)
        if self.noise_sigma > 0:
            noisy_img = self.noise_model(tf.transpose(sensor_img,[3,1,2,0]), self.noise_sigma)
        else:
            noisy_img = tf.transpose(sensor_img,[3,1,2,0])
        
        return noisy_img, psfs ,GT_img, response_curve

def Gaussian_pdf(mu, std):
    x = tf.cast(tf.range(0,31)/30, tf.float32)
    pdf = 1/(std*tf.math.sqrt(2*np.pi))*tf.math.exp(-0.5*((x - tf.clip_by_value(mu,0.01,0.99))/std)**2) 
    return tf.expand_dims(pdf, axis = 1)

def aperture_layer(input_field):
    input_shape = input_field.shape.as_list()
    [x, y] = np.mgrid[-input_shape[1] // 2: input_shape[1] // 2,
             -input_shape[2] // 2: input_shape[2] // 2].astype(np.float64)

    max_val = np.amax(x)

    r = np.sqrt(x ** 2 + y ** 2)[None, :, :, None]
    aperture = (r < max_val).astype(np.float64)
    return aperture * input_field


