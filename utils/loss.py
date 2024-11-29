import tensorflow as tf
import numpy as np

def Perc_loss(G_img, gt_img, vgg_model, loss_weight):
    if loss_weight == 0:
        return 0
    else:
        preprocessed_G_img  = tf.keras.applications.vgg19.preprocess_input(G_img*255.0)
        preprocessed_gt_img = tf.keras.applications.vgg19.preprocess_input(gt_img*255.0)

        G_layer_outs = vgg_model(preprocessed_G_img)
        gt_layer_outs = vgg_model(preprocessed_gt_img)

        perc_loss = tf.add_n([tf.reduce_mean(tf.abs(G_layer_out-gt_layer_out)) 
                              for G_layer_out, gt_layer_out in zip(G_layer_outs, gt_layer_outs)])
        return perc_loss* loss_weight

def L1_loss(real_image, same_image,loss_weight = 5):
    if loss_weight == 0:
        return 0
    else:
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return loss * loss_weight

def sam_loss(preds, target,loss_weight = 5):
    dot_product = tf.reduce_sum(preds * target,-1)
    preds_norm = tf.reduce_sum(preds ** 2,-1)
    target_norm = tf.reduce_sum(target ** 2,-1)
    sam_score = tf.math.acos(tf.clip_by_value(dot_product / (tf.math.sqrt(preds_norm * target_norm)+1e-10), -1, 1))
    return tf.reduce_mean(sam_score) * loss_weight
    
def center_sum_regularizer(psf, loss_weight, lens = 100, target = 1,isfall = False):
    if np.allclose(loss_weight, 0.):
        return 0
    else:
        if isfall is True:
            _,M,_,_= psf.shape.as_list()               
            gather_range = np.arange((M//2-lens+1),(M//2+lens+2))
        else:
            gather_range = np.arange(lens)
        new_tensor = tf.gather(psf, gather_range, axis =1)

        return loss_weight * tf.reduce_mean(tf.clip_by_value(target - tf.math.divide(tf.reduce_sum(new_tensor,axis = 1), tf.reduce_sum(psf,axis = 1)),0,1))
