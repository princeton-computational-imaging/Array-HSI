import numpy as np
import tensorflow as tf
import argparse
import os

from utils.loss import *
import utils.edof_reader as edof_reader
import param.param as param
from utils.save import *

def build_model(network, args):
    if network == 'optical_forward':
        from models.optics_forward import forward_model
        model = forward_model(args.patch_size, args.param)
    elif network == 'HR_UNet':
        from models.HR_UNet import HR_UNet
        model = HR_UNet(args.patch_size, args.param)
    elif network == 'HR_UNet_duotask':
        from models.HR_UNet import HR_UNet_duotask
        model = HR_UNet_duotask(args.patch_size, args.param)

    return model

def log_eval(F,G,VGG, X_val, epoch, summary_writer,args):
    output_image_sensor_org, [psfs,psf_temps], GT_img, response_curve, [height_map, height_map_2D] = F([args.param.input_field1, X_val], training = False)
    output_image, output_image_rgb = G(output_image_sensor_org, training = False)
    
    GTs_ms = tf.concat(tf.unstack(GT_img, axis=-1), axis = -1)[...,None]
    outputs_ms = tf.concat(tf.unstack(output_image, axis=-1), axis = -1)[...,None]
    
    GT_img_rgb =  tf.gather(tf.nn.conv2d(GT_img,args.param.response_weight_rgb,strides=[1,1,1,1],padding = 'SAME'), [2,1,0], axis = -1)
    output_image_rgb =  tf.gather(tf.clip_by_value(output_image_rgb,0.,1.), [2,1,0], axis = -1)
    
    GT_max_intensity = tf.reduce_max(GT_img_rgb)
    forward_out = tf.concat(tf.unstack(output_image_sensor_org, axis=-1), axis = -1)[...,None]

    if args.perc_weight > 0:
        perc_loss = Perc_loss(output_image_rgb/GT_max_intensity, GT_img_rgb/GT_max_intensity, VGG, args.perc_weight)
    else:
        perc_loss = 0

    center_sum_reg = center_sum_regularizer(psf_temps, args.reg_weight, target = 0.9)
    l1_loss_ms = sam_loss(GT_img, output_image, args.l1_weight_ms) 
    l1_loss_rgb = L1_loss(GT_img_rgb, output_image_rgb, args.l1_weight_rgb)
    total_loss = l1_loss_ms + l1_loss_rgb + perc_loss + center_sum_reg

    response_curve_fig = pyplot.figure(figsize=(10,10))
    pyplot.plot(response_curve[0,0].numpy())

    with summary_writer.as_default():
        tf.summary.scalar(name = 'val_loss/l1_loss', data = l1_loss_ms, step=epoch)
        tf.summary.scalar(name = 'val_loss/l2_loss', data = l1_loss_rgb, step=epoch)
        tf.summary.scalar(name = 'val_loss/perc_loss', data = perc_loss, step=epoch)
        tf.summary.scalar(name = 'val_loss/reg_loss', data = center_sum_reg, step=epoch)
        tf.summary.scalar(name = 'val_loss/total_loss', data = total_loss, step=epoch)

        tf.summary.image(name = 'out_RGB', data = output_image_rgb[0:1]/GT_max_intensity,step=epoch)
        tf.summary.image(name = 'out_31c', data = outputs_ms**(1/2.2), step=epoch)
        tf.summary.image(name = 'forward_out', data = forward_out / tf.reduce_max(forward_out), step=epoch)
        tf.summary.image(name = 'psfs_center', data = tf.concat(tf.unstack(tf.concat(tf.unstack(tf.image.central_crop(psfs/np.amax(psfs,axis=(1,2),keepdims=True), 1/16), axis=-1), axis = -1), axis=0), axis = 0)[None,...,None], step=epoch)
        tf.summary.image(name = 'response_curve', data = plot_to_image(response_curve_fig), step=epoch)
        
        if epoch == 0:
            tf.summary.image(name = 'GT_RGB', data = GT_img_rgb[0:1]/GT_max_intensity,step=epoch)
            tf.summary.image(name = 'GT_31c', data = GTs_ms**(1/2.2), step=epoch)


def train(args):
    param = args.param
    param.quantization = False
    if not args.use_noise:
        param.noise_max = 0
    if not args.train_curve:
        param.train_response_curve = False
    if args.finetune_head:
        assert args.pretrained_G is not None
        args.train_F=False
        
    # create models
    F = build_model(args.forward_model, args)    
    H_optimizer = tf.keras.optimizers.Adam(args.H_lr, beta_1=0.9) # Height Map
    C_optimizer = tf.keras.optimizers.Adam(args.C_lr, beta_1=0.9) # Color Filter
    if args.pretrained_F is not None:
        print('Loading pretrained F from %s' %args.pretrained_F)
        F_checkpoint = tf.train.Checkpoint(F = F)
        F_manager = tf.train.CheckpointManager(F_checkpoint, directory=args.pretrained_F, max_to_keep=10)
        status = F_checkpoint.restore(F_manager.latest_checkpoint).expect_partial()

    G = build_model(args.generator, args)
    G_optimizer = tf.keras.optimizers.Adam(args.G_lr, beta_1=0.9)
    if args.pretrained_G is not None:
        G_checkpoint = tf.train.Checkpoint(G = G)
        G_manager = tf.train.CheckpointManager(G_checkpoint, directory=args.pretrained_G, max_to_keep=10)
        status = G_checkpoint.restore(G_manager.latest_checkpoint).expect_partial()
        print('Loading pretrained G from %s' % G_manager.latest_checkpoint)
        if args.finetune_head:
            print('Freeze all feature layers')
            for layer in G.layers[:-1]:
                layer.trainable =  False

    if args.perc_weight > 0: 
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        VGG = tf.keras.Model(inputs=vgg.input, outputs=[vgg.get_layer(name).output for name in args.vgg_layers.split(',')])
    else:
        VGG = None

    checkpoint = tf.train.Checkpoint(G = G, G_optimizer = G_optimizer, F = F, H_optimizer = H_optimizer, C_optimizer = C_optimizer)
    manager = tf.train.CheckpointManager(checkpoint, directory=args.result_path, max_to_keep=10)
    summary_writer = tf.summary.create_file_writer(args.result_path)
    save_settings(args, param)

    # load data
    if args.debug:
        train_image = edof_reader.load_CAVE(os.path.join(param.data_dir, 'CAVE'))
        dataset_length = len(train_image)
        val_image = train_image
    else:
        train_image = []
        hsdb_image = edof_reader.load_hsdb(os.path.join(param.data_dir, 'hsdb'))
        train_image += hsdb_image

        icvl_image = edof_reader.load_ICVL(os.path.join(param.data_dir, 'ICVL'))
        train_image += icvl_image

        dataset_length = len(train_image)

        val_image = edof_reader.load_CAVE(os.path.join(param.data_dir, 'CAVE'))

    # fix val
    val_idx = 30
    args.val_idx = val_idx
    X_val = edof_reader.dataset_preprocess(val_image[val_idx], patch_size = args.patch_size, num_depths= param.num_depths, is_val=True)
    if args.linear_rgb: 
        x_input = x_input**2.2

    for epoch in range(args.n_epochs):
        if epoch % args.save_freq == 0:
            manager.save()
            log_eval(F,G,VGG, X_val, epoch, summary_writer,args)
        train_idx = np.random.permutation(dataset_length)
        X_train = []
        train_l1_loss = []
        train_l2_loss = []
        train_perc_loss = []
        train_reg_loss = []
        train_total_loss = []
        
        for i in range(dataset_length):
            x_input = train_image[train_idx[i]]
            X_train = edof_reader.dataset_preprocess(x_input, patch_size = args.patch_size, num_depths= param.num_depths)
            with tf.GradientTape(persistent=True) as tape:
                output_image_sensor_org, [psfs, psf_temps], GT_img, _, _ = F([param.input_field1, X_train], training = args.train_F)
                output_image, output_image_rgb = G(output_image_sensor_org, training = True)
                
                GT_img_rgb =  tf.nn.conv2d(GT_img,args.param.response_weight_rgb,strides=[1,1,1,1],padding = 'SAME')
                max_intensity = tf.reduce_max(GT_img_rgb)

                if args.perc_weight > 0:
                    perc_loss = Perc_loss(output_image_rgb/max_intensity, GT_img_rgb/max_intensity, VGG, args.perc_weight)
                else:
                    perc_loss = 0
                center_sum_reg = center_sum_regularizer(psf_temps, args.reg_weight, target = 0.9)
                l1_loss_ms = sam_loss(GT_img, output_image, args.l1_weight_ms) 
                l1_loss_rgb = L1_loss(GT_img_rgb, output_image_rgb, args.l1_weight_rgb)
                total_loss = l1_loss_ms + l1_loss_rgb + perc_loss + center_sum_reg
                
            G_gradients = tape.gradient(total_loss, G.trainable_variables)
            G_optimizer.apply_gradients(zip(G_gradients, G.trainable_variables))
            
            if args.train_F:
                H_gradients = tape.gradient(total_loss, F.trainable_variables[0:1])
                H_optimizer.apply_gradients(zip(H_gradients, F.trainable_variables[0:1]))
                if args.train_curve:
                    C_gradients = tape.gradient(total_loss, F.trainable_variables[1:])
                    C_optimizer.apply_gradients(zip(C_gradients, F.trainable_variables[1:]))
               
            train_l1_loss.append(l1_loss_ms)
            train_l2_loss.append(l1_loss_rgb)
            train_perc_loss.append(perc_loss)
            train_reg_loss.append(center_sum_reg)
            train_total_loss.append(total_loss)
        
        with summary_writer.as_default():
            tf.summary.scalar(name = 'train_loss/l1_loss', data = tf.reduce_mean(tf.stack(train_l1_loss, axis = 0)), step=epoch+1)
            tf.summary.scalar(name = 'train_loss/l2_loss', data = tf.reduce_mean(tf.stack(train_l2_loss, axis = 0)), step=epoch+1)
            tf.summary.scalar(name = 'train_loss/perc_loss', data = tf.reduce_mean(tf.stack(train_perc_loss, axis = 0)), step=epoch+1)
            tf.summary.scalar(name = 'train_loss/reg_loss', data = tf.reduce_mean(tf.stack(train_reg_loss, axis = 0)), step=epoch+1)
            tf.summary.scalar(name = 'train_loss/total_loss', data = tf.reduce_mean(tf.stack(train_total_loss, axis = 0)), step=epoch+1)

    # save final results
    manager.save()
    log_eval(F,G,VGG, X_val, epoch, summary_writer,args)

def main():
    parser = argparse.ArgumentParser()
    def str2bool(v):
        assert(v == 'True' or v == 'False')
        return v.lower() in ('true')

    def none_or_str(value):
        if value.lower() == 'none':
            return None
        return value
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action="store_true",
                        help='debug mode, train on validation data to speed up the process')
    parser.add_argument('--linear_rgb', action="store_true",
                        help='whether to train the model in linear space')    
    parser.add_argument('--result_path', type=str, required=True,
                        help='Directory that checkpoints and tensorboard logfiles will be written to.')
    parser.add_argument('--use_noise', type=str2bool, default = True,
                        help='whether add noise to simulation')
    parser.add_argument('--train_curve', type=str2bool, default = True,
                        help='whether train response curve')
    parser.add_argument('--train_F', type=str2bool, default=True,
                        help='whether train the forward model') 
    parser.add_argument('--pretrained_F', type=none_or_str, default=None,
                        help='ckpt dir of a pretrained forward model')
    parser.add_argument('--pretrained_G', type=none_or_str, default=None,
                        help='ckpt dir of a pretrained reconstruction model')
    parser.add_argument('--finetune_head', type=str2bool, default=False,
                        help='finetune the output head, assuming pretrained_G is not None')
    parser.add_argument('--l1_weight_ms', type=float, default = 1e2,
                        help='Weight on L1 loss on MultiSpectral')
    parser.add_argument('--l1_weight_rgb', type=float, default = 1e2,
                        help='Weight on L1 loss on RGB')
    parser.add_argument('--vgg_layers', type=str, default='block2_conv1,block3_conv1,block4_conv1',
                        help = 'layers used for perceptual loss, seperated by , w/o space')
    parser.add_argument('--perc_weight', type=float, default=0.01,
                        help='Weight on Percptual loss')
    parser.add_argument('--reg_weight', type=float, default=0,
                        help = 'center sum regulization weight')
    parser.add_argument('--patch_size', type=int, default = 512, 
                        help = 'training patch size')
    parser.add_argument('--n_epochs', type=int, default = 200, 
                        help = 'total training epochs')
    parser.add_argument('--save_freq', type=int, default = 10, 
                        help = 'saving frequency (epoch)')
    parser.add_argument('--forward_model', type=str, default = 'optical_forward', 
                        help = 'forward model arch')
    parser.add_argument('--generator', type=str, default = 'UNet', 
                        help = 'generator arch')
    parser.add_argument('--H_lr', type=float, default = 1e-4,
                        help='Height Map learning rate')
    parser.add_argument('--C_lr', type=float, default = 1e-4,
                        help='Color Filter learning rate')
    parser.add_argument('--G_lr', type=float, default = 1e-4,
                        help='Generator learning rate')
    args = parser.parse_args()


    num_GPUs = len(tf.config.list_physical_devices('GPU'))
    args.num_GPUs = num_GPUs
    print("Num GPUs Available: ", num_GPUs)
    args.param = param

    train(args)

if __name__ == '__main__':
    main()


