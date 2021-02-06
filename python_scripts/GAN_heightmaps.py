import torch
import numpy as np
import sys
import os
from stylegan2_pytorch import ModelLoader, Trainer

def load_latest_model():
    model_args = dict(
        name = 'default',
        results_dir = './results',
        models_dir = './saved_models/GAN_heightmaps',
        batch_size = 8,
        gradient_accumulate_every = 6,
        image_size = 128,
        network_capacity = 4,
        fmap_max = 512,
        transparent = False,
        lr = 2e-4,
        lr_mlp = 0.1,
        ttur_mult = 1.5,
        rel_disc_loss = False,
        num_workers = 16,
        save_every = 1000,
        evaluate_every = 1000,
        trunc_psi = 0.75,
        fp16 = False,
        cl_reg = False,
        fq_layers = [],
        fq_dict_size = 256,
        attn_layers = [],
        no_const = False,
        aug_prob = 0.,
        aug_types = ['translation', 'cutout'],
        top_k_training = False,
        generator_top_k_gamma = 0.99,
        generator_top_k_frac = 0.5,
        dataset_aug_prob = 0.,
        calculate_fid_every = None,
        mixed_prob = 0.9,
        log = False
    )
    model = Trainer(**model_args)
    model.load(-1)
    model.GAN.train(False)
    return model

def exists(val):
    return val is not None

def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)

def image_noise(n, im_size, device="cpu"):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).to(device)

def get_img_from_tensor(t, nrow=1):
    t = torch.clamp(t,0.0,1.0)
    imgs = t[0].detach().cpu().numpy().swapaxes(0,1).swapaxes(1,2)
    imgs *= 255 
    imgs = imgs.astype(np.uint8)
    return imgs

def noise_to_styles(model, noise, trunc_psi = None):
    w = model.GAN.S(noise)
    if exists(trunc_psi):
        #print("truncing")
        w = model.truncate_style(w)
    return w

def styles_to_images(model, w, noise):
    num_layers = model.GAN.G.num_layers
    w_def = [(w, num_layers)]

    w_tensors = styles_def_to_tensor(w_def)
    image = model.GAN.G(w_tensors, noise)
    image.clamp_(0., 1.)
    return image

def feed_forward(model, noise, img_noises):
    styles  = noise_to_styles(model, noise, trunc_psi = None)
    images  = styles_to_images(model, styles, img_noises) 
    print(images.shape)
    images = images[:,0:1,:,:]
    return get_img_from_tensor(images, nrow=1)

def generate_heightmap():
    print("Loading model")
    model = load_latest_model()

    noise = torch.randn(1, 512)
    img_noises = image_noise(1, 128)

    img = feed_forward(model, noise, img_noises)
    print(img.shape)

    return img
    