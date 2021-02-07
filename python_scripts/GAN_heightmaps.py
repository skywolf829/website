import torch
import numpy as np
import sys
import os
#from stylegan2_pytorch import ModelLoader, Trainer
import imageio
from python_scripts.StyleGan2 import StyleGAN2

def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "saved_models")

    model = StyleGAN2()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    state_dict = torch.load(os.path.join(models_dir, "GAN_heightmaps", "model.pt"))
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.train(False)
    
    return model

def image_noise(n, im_size, device="cpu"):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).to(device)

def get_img_from_tensor(t, nrow=1):
    t = torch.clamp(t,0.0,1.0)
    imgs = t[0].detach().cpu().numpy().swapaxes(0,1).swapaxes(1,2)
    imgs *= 255 
    imgs = imgs.astype(np.uint8)
    return imgs

def generate_heightmap(model):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    noise = torch.randn(1, 512).to(device)
    img_noises = image_noise(1, 128).to(device)

    img = model(noise, img_noises)    
    img = get_img_from_tensor(img, 1)
    return img
    
    