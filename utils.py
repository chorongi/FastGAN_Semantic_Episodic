import numpy as np
import torch
import math
from tqdm import tqdm, trange

from torchvision import utils as vutils
import torch.nn.functional as F
from pathlib import Path

def resize(img):
    return F.interpolate(img, size=256)

def batch_generate(zs, netG, batch_size=8, device = "cuda"):
    # generates images using netG given noise vectors zs
    # zs is the input noise generated as the following
    # noise = torch.randn(batch_size, noise_dim).to(device)
    g_images = []
    netG = netG.to(device)
    with torch.no_grad():
        for i in range(math.ceil(len(zs)/batch_size)):
            batch = zs[i*batch_size:min((i+1)*batch_size, len(zs))].to(device)
            g_images.append(netG(batch))
            
    # May use the following line to interpolate images to a given dimension
    # g_imgs = F.interpolate(g_imgs, 512)
    return torch.cat(g_images)

def batch_save(images, img_folder):
    save_dir = Path(img_folder)
    save_dir.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        vutils.save_image(revert_scale(image), f"{img_folder}/{i}.jpg")

def scale(image):
    # image is normalized image (0~1)
    # convert image to -1~1 scale so we have same scale as noise
    return image.mul(2).add(-1)

def revert_scale(image):
    return image.add(1).mul(0.5)

def crop_img_by_part(img, part):
    hw = img.shape[2]//2
    if part==0:
        return img[:,:,:hw,:hw]
    if part==1:
        return img[:,:,:hw,hw:]
    if part==2:
        return img[:,:,hw:,:hw]
    if part==3:
        return img[:,:,hw:,hw:]