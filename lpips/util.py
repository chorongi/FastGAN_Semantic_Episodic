import torch
import numpy as np
from skimage import measure
from torch.autograd import Variable



def normalize_tensor(in_feature, eps = 1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feature**2, dim = 1, keepdim = True))
    return in_feature / (norm_factor + eps)

def l2(p0, p1, max_val = 255.0):
    return 0.5 * np.mean((p0 / max_val - p1 / max_val)**2)

def psnr(p0, p1, peak = 255.0):
    p0 = np.array(p0)
    p1 = np.array(p1)
    return 10 * np.log10(peak**2 / np.mean((p0.astype(np.float32) - p1.astype(np.float32))**2))

def dssim(p0, p1, max_val = 255.0):
    return (1 - measure.compare_ssim(p0, p1, data_range = max_val, multichannel = True)) / 2

def rgb2lab(in_img, mean_cent = False):
    from skimage import color
    return color.rgb2lab(in_img / 255.0)

def np2tensor(np_obj):
    return torch.Tensor(np_obj[:,:,:,np.newaxis].transpose((3,2,0,1)))

def tensor2tensorlab(image_tensor,to_norm=True,mc_only=False):
    # image tensor to lab tensor
    from skimage import color

    img = tensor2img(image_tensor)
    img_lab = color.rgb2lab(img)
    if(mc_only):
        img_lab[:,:,0] = img_lab[:,:,0]-50
    if(to_norm and not mc_only):
        img_lab[:,:,0] = img_lab[:,:,0]-50
        img_lab = img_lab/100.

    return np2tensor(img_lab)

def tensorlab2tensor(lab_tensor,return_inbnd=False):
    from skimage import color
    import warnings
    warnings.filterwarnings("ignore")

    lab = tensor2np(lab_tensor)*100.
    lab[:,:,0] = lab[:,:,0]+50

    rgb_back = 255.*np.clip(color.lab2rgb(lab.astype('float')),0,1)
    if(return_inbnd):
        # convert back to lab, see if we match
        lab_back = color.rgb2lab(rgb_back.astype('uint8'))
        mask = 1.*np.isclose(lab_back,lab,atol=2.)
        mask = np2tensor(np.prod(mask,axis=2)[:,:,np.newaxis])
        return (img2tensor(rgb_back),mask)
    else:
        return img2tensor(rgb_back)

def tensor2img(img_tensor, imtype = np.uint8, center = 1, factor = 255.0 / 2.0):
    img_np = img_tensor[0].cpu().float().numpy()
    img_np = (np.transpose(img_np, (1,2,0)) + center) * factor
    return img_np.astype(imtype)

def img2tensor(img, imtype = np.uint8, center = 1, factor = 255.0 / 2.0):
    return torch.Tensor((img / factor - center)[:,:,:,np.newaxis].transpose((3,2,0,1)))