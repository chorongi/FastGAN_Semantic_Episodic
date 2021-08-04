import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import random
from attention import attention
from utils import crop_img_by_part
from torch.nn.utils import spectral_norm
from copy import deepcopy

# Spectral normalization stabilizes the training of discriminators (critics) in Generative Adversarial Networks (GANs) 
# by rescaling the weight tensor with spectral norm σ of the weight matrix calculated using power iteration method

# We assume that the image / feature input is (Batch_Size, channel, height, width) through out all computations
##########################
####### Operations #######
##########################

def weights_init(m):
    classname = m.__class__.__name__
    # String find returns -1 if it doesn't it does not contain substring
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def copy_model_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

###################################
### Define Layers for GAN model ###
###################################

def conv2d(*args, **kwargs):
    # in_channels, out_channels, kernel_size, stride=1, padding=0 ...
    return spectral_norm(nn.Conv2d(*args, **kwargs))

def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))

def linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))

def batchNorm2d(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)

class PixelNorm(nn.Module):
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)
        # rsqrt = 1/sprt(input) - elementwise

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.target_shape = shape

    def forward(self, feature):
        batch = feature.shape[0]
        return feature.view(batch, *self.target_shape)        

# Gated Linear Unit (GLU) Activation 
# https://arxiv.org/pdf/1612.08083.pdf
# The GLU also has non-linear capabilities, but has a linear path for the gradient so diminishes the vanishing gradient problem.
# GLU can be thought as a multiplicative skip connection which helps gradients flow through layers
# Used in InitLayer and UpBlock
class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'number of channels should not divide by 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])

# Noise injection for better generalization (similar to weight decay & early stopping)
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2771718/pdf/MPHYA6-000036-004810_1.pdf
# Used in UpblockComp_control
class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize noise weight as 0
        self.weight = nn.Parameter(torch.zeros(1), requires_grad = True)

    def forward(self, feature, noise = None):
        if noise is None:
            batch, _, height, width = feature.shape
            noise = torch.randn(batch, 1, height, width).to(feature.device)
            # We need to inject the same noise to every channel
            return feature + self.weight * noise
        else:
            return

# Activation function SWISH
# https://arxiv.org/pdf/1710.05941.pdf
# Used in SEBlock
class Swish(nn.Module):
    def forward(self, feature):
        return feature * torch.sigmoid(feature)

# Squeeze and Excitation Blocks
# https://arxiv.org/pdf/1709.01507.pdf
# Explicitly models the interdependencies between the channels of its convolutional features
class SEBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.layers = nn.Sequential(nn.AdaptiveAvgPool2d(4), 
                                    conv2d(in_channel, out_channel, 4, 1, 0, bias = False), 
                                    Swish(), 
                                    conv2d(out_channel, out_channel, 1, 1, 0, bias = False), 
                                    nn.Sigmoid() )

    def forward(self, small_feature, big_feature):
        return big_feature * self.layers(small_feature)

class InitLayer(nn.Module):
    def __init__(self, nz, channel):
        super().__init__()
        self.layers = nn.Sequential(convTranspose2d(nz, channel * 2, 4, 1, 0, bias = False),
                                    batchNorm2d(channel * 2),
                                    GLU())

    def forward(self, noise):
        batch_size = noise.shape[0]
        noise = noise.view(batch_size, -1, 1, 1)
        return self.layers(noise)


# Upsample - Conv2d - BN - GLU
def UpBlock(in_planes, out_planes):
    # Note that the output channel of conv2d is out_planes * 2
    # because GLU halves the number of out channels
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        conv2d(in_planes, out_planes * 2, 3, 1, 1, bias = False),
        batchNorm2d(out_planes * 2), 
        GLU()
    )

# Upsample - Conv2d - Noise Inj - BN - GLU - Conv2d - Noise Inj - BN - GLU
class UpBlock_Computation(nn.Module):
    def __init__(self, in_planes, out_planes, noise = False):
        super(UpBlock_Computation, self).__init__()
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False)
        self.noise1 = NoiseInjection()
        self.bn1 = batchNorm2d(out_planes*2)
        self.glu1 = GLU()
        self.conv2 = conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False)
        self.noise2 = NoiseInjection()
        self.bn2 = batchNorm2d(out_planes*2)
        self.glu2 = GLU()
        
        self.noise = noise
    
    def forward(self, x):
        x = self.up1(x)
        x = self.conv1(x)
        if self.noise:
            x = self.noise1(x)
        x = self.bn1(x)
        x = self.glu1(x)
        x = self.conv2(x)
        if self.noise:
            x = self.noise2(x)
        x = self.bn2(x)
        x = self.glu2(x)
        return x

class Generator(nn.Module):
    def __init__(self, config, ngf=64, nz=100, nc=3, im_size=1024, noise=True):
        # ngf : number dimensions for generator feature
        # nz : number dimensions for noise
        # nc : number of channels
        super(Generator, self).__init__()

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        self.im_size = im_size

        self.init_layer = InitLayer(nz, channel=nfc[4])
        self.upBlock_8 = UpBlock_Computation(nfc[4], nfc[8], noise = noise)
        self.upBlock_16 = UpBlock(nfc[8], nfc[16]) # in_planes, out_planes
        self.upBlock_32 = UpBlock_Computation(nfc[16], nfc[32], noise = noise)
        self.upBlock_64 = UpBlock(nfc[32], nfc[64])
        self.upBlock_128 = UpBlock_Computation(nfc[64], nfc[128], noise = noise)
        self.upBlock_256 = UpBlock(nfc[128], nfc[256]) 
        self.upBlock_512 = UpBlock_Computation(nfc[256], nfc[512], noise = noise)
        self.upBlock_1024 = UpBlock(nfc[512], nfc[1024])


        # different - original fastGAN uses different number of seBlocks depending on im_size.
        # Here I just choose to always use all blocks
        self.seBlock_64  = SEBlock(nfc[4], nfc[64]) # small Feature, big Feature
        self.seBlock_128 = SEBlock(nfc[8], nfc[128])
        self.seBlock_256 = SEBlock(nfc[16], nfc[256])
        self.seBlock_512 = SEBlock(nfc[32], nfc[512])

        self.feat_to_128 = conv2d(nfc[128], nc, 1, 1, 0, bias = False)
        self.feat_to_big = conv2d(nfc[im_size], nc, 3, 1, 0, bias = False)

        if im_size > 256:
            self.feat_512 = UpBlockComp(nfc[256], nfc[512], noise=noise) 
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if im_size > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])  

        ### Attention ###
        # TODO : instantiate attention layer according to config param
        self.attention_layer = None
        # self.attention = attention()

    
    def forward(self, x, output_attention_layer = False, eval_noise_weight = 0):
        # eval_noise_weight used to inject noise during eval time

        feat_4   = self.init_layer(x)
        feat_8   = self.upBlock_8(feat_4)
        feat_16  = self.upBlock_16(feat_8)
        feat_32  = self.upBlock_32(feat_16)
        feat_64  = self.upBlock_64(feat_32)
        feat_64  = self.seBlock_64(feat_4, feat_64)

        eval_time_noise = (torch.rand(feat_64.shape) * eval_noise_weight).to(feat_64.device) # Different
        feat_64 += eval_time_noise

        # TODO
        # Include attention layer depending on which resolution you want to apply attention to

        feat_128 = self.upBlock_128(feat_64)
        feat_128 = self.seBlock_128(feat_8, feat_128)
        
        feat_256 = self.upBlock_256(feat_128)
        feat_256 = self.seBlock_256(feat_16, feat_256)

        final_feat = feat_256

        if(self.im_size > 256):
            feat_512 = self.upBlock_512(feat_256)
            feat_512 = self.seBlock_512(feat_32, feat_512)
            final_feat = feat_512
        
        if(self.im_size > 512):
            feat_1024 = self.upBlock_1024(feat_512)
            final_feat = feat_1024

        img_128 = torch.tanh(self.feat_to_128(feat_128))
        img_big = torch.tanh(self.feat_to_big(final_feat))

        return [img_big, img_128]

# conv2d - BN - LeakyReLU
class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlock, self).__init__()

        self.layers = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias = False),
            batchNorm2d(out_planes), 
            nn.LeakyReLU(0.2, inplace = True)
        )

    def forward(self, x):
        return self.layers(x)

# conv2d - BN - LeakyReLU - conv2d - BN -LeakyReLU
class DownBlock_Computation(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlock_Computation, self).__init__()

        self.layers = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias = False),
            batchNorm2d(out_planes),
            nn.LeakyReLU(0.2, inplace = True),
            conv2d(out_planes, out_planes, 3, 1, 1, bias = False),
            batchNorm2d(out_planes),
            nn.LeakyReLU(0.2)
        )

        self.direct = nn.Sequential(
            nn.AvgPool2d(2,2),
            conv2d(in_planes, out_planes, 1, 1, 0, bias = False),
            batchNorm2d(out_planes),
            nn.LeakyReLU(0.2)
        )   
    
    def forward(self, x):
        return (self.layers(x) + self.direct(x)) / 2


class Discriminator(nn.Module):
    def __init__(self, config, ndf=64, nc=3, im_size=512):
        # ndf : feature dimension for discriminator
        # nc : number of channels of image
        super(Discriminator, self).__init__()

        self.ndf = ndf
        self.im_size = im_size

        nfc_multi = {4:16, 8:16, 16:8, 32:4, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ndf)
        
        self.big_down_layers = self.gen_down_layer(nfc, nc, im_size, type = "big")
        self.small_down_layers = self.gen_down_layer(nfc, nc, im_size, type = "small")

        self.downBlock_4 = DownBlock_Computation(nfc[512], nfc[256])
        self.downBlock_8 = DownBlock_Computation(nfc[256], nfc[128])
        self.downBlock_16 = DownBlock_Computation(nfc[128], nfc[64])
        self.downBlock_32 = DownBlock_Computation(nfc[64], nfc[32])
        self.downBlock_64 = DownBlock_Computation(nfc[32], nfc[16])

        self.rf_small = conv2d(nfc[32], 1, 4, 1, 0, bias=False)
        self.rf_big = nn.Sequential(
            conv2d(nfc[16], nfc[8], 1, 1, 0, bias=False),
            batchNorm2d(nfc[8]),
            nn.LeakyReLU(0.2, inplace=True),
            conv2d(nfc[8], 1, 4, 1, 0, bias=False)
        )
    
        self.seBlock_2_16 = SEBlock(nfc[512], nfc[64])
        self.seBlock_4_32 = SEBlock(nfc[256], nfc[32])
        self.seBlock_8_64 = SEBlock(nfc[128], nfc[16])

        self.decoder_big = SimpleDecoder(nfc[16], nc)
        self.decoder_small = SimpleDecoder(nfc[32], nc)
        self.decoder_part = SimpleDecoder(nfc[32], nc)
        
    def gen_down_layer(self, nfc, nc, im_size, type = "big"):
        if type == "big":
            if im_size == 1024:
                return nn.Sequential(
                    conv2d(nc, nfc[1024], 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    conv2d(nfc[1024], nfc[512], 4, 2, 1, bias=False),
                    batchNorm2d(nfc[512]),
                    nn.LeakyReLU(0.2, inplace=True)
                )            
            elif im_size == 512:
                return nn.Sequential(
                    conv2d(nc, nfc[512], 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            elif im_size == 256:
                return nn.Sequential(
                    conv2d(nc, nfc[512], 3, 1, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            else:
                print("[*] ERROR: Unexpected Image Size")
                exit()
        elif type == "small":
            return nn.Sequential(
                conv2d(nc, nfc[256], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                DownBlock(nfc[256], nfc[128]),
                DownBlock(nfc[128], nfc[64]),
                DownBlock(nfc[64], nfc[32])
            )
        else:
            print("[*] ERROR: Unexpected Type of Down sample Init Layer")
            exit()

    def forward(self, imgs, label):
        # Convert image to Big / Small(128) standard versions
        if type(imgs) is not list:
            imgs = [F.interpolate(imgs, size=self.im_size), F.interpolate(imgs,size=128)]
        
        feat_2 = self.big_down_layers(imgs[0])
        feat_4 = self.downBlock_4(feat_2)
        feat_8 = self.downBlock_8(feat_4)
        feat_16 = self.downBlock_16(feat_8)
        feat_16 = self.seBlock_2_16(feat_2, feat_16)
        feat_32 = self.downBlock_32(feat_16)
        feat_32 = self.seBlock_4_32(feat_4, feat_32)
        feat_64 = self.downBlock_64(feat_32)
        
        feat_big = self.seBlock_8_64(feat_8, feat_64)
        feat_small = self.small_down_layers(imgs[1])
        
        rf_0 = self.rf_big(feat_big).view(-1)
        rf_1 = self.rf_small(feat_small).view(-1)
        
        if label == "real":
            part = random.randint(0, 3)
            rec_img_big = self.decoder_big(feat_big)
            rec_img_small = self.decoder_small(feat_small)
            rec_img_part = self.decoder_part(crop_img_by_part(feat_32, part))
            return torch.cat([rf_0, rf_1]), [rec_img_big, rec_img_small, rec_img_part], part
        else:
            return torch.cat([rf_0, rf_1])


class SimpleDecoder(nn.Module):

    def __init__(self, nfc_in=64, nc=3, ndf=32):
        super(SimpleDecoder, self).__init__()

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ndf)
        
        self.layers = nn.Sequential( nn.AdaptiveAvgPool2d(8),
                                     UpBlock(nfc_in, nfc[16]),
                                     UpBlock(nfc[16], nfc[32]),
                                     UpBlock(nfc[32], nfc[64]),
                                     UpBlock(nfc[64], nfc[128]),
                                     conv2d(nfc[128], nc, 3, 1, 1, bias=False),
                                     nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)


class TextureDiscriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=512):
        super(TextureDiscriminator, self).__init__()
        self.ndf = ndf
        self.im_size = im_size
        nfc_multi = {4:16, 8:8, 16:8, 32:4, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ndf)

        self.small_down_layers = nn.Sequential(
                                    conv2d(nc, nfc[256], 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    DownBlock(nfc[256], nfc[128]),
                                    DownBlock(nfc[128], nfc[64]),
                                    DownBlock(nfc[64], nfc[32]),
        )
        self.rf_small = nn.Sequential(conv2d(nfc[16], 1, 4, 1, 0, bias=False))
        self.decoder_small = SimpleDecoder(nfc[32], nc)

    def forward(self, img, label):
        img = self.random_crop(img, size=128)
        feat_small = self.small_down_layers(img)
        rf = self.rf_small(feat_small).view(-1)
        if label == "real":
            rec_img_small = self.decoder_small(feat_small)
            return rf, rec_img_small, img
        else:
            return rf



    def random_crop(self, img, size):
        h, w = image.shape[2:]
        ch = random.randint(0, h-size-1)
        cw = random.randint(0, w-size-1)
        return image[:,:,ch:ch+size,cw:cw+size]

####################################
##      Encoder Definitions       ##  
####################################

# ResNET-18 based Encoder
class BasicEncoder(nn.Module):
    def __init__(self, feature_dim):
        super(BasicEncoder, self).__init__()
        self.encoder = torchvision.models.resnet18() # output of resnet18 is a 1000 dimension
        self.head = nn.Linear(1000, feature_dim)

    def forward(self, x):
        feature = self.encoder(x)
        out = self.head(feature)
        return out

        self.adaptive_pool1 = nn.AdaptiveAvgPool2d((1,1))
        self.adaptive_pool2 = nn.AdaptiveAvgPool2d((1,1))
        self.last_dim = nfc[16] + nfc[32]
        self.mean_head = nn.Linear(self.last_dim, noise_dim)

