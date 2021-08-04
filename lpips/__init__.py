from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from lpips import dist_model
from utils import scale

class PerceptualLoss(torch.nn.Module):
    def __init__(self, model="net-lin", net="alex", colorspace="rgb", spatial=False, use_gpu=True, gpu_ids=[0]):

        super(PerceptualLoss, self).__init__()
        print("Setting up Perceptual Loss...")
        self.use_gpu = use_gpu
        self.spatial = spatial
        self.gpu_ids = gpu_ids
        self.model = dist_model.DistModel()
        self.model.initialize(model = model, net = net, use_gpu = use_gpu, colorspace = colorspace, spatial = spatial, gpu_ids = gpu_ids)
        print(f"...[{self.model.name()}] initialized")
        print("...Done")

    def forward(self, pred, target, normalize = False):
        if normalize:
            target = scale(target)
            pred = scale(pred)
    
        return self.model.forward(target, pred)
