import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.model_2dcnn import _VideoCNN2d
learningR = 0.0001
class _Model_infer(object):
    def __init__(self, GPU_mode =True,num_gpus=1):
        self.VideoNets = _VideoCNN2d()
        resnet18 = models.resnet18(pretrained=True)
        
        # Remove the fully connected layers at the end
        partial = nn.Sequential(*list(resnet18.children())[0:-6])
        
        # Modify the last layer to produce the desired feature map size
        self.resnet = nn.Sequential(
            partial,
            nn.ReLU()
        )
        # if GPU_mode ==True:
        #     self.VideoNets.cuda()
        if GPU_mode ==True:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        else:
            device = torch.device("cpu")
        if GPU_mode == True:
            if num_gpus > 1:
                self.VideoNets = torch.nn.DataParallel(self.VideoNets)
                self.resnet  = torch.nn.DataParallel(self.resnet )
        self.VideoNets.to(device)
        self.resnet .to(device)
        # self.customeBCE = torch.nn.BCEWithLogitsLoss().to(device)
        self.customeBCE = torch.nn.BCELoss().to(device)

        self.optimizer = torch.optim.Adam([
            # {'params': self.netG.Unet_back.parameters()},
            {'params': self.VideoNets .parameters()}
        ], lr=learningR, betas=(0.5, 0.999))
        # if GPU_mode ==True:
        #     if num_gpus > 1:
        #         self.optimizer = torch.nn.DataParallel(optself.optimizerimizer)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    def forward(self,input3d):
        bz, ch, D, H, W = input3d.size()

        input = input3d[:,:,1,:,:] # first images
        self.res_f = self.resnet(input)
        self.output,  self. cam3D= self.VideoNets(self.res_f)
    def optimization(self, label):
        self.optimizer.zero_grad()
        self.set_requires_grad(self.VideoNets, True)
        self.set_requires_grad(self.resnet, False)

        self.loss=  self.customeBCE(self.output.view(label.size(0), -1), label)
        # self.lossEa.backward(retain_graph=True)
        self.loss.backward()

        self.optimizer.step()
        self.lossDisplay = self.loss. data.mean()