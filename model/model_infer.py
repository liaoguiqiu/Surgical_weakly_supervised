import torch
import torch.nn.functional as F

from model.model_3dcnn import _VideoCNN
learningR = 0.00001
class _Model_infer(object):
    def __init__(self,GPU_mode =True):
        self.VideoNets = _VideoCNN()
        if GPU_mode ==True:
            self.VideoNets.cuda()
        self.customeBCE = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam([
            # {'params': self.netG.Unet_back.parameters()},
            {'params': self.VideoNets .parameters()}
        ], lr=learningR, betas=(0.5, 0.999))

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
    def forward(self,input):
        self.output, self.slice_valid, self. cam3D= self.VideoNets(input)
    def optimization(self, label):
        self.optimizer.zero_grad()
        self.set_requires_grad(self.VideoNets, True)
        self.loss=  self.customeBCE(self.output[:,:,0,0,0], label)
        # self.lossEa.backward(retain_graph=True)
        self.loss.backward( )

        self.optimizer.step()
        self.lossDisplay = self.loss. data.mean()