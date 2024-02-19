import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from model.model_2dcnn_self import _VideoCNN2d
from working_dir_root import learningR,learningR_res,Evaluation
from dataset.dataset import class_weights
# learningR = 0.0001
class _Model_infer(object):
    def __init__(self, GPU_mode =True,num_gpus=1,Name = None):
        self.imageNets = _VideoCNN2d()
        self.input_size = 512+128
        resnet18 = models.resnet18(pretrained=True)
        self.gradcam = None
        
        # Remove the fully connected layers at the end
        partial = nn.Sequential(*list(resnet18.children())[0:-3])
        
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
                self.imageNets = torch.nn.DataParallel(self.imageNets)
                self.resnet  = torch.nn.DataParallel(self.resnet )
        self.imageNets.to(device)
        self.resnet .to(device)
        self.imageNets.train(True)
        self.resnet.train(True)
        if Evaluation:
             self.imageNets.eval( )
             self.resnet.eval( )

        
        weight_tensor = torch.tensor(class_weights, dtype=torch.float)
        self.customeBCE = torch.nn.BCEWithLogitsLoss().to(device)
        # self.customeBCE = torch.nn.BCELoss(weight=weight_tensor).to(device)

        self.optimizer = torch.optim.Adam([
            {'params': self. resnet.parameters(),'lr': learningR_res},
            {'params': self.imageNets .parameters(),'lr': learningR}
        ] )
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
    def forward(self,input):
        # self.res_f = self.resnet(input)
        bz, ch, D, H, W = input.size()

        self.input_resample =   F.interpolate(input,  size=(D, self.input_size, self.input_size), mode='trilinear', align_corners=False)
        # self.
        flattened_tensor = self.input_resample.permute(0,2,1,3,4)
        flattened_tensor = flattened_tensor.reshape(bz * D, ch, self.input_size, self.input_size)
        flattened_tensor = self.resnet((flattened_tensor-128.0)/60.0)
        self.output, self.cam2d= self.imageNets(flattened_tensor)


        new_bz, new_ch, new_H, new_W = self.cam2d.size()
        self.cam3D=  self.cam2d.reshape (bz,D,new_ch,new_H, new_W).permute(0,2,1,3,4)
        # self.output, self.cam2d= self.imageNets(self.f)


    def optimization(self, frame_label):
        new_bz, D, ch= frame_label.size()
        frame_label = frame_label.reshape(new_bz*D,ch)
        self.optimizer.zero_grad()
        # self.set_requires_grad(self.imageNets, True)
        # self.set_requires_grad(self.resnet, True)

        self.loss=  F.multilabel_soft_margin_loss(self.output.reshape(frame_label.size(0), frame_label.size(1)), frame_label)
        # self.lossEa.backward(retain_graph=True)
        self.loss.backward( )

        self.optimizer.step()
        self.lossDisplay = self.loss. data.mean()