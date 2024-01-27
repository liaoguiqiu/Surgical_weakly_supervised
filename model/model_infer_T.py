import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from model.model_3dcnn_linear import _VideoCNN
from working_dir_root import learningR,learningR_res,SAM_pretrain_root
from dataset.dataset import class_weights
from SAM.segment_anything import  SamPredictor, sam_model_registry
# learningR = 0.0001
class _Model_infer(object):
    def __init__(self, GPU_mode =True,num_gpus=1):
        if GPU_mode ==True:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        else:
            device = torch.device("cpu")
        sam_checkpoint = SAM_pretrain_root+"sam_vit_h_4b8939.pth"
        sam_checkpoint = SAM_pretrain_root+"sam_vit_l_0b3195.pth"
        sam_checkpoint =SAM_pretrain_root+ "sam_vit_b_01ec64.pth"

        model_type = "vit_h"
        model_type = "vit_l"
        model_type = "vit_b"
        
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam) 
        self.VideoNets = _VideoCNN()
        self.input_size = 1024
        resnet18 = models.resnet18(pretrained=True)
        self.gradcam = None
        
        # Remove the fully connected layers at the end
        partial = nn.Sequential(*list(resnet18.children())[0:-2])
        
        # Modify the last layer to produce the desired feature map size
        self.resnet = nn.Sequential(
            partial,
            nn.ReLU()
        )
        # if GPU_mode ==True:
        #     self.VideoNets.cuda()
        
        if GPU_mode == True:
            if num_gpus > 1:
                self.VideoNets = torch.nn.DataParallel(self.VideoNets)
                self.resnet  = torch.nn.DataParallel(self.resnet )
                self.sam  = torch.nn.DataParallel(self.sam )
        self.VideoNets.to(device)
        self.resnet .to(device)
        
        weight_tensor = torch.tensor(class_weights, dtype=torch.float)
        self.customeBCE = torch.nn.BCEWithLogitsLoss().to(device)
        # self.customeBCE = torch.nn.BCELoss(weight=weight_tensor).to(device)

        self.optimizer = torch.optim.Adam([
            # {'params': self. sam.parameters(),'lr': learningR_res},
            {'params': self.VideoNets .parameters(),'lr': learningR}
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
    def forward(self,input,input_flows):
        # self.res_f = self.resnet(input)
        bz, ch, D, H, W = input.size()

        self.input_resample =   F.interpolate(input,  size=(D, self.input_size, self.input_size), mode='trilinear', align_corners=False)
        # self.
        flattened_tensor = self.input_resample.permute(0,2,1,3,4)
        flattened_tensor = flattened_tensor.reshape(bz * D, ch, self.input_size, self.input_size)
        flattened_tensor = (flattened_tensor-124.0)/60.0
        flattened_tensor = self.sam.image_encoder(flattened_tensor)
        new_bz, new_ch, new_H, new_W = flattened_tensor.size()
        self.f = flattened_tensor.reshape (bz,D,new_ch,new_H, new_W).permute(0,2,1,3,4)
        self.output, self.slice_valid, self. cam3D= self.VideoNets(self.f,input_flows)
    def optimization(self, label):
        self.optimizer.zero_grad()
        self.set_requires_grad(self.VideoNets, True)
        # self.set_requires_grad(self.resnet, True)

        self.loss=  self.customeBCE(self.output.view(label.size(0), -1), label)
        # self.lossEa.backward(retain_graph=True)
        self.loss.backward( )

        self.optimizer.step()
        self.lossDisplay = self.loss. data.mean()