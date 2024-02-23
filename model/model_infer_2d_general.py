import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models_torch

from model.model_3dcnn_linear_MCT import _VideoCNN
from working_dir_root import learningR,learningR_res,SAM_pretrain_root,Load_feature,Weight_decay,Evaluation,Display_final_SAM
from dataset.dataset import class_weights,Obj_num
from SAM.segment_anything import  SamPredictor, sam_model_registry

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler
from model import model_operator
import torchvision.models as models

from torchcam.methods import CAM,LayerCAM,GradCAM
Show_cam = False
# learningR = 0.0001
class _Model_infer(object):
    def __init__(self, GPU_mode =True,num_gpus=1,Name= None):
        self.inter_bz =29

        if GPU_mode ==True:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        else:
            device = torch.device("cpu")
        resnet = models.resnet18(pretrained=True)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, Obj_num)  # Assuming you have 7 output classes

        # self.predictor = SamPredictor(self.sam) 
        # self.Vit_encoder = model
        # self.set_requires_grad(self.Vit_encoder, True)

       
        self.input_size = 512
        # resnet18 = models_torch.resnet18(pretrained=True)
        self.gradcam = None
        # Remove the fully connected layers at the end
       
        # Modify the last layer to produce the desired feature map size
        self.resnet =  resnet
        # if GPU_mode ==True:
        #     self.VideoNets.cuda()
        
        if GPU_mode == True:
            if num_gpus > 1:
                pass
                # self.resnet  = torch.nn.DataParallel(self.resnet )
               
         
        self.resnet .to(device)
        self.cam = LayerCAM(self.resnet, 'layer4')
        if Evaluation:
            self.resnet.eval()
        else:
            self.resnet.train(True)
           
 
        self.customeBCE = torch.nn.BCEWithLogitsLoss().to(device)
        # self.customeBCE = F.multilabel_soft_margin_loss()

        # self.customeBCE = torch.nn.BCEWithLogitsLoss(weight=weight_tensor).to(device)

        # self.customeBCE = torch.nn.BCELoss(weight=weight_tensor).to(device)

        self.optimizer = torch.optim.AdamW([
            {'params': self.resnet.parameters(),'lr': learningR_res},
            
        ] , weight_decay=Weight_decay)
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
    def forward(self,input,input_flows, features):
        # self.res_f = self.resnet(input)
        bz, ch, D, H, W = input.size()

        self.input_resample =   F.interpolate(input,  size=(D, self.input_size, self.input_size), mode='trilinear', align_corners=False)
        # self.
        
        flattened_tensor = self.input_resample.permute(0,2,1,3,4)
        flattened_tensor = flattened_tensor.reshape(bz * D, ch, self.input_size, self.input_size)
        self.concatenated_tensor = self.resnet((flattened_tensor-128.0)/60.0)
        # new_bz, new_ch, new_H, new_W = flattened_tensor.size()
        cam_tensors = []
        if Evaluation:
            for id in range(Obj_num):
                # self.optimizer.zero_grad()
                        # Call the cam function for the current class index
                self.concatenated_tensor = self.resnet((flattened_tensor-128.0)/60.0)
                
                cam_tensor = self.cam(class_idx=id,scores=self.concatenated_tensor)  # Assuming cam() is a function defined elsewhere
                # Append the cam tensor to the list
                cam_tensors.append(cam_tensor[0])

            cam_tensor_stack = torch.stack(cam_tensors, dim=0)
            new_ch, new_bz, new_H, new_W = cam_tensor_stack.size()
            self.cam3D= cam_tensor_stack.reshape(bz, new_ch,D, new_H, new_W )
            self.raw_cam = self.cam3D

        new_bz, class_num= self.concatenated_tensor.size()
        self.logits = self.concatenated_tensor.reshape (bz,D,class_num).permute(0,2,1)
        self.output = self.logits.max(dim=2)[0].reshape(bz, Obj_num,1,1,1)
        if Display_final_SAM:
            with torch.no_grad():
                activationLU = nn.ReLU()

                post_processed_masks=model_operator.Cam_mask_post_process(activationLU(self.cam3D), input,self.output)
                # self.sam_mask_prompt_decode(activationLU(self.cam3D),self.f,input)

                
                self.cam3D = post_processed_masks 
        # self.output, self.slice_valid, self. cam3D= self.VideoNets(self.f,self.c_logits,self.p_logits)
    def optimization(self, label,frame_label):
        new_bz, D, ch= frame_label.size()
        frame_label = frame_label.reshape(new_bz*D,ch)
        self.optimizer.zero_grad()
        self.set_requires_grad(self.resnet, True)

        # self.set_requires_grad(self.Vit_encoder, True)
        # c_out,_= torch.max (self.c_logits,dim=2)
        # p_out,_= torch.max (self.p_logits,dim=2)
        # self.loss=  self.customeBCE(self.slice_valid, frame_label)
        # loss_c = F.multilabel_soft_margin_loss(self.concatenated_tensor, frame_label)
        loss_c = F.multilabel_soft_margin_loss(self.concatenated_tensor, frame_label)

        # loss_p = F.multilabel_soft_margin_loss(self.concatenated_x_patch_logits, frame_label)
        # self.loss = loss_c +loss_p
        self.loss = loss_c  

        # self.lossEa.backward(retain_graph=True)
        self.loss.backward()

        self.optimizer.step()
        self.lossDisplay = self.loss. data.mean()