import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models_torch

from model.model_3dcnn_linear_MCT import _VideoCNN
from working_dir_root import learningR,learningR_res,SAM_pretrain_root,Load_feature,Weight_decay
from dataset.dataset import class_weights,Obj_num
from SAM.segment_anything import  SamPredictor, sam_model_registry

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler
import torchvision.models as models

from torchcam.methods import CAM
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

       
        self.input_size = 256
        # resnet18 = models_torch.resnet18(pretrained=True)
        self.gradcam = None
        # Remove the fully connected layers at the end
       
        # Modify the last layer to produce the desired feature map size
        self.resnet =  resnet
        # if GPU_mode ==True:
        #     self.VideoNets.cuda()
        
        if GPU_mode == True:
            if num_gpus > 1:
                 
                self.resnet  = torch.nn.DataParallel(self.resnet )
               
         
        self.resnet .to(device)
        self.cam = CAM(resnet, 'layer4', 'fc')

 
        self.customeBCE = torch.nn.BCEWithLogitsLoss().to(device)
        # self.customeBCE = F.multilabel_soft_margin_loss()

        # self.customeBCE = torch.nn.BCEWithLogitsLoss(weight=weight_tensor).to(device)

        # self.customeBCE = torch.nn.BCELoss(weight=weight_tensor).to(device)

        self.optimizer = torch.optim.AdamW([
            {'params': self.resnet.parameters(),'lr': learningR_res},
            
        ] , betas=(0.9, 0.999),weight_decay=Weight_decay)
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
        Load_feature = False#########################33333
        if Load_feature == False:
            flattened_tensor = self.input_resample.permute(0,2,1,3,4)
            flattened_tensor = flattened_tensor.reshape(bz * D, ch, self.input_size, self.input_size)
            flattened_tensor = (flattened_tensor-124.0)/60.0

            num_chunks = (bz*D + self.inter_bz - 1) // self.inter_bz
        
            # List to store predicted tensors
            predicted_tensors = []
            predicted_tensors_x_cls_logits=[]
            predicted_tensors_x_patch_logits=[]
            cams = []
            
            # Chunk input tensor and predict
            # with torch.cuda.amp.autocast():
            for i in range(num_chunks):
                start_idx = i * self.inter_bz
                end_idx = min((i + 1) * self.inter_bz, bz*D)
                input_chunk = flattened_tensor[start_idx:end_idx]
                output_chunk = self.resnet(input_chunk)

                cam_tensors = []

                # Iterate over the range of class indices
                for id in range(Obj_num):
                    # Call the cam function for the current class index
                    cam_tensor = self.cam(class_idx=id)  # Assuming cam() is a function defined elsewhere
                    # Append the cam tensor to the list
                    cam_tensors.append(cam_tensor[0])

                cam_tensor_stack = torch.stack(cam_tensors, dim=0)
                # patch_attn = torch.sum(patch_attn, dim=0)
                # cls_attentions = torch.matmul(patch_attn.unsqueeze(1), cls_attentions.view(cls_attentions.shape[0],cls_attentions.shape[1], -1, 1)).reshape(cls_attentions.shape[0],cls_attentions.shape[1], 14, 14)
                predicted_tensors.append(output_chunk)
                cams.append(cam_tensor_stack)

                    # torch.cuda.empty_cache()  # Release memory
            
            # Concatenate predicted tensors along batch dimension
            self.concatenated_tensor = torch.cat(predicted_tensors, dim=0)
            self.concatenated_cams =  torch.cat(cams, dim=0)
        #     self.concatenated_x_cls_logits = torch.cat(predicted_tensors_x_cls_logits, dim=0)
        #     self.concatenated_x_patch_logits = torch.cat(predicted_tensors_x_patch_logits, dim=0)

        #     new_bz, new_ch, new_H, new_W = self.concatenated_tensor.size()
        #     self.f = self.concatenated_tensor.reshape (bz,D,new_ch,new_H, new_W).permute(0,2,1,3,4)

        #     new_bz, class_num= self.concatenated_x_cls_logits.size()
        #     self.c_logits = self.concatenated_x_cls_logits.reshape (bz,D,class_num).permute(0,2,1)

        #     new_bz, class_num= self.concatenated_x_patch_logits.size()
        #     self.p_logits = self.concatenated_x_patch_logits.reshape (bz,D,class_num).permute(0,2,1)
        # else:
        #     self.f = features
            avgpool = nn.AdaptiveAvgPool2d(1)
            self.cam_logits =avgpool(self.concatenated_cams) .squeeze(3).squeeze(2)
            new_bz, new_ch, new_H, new_W = self.concatenated_cams.size()
            self. cam3D = self.concatenated_cams.reshape (bz,Obj_num,D,new_H, new_W).permute(0,1,2,3,4)
            # self.output = torch.max( self.concatenated_tensor,dim=0).view(bz,Obj_num)
            new_bz, class_num= self.concatenated_tensor.size()
            self.logits = self.concatenated_tensor.reshape (bz,D,class_num).permute(0,2,1)
            self.output = self.logits.max(dim=2)[0].view(bz, Obj_num,1,1)

        # self.output, self.slice_valid, self. cam3D= self.VideoNets(self.f,self.c_logits,self.p_logits)
    def optimization(self, label,frame_label):
        new_bz, D, ch= frame_label.size()
        frame_label = frame_label.view(new_bz*D,ch)
        self.optimizer.zero_grad()
        self.set_requires_grad(self.resnet, True)
        # self.set_requires_grad(self.Vit_encoder, True)
        # c_out,_= torch.max (self.c_logits,dim=2)
        # p_out,_= torch.max (self.p_logits,dim=2)
        # self.loss=  self.customeBCE(self.slice_valid, frame_label)
        loss_c = F.multilabel_soft_margin_loss(self.concatenated_tensor, frame_label)
        # loss_p = F.multilabel_soft_margin_loss(self.concatenated_x_patch_logits, frame_label)
        # self.loss = loss_c +loss_p
        self.loss = loss_c  

        # self.lossEa.backward(retain_graph=True)
        self.loss.backward()

        self.optimizer.step()
        self.lossDisplay = self.loss. data.mean()