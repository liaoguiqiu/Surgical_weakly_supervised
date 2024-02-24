import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models_torch

from model.model_3dcnn_linear_MCT import _VideoCNN
from working_dir_root import learningR,learningR_res,SAM_pretrain_root,Load_feature,Weight_decay,Evaluation,Display_final_SAM
from dataset.dataset import class_weights,Obj_num
from SAM.segment_anything import  SamPredictor, sam_model_registry
from model import model_operator
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler
from MCTformer import models
# learningR = 0.0001
class _Model_infer(object):
    def __init__(self, GPU_mode =True,num_gpus=1):
        self.inter_bz =29

        if GPU_mode ==True:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        else:
            device = torch.device("cpu")
        model = create_model(
        'deit_small_MCTformerV2_patch16_224',
        pretrained=True,
        num_classes= Obj_num,
        drop_rate= 0.0,
        drop_path_rate=0.0,
        drop_block_rate=None
        )
        # model.train(True)
        
        # self.predictor = SamPredictor(self.sam) 
        self.Vit_encoder = model
        # self.set_requires_grad(self.Vit_encoder, True)

        self.VideoNets = _VideoCNN()
        self.input_size = 224
        resnet18 = models_torch.resnet18(pretrained=True)
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
                self.Vit_encoder   = torch.nn.DataParallel(self.Vit_encoder  )
        self.VideoNets.to(device)
        self.resnet .to(device)
        self.Vit_encoder.to(device)
        if Evaluation == False:
            self.Vit_encoder.train(True)
        else:
            self.Vit_encoder.eval()
        weight_tensor = torch.tensor(class_weights, dtype=torch.float)
        self.customeBCE = torch.nn.BCEWithLogitsLoss().to(device)
        # self.customeBCE = F.multilabel_soft_margin_loss()

        # self.customeBCE = torch.nn.BCEWithLogitsLoss(weight=weight_tensor).to(device)

        # self.customeBCE = torch.nn.BCELoss(weight=weight_tensor).to(device)

        self.optimizer = torch.optim.AdamW([
            {'params': self.Vit_encoder.parameters(),'lr': learningR_res},
            {'params': self.VideoNets .parameters(),'lr': learningR}
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

            
            # Chunk input tensor and predict
            # with torch.cuda.amp.autocast():
            for i in range(num_chunks):
                start_idx = i * self.inter_bz
                end_idx = min((i + 1) * self.inter_bz, bz*D)
                input_chunk = flattened_tensor[start_idx:end_idx]
                x_cls_logits, cls_attentions, patch_attn,x_patch_logits = self.Vit_encoder(input_chunk,return_att=True)

                patch_attn = torch.sum(patch_attn, dim=0)
                # cls_attentions = torch.matmul(patch_attn.unsqueeze(1), cls_attentions.view(cls_attentions.shape[0],cls_attentions.shape[1], -1, 1)).reshape(cls_attentions.shape[0],cls_attentions.shape[1], 14, 14)
                predicted_tensors.append(cls_attentions)
                predicted_tensors_x_cls_logits.append(x_cls_logits)
                predicted_tensors_x_patch_logits.append(x_patch_logits)


                    # torch.cuda.empty_cache()  # Release memory
            
            # Concatenate predicted tensors along batch dimension
            self.concatenated_tensor = torch.cat(predicted_tensors, dim=0)
            self.concatenated_x_cls_logits = torch.cat(predicted_tensors_x_cls_logits, dim=0)
            self.concatenated_x_patch_logits = torch.cat(predicted_tensors_x_patch_logits, dim=0)

            new_bz, new_ch, new_H, new_W = self.concatenated_tensor.size()
            self.f = self.concatenated_tensor.reshape (bz,D,new_ch,new_H, new_W).permute(0,2,1,3,4)

            new_bz, class_num= self.concatenated_x_cls_logits.size()
            self.c_logits = self.concatenated_x_cls_logits.reshape (bz,D,class_num).permute(0,2,1)

            new_bz, class_num= self.concatenated_x_patch_logits.size()
            self.p_logits = self.concatenated_x_patch_logits.reshape (bz,D,class_num).permute(0,2,1)
        else:
            self.f = features
        self.output, self.slice_valid, self. cam3D= self.VideoNets(self.f,self.c_logits,self.p_logits)
        max_p,_ =  torch.max(self.c_logits,dim=2)
        self.output =max_p.reshape(bz, class_num,1,1,1)
        self.raw_cam = self.cam3D
        if Display_final_SAM:
            with torch.no_grad():
                activationLU = nn.ReLU()

                post_processed_masks=model_operator.Cam_mask_post_process(activationLU(self.cam3D), input,self.output)
                # self.sam_mask_prompt_decode(activationLU(self.cam3D),self.f,input)

                
                self.cam3D = post_processed_masks 
    def optimization(self, label,frame_label):
        new_bz, D, ch= frame_label.size()
        frame_label = frame_label.reshape(new_bz*D,ch)
        self.optimizer.zero_grad()
        # self.set_requires_grad(self.VideoNets, True)
        # self.set_requires_grad(self.Vit_encoder, True)
        # c_out,_= torch.max (self.c_logits,dim=2)
        # p_out,_= torch.max (self.p_logits,dim=2)
        # self.loss=  self.customeBCE(self.slice_valid, frame_label)
        loss_c = F.multilabel_soft_margin_loss(self.concatenated_x_cls_logits, frame_label)
        loss_p = F.multilabel_soft_margin_loss(self.concatenated_x_patch_logits, frame_label)
        self.loss = loss_c +loss_p
        # self.loss = loss_p  

        # self.lossEa.backward(retain_graph=True)
        self.loss.backward()

        self.optimizer.step()
        self.lossDisplay = self.loss. data.mean()