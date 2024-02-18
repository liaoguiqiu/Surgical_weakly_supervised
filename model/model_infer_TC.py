import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
from model.model_3dcnn_linear_TC import _VideoCNN
from model.model_3dcnn_linear_ST import _VideoCNN_S
from working_dir_root import learningR,learningR_res,SAM_pretrain_root,Load_feature,Weight_decay,Evaluation,Display_student
from dataset.dataset import class_weights
import numpy as np
from image_operator import basic_operator   
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from SAM.segment_anything import  SamPredictor, sam_model_registry
from working_dir_root import Enable_student

# from MobileSAM.mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from dataset.dataset import label_mask,Mask_out_partial_label
if Evaluation == True:
    learningR=0
    Weight_decay=0
# learningR = 0.0001
class _Model_infer(object):
    def __init__(self, GPU_mode =True,num_gpus=1):
        if GPU_mode ==True:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        else:
            device = torch.device("cpu")
        self.device = device
        sam_checkpoint = SAM_pretrain_root+"sam_vit_h_4b8939.pth"
        sam_checkpoint = SAM_pretrain_root+"sam_vit_l_0b3195.pth"
        sam_checkpoint =SAM_pretrain_root+ "sam_vit_b_01ec64.pth"
        self.inter_bz =2
        model_type = "vit_h"
        model_type = "vit_l"
        model_type = "vit_b"
        
        # model_type = "vit_t"
        # sam_checkpoint = "./MobileSAM/weights/mobile_sam.pt"
        
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        # self.predictor = SamPredictor(self.sam) 
        self.Vit_encoder = sam.image_encoder
        sam_predictor = SamPredictor(sam)
        self.sam_model = sam_predictor.model
        self.VideoNets = _VideoCNN()
        self.VideoNets_S = _VideoCNN_S()
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
                # self.VideoNets.classifier = torch.nn.DataParallel(self.VideoNets.classifier)
                # self.VideoNets.blocks = torch.nn.DataParallel(self.VideoNets.blocks)
                self.VideoNets = torch.nn.DataParallel(self.VideoNets)
                self.VideoNets_S = torch.nn.DataParallel(self.VideoNets_S)


                self.resnet  = torch.nn.DataParallel(self.resnet )
                self.Vit_encoder   = torch.nn.DataParallel(self.Vit_encoder  )
                self.sam_model  = torch.nn.DataParallel(self.sam_model )
        self.VideoNets.to(device)
        self.VideoNets_S.to(device)


        # self.VideoNets.classifier.to(device)
        # self.VideoNets.blocks.to(device)


        self.resnet .to(device)
        self.Vit_encoder.to(device)
        self.sam_model .to (device)
        if Evaluation:
             self.VideoNets.eval()
             self.VideoNets_S.eval()
        else:
            self.VideoNets.train(True)
            self.VideoNets_S.train(True)

        
        weight_tensor = torch.tensor(class_weights, dtype=torch.float)
        # self.customeBCE = torch.nn.BCEWithLogitsLoss().to(device)
        # self.customeBCE = torch.nn.BCEWithLogitsLoss(weight=weight_tensor).to(device)
        self.customeBCE = torch.nn.BCEWithLogitsLoss().to(device)
        self.customeBCE_S = torch.nn.BCEWithLogitsLoss().to(device)



        self.customeBCE_mask = torch.nn.MSELoss( ).to(device)

        # self.customeBCE = torch.nn.BCELoss(weight=weight_tensor).to(device)
        
        # self.optimizer = torch.optim.Adam([
        # {'params': self.VideoNets.parameters(),'lr': learningR}
        # # {'params': self.VideoNets.blocks.parameters(),'lr': learningR*0.9}
        # ], weight_decay=0.1)
        # self.optimizer = torch.optim.Adam([
        # {'params': self.VideoNets.parameters(),'lr': learningR}
        # # {'params': self.VideoNets.blocks.parameters(),'lr': learningR*0.9}
        # ])
        self.optimizer = torch.optim.AdamW ([
        {'params': self.VideoNets.parameters(),'lr': learningR}
        # {'params': self.VideoNets.blocks.parameters(),'lr': learningR*0.9}
        ], weight_decay=Weight_decay)
        self.optimizer_s = torch.optim.AdamW ([
        {'params': self.VideoNets_S.parameters(),'lr': learningR}
        # {'params': self.VideoNets.blocks.parameters(),'lr': learningR*0.9}
        ], weight_decay=Weight_decay)
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
    def forward(self,input,input_flows, features,Enable_student):
        # self.res_f = self.resnet(input)
        bz, ch, D, H, W = input.size()
        activationLU = nn.ReLU()


        self.input_resample =   F.interpolate(input,  size=(D, self.input_size, self.input_size), mode='trilinear', align_corners=False)
        # self.
        if Load_feature == False:
            flattened_tensor = self.input_resample.permute(0,2,1,3,4)
            flattened_tensor = flattened_tensor.reshape(bz * D, ch, self.input_size, self.input_size)
            flattened_tensor = (flattened_tensor-124.0)/60.0

            num_chunks = (bz*D + self.inter_bz - 1) // self.inter_bz
        
            # List to store predicted tensors
            predicted_tensors = []
            
            # Chunk input tensor and predict
            with torch.no_grad():
                for i in range(num_chunks):
                    start_idx = i * self.inter_bz
                    end_idx = min((i + 1) * self.inter_bz, bz*D)
                    input_chunk = flattened_tensor[start_idx:end_idx]
                    output_chunk = self.Vit_encoder(input_chunk)
                    predicted_tensors.append(output_chunk)
                    # torch.cuda.empty_cache()
               
        
            # Concatenate predicted tensors along batch dimension
            concatenated_tensor = torch.cat(predicted_tensors, dim=0)

            flattened_tensor = concatenated_tensor
            new_bz, new_ch, new_H, new_W = flattened_tensor.size()
            self.f = flattened_tensor.reshape (bz,D,new_ch,new_H, new_W).permute(0,2,1,3,4)
        else:
            with torch.no_grad():
                self.f = features
        self.output, self.slice_valid, self. cam3D= self.VideoNets(self.f,input_flows)
        with torch.no_grad():
            self.slice_hard_label,self.binary_masks= self.CAM_to_slice_hardlabel(activationLU(self.cam3D))
            self.cam3D_target = self.cam3D.detach().clone()
        if Enable_student:
            self.output_s,self.slice_valid_s,self.cam3D_s = self.VideoNets_S(self.f,input_flows)
        # self.sam_mask_prompt_decode(activationLU(self.cam3D_s),self.f,input)
        # stack = self.cam3D_s -torch.min(self.cam3D_s)
        # stack = stack /(torch.max(stack)+0.0000001)
        if Display_student:
            with torch.no_grad():
                self.cam3D = self.cam3D_s.detach().clone()
        # self. sam_mask =   F.interpolate(self. sam_mask,  size=(D, 32, 32), mode='trilinear', align_corners=False)
        # self.cam3D = self. sam_mask.to(self.device)  
        # self.cam3D = self.cam3D+stack
        # self.cam3D = self. post_processed_masks

    def CAM_to_slice_hardlabel(self,cam):
        bz, ch, D, H, W = cam.size()
        raw_masks = cam -torch.min(cam)
        raw_masks = raw_masks /(torch.max(raw_masks)+0.0000001)        
        binary_mask = (raw_masks >0.1)*1.0
        binary_mask = self. clear_boundary(binary_mask)
        # flatten_mask = binary_mask.view(bz,ch)
        count_masks = torch.sum(binary_mask, dim=(-1, -2), keepdim=True)
        slice_hard_label = (count_masks>10)*1.0
        return slice_hard_label,binary_mask

    def sam_mask_prompt_decode(self,raw_masks,features,input,multimask_output: bool = False):
        bz_i, ch_i, D_i, H_i, W_i = input.size()

        bz, ch, D, H, W = raw_masks.size()
        bz_f, ch_f, D_f, H_f, W_f = features.size()

        raw_masks = raw_masks -torch.min(raw_masks)
        raw_masks = raw_masks /(torch.max(raw_masks)+0.0000001) 
        self.mask_resample =   F.interpolate(raw_masks,  size=(D, H_i, W_i), mode='trilinear', align_corners=False)
        binary_mask =  self.mask_resample 
        # binary_mask = (self.mask_resample >0.05)*1.0

        # binary_mask =  self.mask_resample 

        # binary_mask = binary_mask.float(). to (self.device)
        # flattened_tensor = binary_mask.reshape(bz *ch* D,  256, 256)
        flattened_video= input.permute(0,2,1,3,4)
        flattened_video = flattened_video.reshape(bz * D, ch_i, H_i, W_i)

        flattened_feature = features.permute(0,2,1,3,4)
        flattened_feature = flattened_feature.reshape(bz_f * D_f, ch_f, H_f, W_f)

        flattened_mask= binary_mask.permute(0,2,1,3,4)
        flattened_mask = flattened_mask.reshape(bz * D, ch, H_i, W_i)



        output_mask = torch.zeros(bz * D, ch, 256, 256)
        post_process_mask = torch.zeros((bz * D, ch, H_i, W_i))
        with torch.no_grad():
                for i in range(ch):
                    for j in range (bz*D):
                        this_input_image=  flattened_video[j,:,:,:]

                        this_input_mask =  flattened_mask[j,i,:,:]
                        this_feature= flattened_feature[j:j+1,:,:,:]
                        # this_input_mask= torch.tensor(self.post_process_softmask(this_input_mask,this_input_image))
                        this_input_mask =(this_input_mask >0.1)*1.0
                        # this_input_mask =(this_input_mask>125)*1.0
                        post_process_mask[j,i,:,:] = this_input_mask
                        # coordinates = torch.ones(bz * D,1,2)*512.0
                        # coordinates= coordinates.cuda()
                        # labels = torch.ones(bz * D,1)
                        forground_num =  int(torch.sum(this_input_mask).item())
                        if forground_num>30:
                            foreground_indices = torch.nonzero(this_input_mask > 0.5, as_tuple=False)
                            cntral = self.extract_central_point_coordinates(this_input_mask)
                                # Extract coordinates from indices
                            foreground_coordinates = foreground_indices[:, [1, 0]]  # Swap x, y to get (y, x) format
                            mask = self.decode_mask_with_multi_coord(foreground_coordinates*1024/H_i,this_feature)
                            # mask = self.decode_mask_with_multi_coord(cntral[0]*1024/H_i,this_feature)

                            output_mask[j,i,:,:] = mask
                        


        # self.f = flattened_tensor.reshape (bz,D,new_ch,new_H, new_W).permute(0,2,1,3,4)
        self.sam_mask = output_mask.reshape (bz,D,ch,256,256).permute(0,2,1,3,4)
        self.post_processed_masks = post_process_mask.reshape (bz,D,ch,H_i,W_i).permute(0,2,1,3,4)
        # self.sam_mask = binary_mask

        pass
    def post_process_softmask(self,mask,image):
        def apply_opening(mask, kernel_size=3):
            """
            Apply opening operation to the mask.
            
            Parameters:
                mask (ndarray): Binary mask array.
                kernel_size (int): Size of the kernel for morphological opening.
            
            Returns:
                ndarray: Mask after applying morphological opening.
            """
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        image= image.permute(1,2,0)
        # mask = mask -torch.min(mask)
        # mask = mask /(torch.max(mask)+0.0000001) 
        mask= mask*4
        # mask = (mask>0.3)*1.0
        mask =mask.cpu().detach().numpy()  
        image= image.cpu().detach().numpy()  
        mask= np.clip(mask,0,1)
        mask=basic_operator .DCRF (image,mask)
        final_seg = np.argmax(mask, axis=0)
        # mask_uint8 = np.uint8(mask * 255)
        # # Ensure it's binary
        # _, mask = cv2.threshold(mask_uint8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # # Clear boundary errors if needed
        # # mask_cleaned = clear_boundary_errors(mask, boundary_size=5)

        # # Apply morphological opening if needed
        final_seg = apply_opening(final_seg.astype(np.uint8), kernel_size=3)
        return final_seg
    def decode_mask_with_multi_coord(self,foreground_coordinates,this_feature):
        N = foreground_coordinates.size(0)

# Calculate the step size
        step = N // 30

        # Sample coordinates using the step size
        if step == 0:
            sampled_coordinates= foreground_coordinates
        else:
            sampled_coordinates = foreground_coordinates[::step]
        # sampled_coordinates = foreground_coordinates 

        labels = torch.ones(1,1)
        # coordinates = cntral.view(1,1,2)*4
        # coordinates= coordinates.cuda()
        labels = labels.cuda()

        masks=[]
        N= len(sampled_coordinates)
        for i in range(len(sampled_coordinates)):
            coordinates = sampled_coordinates[i,:].view(1,1,2)
            coordinates= coordinates.cuda() 

            points = (coordinates, labels)

            # Embed prompts
            sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
            )
            # Predict masks
            low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                image_embeddings= this_feature,
                image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            this_mask = (low_res_masks[0,0,:,:]>0)*1.0
            masks.append(this_mask)
        masks = torch.stack(masks)
        sum_mask = torch.sum(masks,dim=0)
        out_mask= (sum_mask>(N*0.5))*1.0
        # out_mask = this_mask
        return out_mask
        pass
    def sample_points(self,mask, num_points=16):
    # Get mask shape
        bz, H, W = mask.shape

        # Generate coordinates for sampling
        x_coordinates = torch.linspace(0, W-1, num_points).long()
        y_coordinates = torch.linspace(0, H-1, num_points).long()

        # Generate grid of coordinates
        x_grid, y_grid = torch.meshgrid(x_coordinates, y_coordinates)

        # Flatten the grid coordinates
        coordinates = torch.stack((y_grid.flatten(), x_grid.flatten()), dim=1)

        # Get mask values at coordinates
        mask_values = mask[:, y_grid.flatten(), x_grid.flatten()]

        # Threshold mask values to determine foreground or background
        labels = (mask_values > 0.5).float()

        # Reshape coordinates and labels
        coordinates = coordinates.unsqueeze(0).repeat(bz, 1, 1)
        # coordinates = coordinates.permute(0,2,1)
        labels = labels.view(bz, num_points * num_points)

        return coordinates, labels
    def clear_boundary(self,masks):
        boundary_size =5
        masks[:,:,:,:boundary_size, :] = 0
        masks[:,:,:,-boundary_size:, :] = 0
        masks[:,:,:,:, :boundary_size] = 0
        masks[:,:,:,:, -boundary_size:] = 0
        return masks
        
    def extract_central_point_coordinates(self,masks):
        boundary_size =10
        masks[:boundary_size, :] = 0
        masks[-boundary_size:, :] = 0
        masks[:, :boundary_size] = 0
        masks[:, -boundary_size:] = 0
        foreground_indices = torch.nonzero(masks > 0.5, as_tuple=False)

# Extract coordinates from indices
        foreground_coordinates = foreground_indices[:, [1, 0]]  # Swap x, y to get (y, x) format

        # Compute centroid of foreground coordinates
        centroid = torch.mean(foreground_coordinates.float(), dim=0)
        
        # Return centroid coordinates reshaped to [bz, 1, 2]
        return centroid.view(1, 1, 2)
    def loss_of_one_scale(self,output,label,BCEtype = 1):
        out_logits = output.view(label.size(0), -1)
        bz,length = out_logits.size()

        label_mask_torch = torch.tensor(label_mask, dtype=torch.float32)
        label_mask_torch = label_mask_torch.repeat(bz, 1)
        label_mask_torch = label_mask_torch.to(self.device)
        if BCEtype == 1:
            loss = F.multilabel_soft_margin_loss(out_logits * label_mask_torch, label * label_mask_torch)
        else:
            loss = F.multilabel_soft_margin_loss (out_logits * label_mask_torch, label * label_mask_torch)
        return loss

    def optimization(self, label,Enable_student):
        # for param_group in  self.optimizer.param_groups:
        #     param_group['lr'] = lr 
        self.optimizer.zero_grad()
        self.optimizer_s.zero_grad()
        # torch.autograd.set_detect_anomaly(True)
        # self.set_requires_grad(self.VideoNets, True)

      
        self.loss = self.loss_of_one_scale(self.output,label)


        # self.lossEa.backward(retain_graph=True)
        self.loss.backward( )

        self.optimizer.step()
        self.lossDisplay = self.loss. data.mean()


        # out_logits_s = self.output_s.view(label.size(0), -1)
        if Enable_student:
            self.set_requires_grad(self.VideoNets_S,True)

            self.loss_s_v = self.loss_of_one_scale(self.output_s,label,BCEtype=2)  


            bz, ch, D, H, W = self.cam3D_s.size()

            valid_masks_repeated = self.slice_hard_label.repeat(1, 1, 1, H, W)
            predit_mask= self.cam3D_s * valid_masks_repeated
            target_mask= self.cam3D_target  * valid_masks_repeated
            # self.loss_s_pix = self.customeBCE_mask(predit_mask, self.binary_masks * target_mask)
            self.loss_s_pix = self.customeBCE_mask(predit_mask,  target_mask)

            # self.loss_s_pix = self.customeBCE_mask(self.cam3D_s_low  , self.cam3D_target   )

            self.loss_s = self.loss_s_v  + 0.002*self.loss_s_pix
            # self.set_requires_grad(self.VideoNets, False)
            self.loss_s.backward()
            self.optimizer_s.step()
            self.lossDisplay_s = self.loss_s. data.mean()

    def optimization_slicevalid(self):

        pass