
# the model
import cv2
import numpy

import os
import shutil
# from train_display import *
# the model
# import arg_parse
import cv2
import numpy as np
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from model import  model_experiement, model_infer_TC
from working_dir_root import Output_root,Save_flag,Load_flow,Test_on_cholec_seg8k,Display_images
from dataset.dataset import myDataloader,categories,category_colors
from dataset import io
import eval
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Save_flag =False
def save_img_to_folder(this_save_dir,ID,img):
    # this_save_dir = Output_root + "1out_img/" + Model_key + "/ground_circ/"
    if not os.path.exists(this_save_dir):
        os.makedirs(this_save_dir)
    cv2.imwrite(this_save_dir +
                str(ID) + ".jpg", img)

class Display(object):
    def __init__(self,GPU=False):
        self.Model_infer = model_infer_TC._Model_infer(GPU)
        self.dataLoader = myDataloader()
        self.show_num=8

    def train_display(self,MODEL_infer,mydata_loader, read_id,Output_root):
        # copy all the input videos and labels
        # cv2.destroyAllWindows()
        if type(MODEL_infer.final_output) is list:
            self.Model_infer.output= MODEL_infer.final_output[0]

            print("It's a list!")
        else:
            self.Model_infer.output= MODEL_infer.final_output

            print("It's not a list.")
        # self.Model_infer.slice_valid = MODEL_infer.slice_valid
        self.Model_infer.cam3D = MODEL_infer.cam3D
        self.Model_infer.raw_cam = MODEL_infer.raw_cam

        self.dataLoader.input_videos = mydata_loader.input_videos
        self.dataLoader.labels = mydata_loader.labels
        self.dataLoader.input_flows = mydata_loader.input_flows
        self.Model_infer.input_resample = MODEL_infer.input_resample
        self.dataLoader.all_raw_labels = mydata_loader.all_raw_labels
       

        self.Model_infer.direct_frame_output  = MODEL_infer.direct_frame_output

        if Test_on_cholec_seg8k:
            self.dataLoader.this_label_mask = mydata_loader.this_label_mask
            self.dataLoader.this_frame_label = mydata_loader.this_frame_label
            self.dataLoader.this_video_label = mydata_loader.this_video_label

            label_mask = torch.from_numpy(np.float32(self.dataLoader.this_label_mask )).to (device)
            frame_label = torch.from_numpy(np.float32(self.dataLoader.this_frame_label )).to (device)
            video_label = torch.from_numpy(np.float32(self.dataLoader.this_video_label )).to (device)


            # self.Model_infer.cam3D[0,2:7,:,:]*=0
            # label_mask[2:7,:,:]*=0
            eval.cal_all_metrics(read_id,Output_root,label_mask,frame_label,video_label, 
                                 self.Model_infer.cam3D[0],self.Model_infer.output[0,:,0,0,0].detach(),self.Model_infer.direct_frame_output[0])
            
            
            # print("iou" + str(this_iou))
            # self.Model_infer.cam3D[0] = label_mask

        if hasattr(MODEL_infer, 'sam_mask'):
            self.Model_infer. sam_mask =  MODEL_infer.sam_mask
             
        else:
            print("Parameter sam mask does not exist or is NaN")
        if Load_flow == True:
            Gray_video = self.dataLoader.input_flows[0,:,:,:] # RGB together
            Ori_D,Ori_H,Ori_W = Gray_video.shape
            step_l = int(Ori_D/self.show_num)+1
            for i in range(0,Ori_D,step_l):
                if i ==0:
                    stack = Gray_video[i]
                else:
                    stack = np.hstack((stack,Gray_video[i]))

            # Display the final image
            # cv2.imshow('Stitched in put flows', stack.astype((np.uint8)))
            # cv2.waitKey(1)


        # Gray_video = self.Model_infer.input_resample[0,2,:,:,:].cpu().detach().numpy()# RGB together
            ### OG video #################################
        Gray_video = self.dataLoader.input_videos[0,:,:,:,:] # RGB together
        ch,Ori_D,Ori_H,Ori_W = Gray_video.shape
        Gray_video = np.transpose(Gray_video,(1,2,3,0))
        step_l = int(Ori_D/self.show_num)+1
        for i in range(0,Ori_D,step_l):
            if i ==0:
                stack1 =  Gray_video[i] 
            else:
                stack1 = np.hstack((stack1,Gray_video[i]))
        # stack1 = np.array(cv2.merge((stack1, stack1, stack1)))

        # Display the final image
        # cv2.imshow('Stitched in put Image', stack1.astype((np.uint8)))
        # cv2.waitKey(1)

        if Save_flag == True:
            io.save_img_to_folder(Output_root + "image/original/" ,  read_id, stack1.astype((np.uint8)) )
        # Combine the rows vertically to create the final 3x3 arrangement
        Cam3D=self.Model_infer.raw_cam[0]
        final_mask = self.Model_infer.cam3D[0].cpu().detach().numpy()
        label_0 = self.dataLoader.labels[0]
        if len (Cam3D.shape) == 3:
            Cam3D = Cam3D.unsqueeze(1)
        ch, D, H, W = Cam3D.size()
        

        # activation = nn.Sigmoid()
        # Cam3D =  activation( Cam3D)
        # average_tensor = Cam3D.mean(dim=[1,2,3], keepdim=True)
        # _, sorted_indices = average_tensor.sort(dim=0)
        if len (self.Model_infer.output.shape) == 5:
            output_0 = self.Model_infer.output[0,:,0,0,0].cpu().detach().numpy()
        else:
            output_0 = self.Model_infer.output[0,:,0,0].cpu().detach().numpy()
        step_l = int(D/self.show_num)+1
        stitch_i =0
        stitch_im  = np.zeros((H,W))
        stitch_over = np.zeros((H,W))
        # ch, D, H_m, W_m = final_mask.shape
        # color_mask = np.zeros((D,H_m,W_m,3))
        # stack_color_mask = np.zeros((H,W))
        for j in range(len(categories)):
            # j=sorted_indices[13-index,0,0,0].cpu().detach().numpy()
            this_grayVideo = Cam3D[j].cpu().detach().numpy()
            # this_mask_channel = final_mask[j].cpu().detach().numpy()
            if (output_0[j]>0.5 or label_0[j]>0.5):
                for i in range(0, D, step_l):
                    this_image = this_grayVideo[i]
                    this_image =  cv2.resize(this_image, (Ori_H, Ori_W), interpolation = cv2.INTER_LINEAR)
                     
                    if i == 0:
                        stack = this_image
                        
                    else:
                        stack = np.hstack((stack, this_image))
                        
                stack= (stack>0)*stack
                stack = stack -np.min(stack)
                stack = stack /(np.max(stack)+0.0000001)*254 
                # stack
                # stack = (stack>20)*stack
                # stack = (stack>0.5)*128
                stack = np.clip(stack,0,254)
                stack = cv2.applyColorMap(stack.astype((np.uint8)), cv2.COLORMAP_JET)
                # stack = cv2.merge((stack, stack, stack))

                alpha= 0.5
                overlay = cv2.addWeighted(stack1.astype((np.uint8)), 1 - alpha, stack.astype((np.uint8)), alpha, 0)
                # stack =  stack - np.min(stack)
                infor_image = this_image*0
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                font_color = (255, 255, 255)  # White color

                text1 = str(j) + "S"+ "{:.2f}".format(output_0[j])  
                
                text2="G"+ str(label_0[j])
                text3 = categories[j]
                # Define the position where you want to put the text (bottom-left corner)
                text_position = (5, 20)
                # Use cv2.putText() to write the text on the image
                cv2.putText(infor_image, text1, text_position, font, font_scale, font_color, font_thickness)
                text_position = (5, 30)
                # Use cv2.putText() to write the text on the image
                cv2.putText(infor_image, text2, text_position, font, font_scale, font_color, font_thickness)
                text_position = (5, 40)
                # Use cv2.putText() to write the text on the image
                cv2.putText(infor_image, text3, text_position, font, font_scale, font_color, font_thickness)
                # stack = stack -np.min(stack)
                # stack = stack /(np.max(stack)+0.0000001)*254
                infor_image = cv2.merge((infor_image, infor_image, infor_image))

                stack = np.hstack((infor_image, stack))
                overlay = np.hstack((infor_image, overlay))
               
                # Display the final image
                # cv2.imshow( str(j) + "score"+ "{:.2f}".format(output_0[j]) + "GT"+ str(label_0[j])+categories[j], stack.astype((np.uint8)))
                # cv2.waitKey(1)
                if stitch_i ==0:
                    stitch_im = stack
                    stitch_over = overlay
                else:
                    stitch_im = np.vstack((stitch_im, stack))
                    stitch_over = np.vstack((stitch_over, overlay))

                stitch_i+=1
        
        stack_color_mask=stack_to_color_mask (final_mask,Ori_H,Ori_W,output_0,label_0,step_l)
        if stack_color_mask is not None:
            alpha= 0.5
            stack_color_mask = cv2.addWeighted(stack1.astype((np.uint8)), 1 - alpha, stack_color_mask.astype((np.uint8)), alpha, 0)
            io.save_img_to_folder(Output_root + "image/predict_color_mask/" ,  read_id, stack_color_mask.astype((np.uint8)) )

        if Test_on_cholec_seg8k:
            final_mask = label_mask.cpu().detach().numpy()
            gt_color_mask=stack_to_color_mask (final_mask,Ori_H,Ori_W,output_0,label_0,step_l)
            if gt_color_mask is not None:

                alpha= 0.5
                gt_color_mask = cv2.addWeighted(stack1.astype((np.uint8)), 1 - alpha, gt_color_mask.astype((np.uint8)), alpha, 0)
                io.save_img_to_folder(Output_root + "image/GT_color_mask/" ,  read_id, gt_color_mask.astype((np.uint8)) )
        # for j in range(len(categories)):
        #     # j=sorted_indices[13-index,0,0,0].cpu().detach().numpy()
           
        #     this_mask_channel = final_mask[j]
        #     color_mask[this_mask_channel > 0.5] = category_colors[categories[j]]
        #     if (output_0[j]>0.5 or label_0[j]>0.5):
        #         for i in range(0, D, step_l):
                    
        #             this_mask_channel_frame = color_mask[i]
        #             this_mask_channel_frame =  cv2.resize(this_mask_channel_frame, (Ori_H, Ori_W), interpolation = cv2.INTER_LINEAR)
                     
        #             if i == 0:
                      
        #                 stack_color_mask = this_mask_channel_frame
        #             else:
                         
        #                 stack_color_mask = np.hstack((stack_color_mask, this_mask_channel_frame))
        image_all = np.vstack((stitch_over,stitch_im))
        if Display_images:
            cv2.imshow( 'all', image_all.astype((np.uint8)))
            # cv2.imshow( 'overlay', stitch_over.astype((np.uint8)))

            cv2.waitKey(1)
        if Save_flag == True:

            io.save_img_to_folder(Output_root + "image/predict/" ,  read_id, stitch_over.astype((np.uint8)) )
            io.save_img_to_folder(Output_root + "image/predict_overlay/" ,  read_id, image_all.astype((np.uint8)) )




        if MODEL_infer.gradcam is not None:
            heatmap = MODEL_infer.gradcam[0,0,:,:].cpu().detach().numpy()

            # heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-5)

                # Resize the heatmap to the original image size
            # heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))

            # Apply colormap to the heatmap
            heatmap_colormap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

            # Superimpose the heatmap on the original image
            # result = cv2.addWeighted(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), 0.7, heatmap_colormap, 0.3, 0)

            # Display the result
            cv2.imshow('Grad-CAM', heatmap_colormap)
            cv2.waitKey(1)
        # Cam3D = nn.functional.interpolate(side_out_low, size=(1, Path_length), mode='bilinear')


def stack_to_color_mask (final_mask,Ori_H,Ori_W,output_0,label_0,step_l):
    ch, D, H_m, W_m = final_mask.shape
    color_mask = np.zeros((D,H_m,W_m,3))
    stack_color_mask = None
    for j in range(len(categories)):
            # j=sorted_indices[13-index,0,0,0].cpu().detach().numpy()
           
            this_mask_channel = final_mask[j]
            color_mask[this_mask_channel > 0.5] = category_colors[categories[j]]
            if (output_0[j]>0.5 or label_0[j]>0.5):
                for i in range(0, D, step_l):
                    
                    this_mask_channel_frame = color_mask[i]
                    this_mask_channel_frame =  cv2.resize(this_mask_channel_frame, (Ori_H, Ori_W), interpolation = cv2.INTER_LINEAR)
                     
                    if i == 0:
                      
                        stack_color_mask = this_mask_channel_frame
                    else:
                         
                        stack_color_mask = np.hstack((stack_color_mask, this_mask_channel_frame))
    return stack_color_mask