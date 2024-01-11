
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
from model import  model_experiement, model_infer
from working_dir_root import Output_root
from dataset.dataset import myDataloader,categories
from dataset import io

def save_img_to_folder(this_save_dir,ID,img):
    # this_save_dir = Output_root + "1out_img/" + Model_key + "/ground_circ/"
    if not os.path.exists(this_save_dir):
        os.makedirs(this_save_dir)
    cv2.imwrite(this_save_dir +
                str(ID) + ".jpg", img)

class Display(object):
    def __init__(self,GPU=False):
        self.Model_infer = model_infer._Model_infer(GPU)
        self.dataLoader = myDataloader()


    def train_display(self,MODEL_infer,mydata_loader, read_id):
        # copy all the input videos and labels
        # cv2.destroyAllWindows()
        self.Model_infer.output= MODEL_infer.output
        # self.Model_infer.slice_valid = MODEL_infer.slice_valid
        self.Model_infer.cam3D = MODEL_infer.cam3D
        self.dataLoader.input_videos = mydata_loader.input_videos
        self.dataLoader.labels = mydata_loader.labels
        Gray_video = self.dataLoader.input_videos[0,0,:,:,:] # RGB together
        Ori_D,Ori_H,Ori_W = Gray_video.shape
        step_l = int(Ori_D/6)
        for i in range(0,Ori_D-1,step_l):
            if i ==0:
                stack = Gray_video[i]
            else:
                stack = np.hstack((stack,Gray_video[i]))

        # Display the final image
        cv2.imshow('Stitched in put Image', stack.astype((np.uint8)))
        cv2.waitKey(1)
        io.save_img_to_folder(Output_root + "image/original/" ,  read_id, stack.astype((np.uint8)) )
        # Combine the rows vertically to create the final 3x3 arrangement
        Cam3D= self.Model_infer.cam3D[0]
        label_0 = self.dataLoader.labels[0]
        if len (Cam3D.shape) == 3:
            Cam3D = Cam3D.unsqueeze(1)
        ch, D, H, W = Cam3D.size()
        # average_tensor = Cam3D.mean(dim=[1,2,3], keepdim=True)
        # _, sorted_indices = average_tensor.sort(dim=0)
        if len (self.Model_infer.output.shape) == 5:
            output_0 = self.Model_infer.output[0,:,0,0,0].cpu().detach().numpy()
        else:
            output_0 = self.Model_infer.output[0,:,0,0].cpu().detach().numpy()
        step_l = int(D/6)+1
        stitch_i =0
        for j in range(13):
            # j=sorted_indices[13-index,0,0,0].cpu().detach().numpy()
            this_grayVideo = Cam3D[j].cpu().detach().numpy()
            if (output_0[j]>0.5 or label_0[j]>0.5):
                for i in range(0, D, step_l):
                    this_image = this_grayVideo[i]
                    this_image =  cv2.resize(this_image, (Ori_H, Ori_W), interpolation = cv2.INTER_LINEAR)
            
                    if i == 0:
                        stack = this_image
                    else:
                        stack = np.hstack((stack, this_image))
                # stack =  stack - np.min(stack)
                infor_image = this_image*0
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.3
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
                stack = stack -np.min(stack)
                stack = stack /(np.max(stack)+0.0000001)*254
                # stack =  stack*254
                stack = np.clip(stack,0,254)
                stack = np.hstack((infor_image, stack))
                # Display the final image
                # cv2.imshow( str(j) + "score"+ "{:.2f}".format(output_0[j]) + "GT"+ str(label_0[j])+categories[j], stack.astype((np.uint8)))
                # cv2.waitKey(1)
                if stitch_i ==0:
                    stitch_im = stack
                else:
                    stitch_im = np.vstack((stitch_im, stack))
                stitch_i+=1
        cv2.imshow( 'all', stitch_im.astype((np.uint8)))
        cv2.waitKey(1)
        io.save_img_to_folder(Output_root + "image/predict/" ,  read_id, stitch_im.astype((np.uint8)) )
        # Cam3D = nn.functional.interpolate(side_out_low, size=(1, Path_length), mode='bilinear')


