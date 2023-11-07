
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
from dataset.dataset import myDataloader,img_size

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
        cv2.destroyAllWindows()
        self.Model_infer.output= MODEL_infer.output
        self.Model_infer.slice_valid = MODEL_infer.slice_valid
        self.Model_infer.cam3D = MODEL_infer.cam3D
        self.dataLoader.input_videos = mydata_loader.input_videos
        self.dataLoader.labels = mydata_loader.labels
        Gray_video = self.dataLoader.input_videos[0,0,:,:,:] # RGB together

        for i in range(0,27,3):
            if i ==0:
                stack = Gray_video[i]
            else:
                stack = np.hstack((stack,Gray_video[i]))

        # Display the final image
        cv2.imshow('Stitched in put Image', stack.astype((np.uint8)))
        cv2.waitKey(1)

        # Combine the rows vertically to create the final 3x3 arrangement
        Cam3D= self.Model_infer.cam3D[0,:,:,:,:]
        ch, D, H, W = Cam3D.size()
        average_tensor = Cam3D.mean(dim=[1,2,3], keepdim=True)
        _, sorted_indices = average_tensor.sort(dim=0)
        for index in range(6):
            j=sorted_indices[13-index,0,0,0].cpu().detach().numpy()
            this_grayVideo = Cam3D[j].cpu().detach().numpy()
            for i in range(0, 27, 3):
                if i == 0:
                    stack = this_grayVideo[i]
                else:
                    stack = np.hstack((stack, this_grayVideo[i]))
            # stack =  stack - np.min(stack)
            stack = stack -np.min(stack)
            stack = stack /(np.max(stack)+0.001)*400
            # stack =  stack*254
            stack = np.clip(stack,0,254)
            # Display the final image
            cv2.imshow('Stitched Image' + str(j), stack.astype((np.uint8)))
            cv2.waitKey(1)
        # Cam3D = nn.functional.interpolate(side_out_low, size=(1, Path_length), mode='bilinear')


