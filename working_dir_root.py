
# PC
# working_root = "/media/guiqiu/Installation/database/surgtoolloc2022_dataset/"
#
# Dataset_video_root =  working_root + "_release/training_data/video_clips/"
# Dataset_video_pkl_root = working_root + "_release/training_data/video_clips_pkl/"
# Dataset_label_root =  working_root + "_release/training_data/"
# config_root =   working_root + "config/"
# Output_root =   "/media/guiqiu/Installation/database/output/"
# #

# Remote

working_root = "/home/guiqiu/GQ_project/weakly_supervised/Data/"
working_root = "C:/2data/"

Dataset_video_root =  working_root + "training_data/video_clips/"
Dataset_video_pkl_root = working_root + "training_data/video_clips_pkl/"
Dataset_video_pkl_flow_root = working_root + "training_data/video_clips_pkl_flow/"
Dataset_video_pkl_cholec = working_root + "training_data/video_clips_pkl_cholec/"
Dataset_label_root =  working_root + "training_data/"
config_root =   working_root + "config/"
Output_root =   working_root+"output/"


img_size = 64
GPU_mode= True
Continue_flag = False
Visdom_flag = False
Display_flag = True
Save_flag =False
loadmodel_index = '5.pth'

Batch_size =2
Random_mask = False
Load_flow = False
learningR = 0.001
Call_gradcam = False 

class Para(object):
    def __init__(self):
        
        self.x=0