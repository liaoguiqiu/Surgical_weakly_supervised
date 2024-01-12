
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
# working_root = "C:/2data/"

Dataset_video_root =  working_root + "training_data/video_clips/"
Dataset_video_pkl_root = working_root + "training_data/video_clips_pkl128/"
Dataset_label_root =  working_root + "training_data/"
config_root =   working_root + "config/"
Output_root =   working_root+"output/"

GPU_mode= True
Continue_flag = False
Visdom_flag = False
Display_flag = True
loadmodel_index = '3.pth'

Batch_size =20


learningR = 0.001
Call_gradcam = False 

class Para(object):
    def __init__(self):
        
        self.x=0