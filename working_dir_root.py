
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
SAM_pretrain_root = working_root+"output/SAM/"
output_folder_sam_feature = working_root+ "cholec80/output_sam_features/"

train_test_list_dir = working_root + "output/train_test_list/"
train_sam_feature_dir = working_root+ "cholec80/train_sam_feature/"

img_size = 64
GPU_mode= True
Continue_flag = True
Visdom_flag = False
Display_flag = True
Save_flag =False
loadmodel_index = '3.pth'

Batch_size =1
Data_aug = False
Random_mask = False
Random_Full_mask = False
Load_feature = True
Load_flow = False
Max_lr = 0.0001
learningR = 0.0001
learningR_res = 0.0001
Call_gradcam = False 

class Para(object):
    def __init__(self):
        
        self.x=0