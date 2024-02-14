
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

# path='C:/2data/cholecseg8k_working/cholecseg8k/'
# op_path='C:/2data/cholecseg8k_working/'
Dataset_video_pkl_cholec8k =  "C:/2data/cholecseg8k_working/output_pkl/"
output_folder_sam_feature_cholec8k =  "C:/2data/cholecseg8k_working/output_sam_features/"

train_test_list_dir = working_root + "output/train_test_list/"
train_sam_feature_dir = working_root+ "cholec80/train_sam_feature/"
sam_feature_OLG_dir= working_root+ "cholec80/sam_feature_OLG/"

Evaluation = True
img_size = 64
GPU_mode= True

Continue_flag = True
Test_on_cholec_seg8k= True

Visdom_flag = True
if Evaluation == True:
    Continue_flag = True
    Visdom_flag= False
Display_flag = True
Save_flag =False
loadmodel_index = '3.pth'

Batch_size =1
Data_aug = False
Random_mask = False
Random_Full_mask = False
Load_feature = True
Save_feature_OLG = True

if Load_feature == True:
   Save_feature_OLG= False
if Save_feature_OLG == True:
    Batch_size=1

Load_flow = False
Weight_decay = 0.0001
Max_lr = 0.0001
learningR = 0.0001
learningR_res = 0.001
Call_gradcam = False 

class Para(object):
    def __init__(self):
        
        self.x=0