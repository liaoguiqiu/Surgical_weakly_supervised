import os
import numpy as np
import h5py
import cv2
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from SAM.segment_anything import  SamPredictor, sam_model_registry
# from MobileSAM.mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from working_dir_root import learningR,learningR_res,SAM_pretrain_root,train_test_list_dir
Namelist = ['video01', 'video09', 'video12', 'video17', 'video18', 'video20', 'video24', 'video25', 'video26', 'video27', 'video28', 'video35', 'video37', 'video43', 'video48', 'video52', 'video55']

Create_sam_feature = False
GPU_mode = True
if GPU_mode ==True:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

else:
    device = torch.device("cpu")
sam_checkpoint = SAM_pretrain_root+"sam_vit_h_4b8939.pth"
sam_checkpoint = SAM_pretrain_root+"sam_vit_l_0b3195.pth"
sam_checkpoint =SAM_pretrain_root+ "sam_vit_b_01ec64.pth"
# self.inter_bz =1
model_type = "vit_h"
model_type = "vit_l"
model_type = "vit_b"

# model_type = "vit_t"
# sam_checkpoint = "./MobileSAM/weights/mobile_sam.pt"

# mobile SAM
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# self.predictor = SamPredictor(self.sam) 
Vit_encoder = sam.image_encoder
Vit_encoder.to(device)

# Folder paths
dataset_label_root = "C:/2data/cholec80/tool_annotations/"
dataset_video_root = "C:/2data/cholec80/frames/"
output_folder_hdf5 = "C:/2data/cholec80/output_hdf5/"
output_folder_pkl = "C:/2data/cholec80/output_pkl/"
output_folder_sam_feature = "C:/2data/cholec80/output_sam_features/"

img_size = (256, 256)  # Specify the desired size
video_buffer_len = 29

# Function to read labels from a text file
def read_labels(file_path):
    labels = np.genfromtxt(file_path, skip_header=1, usecols=(1, 2, 3, 4, 5, 6, 7), dtype=int)
    return labels

# Function to read frames from a video folder and resize
def read_frames(video_folder, img_size):
    frame_paths = [os.path.join(video_folder, frame) for frame in sorted(os.listdir(video_folder))]
    frames = [cv2.resize(cv2.imread(frame_path), img_size) for frame_path in frame_paths]
    return frames

# Function to perform "or" operation on a group of labels
def merge_labels(label_group):
    return np.max(label_group, axis=0)

# Counter for naming files
file_counter = 0

# Iterate through text files
training_list=[]
testing_list=[]


for file_name in sorted(os.listdir(dataset_label_root)):
    if file_name.endswith("-tool.txt"):
        file_path = os.path.join(dataset_label_root, file_name)
        video_name = file_name.split("-")[0]

        # Read labels
        labels = read_labels(file_path)

        # Read frames and resize
        video_folder = os.path.join(dataset_video_root, video_name)
        frames = read_frames(video_folder, img_size)

        all_data = []

        for this_frame, this_label in zip(frames, labels):
            # Organize as a dictionary or structure
            data_pair = {'frame': this_frame, 'label': this_label}
            all_data.append(data_pair)

            # Check if buffer is not empty and has reached the desired length
            if len(all_data) > 0 and len(all_data) == video_buffer_len:
                # Convert list of dictionaries to a dictionary of arrays
                 

                # Save frames and labels to HDF5 file
                # hdf5_file_name = f"clip_{file_counter:06d}.h5"
                # hdf5_file_path = os.path.join(output_folder_hdf5, hdf5_file_name)

                # with h5py.File(hdf5_file_path, 'w') as file:
                #     for key, value in data_dict.items():
                #         file.create_dataset(key, data=value)

                # Save frames and labels to PKL file
                pkl_file_name = f"clip_{file_counter:06d}.pkl"
                if video_name in Namelist:
                    testing_list.append(pkl_file_name)
                    print("test" + pkl_file_name)
                else:
                    training_list.append(pkl_file_name)
                    print("train" + pkl_file_name)




                # Increment the file counter
                file_counter += 1

                # Clear data for the next batch
                all_data = []

# Example: Print the total number of files created
train_file_name = "train_set_no8k.pkl"
train_file_path = os.path.join(train_test_list_dir, train_file_name)
with open(train_file_path, 'wb') as file:
                pickle.dump(training_list, file)
                print("sam Pkl file created:" + train_file_name)

test_file_name = "test_set_no8k.pkl"
test_file_path = os.path.join(train_test_list_dir, test_file_name)
with open(test_file_path, 'wb') as file:
                pickle.dump(testing_list, file)
                print("sam Pkl file created:" + test_file_name)
print("Total files created:", file_counter)
