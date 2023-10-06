# update on 26th July
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
# from model import CE_build3  # the mmodel
import os
import shutil
# from train_display import *
# the model
# import arg_parse
import cv2
import numpy
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from model import  model_experiement
from dataset.dataset import myDataloader
Continue_flag = False
 # create the model
def is_external_drive(drive_path):
    # Check if the drive is a removable drive (usually external)
    return os.path.ismount(drive_path) and shutil.disk_usage(drive_path).total > 0

def find_external_drives():
    # List all drives on the system
    drives = [d for d in os.listdir('/') if os.path.isdir(os.path.join('/', d))]

    # Filter out external drives and exclude certain paths
    external_drives = [drive for drive in drives if is_external_drive(os.path.join('/', drive))
                       and not drive.startswith(('media', 'run', 'dev'))]

    return external_drives

# weight init
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    pass
############ for the linux to find the extenral drive
external_drives = find_external_drives()

if external_drives:
    print("External drives found:")
    for drive in external_drives:
        print(drive)
else:
    print("No external drives found.")
############ for the linux to find the extenral drive

Model = model_experiement._netPath()
dataLoader = myDataloader()

if Continue_flag == False:
    Model.apply(weights_init)


read_id = 0

epoch = 0
# transform = BaseTransform(  Resample_size,(104/256.0, 117/256.0, 123/256.0))
# transform = BaseTransform(  Resample_size,[104])  #gray scale data
iteration_num = 0
#################
#############training
while (1):

    images, labels= dataLoader.read_a_batch()

    # print(labels)

    # pass





















