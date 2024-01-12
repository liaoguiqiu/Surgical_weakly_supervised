# update on 26th July
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
# from model import CE_build3  # the mmodel
from time import time
import os
print("Current working directory:", os.getcwd())
import shutil
# from train_display import *
# the model
# import arg_parse
import cv2
import numpy as np
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from model import  model_experiement, model_infer, model_infer2d
from working_dir_root import Output_root
from dataset.dataset import myDataloader
from display import Display
import torch.nn.parallel
import torch.distributed as dist
from dataset import io
Output_root = Output_root + "2dcnn/"
io.self_check_path_create(Output_root)
GPU_mode= True
Continue_flag = True
Visdom_flag = False
Display_flag = True
loadmodel_index = '3.pth'

if torch.cuda.is_available():
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
   
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.is_available())
    num_gpus = torch.cuda.device_count()
    print("Number of GPUs available:", num_gpus)
if GPU_mode ==True:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

else:
    device = torch.device("cpu")

# dataroot = "../dataset/CostMatrix/"
torch.set_num_threads(2)
 # create the model

if Visdom_flag == True:
    from visual import VisdomLinePlotter

    plotter = VisdomLinePlotter(env_name='path finding training Plots')

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

Model_infer = model_infer2d ._Model_infer(GPU_mode,num_gpus)
# if GPU_mode == True:
#     if num_gpus > 1:
#         Model_infer.VideoNets = torch.nn.DataParallel(Model_infer.VideoNets)
#     Model_infer.VideoNets.to(device)

# Model.cuda()
dataLoader = myDataloader()

if Continue_flag == False:
    Model_infer.VideoNets.apply(weights_init)
else:
    pretrained_dict = torch.load(Output_root + 'outNets' + loadmodel_index)
    model_dict = Model_infer.VideoNets.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict_trim = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict_trim)
    # 3. load the new state dict
    Model_infer.VideoNets.load_state_dict(model_dict)


    pretrained_dict = torch.load(Output_root + 'outResNets' + loadmodel_index)
    model_dict = Model_infer.resnet.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict_trim = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict_trim)
    # 3. load the new state dict
    Model_infer.resnet.load_state_dict(model_dict)

read_id = 0

epoch = 0
# transform = BaseTransform(  Resample_size,(104/256.0, 117/256.0, 123/256.0))
# transform = BaseTransform(  Resample_size,[104])  #gray scale data
iteration_num = 0
#################
#############training
saver_id =0
displayer = Display(GPU_mode)
epoch =0
while (1):
    start_time = time()
    input_videos, labels= dataLoader.read_a_batch()
    input_videos_GPU = torch.from_numpy(np.float32(input_videos))
    labels_GPU = torch.from_numpy(np.float32(labels))
    input_videos_GPU = input_videos_GPU.to (device)
    labels_GPU = labels_GPU.to (device)
    Model_infer.forward((input_videos_GPU-128.0)/60.0)
    Model_infer.optimization(labels_GPU)
    if Display_flag == True:
        displayer.train_display(Model_infer,dataLoader,read_id)
        pass

    if dataLoader.all_read_flag ==1:
        #remove this for none converting mode
        epoch +=1

        print("finished epoch" + str (epoch) )
        dataLoader.all_read_flag = 0
        read_id=0

        # break
    if read_id % 1== 0   :
        print(" epoch" + str (epoch) )
    if read_id % 100== 0 and Visdom_flag == True  :
        
        plotter.plot('l0', 'l0', 'l0', read_id, Model_infer.lossDisplay.cpu().detach().numpy())
    if (read_id % 1000) == 0  :
        torch.save(Model_infer.VideoNets.state_dict(), Output_root + "outNets" + str(saver_id) + ".pth")
        torch.save(Model_infer.resnet.state_dict(), Output_root + "outResNets" + str(saver_id) + ".pth")
        
        saver_id +=1
        if saver_id >5:
            saver_id =0

        end_time = time()

        print("time is :" + str(end_time - start_time))

    read_id+=1
    # print(labels)

    # pass





















