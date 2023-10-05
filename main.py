# update on 26th July
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
# from model import CE_build3  # the mmodel

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

# weight init
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    pass
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





















