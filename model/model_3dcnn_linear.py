import torch
import torch.nn as nn
import numpy as np
from dataset.dataset import myDataloader, img_size
import model.base_models as block_buider
from dataset.dataset import Obj_num, Seperate_LR
# Seperate_LR = True # seperate left and right

class _VideoCNN(nn.Module):
    # output width=((W-F+2*P )/S)+1

    def __init__(self, inputC=3,base_f=8):
        super(_VideoCNN, self).__init__()
        ## depth rescaler: -1~1 -> min_deph~max_deph

        # a side branch predict with original iamge with rectangular kernel
        # 256*256 - 128*256
        # limit=1024
        self.blocks = nn.ModuleList()

        #

        self.blocks.append(block_buider.conv_keep_all(inputC, base_f))

        # 16*256  - 8*256
        # self.side_branch1.append(  conv_keep_all(base_f, base_f))
        # self.side_branch1.append(  conv_keep_all(base_f, base_f))
        self.blocks.append(block_buider.conv_keep_all(base_f, base_f))
        self.blocks.append(block_buider.conv_keep_all(base_f, base_f*2))
        base_f = base_f * 2
        # 8*256  - 4*256
        self.blocks.append(block_buider.conv_keep_all(base_f, base_f))
        self.blocks.append(block_buider.conv_keep_all(base_f, base_f*2))  # 4*256
        base_f = base_f * 2

        self.blocks.append(block_buider.conv_keep_all(base_f, base_f))
        self.blocks.append(block_buider.conv_dv_WH(base_f, base_f*2))  # 2*256
        base_f = base_f*2

        self.blocks.append(block_buider.conv_keep_all(base_f, base_f))
        self.blocks.append(block_buider.conv_keep_all(base_f, base_f * 2))  # 4*256
        base_f = base_f * 2

        self.blocks.append(block_buider.conv_dv_WH(base_f, base_f * 2))  # 4*256
        base_f = base_f * 2

        self.blocks.append(block_buider.conv_keep_all(base_f, base_f))
        self.blocks.append(block_buider.conv_keep_all(base_f, base_f * 2))  # 4*256
        base_f = base_f * 2
        self.depth = base_f
        # final equal to class
        # if Seperate_LR == True:
        #     self.blocks.append(block_buider.conv_keep_all(base_f, Obj_num * 2,final=True))  # 4*256
        # else:
        #     self.blocks.append(block_buider.conv_keep_all(base_f, Obj_num,final=True))  # 4*256
        # if Seperate_LR == True:
        #     self.classifier = nn.Linear(base_f, Obj_num * 2) # 4*256
        # else:
        #     self.classifier = nn.Linear(base_f, Obj_num )  # 4*256
        if Seperate_LR == True:
            self.classifier = nn.Conv3d(base_f, Obj_num *2, (1,1,1), (1,1,1), (0,0,0), bias=False) # 4*256
        else:
            self.classifier = nn.Conv3d(base_f, Obj_num , (1,1,1), (1,1,1), (0,0,0), bias=False)  # 4*256


    def maxpooling(self,input):

        Avg_pool = nn.AvgPool3d((1,2,2),stride=(1,2,2))
        # input = Avg_pool(input)

        input = Avg_pool (input)
        bz, ch, D, H, W = input.size()
        # activation = nn.Sigmoid()
        # activation = nn.ReLU()

        # input = Drop( input)
        # Drop = nn.Dropout(0.1)
        # input = Drop(input)
        # input = activation(input)
        # Maxpool_keepD = nn.MaxPool3d((1,H,W),stride=(1,1,1))
        # Maxpool_keepC = nn.MaxPool3d((D,1,1),stride=(1,1,1))
        Maxpool_keepD = nn.MaxPool3d((1,H,W),stride=(1,1,1))
        Maxpool_keepC = nn.MaxPool3d((D,1,1),stride=(1,1,1))
        
        slice_valid = Maxpool_keepD(input)
        final = Maxpool_keepC(slice_valid)
        #Note: how about add a number of object loss here ??
        # activation = nn.Sigmoid()
        # final = activation(final)
        # slice_valid = activation(slice_valid)

        return final, slice_valid
    def forward(self, x):
        out = x
        for j, name in enumerate(self.blocks):
            out = self.blocks[j](out)
        activation = nn.Sigmoid()
        activationLU = nn.ReLU()
        pooled, slice_valid = self.maxpooling(out)
        # pooled = pooled.view(out.size(0), -1)
        # Check the size of the final feature map
        final = self.classifier(pooled)
        cam = activationLU(self.classifier(out))
        # bz, ch, D, H, W = out.size()
        # final, slice_valid = self.maxpooling(out)
        final = activation(final)
        return final, slice_valid, cam