import torch
import torch.nn as nn

def build_3dconv_block(indepth, outdepth, k, s, p, Drop_out = False, final=False):
    if final == False:
        module = nn.Sequential(
            # nn.ReflectionPad2d((p[1],p[1],p[0],p[0])),
            # nn.Conv2d(indepth, outdepth,k, s, (0,0), bias=False),
            nn.Conv3d(indepth, outdepth, k, s, p, bias=False),

            nn.BatchNorm3d(outdepth),
            # nn.GroupNorm(4*int(outdepth/basic_feature),outdepth),

            nn.LeakyReLU(0.1, inplace=True),
            # nn.Dropout(0.1)
        )
        if Drop_out == True:
            module= torch.nn.Sequential(module,nn.Dropout(0.5))
    else:
        module = nn.Sequential(
            # nn.ReflectionPad2d((p[1],p[1],p[0],p[0])),
            # nn.Conv2d(indepth, outdepth,k, s, (0,0), bias=False),
            nn.Conv3d(indepth, outdepth, k, s, p, bias=False),
            # nn.Tanh()
            # nn.LeakyReLU(0.1, inplace=True)
        )
    return module



