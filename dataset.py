#THe data set read the PKL file for Contour with sheath so the contact can be detected
import cv2
import numpy as np
import os
# from analy import MY_ANALYSIS
# from dataTool import generator_contour
# from dataTool import generator_contour_ivus

# from dataTool.generator_contour import  Generator_Contour,Save_Contour_pkl,Communicate
# from  dataTool.generator_contour_ivus import  Generator_Contour_sheath,Communicate,Save_Contour_pkl
from working_dir_root import Dataset_root

class myDataloader(object):
    def __init__(self, batch_size,image_size,path_size,validation= False,OLG=False):
        self.OLG_flag = OLG
        self.GT = True
        self.noisyflag = False
        self.Random_rotate = True
        self.Random_vertical_shift = True
        self.save_id =0
        #Guiqiu modified for my computer
        # self.com_dir =  Generator_Contour_sheath().com_dir # this dir is for the OLG
        # if self.OLG_flag == True:
        #      # initial lizt the
        #     self.talker = Communicate()

    def read_a_batch(self):
        read_start = self.read_record

        # return self.input_image,self.input_path# if out this folder boundary, just returen
        this_pointer = 0
        i = read_start
        this_folder_list = self.folder_list[self.folder_pointer]
        # read_end  = self.read_record+ self.batch_size
        this_signal = self.signal[self.folder_pointer]