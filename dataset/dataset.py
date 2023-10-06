#THe data set read the PKL file for Contour with sheath so the contact can be detected
import cv2
import numpy as np
import csv
import re
import os
# from analy import MY_ANALYSIS
# from dataTool import generator_contour
# from dataTool import generator_contour_ivus

# from dataTool.generator_contour import  Generator_Contour,Save_Contour_pkl,Communicate
# from  dataTool.generator_contour_ivus import  Generator_Contour_sheath,Communicate,Save_Contour_pkl
from working_dir_root import Dataset_video_root, Dataset_label_root, Output_root
img_size = 300
input_ch = 3 # input channel of each image/video
Display_loading_video = True
categories = [
    'bipolar dissector',
    'bipolar forceps',
    'cadiere forceps',
    'clip applier',
    'force bipolar',
    'grasping retractor',
    'monopolar curved scissors',
    'needle driver',
    'permanent cautery hook/spatula',
    'prograsp forceps',
    'stapler',
    'suction irrigator',
    'tip-up fenestrated grasper',
    'vessel sealer'
]
class myDataloader(object):
    def __init__(self, OLG=False):
        self.batch_size = 2
        self.obj_num = 14
        self.video_down_sample = 60  # 60 FPS
        self.video_buff_size = int(60/self.video_down_sample) * 30 # each video has 30s
        self.OLG_flag = OLG
        self.GT = True
        self.noisyflag = False
        self.Random_rotate = True
        self.Random_vertical_shift = True
        self.input_images= np.zeros((self.batch_size, 1, img_size, img_size))
        # the number of the contour has been increased, and another vector has beeen added
        self.labels = np.zeros((self.batch_size, self.obj_num, 2))  # predifine the path number is 2
        self.save_id =0
        self.read_record = 0
        self.all_labels = self.load_all_lables()
        self.all_video_dir_list = os.listdir(Dataset_video_root)
        self.video_num = len (self.all_video_dir_list)
        #Guiqiu modified for my computer
        # self.com_dir =  Generator_Contour_sheath().com_dir # this dir is for the OLG
        # if self.OLG_flag == True:
        #      # initial lizt the
        #     self.talker = Communicate()
    def load_all_lables(self): # load all labels and save then as dictionary format
        csv_file_path = Dataset_label_root + "labels.csv"

        # Initialize an empty list to store the data from the CSV file
        data = []

        # Open the CSV file and read its contents
        try:
            with open(csv_file_path, 'r', newline='') as csvfile:
                csvreader = csv.reader(csvfile)

                # Read the header row (if any)
                header = next(csvreader)

                # Read the remaining rows and append them to the 'data' list
                for row in csvreader:
                    data.append(row)
        except FileNotFoundError:
            print(f"File not found at path: {csv_file_path}")
            exit()
        except Exception as e:
            print(f"An error occurred: {e}")
            exit()

        # Now you have the data from the CSV file in the 'data' list
        # You can manipulate or process the data as needed

        # Example: Printing the first few rows
        for row in data[:5]:
            print("all data is loaded and here are some samples:")
            print(row)

        labels = data
        # conver label list into dictionary that can used key for fast lookingup
        label_dict = {label_info[1]: label_info[2] for label_info in labels}  # use the full name as the dictionary key
        label_dict_number = {label_info[0]: label_info[2] for label_info in
                             labels}  # using the number and dictionary keey instead

        all_labels = label_dict
        return all_labels
    def convert_left_right_v(self,this_label):
        label_element = re.findall(r'\w+(?:\s\w+)*|nan',
                                   this_label)  # change to vector format instead of string

        # Initialize the label vector with 'nan' values
        label_vector = ['nan'] * 4  # Assuming a fixed length of 4 elements
        for i, element in enumerate(label_element):
            label_vector[i] = element
        binary_vector_l = np.zeros(len(categories), dtype=int)
        binary_vector_r = np.zeros(len(categories), dtype=int)

        # Iterate through the label vector and set corresponding binary values
        for i in range(len(label_vector)):
            this_ele = label_vector[i]
            if this_ele in categories:
                category_index = categories.index(this_ele)
                # Determine if the category is in the left or right direction
                if i < (len(label_vector) / 2):
                    binary_vector_l[category_index] = 1
                else:
                    binary_vector_r[category_index] = 1
        # readd
        return binary_vector_l, binary_vector_r
    def load_this_video_buffer(self,video_path,this_label):
        cap = cv2.VideoCapture(video_path)

        # Read frames from the video clip
        frame_count = 0
        buffer_count = 0
        # Read frames from the video clip
        video_buffer = np.zeros((self.video_buff_size,3,img_size,img_size))
        while True:
            ret, frame = cap.read()
            if not ret:
                break


            # Sample one frame per second (assuming original frame rate is 60 fps)
            if frame_count % (self.video_down_sample) == 0:
                # cv2.imwrite(Output_root +
                #             str(frame_count) + ".jpg", frame)
                H,W,_ = frame.shape
                crop = frame[0:H, 192:1088]
                this_resize = cv2.resize(crop, (  img_size, img_size), interpolation=cv2.INTER_AREA)
                reshaped= np.transpose(this_resize, (2, 0, 1))
                video_buffer[buffer_count,:,:,:] = reshaped
                # video_buffer[frame_count,:,:] = this_resize
                # frames_array.append(frame)
                # video_buffer

                buffer_count +=1
                if buffer_count>=self.video_buff_size:
                    buffer_count=0
                    break

            frame_count += 1

        cap.release()
        # Squeeze the RGB channel
        squeezed = np.reshape(video_buffer, (self.video_buff_size*3, img_size, img_size))
        if Display_loading_video == True:
            x, y = 0, 10  # Position of the text
            # Font settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_color = (255, 255, 255)  # White color
            font_thickness = 1
            # cv2.putText(this_resize, this_label, (x, y), font, font_scale, font_color, font_thickness)
            cv2.imshow("First Frame", squeezed[0, :, :].astype((np.uint8)))
            cv2.waitKey(1)
    def read_a_batch(self):
        folder_path = Dataset_video_root

        for i in range(self.batch_size): # load a batch of images
            index = self.read_record
            filename = self.all_video_dir_list[index]
            if filename.endswith(".mp4"):
                # Extract clip ID from the filename
                clip_id = int(filename.split("_")[1].split(".")[0])
                clip_name = filename.split('.')[0]
                # label_Index = labels.index("clip_"+str(clip_id))
                # Check if the clip ID is within the range you want to read
                # if clip_id <= num_clips_to_read:
                # Construct the full path to the video clip
                video_path = os.path.join(folder_path, filename)
                this_label = self.all_labels[clip_name]
                binary_vector = np.array([1 if category in this_label else 0 for category in categories], dtype=int)
                # seperate the binary vector as left and right channel, so that when the image is fliped, two vector will exchange
                binary_vector_l, binary_vector_r = self. convert_left_right_v(this_label)
                self.load_this_video_buffer(video_path,this_label)

            print(filename)
            print(this_label)
            print(self.read_record)

            self.read_record +=1
            if self.read_record>= self.video_num:
                print("all videos have been readed")
                self.read_record =0

            pass

        # return self.input_image,self.input_path# if out this folder boundary, just returen
        this_pointer = 0
        # i = self.read_record
        # this_folder_list = self.folder_list[self.folder_pointer]
        # # read_end  = self.read_record+ self.batch_size
        # this_signal = self.signal[self.folder_pointer]

        return self.input_images, self.labels