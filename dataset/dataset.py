#THe data set read the PKL file for Contour with sheath so the contact can be detected
import cv2
import numpy as np
import csv
import re
import os
print("Current working directory:", os.getcwd())
from time import  time
import dataset.io as io
import random
import image_operator.basic_operator as basic_operator
# import imageio
# import imageio_ffmpeg as ffmpeg
# # from decord import VideoReader
# # from decord import cpu
# import imageio
# from analy import MY_ANALYSIS
# from dataTool import generator_contour
# from dataTool import generator_contour_ivus

# from dataTool.generator_contour import  Generator_Contour,Save_Contour_pkl,Communicate
# from  dataTool.generator_contour_ivus import  Generator_Contour_sheath,Communicate,Save_Contour_pkl
from working_dir_root import Dataset_video_root, Dataset_label_root, Dataset_video_pkl_root,Output_root
Seperate_LR = False
img_size = 64
input_ch = 3 # input channel of each image/video
Display_loading_video = False
Read_from_pkl= True
Save_pkl = False
categories = [
    'bipolar dissector', #0   - 17
    'bipolar forceps', #1     -13163
    'cadiere forceps', #2     -17440
    'clip applier', #3        -576
    'force bipolar',#4         - 1698
    'grasping retractor',#5     -4413
    'monopolar curved scissors',#6     -11924
    'needle driver', #7                 -10142
    'permanent cautery hook/spatula', # 8     -866
    'prograsp forceps', #9                -2992
    'stapler', #10                      -131
    'suction irrigator', #11             -17
    'tip-up fenestrated grasper', #12       -181
    'vessel sealer' #13                  -1026
]
Obj_num = len(categories)
class myDataloader(object):
    def __init__(self, OLG=False):
        print("GPU function is : "+ str(cv2.cuda.getCudaEnabledDeviceCount()))
        self.batch_size = 4
        self.obj_num = Obj_num
        self.video_down_sample = 60  # 60 FPS
        self.video_buff_size = int(60/self.video_down_sample) * 30 # each video has 30s
        self.OLG_flag = OLG
        self.GT = True
        self.noisyflag = False
        self.Random_rotate = True
        self.Random_vertical_shift = True
        self.input_images= np.zeros((self.batch_size, 1, img_size, img_size))
        self.input_videos = np.zeros((self.batch_size,3,self.video_buff_size,img_size,img_size )) # RGB together
        # the number of the contour has been increased, and another vector has beeen added
        self.labels_LR= np.zeros((self.batch_size,2*self.obj_num))  # predifine the path number is 2 to seperate Left and right
        self.labels= np.zeros((self.batch_size, self.obj_num))  # left right merge

        self.all_read_flag =0
        self.save_id =0
        self.read_record = 0
        self.all_labels = self.load_all_lables()
        if Read_from_pkl == False:
            self.all_video_dir_list = os.listdir(Dataset_video_root)
        else:
            self.all_video_dir_list = os.listdir(Dataset_video_pkl_root)

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
        sum = np.zeros(self.obj_num)
        # Open the CSV file and read its contents
        try:
            with open(csv_file_path, 'r', newline='') as csvfile:
                csvreader = csv.reader(csvfile)

                # Read the header row (if any)
                header = next(csvreader)

                # Read the remaining rows and append them to the 'data' list
                for row in csvreader:
                    data.append(row)
                    binary_vector = np.array([1 if category in row[2] else 0 for category in categories], dtype=int)
                    sum = sum+ binary_vector
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
        # label_element = re.findall(r'[^,]+|nan', this_label)  # change to vector format instead of string
        split_string = this_label.split(',', 2)
        this_label_l = ','.join(split_string[:2])
        this_label_r =  split_string[2]
        binary_vector_l = np.array([1 if category in this_label_l else 0 for category in categories], dtype=int)
        binary_vector_r = np.array([1 if category in this_label_r else 0 for category in categories], dtype=int)

        # readd
        return binary_vector_l, binary_vector_r

    # load one video buffer (self.video_buff_size , 3, img_size, img_size),
    # and its squeesed which RGB are put together (self.video_buff_size * 3, img_size, img_size),
    def load_this_video_buffer(self,video_path ):
        cap = cv2.VideoCapture(video_path)

        # Read frames from the video clip
        frame_count = 0
        buffer_count = 0
        # Read frames from the video clip
        video_buffer = np.zeros((3,self.video_buff_size,  img_size, img_size))
        frame_number =0
        Valid_video=False
        while True:
            # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            if (frame_count % self.video_down_sample) ==0:
                # start_time = time()

                ret, frame = cap.read()
                # end_time = time()
                # print("inner time" + str(end_time - start_time))


            # Sample one frame per second (assuming original frame rate is 60 fps)

            # if (frame_count % self.video_down_sample) == 0:

                # cv2.imwrite(Output_root +
                #             str(frame_count) + ".jpg", frame)
                if ret == True:
                    H, W, _ = frame.shape
                    crop = frame[0:H, 192:1088]
                    if Display_loading_video == True:

                        cv2.imshow("crop", crop.astype((np.uint8)))
                        cv2.waitKey(1)

                    this_resize = cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_AREA)
                    reshaped = np.transpose(this_resize, (2, 0, 1))
                    video_buffer[:, buffer_count, :, :] = reshaped
                    # video_buffer[frame_count,:,:] = this_resize
                    # frames_array.append(frame)
                    # video_buffer

                    buffer_count += 1
                    if buffer_count >= self.video_buff_size:
                        buffer_count = 0
                        Valid_video =True
                        break
            else:
                ret = cap.grab()
                # counter += 1
            if not ret:
                break
            frame_count += 1
            frame_number +=1

        cap.release()
        # Squeeze the RGB channel
        squeezed = np.reshape(video_buffer, (self.video_buff_size * 3, img_size, img_size))
        if Display_loading_video == True:
            # x, y = 0, 10  # Position of the text
            # # Font settings
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # font_scale = 0.4
            # font_color = (255, 255, 255)  # White color
            # font_thickness = 1
            # cv2.putText(this_resize, this_label, (x, y), font, font_scale, font_color, font_thickness)
            cv2.imshow("First Frame R", squeezed[0, :, :].astype((np.uint8)))
            cv2.imshow("First Frame G", squeezed[1, :, :].astype((np.uint8)))
            cv2.imshow("First Frame B", squeezed[2, :, :].astype((np.uint8)))
            cv2.imshow("First Frame R1", squeezed[30, :, :].astype((np.uint8)))
            cv2.imshow("First Frame G1", squeezed[31, :, :].astype((np.uint8)))
            cv2.imshow("First Frame B1", squeezed[32, :, :].astype((np.uint8)))
            cv2.imshow("First Frame R2", squeezed[60, :, :].astype((np.uint8)))
            cv2.imshow("First Frame G2", squeezed[61, :, :].astype((np.uint8)))
            cv2.imshow("First Frame B2", squeezed[62, :, :].astype((np.uint8)))
            cv2.waitKey(1)
        return video_buffer, squeezed,Valid_video
    def read_a_batch(self):
        if Read_from_pkl == False:
            folder_path = Dataset_video_root
            file_name_extention = ".mp4"
        else:
            folder_path = Dataset_video_pkl_root
            file_name_extention = ".pkl"

        for i in range(self.batch_size): # load a batch of images
            start_time = time()

            index = self.read_record
            filename = self.all_video_dir_list[index]
            print(filename)

            if filename.endswith(file_name_extention):
                # Extract clip ID from the filename
                clip_id = int(filename.split("_")[1].split(".")[0])
                clip_name = filename.split('.')[0]
                # clip_name = 'clip_001714'
                # filename =  'clip_001714.mp4'
                # label_Index = labels.index("clip_"+str(clip_id))
                # Check if the clip ID is within the range you want to read
                # if clip_id <= num_clips_to_read:
                # Construct the full path to the video clip
                video_path = os.path.join(folder_path, filename)
                if Read_from_pkl == False:
                    self.video_buff, self.video_buff_s, Valid_video_flag = self.load_this_video_buffer(video_path)

                    if Save_pkl == True and Valid_video_flag == True:
                        this_video_buff = self.video_buff.astype((np.uint8))
                        io.save_a_pkl(Dataset_video_pkl_root, clip_name, this_video_buff)
                else:
                    this_video_buff = io.read_a_pkl(Dataset_video_pkl_root, clip_name)
                    self.video_buff = this_video_buff
                    Valid_video_flag = True
                # clip_name= 'test'

                if clip_name in self.all_labels and Valid_video_flag==True:
                    this_label = self.all_labels[clip_name]
                    print(this_label)

                    binary_vector = np.array([1 if category in this_label else 0 for category in categories], dtype=int)
                    # seperate the binary vector as left and right channel, so that when the image is fliped, two vector will exchange
                    binary_vector_l, binary_vector_r = self.convert_left_right_v(this_label)
                    # load the squess and unsquess

                    if Display_loading_video == True:
                        cv2.imshow("SS First Frame R", this_video_buff[0,15, :, :].astype((np.uint8)))
                        cv2.imshow("SS First Frame G", this_video_buff[1,15, :, :].astype((np.uint8)))
                        cv2.imshow("SS First Frame B", this_video_buff[2, 15,:, :].astype((np.uint8)))
                        cv2.waitKey(1)

                    # fill the batch
                    # if Valid_video_flag == True:
                    self.video_buff = basic_operator.random_verse_the_video(self.video_buff)
                    flip_flag = random.choice([True, False])

                    # flip_flag = True
                    if flip_flag == False:
                        self.input_videos[i,:, :, :, :] = self.video_buff
                        self.labels[i, :] = binary_vector
                        self.labels_LR[i, :] = np.concatenate([binary_vector_l, binary_vector_r])
                    # self.labels_LR[i, 1, :] = binary_vector_r
                    else:
                        self.input_videos[i, :, :, :, :] = np.flip(self.video_buff, axis=3)
                        self.labels[i, :] = binary_vector
                        self.labels_LR[i, :] = np.concatenate([binary_vector_r, binary_vector_l])


                else:
                    print("Key does not exist in the dictionary.")

            end_time = time()


            print(self.read_record)
            # print("time is :" + str(end_time - start_time))
            self.read_record +=1
            if self.read_record>= self.video_num:
                print("all videos have been readed")
                self.all_read_flag = 1
                self.read_record =0

            pass

        # return self.input_image,self.input_path# if out this folder boundary, just returen
        this_pointer = 0
        # i = self.read_record
        # this_folder_list = self.folder_list[self.folder_pointer]
        # # read_end  = self.read_record+ self.batch_size
        # this_signal = self.signal[self.folder_pointer]
        if Seperate_LR == False:
            return self.input_videos, self.labels
        else:
            return self.input_videos, self.labels_LR
