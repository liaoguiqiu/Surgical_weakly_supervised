import cv2
import numpy as np
import os
import random

def random_verse_the_video(video,max=255):
    flag = random.choice([True, False])
    if flag == True:
        output_v = max - video 
    else:
        output_v = video
    return output_v

