import cv2
import numpy as np
import os
from random import random, choice
# import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch
def random_verse_the_video(video,max=255):
    flag = choice([True, False])
    if flag == True:
        output_v = max - video 
    else:
        output_v = video
    return output_v


def random_augment(video):
    flag = choice([True, False])
    if flag ==False:
        return video
    ch, D, H, W= video.shape
    alpha = random() * 0.8 + 0.2  # random between 0.2 and 0.5
    angle = choice([0,90,180,270 ])  # random choose from 0, 90, -90

    # Reshape the stack of images to (180, 256, 256)
    images = video.reshape(ch * D, H, W)

    # Function to apply data augmentation
    def apply_data_augmentation(image):
        # Convert to PIL Image
        pil_image = Image.fromarray(np.uint8(image ))

        # Data augmentation: Random Saturation and Random Rotation
        data_augmentation = transforms.Compose([
            transforms.ColorJitter(saturation=alpha),
             
        ])
        augmented_image= transforms.functional.rotate (pil_image, angle)
        # Apply data augmentation
        augmented_image = data_augmentation(augmented_image)
         
        # Convert back to NumPy array
        augmented_image = np.array(augmented_image) 

        return augmented_image

    # Apply data augmentation to each image in the stack
    augmented_images = np.stack([apply_data_augmentation(image) for image in images])

    # Reshape augmented_images back to (3, 60, 256, 256)
    augmented_images = augmented_images.reshape(ch, D, H, W)

    return augmented_images



def hide_patch(video, patch_num=32, hide_prob=0.3, mean=128):
    # assume patch_num is int**2
    if patch_num == 1: return video
    flag = choice([True, False])
    
    if flag == True:
       return video
    ch, D, H, W = video.shape
    pn = int(patch_num ** (1/2))
    patch_size = int(W // pn)
    patch_offsets = [(x * patch_size, y * patch_size) for x in range(pn) for y in range(pn)]

    # if np.random.uniform() < hide_prob:
    #     for d in range(D):
    #         for (px, py) in patch_offsets:
    #             video[:, d, px:px + patch_size, py:py + patch_size] = mean
    
      
    for (px, py) in patch_offsets:
        if np.random.uniform() < hide_prob:
                video[:, :, px:px + patch_size, py:py + patch_size] = mean
    return video

