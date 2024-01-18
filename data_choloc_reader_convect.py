import os
import numpy as np
import h5py
import cv2

# Folder paths
dataset_label_root = "C:/2data/cholec80/tool_annotations/"
dataset_video_root = "C:/2data/cholec80/frames/"
output_folder = "C:/2data/cholec80/output_hdf5/"
img_size = (64, 64)  # Specify the desired size
video_buffer_len = 29
# Function to read labels from a text file
def read_labels(file_path):
    labels = np.genfromtxt(file_path, skip_header=1, usecols=(1, 2, 3, 4, 5, 6, 7), dtype=int)
    return labels

# Function to read frames from a video folder and resize
def read_frames(video_folder, img_size):
    frame_paths = [os.path.join(video_folder, frame) for frame in sorted(os.listdir(video_folder))]
    frames = [cv2.resize(cv2.imread(frame_path), img_size) for frame_path in frame_paths]
    return frames

# Function to perform "or" operation on a group of labels
def merge_labels(label_group):
    return np.max(label_group, axis=0)

# Counter for naming HDF5 files
file_counter = 0

# Initialize empty arrays
all_labels = []
all_frames = []

# Iterate through text files
for file_name in sorted(os.listdir(dataset_label_root)):
    
    label_stacks = []
    frame_stacks = []
    if file_name.endswith("-tool.txt"):
        file_path = os.path.join(dataset_label_root, file_name)
        video_name = file_name.split("-")[0]

        # Read labels
        labels = read_labels(file_path)

        # Read frames and resize
        video_folder = os.path.join(dataset_video_root, video_name)
        frames = read_frames(video_folder, img_size)

        for this_frame in frames:
            # Store in arrays
            label_stacks.append(labels)
            label_stacks.append(frames)
            # Check if buffer is not empty and has reached 30 groups
            if len(all_labels) > 0 and len(all_labels) == 30:
                # Convert lists to numpy arrays
                labels_array = np.array(all_labels)
                frames_array = np.array(all_frames)

                # Perform "or" operation to merge labels
                merged_labels = merge_labels(labels_array)

                # Save frames and labels to HDF5 file
                hdf5_file_name = f"clip_{file_counter:06d}.h5"
                hdf5_file_path = os.path.join(output_folder, hdf5_file_name)

                with h5py.File(hdf5_file_path, 'w') as file:
                    file.create_dataset('frames', data=frames_array)
                    file.create_dataset('labels', data=merged_labels)

                # Increment the file counter
                file_counter += 1

                # Clear arrays for the next batch
                label_stacks = []
                frame_stacks = []

            # # If video changes, start with an empty buffer
            # if not video_name == current_video_name:
            #     label_stacks = []
            #     frame_stacks = []

        

        # Update the current video name
        # current_video_name = video_name

# # If there are remaining groups less than 30
# if len(all_labels) > 0 and len(all_labels) == 30:
#     labels_array = np.array(all_labels)
#     frames_array = np.array(all_frames)

#     # Perform "or" operation to merge labels
#     merged_labels = merge_labels(labels_array)

#     # Save frames and labels to HDF5 file
#     hdf5_file_name = f"clip_{file_counter:06d}.h5"
#     hdf5_file_path = os.path.join(output_folder, hdf5_file_name)

#     with h5py.File(hdf5_file_path, 'w') as file:
#         file.create_dataset('frames', data=frames_array)
#         file.create_dataset('labels', data=merged_labels)

#     # Increment the file counter
#     file_counter += 1

# Example: Print the total number of HDF5 files created
print("Total HDF5 files created:", file_counter)
