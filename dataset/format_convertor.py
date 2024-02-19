import numpy as np

class_name_Cholec_8k={0: 'Black Background',
                    1: 'Abdominal Wall',
                    2: 'Liver',
                    3: 'Gastrointestinal Tract',
                    4: 'Fat',
                    5: 'Grasper',
                    6: 'Connective Tissue',
                    7: 'Blood',
                    8: 'Cystic Duct',
                    9: 'L-hook Electrocautery',
                    10: 'Gallbladder',
                    11: 'Hepatic Vein',
                    12: 'Liver Ligament'}

categories = [
        'Grasper', #0   
        'Bipolar', #1    
        'Hook', #2    
        'Scissors', #3      
        'Clipper',#4       
        'Irrigator',#5    
        'SpecimenBag',#6                  
    ]

def label_from_seg8k_2_cholec(inputlabel): #(13,29,256,256)
    in_ch,in_D,H,W =  inputlabel.shape
    inputlabel=np.transpose(inputlabel , (1, 0, 2, 3)) 
    lenth = len(categories)
    new_label = np.zeros((in_D,lenth,H,W))
    new_label[:,0,:,:] = inputlabel[:,5,:,:]
    new_label[:,2,:,:] = inputlabel[:,9,:,:]
    frame_label=np.sum(new_label,axis=(2,3))
    frame_label=(frame_label>20)*1.0
    video_label=np.max(frame_label, axis=0)
    mask = new_label
    return mask,frame_label,video_label
     




