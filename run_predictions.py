import os
import numpy as np
import json
from PIL import Image

def detect_red_light(I, fname):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    
    
    bb = [] # This should be a list of lists, each of length 4. See format example below. 
    
    '''
    BEGIN YOUR CODE
    '''

    # Blur the images
    s = 3
    kernel = np.ones(s) / s
    for d in range(I.shape[2]):
        I[:, :, d] = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 0, I[:, :, d])
        I[:, :, d] = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 1, I[:, :, d])

    i = 0
    while i < I.shape[0]:
        j = 0
        while j < I.shape[1]:
            # Check if pixel has red hue
            if I[i][j][0] > 55 + max(I[i][j][1], I[i][j][2]):
                # Add bounding box
                tl_row = max(0, i - 3)
                tl_col = max(0, j - 3)
                br_row = min(i + 6, I.shape[0] - 1)
                br_col = min(j + 3, I.shape[1] - 1)

                bb.append([tl_row, tl_col, br_row, br_col])
        
            j += 3
        i += 3

    # Merge boxes
    done = False
    while not done:
        done = True
        for i in range(len(bb) - 1):
            for j in range(i + 1, len(bb)):
                # Check if bounding boxes overlap
                if ((bb[j][0] <= bb[i][0] and bb[i][0] <= bb[j][2]) or (bb[i][0] <= bb[j][0] and bb[j][0] <= bb[i][2])) and \
                   ((bb[j][1] <= bb[i][1] and bb[i][1] <= bb[j][3]) or (bb[i][1] <= bb[j][1] and bb[j][1] <= bb[i][3])):

                    # Get new box and start over.
                    new_box = [min(bb[i][0], bb[j][0]), min(bb[i][1], bb[j][1]), max(bb[i][2], bb[j][2]), max(bb[i][3], bb[j][3])]
                    del bb[j], bb[i]

                    bb.append(new_box)

                    done = False
                    break
            if not done:
                break
    
    '''
    END YOUR CODE
    '''
    
    for i in range(len(bb)):
        assert len(bb[i]) == 4
    
    return bb

# set the path to the downloaded data: 
data_path = 'data/RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = 'data/hw01_preds' 
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

preds = {}
for i in range(len(file_names)):
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    I = np.asarray(I)
    
    preds[file_names[i]] = detect_red_light(I, file_names[i])

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'), 'w') as f:
    json.dump(preds, f)
