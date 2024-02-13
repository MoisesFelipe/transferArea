# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:47:32 2023

@author: r334256
"""

# Importing all necessary libraries
import cv2
import os
import numpy as np


path = './drone_events/smaller_sensor/'
filename = 'val0002-0095_50.25._2023_09_12_12_34_20_'
ext = '.avi'

# Read the video from specified path
cam = cv2.VideoCapture(path+filename+ext)
fps = cam.get(cv2.CAP_PROP_FPS)

try:
      
    # creating a folder named data
    if not os.path.exists(path+filename+'_frames'):
        os.makedirs(path+filename+'_frames')
  
# if not created then raise error
except OSError:
    print ('Error: Creating directory of data')
  
# frame
currentframe = 0
frames_timestamps = []

while(True):
      
    # reading from frame
    ret,frame = cam.read()
  
    if ret:
        # if video is still left continue creating images
        name = './' + filename + '/frame' + str(currentframe) + '.png'
        name = path+filename+'_frames' + '/frame' + str(currentframe) + '.png'
        print ('Creating...' + name)
  
        # writing the extracted images
        cv2.imwrite(name, frame)
        
        frames_timestamps.append(currentframe/fps) # in seconds
        
        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
        
    else:
        break
  
# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()


frames_timestamps = np.array(frames_timestamps)
np.save(path+filename+'_frames' + '/frames_timestamps_' + filename, frames_timestamps)


