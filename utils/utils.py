# -*- coding: utf-8 -*-
"""
Created on Sun May  9 16:21:30 2021

@author: dan
"""

import cv2;
import h5py;
import numpy as np;

def calc_cent_scale(img):
    imgw = img.shape[1];
    imgh = img.shape[0];
    
    c = (int(imgw / 2), int(imgh / 2));
    s = imgh / 200;
    
    return c , s;

def highlightPoint(img,point,label):
    y = int(point[1]);
    x = int(point[0]);
    
    #Add a square around the selected pixel
    cv2.rectangle(img,(x-1,y-1),(x+1,y+1),(255,255,255));
            
    #Add a text label at the selected pixel
    cv2.putText(img,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255));
            
    return img;

def read_heatmap(fn):
    f = h5py.File(fn,'r');
    ds = f['heatmap'];
    ds = np.array(ds);
    f.close();
    ds = np.array(ds);
    #Equivilent to the reverse way ML reads data + the permute
    ds = np.transpose(ds,(1,2,0))
    return ds;

def convert_keypoints(kps,inSize,outSize):
    scale = inSize / outSize;    
    for kp in kps:
        kp[0] /= scale;
        kp[1] /= scale;
        
        kp[0] = min(kp[0],outSize);
        kp[0] = max(0,kp[0]);
        
        kp[1] = min(kp[1],outSize);
        kp[1] = max(0,kp[1]);
        
        kp[0] = int(kp[0]);
        kp[1] = int(kp[1]);
        
    return np.array([kps]);