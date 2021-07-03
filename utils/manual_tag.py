# -*- coding: utf-8 -*-
"""
Created on Tue May 25 16:00:50 2021

@author: dan
"""

import cv2;
import numpy as np;
from hourglass.hg_files.img import crop;
from utils import utils as ut;

def get_click(event,x,y,flags,param):
    global click;
    if event == cv2.EVENT_LBUTTONDOWN:
        print(len(coords),":",x,y);
        coords.append([x,y]);
        click = True;
        
def manual_tag():      
    cv2.namedWindow('Window');
    cv2.setMouseCallback('Window',get_click);
    
    img = np.zeros((256,256,3));
    
    global coords;
    global click;
    coords = [];
    
    #Set click to true to load first frame
    click = True;
    
    cap = cv2.VideoCapture('..\\BL28n.mov');
    
    ret, frame = cap.read();
    c,s = ut.calc_cent_scale(frame);
    
    #cap.set(cv2.CAP_PROP_POS_FRAMES,200);
    
    while len(coords) < 500:
        if click:
            print(cap.get(cv2.CAP_PROP_POS_FRAMES));
            ret, frame = cap.read();
            #f = crop(frame,c,s,(256,256));
            #img = 255 - (f.astype('uint8') * 255);
            cv2.imshow('Window',frame);
            click = False;
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;
            
    cap.release();
    cv2.destroyAllWindows();
    
    return coords;
