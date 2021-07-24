# -*- coding: utf-8 -*-
"""
Created on Tue May 25 16:00:50 2021

@author: dan
"""

import cv2;
from utils import utils as ut;

def get_click(event,x,y,flags,param):
    global click;
    if event == cv2.EVENT_LBUTTONDOWN:
        print(len(coords),":",x,y);
        coords.append([x,y]);
        click = True;
        
def manual_tag(frame0,frame_gen,total):    
    """Takes a stream of frames and returns the location clicked for each frame.
    
    Takes a sample frame (for calulating scale/centre), a generator object to 
    get each frame and the total number of frames to know when to finish."""
    cv2.namedWindow('Window');
    cv2.setMouseCallback('Window',get_click);
    
    global coords;
    global click;
    coords = [];
    
    #Set click to true to load first frame
    click = True;

    c,s = ut.calc_cent_scale(frame0);
    
    while len(coords) < total:
        if click:
            frame = next(frame_gen);
            cv2.imshow('Window',frame);
            click = False;
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;
    cv2.destroyAllWindows();
    
    return coords;
