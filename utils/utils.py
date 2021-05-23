# -*- coding: utf-8 -*-
"""
Created on Sun May  9 16:21:30 2021

@author: dan
"""

import cv2;

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

def pt_to_tup(pt):
    return (int(pt[0]),int(pt[1]));

def draw_stick_figure(frame,ps):
    #Red for left arm/leg
    cv2.line(frame,pt_to_tup(ps[3]),pt_to_tup(ps[4]),(0,0,255),5);
    cv2.line(frame,pt_to_tup(ps[4]),pt_to_tup(ps[5]),(0,0,255),5);
    cv2.line(frame,pt_to_tup(ps[13]),pt_to_tup(ps[14]),(0,0,255),5);
    cv2.line(frame,pt_to_tup(ps[14]),pt_to_tup(ps[15]),(0,0,255),5);
    
    #Green for right arm/leg
    cv2.line(frame,pt_to_tup(ps[0]),pt_to_tup(ps[1]),(0,255,0),5);
    cv2.line(frame,pt_to_tup(ps[1]),pt_to_tup(ps[2]),(0,255,0),5);
    cv2.line(frame,pt_to_tup(ps[10]),pt_to_tup(ps[11]),(0,255,0),5);
    cv2.line(frame,pt_to_tup(ps[11]),pt_to_tup(ps[12]),(0,255,0),5);
    
    #Blue for body
    cv2.line(frame,pt_to_tup(ps[2]),pt_to_tup(ps[6]),(255,0,0),5);
    cv2.line(frame,pt_to_tup(ps[3]),pt_to_tup(ps[6]),(255,0,0),5);
    cv2.line(frame,pt_to_tup(ps[6]),pt_to_tup(ps[7]),(255,0,0),5);
    cv2.line(frame,pt_to_tup(ps[7]),pt_to_tup(ps[8]),(255,0,0),5);
    cv2.line(frame,pt_to_tup(ps[8]),pt_to_tup(ps[9]),(255,0,0),5);
    cv2.line(frame,pt_to_tup(ps[8]),pt_to_tup(ps[12]),(255,0,0),5);
    cv2.line(frame,pt_to_tup(ps[8]),pt_to_tup(ps[13]),(255,0,0),5);