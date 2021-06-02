# -*- coding: utf-8 -*-
"""
Created on Sun May  9 16:21:30 2021

@author: dan
"""

import cv2;
import matplotlib.pyplot as plt;
from mpl_toolkits.mplot3d import Axes3D;
import numpy as np;

plt.ioff()

left_limbs = [[3,4],[4,5],[13,14],[14,15]];
right_limbs = [[0,1],[1,2],[10,11],[11,12]];
body = [[2,6],[3,6],[6,7],[7,8],[8,9],[8,12],[8,13]];

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
    for p0,p1 in left_limbs:
      cv2.line(frame,pt_to_tup(ps[p0]),pt_to_tup(ps[p1]),(0,0,255),5);  
    
    #Green for right arm/leg
    for p0,p1 in right_limbs:
        cv2.line(frame,pt_to_tup(ps[p0]),pt_to_tup(ps[p1]),(0,255,0),5);
    
    #Blue for body
    for p0,p1 in body:
        cv2.line(frame,pt_to_tup(ps[p0]),pt_to_tup(ps[p1]),(255,0,0),5);
        
def get_3d_ps(ps,p0,p1):
    return [ps[p0][0],ps[p1][0]],[ps[p0][2],ps[p1][2]],[-ps[p0][1],-ps[p1][1]];
        
def get_3d_lines(ps):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x');
    ax.set_ylabel('y');
    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-2, 2)
    ax.set_zlim3d(-2, 2)
    for p in left_limbs:
        ax.plot(*get_3d_ps(ps,*p),c='r');
    for p in right_limbs:
        ax.plot(*get_3d_ps(ps,*p),c='g');
    for p in body:
        ax.plot(*get_3d_ps(ps,*p),c='b');
        
    fig.canvas.draw();
    plt.close()
    plot = np.frombuffer(fig.canvas.tostring_rgb(),dtype=np.uint8);
    plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,));
    plot = cv2.cvtColor(plot,cv2.COLOR_RGB2BGR)
    
    return plot;

def remove_outliers(ps):
    for i in range(1,len(ps) - 1):
        a = ps[i-1];
        b = ps[i];
        c = ps[i+1];
        
        m = (a+c) / 2;
        ad = abs(a - m);
        bd = abs(b - m);
        cd = abs(c - m);
        
        if ad < m * 0.1 and cd < m * 0.1 and bd > m * 0.1:
            ps[i] = m;
            
    return ps;

def remove_all_outliers(ps,js):
    #Remove outliers for specified joints (js) in x and y for all points (ps)
    for j in js:
        remove_outliers(ps[:,j,0]);
        remove_outliers(ps[:,j,1]);
    