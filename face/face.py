# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 14:26:03 2021

@author: dan
"""

import dlib;
import cv2;
import utils.utils as ut;
from tqdm import tqdm;
import numpy as np;

def get_det_pre():
    return dlib.get_frontal_face_detector(),dlib.shape_predictor("face\\shape_predictor_68_face_landmarks.dat");

def get_head_tilts(landmarks):
    right_eye = landmarks.part(36);
    left_eye = landmarks.part(45);
    centre = landmarks.part(27);
    left_temp = landmarks.part(16);
    right_temp = landmarks.part(0);
    
    #Basic trig for the angle between horizon and line between eyes
    opp = right_eye.y - left_eye.y;
    adj = right_eye.x - left_eye.x;
    angZ = -np.arctan(opp/adj);
    
    #Relative distance between eye and centre of nose is a proxy for Y tilt
    right_dis = centre.x - right_eye.x;
    left_dis = left_eye.x - centre.x;
    
    rd = right_dis / (right_dis + left_dis);
    ld = left_dis / (right_dis + left_dis);
    
    angY = 1.57 * (rd - ld);
    
    #Average y-value of temples should give baseline for comparison with top of nose for X tilt
    base = (left_temp.y + right_temp.y) / 2;
    opp = centre.y - base;
    adj = right_temp.x - centre.x;
    angX = -np.arctan(opp/adj);
    
    return [angX,angY, angZ];

def est_face_box(head,neck):
    hn_dis = int(neck[1] - head[1]);
    dim = int(hn_dis * (2/3));
    half_dim = int(dim / 2);
    top = int(head[1]);
    bottom = int(head[1] + dim);
    centre = int((head[0] + neck[0]) / 2);
    left = centre - half_dim;
    right = centre + half_dim;
    
    return (left,top),(right,bottom); 

def scale_face_boxes(efs,tls,brs,size):
    sefs = [];
    for i,ef in enumerate(efs):
        left = tls[i][0];
        top = tls[i][1];
        right = brs[i][0];
        bottom = brs[i][1];
        
        w = right - left;
        h = bottom - top;
        
        h_scale = size / w;
        v_scale = size / h;
        
        new_left = int((ef[0][0]-left) * h_scale);
        new_top = int((ef[0][1]-top) * v_scale);
        new_right = int((ef[1][0]-left) * h_scale);
        new_bottom = int((ef[1][1]-top) * v_scale);
        
        sef = ((new_left,new_top),(new_right,new_bottom));
        sefs.append(sef);
        
    return sefs;
        

def id_face(frame,detector,predictor,ef):
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);
    faces = detector(grey);
    if len(faces) > 0:
        face = faces[0];
        rect = ((face.left(), face.top()), (face.right(), face.bottom()));
    else:
        face = dlib.rectangle(*ef[0],*ef[1]);
        rect = None;
        
    landmarks = predictor(grey,face);
    ps = [];
    for l in range(68):
        p = landmarks.part(l);
        ps.append((p.x,p.y));
    tilt = get_head_tilts(landmarks);
        
    return rect,ps,tilt;

def id_faces(frame_gen,tls,brs,uc,efs):
    d,p = get_det_pre();
    rects = [];
    fpss = [];
    tilts = [];
    for i,frame in enumerate(tqdm(frame_gen)):
        frame = ut.extract_head(frame,tls[i],brs[i],uc,256);
        rect,ps,tilt = id_face(frame, d, p, efs[i]);
        rects.append(rect);
        fpss.append(ps);
        tilts.append(tilt);
    return rects,fpss,tilts;