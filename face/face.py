# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 14:26:03 2021

@author: dan
"""

import dlib;
import cv2;
import utils.utils as ut;
from tqdm import tqdm;

def get_det_pre():
    return dlib.get_frontal_face_detector(),dlib.shape_predictor("face\\shape_predictor_68_face_landmarks.dat");

def id_face(frame,detector,predictor):
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);
    faces = detector(grey);
    if len(faces) > 0:
        rect = ((faces[0].left(), faces[0].top()), (faces[0].right(), faces[0].bottom()));
        landmarks = predictor(grey,faces[0]);
        ps = [];
        for l in range(68):
            p = landmarks.part(l);
            ps.append((p.x,p.y));
        tilt = ut.get_head_tilt(landmarks);
    else:
        rect = None;
        ps = None;
        tilt = None;
        
    return rect,ps,tilt;

def id_faces(frame_gen,tls,brs,uc):
    d,p = get_det_pre();
    rects = [];
    fpss = [];
    tilts = [];
    for i,frame in enumerate(tqdm(frame_gen)):
        frame = ut.extract_head(frame,tls[i],brs[i],uc,256);
        rect,ps,tilt = id_face(frame, d, p);
        rects.append(rect);
        fpss.append(ps);
        tilts.append(tilt);
    return rects,fpss,tilts;