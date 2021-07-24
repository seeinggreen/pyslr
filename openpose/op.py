# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 15:06:52 2021

@author: dan
"""

import sys;
import config as cf;
import os;
sys.path.append(cf.openpose_dir + '/build/python/openpose/Release');
os.environ['PATH']  = os.environ['PATH'] + ';' + cf.openpose_dir + '/build/x64/Release;' +  cf.openpose_dir + '/build/bin;'
import pyopenpose as pop;
from tqdm import tqdm;
import numpy as np;
import cv2;
import utils.utils as ut;


wrapper = pop.WrapperPython();
wrapper.configure({"model_folder": cf.openpose_dir + "/models/"})
wrapper.start()

def get_kps(frame):
    # Process Image
    datum = pop.Datum()
    datum.cvInputData = np.ascontiguousarray(frame);
    wrapper.emplaceAndPop(pop.VectorDatum([datum]))
    
    # Display Image
    return datum.poseKeypoints;

def get_kpss(frame_gen,total):
    kpss = [];
    for frame in tqdm(frame_gen,initial=1,total=total):
        kps = get_kps(frame);
        if kps is None:
            kpss.append(np.zeros((25,3)));
        else:
            kpss.append(np.sum(kps,axis=0));
    return np.array(kpss);

def smooth_kpss(kpss):
    kpss = kpss[:,0:8,:];
    for p in range(8):
        if kpss[0][p][2] == 0:
            kpss[0][p] = kpss[1][p];
        if kpss[-1][p][2] == 0:
            kpss[-1][p] = kpss[-2][p];
    for i in range(1,len(kpss)-1):
        for p in range(8):
            if kpss[i][p][2] == 0:
                kpss[i][p][0] = (kpss[i-1][p][0] + kpss[i+1][p][0]) / 2;
                kpss[i][p][1] = (kpss[i-1][p][1] + kpss[i+1][p][1]) / 2;
                kpss[i][p][2] = (kpss[i-1][p][2] + kpss[i+1][p][2]) / 2;
    return kpss;

def mpii_to_openpose(mpii):
    order = [9,8,13,14,15,12,11,10,6,3,4,5,2,1,0];
    return mpii[:,order,:];

def get_get_kp(cap):
    _,frame0 = cap.read();
    def frame_gen(f):
        print(f);
        cap.set(cv2.CAP_PROP_POS_FRAMES,f-5);
        for j in range(6):
            _,frame = cap.read();
            crop = ut.crop(frame,256,320);
            yield crop;
    uc = ut.get_uncrop(frame0,256,320);
    def get_kp(f):
        fg = frame_gen(f);
        raw_kpss = get_kpss(fg,6);
        kps = raw_kpss[-1];
        uc_kps = [uc(kp) for kp in kps];
        kps_with_conf = [[uc_kps[i][0],uc_kps[i][1],kps[i][2]] for i in range(15)];
        return kps_with_conf;
    return get_kp;
            
parts = ['head',
         'neck',
         'rsho',
         'relb',
         'rwri',
         'lsho',
         'lelb',
         'lwri',
         'pelv',
         'rhip',
         'rkne',
         'rank',
         'lhip',
         'lkne',
         'lank'];

def get_part_ids():
    ids = {};
    for i,p in enumerate(parts):
        ids[p] = i;
    return ids;