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

def get_wrapper():
    wrapper = pop.WrapperPython();
    wrapper.configure({"model_folder": cf.openpose_dir + "/models/"})
    wrapper.start()
    return wrapper;

def get_kps(wrapper,frame):
    # Process Image
    datum = pop.Datum()
    datum.cvInputData = np.ascontiguousarray(frame);
    wrapper.emplaceAndPop(pop.VectorDatum([datum]))
    
    # Display Image
    return datum.poseKeypoints;

def get_kpss(wrapper,frame_gen,total):
    kpss = [];
    for frame in tqdm(frame_gen,initial=1,total=total):
        kpss.append(get_kps(wrapper,frame));
    return kpss;