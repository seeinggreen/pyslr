# -*- coding: utf-8 -*-
"""
Created on Sun May  9 14:53:40 2021

@author: dan
"""

import config as cf;
from hourglass import hg;
import os;
import cv2;
import utils.utils as ut;
import numpy as np;
from monocap import mc;


if __name__ == "__main__":
    print("Loading hourglass model...")
    hg_model = hg.load_model(cf.hg_dir);
    
    #Get a list of images to process as frames
    #datapath = 'C:\\Users\\dan\\Documents\\HWU\\MSc\\Project\\testFrames\\';
    datapath = 'C:\\Users\\dan\\Documents\\HWU\\MSc\\Project\\monocap\\data\\tennis\\';
    imagelist = [x for x in os.listdir(datapath) if x.endswith(".jpg")];
    
    #Total number of image frames
    nImg = len(imagelist);
    
    #Define how to get the frame image from the iterator
    def get_frame(path):
        return cv2.imread(datapath + path);
    
    print("Getting keypoints for each frame...");
    kpss = hg.get_kps(hg_model,iter(imagelist),nImg,get_frame);
    
    print("Converting keypoints to heatmaps...");
    hms = hg.gen_heatmaps(kpss, 256, 64);
        
    print("Starting MATLAB engine...");
    eng = mc.get_ml_eng();
    
    print("Converting heatmaps to MATLAB format...")
    mlhms = mc.hms_to_ml(hms);
    
    print("Loading pose dictionary...");
    bDict = mc.get_bone_data(eng)
    
    print("Calculating pose using MATLAB code...");
    output = eng.PoseFromVideo('heatmap',mlhms,'dict',bDict);
    
    print("Extracting 2D and 3D pose estimates...");
    preds_2d, preds_3d = mc.get_mc_pred(output,eng,get_frame(imagelist[0]),nImg);
    
    print("Showing results...");
    cont = True;
    
    while cont:    
        for i,img in enumerate(imagelist):
            frame1 = get_frame(img);
            frame2 = get_frame(img);
        
            #For each point, highlight and label it on the original image
            for p in range(16):
                ut.draw_stick_figure(frame1, preds_2d[i])
                #ut.highlightPoint(frame1,[kpss[i][p][0],kpss[i][p][1]],hg.parts['mpii'][p]);
                #ut.highlightPoint(frame2,[preds_2d[i][p][0],preds_2d[i][p][1]],hg.parts['mpii'][p]);
            
            # Display the resulting frame
            cv2.imshow('frame',np.concatenate([frame1,frame2]));
            
            cv2.waitKey(100);
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cont = False;
                break;
     
    cv2.destroyAllWindows();
