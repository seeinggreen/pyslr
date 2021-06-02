# -*- coding: utf-8 -*-
"""
Created on Sun May  9 14:53:40 2021

@author: dan
"""

import h5py;
import config as cf;
from hourglass import hg;
import os;
import cv2;
import utils.utils as ut;
import numpy as np;
from monocap import mc;
import utils.manual_tag as mt;
from tqdm import tqdm;


if __name__ == "__main__":
    print("Loading hourglass model...")
    #hg_model = hg.load_model(cf.hg_dir);
    
    #Get a list of images to process as frames
    datapath = 'C:\\Users\\dan\\Documents\\HWU\\MSc\\Project\\testFrames\\';
    #datapath = 'C:\\Users\\dan\\Documents\\HWU\\MSc\\Project\\monocap\\data\\tennis\\';
    imagelist = [x for x in os.listdir(datapath) if x.endswith(".png")];
    
    #Total number of image frames
    nImg = len(imagelist);
    
    #Define how to get the frame image from the iterator
    def get_frame(path):
        return cv2.imread(datapath + path);
    
    print("Getting keypoints for each frame...");
    #kpss = hg.get_kps(hg_model,iter(imagelist),nImg,get_frame);
    kpss = np.zeros((100,16,3));
    
    kpss = np.array(kpss);
    for i,p in enumerate(kpss):
        kpss[i][7] = [107,135,0.5];
        kpss[i][6] = [108,188,0.5];
        kpss[i][2] = [60,208,0.5];
        kpss[i][3] = [152,194,0.5];
        kpss[i][4] = [101,201,0.5];
        kpss[i][1] = [130,243,0.5];
        kpss[i][5] = [88,255,0.5];
        kpss[i][0] = [129,255,0.5];
        #ut.remove_all_outliers(kpss,[10,11,12,13,14,15]);
    f = h5py.File('man_tags.h5','r');
    coords = np.array(f['coords']);
    f.close();
        
    kpss[:,8:16,:] = coords;
        
        
    print("Converting keypoints to heatmaps...");
    hms = hg.gen_heatmaps(kpss, 256, 64);
        
    print("Starting MATLAB engine...");
    eng = mc.get_ml_eng();
    
    print("Converting heatmaps to MATLAB format...")
    mlhms = mc.hms_to_ml(hms);
    
    print("Loading pose dictionary...");
    bDict = mc.get_bone_data(eng)
    
    print("Calculating pose using MATLAB code...");
    #th_vis sets the threshold to ignore points under a specified confidence level
    output = eng.PoseFromVideo('heatmap',mlhms,'dict',bDict,'th_vis',0);
    
    print("Extracting 2D and 3D pose estimates...");
    preds_2d, preds_3d = mc.get_mc_pred(output,eng,get_frame(imagelist[0]),nImg);
    
    print("Scaling 3D points...");
    ps_3d = mc.get_3d_points(preds_3d);
    
    print("Calculating plots...");
    plts = [];
    for i in tqdm(range(100)):
        plts.append(ut.get_3d_lines(ps_3d[i])[:,98:-78,:])
    
    print("Showing results...");
    cont = True;
    
    while cont:    
        for i,img in enumerate(imagelist):
            frame1 = get_frame(img);
            frame2 = get_frame(img);
            plot = plts[i];
            
            ut.draw_stick_figure(frame1, preds_2d[i])        
            #For each point, highlight and label it on the original image
            for p in range(16):
                ut.highlightPoint(frame2,[kpss[i][p][0],kpss[i][p][1]],hg.parts['mpii'][p]);
            
            # Display the resulting frame
            cv2.imshow('frame',np.concatenate([frame1,frame2,plot]));
            
            cv2.waitKey(100);
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cont = False;
                break;
     
    cv2.destroyAllWindows();
