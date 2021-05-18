# -*- coding: utf-8 -*-
"""
Created on Sun May  9 14:53:40 2021

@author: dan
"""

import config as cf;
from hourglass import hg;
import os;
import cv2;
import matlab.engine;
import io;
from tqdm import tqdm;
import utils.utils as ut;
import numpy as np;


if __name__ == "__main__":
    print("Loading hourglass model...")
    hg_model = hg.load_model(cf.hg_dir);
    
    #Get a list of images to process as frames
    #datapath = 'C:\\Users\\dan\\Documents\\HWU\\MSc\\Project\\testFrames\\';
    datapath = 'C:\\Users\\dan\\Documents\\HWU\\MSc\\Project\\monocap-master\\data\\tennis\\';
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
    eng = matlab.engine.start_matlab();
    out = io.StringIO();
    err = io.StringIO();
    eng.cd("C:\\Users\\dan\\Documents\\HWU\\MSc\\Project\\monocap-master",nargout=0);
    eng.startup(nargout=0);
    
    print("Converting heatmaps to MATLAB format...")
    mlhms = [];
    for hm in hms:
        mlhm = np.transpose(hm,(1,2,0));
        mlhms.append(np.expand_dims(mlhm,3));
    mlhms = matlab.single(np.concatenate(mlhms,axis=3).tolist());
    
    print("Loading pose dictionary...");
    B = eng.getOutput();
    bDict = {'B':B};
    
    print("Calculating pose using MATLAB code...");
    output = eng.PoseFromVideo('heatmap',mlhms,'dict',bDict,stdout=out,stderr=err)
    
    print("Extracting 2D and 3D pose estimates...");
    preds_2d = [];
    preds_3d = [];
    
    c,s = ut.calc_cent_scale(get_frame(imagelist[0]));
    
    center = matlab.double([list(c)],(1,2));
    scale = matlab.double([s],(1,1));
    
    for i in range(0,nImg):
        preds_2d.append(eng.transformMPII(output["W_final"][2*i:2*i + 2],center,scale,matlab.double([len(hm[0]),len(hm[1])],(1,2)),1));
        preds_3d.append(output["S_final"][3*i:3*i + 3]);
        
    print("Converting estimates to Python format...");
    preds_2d = np.array(preds_2d);
    preds_3d = np.array(preds_3d);
    
    preds_2d = preds_2d.swapaxes(1, 2);
    preds_3d = preds_3d.swapaxes(1, 2);
    
    ############ ^^^ SORTED CODE  ^^^ ##########################
    ############ vvv CODE TO SORT vvv ##########################
"""    
    out = cv2.VideoWriter('..\\videoOut4.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 12, (256,256));
    
    for i in tqdm(range(0,nImg)):
        pred = preds_2d[i];
        img = np.array(images[i]);
        img = 255 - (img.astype('uint8') * 255);
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        #For each point, highlight and label it on the original image
        for p in range(0,len(preds_2d[i][0])):
            ut.highlightPoint(img,[pred[0][p],pred[1][p]],"label");
        
        
        # Display the resulting frame
        cv2.imshow('frame',img);
        out.write(img);
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;
         
    out.release();
    cv2.destroyAllWindows();
    
    import h5py;
    
    for i,hm in tqdm(enumerate(hms)):
        fn = imagelist[i];
        fn = fn[:-3] + 'h5';
        f = h5py.File(datapath + '\\new\\' + fn, 'w');
        ds = f.create_dataset('heatmap',data=hm);
        f.close();
"""