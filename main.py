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
import tqdm.tqdm;
import utils.utils as ut;
import numpy as np;


if __name__ == "__main__":
    #Load the hourglass model
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
    
    #Get the keypoints for each frame
    kpss = hg.get_kps(hg_model,iter(imagelist),nImg,get_frame);
    
    ############ ^^^ SORTED CODE  ^^^ ##########################
    ############ vvv CODE TO SORT vvv ##########################
    
    os.chdir("C:\\Users\\dan\\Documents\\HWU\\MSc\\Project\\monocap-master");
    
    eng = matlab.engine.start_matlab();
    out = io.StringIO();
    err = io.StringIO();
    
    images = [];
    hms = [];
    
    for image in tqdm(imagelist):
        img = cv2.imread(datapath + image);
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);
        images.append(img);
        hm = ut.read_heatmap(datapath + image[:-3] + "h5");
        hms.append(np.expand_dims(hm,3));
    
    mlhms = matlab.single(np.concatenate(hms,axis=3).tolist());
    
    bDict = eng.getOutput();
    
    output = eng.PoseFromVideo('heatmap',mlhms,'dict',bDict,stdout=out,stderr=err)
    
    preds_2d = [];
    preds_3d = [];
    
    center = matlab.double([int(len(images[0][0])/2),int(len(images[0][1])/2)],(1,2));
    scale = matlab.double([len(images[0][0])/200],(1,1));
    
    for i in tqdm(range(0,nImg)):
        preds_2d.append(eng.transformMPII(output["W_final"][2*i:2*i + 2],center,scale,matlab.double([len(hm[0]),len(hm[1])],(1,2)),1));
        preds_3d.append(output["S_final"][3*i:3*i + 3]);
    
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
    
    conKps = [ut.convert_keypoints(kps,256,64) for kps in kpss];
    
    from data.MPII.dp import GenerateHeatmap as GH
    
    hmg = GH(64,16);
    
    hms = [hmg(kps) for kps in conKps];
    
    for i,hm in tqdm(enumerate(hms)):
        kps = kpss[i];
        for j,m in enumerate(hm):
            p = kps[j][2];
            if j not in [9,10,11,12,13,14,15]: p = 0;
            m *= p;
    
    import h5py;
    
    for i,hm in tqdm(enumerate(hms)):
        fn = imagelist[i];
        fn = fn[:-3] + 'h5';
        f = h5py.File(datapath + '\\new\\' + fn, 'w');
        ds = f.create_dataset('heatmap',data=hm);
        f.close();