# -*- coding: utf-8 -*-
"""
Created on Sun May  9 14:53:40 2021

@author: dan
"""

import h5py;
import config as cf;
from hourglass import hg;
from face import face;
from blender import blender;
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
    #datapath = 'C:\\Users\\dan\\Documents\\HWU\\MSc\\Project\\testFrames\\';
    #datapath = 'C:\\Users\\dan\\Documents\\HWU\\MSc\\Project\\monocap\\data\\tennis\\';
    datapath = 'C:\\Users\\dan\\Documents\\HWU\\MSc\\Project\\BF1n.mov'
    #imagelist = [x for x in os.listdir(datapath) if x.endswith(".png")];
    
    #Total number of image frames
    nImg = 100;
    
    #Define how to get the frame image from the iterator
    def get_frame(cap):
        _,frame = cap.read();
        return frame;
    
    cap = cv2.VideoCapture(datapath);
    frame0 = ut.crop(get_frame(cap),256,320);
    cap.release();
    
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
    preds_2d, preds_3d = mc.get_mc_pred(output,eng,frame0,nImg);
    
    print("Scaling 3D points...");
    ps_3d = mc.get_3d_points(preds_3d);
    
    print("Calculating plots...");
    plts = [];
    for i in tqdm(range(nImg)):
        plts.append(ut.get_3d_lines(ps_3d[i])[16:-16,98:-78,:]);

    print("Isolating head...");
    tls = [];
    brs = [];
    for i in tqdm(range(nImg)):
        tl,br = ut.crop_head(kpss[i][9],kpss[i][8]);
        tls.append(tl);
        brs.append(br);
        
    cap = cv2.VideoCapture(datapath);
    frame = get_frame(cap);
    uc = ut.get_uncrop(frame,256,320);
    cap.set(cv2.CAP_PROP_POS_FRAMES,200);
        
    print("Idetifying facial features...");
    rects,fpss,tilts = face.id_faces((get_frame(cap) for i in range(nImg)),tls,brs,uc);
    cap.release();
        
    print("Preparing render...");
    scaled_points = blender.convert_points(ps_3d);
    blender.prepare_render(scaled_points,tilts);
    
    print("Rendering output...");
    blender.render();
        
    print("Showing results...");
    cont = True;
    cap = cv2.VideoCapture(datapath);
    
    #out = cv2.VideoWriter('..\\videoOut5.mov',cv2.VideoWriter_fourcc(*'mjpa'), 20, (768*2,768));
    
    while cont:
        cap.set(cv2.CAP_PROP_POS_FRAMES,200);
        for i in range(nImg):
            render = cv2.imread(cf.renders_path + "{:04d}.png".format(i+1));
            
            frame = get_frame(cap);
            crop = ut.crop(frame,256,320);
            frame1 = np.array(crop);
            frame2 = np.array(crop);
            frame3 = np.array(crop);
            frame4 = np.array(crop);
            plot = plts[i];
            frame8 = np.array(crop);
            frame9 = render[:256,293:549,:];
            
            ut.draw_stick_figure(frame3, preds_2d[i])        
            #For each point, highlight and label it on the original image
            for p in range(16):
                ut.highlightPoint(frame2,[kpss[i][p][0],kpss[i][p][1]],hg.parts['mpii'][p]);
                
            #Draw head box
            cv2.rectangle(frame4,tls[i],brs[i],(255,255,0));
            
            frame5 = ut.extract_head(frame,tls[i],brs[i],uc,256);
            
            frame7 = np.array(frame5);
            frame8 = np.array(frame5);
            cv2.rectangle(frame7, rects[i][0], rects[i][1], (0, 255, 0));
            for p in fpss[i]:
                cv2.circle(frame8,p,3,(255,0,0));
            
            # Display the resulting frame
            row1 = np.concatenate([frame1,frame2,frame3],axis=1);
            row2 = np.concatenate([plot,frame4,frame5],axis=1);
            row3 = np.concatenate([frame7,frame8,frame9],axis=1);
            grid = np.concatenate([row1,row2,row3]);
            
            outFrame = np.concatenate([grid,render],axis=1);
            cv2.imshow('frame',outFrame);
            #out.write(outFrame);
            
            cv2.waitKey(10);
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cont = False;
                break;
            cont = False;
    cv2.destroyAllWindows();
    cap.release();
    #out.release();
