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
#import os;
import cv2;
import utils.utils as ut;
import numpy as np;
from monocap import mc;
import utils.manual_tag as mt;
from tqdm import tqdm;
from openpose import op;

if __name__ == "__main__":
    print("Loading hourglass model...")
    #hg_model = hg.load_model(cf.hg_dir);
    
    #Get a list of images to process as frames
    #datapath = 'C:\\Users\\dan\\Documents\\HWU\\MSc\\Project\\testFrames\\';
    #datapath = 'C:\\Users\\dan\\Documents\\HWU\\MSc\\Project\\monocap\\data\\tennis\\';
    #datapath = 'C:\\Users\\dan\\Documents\\HWU\\MSc\\Project\\BL28n.mov';
    #datapath = 'C:\\Users\\dan\\Documents\\HWU\\MSc\\Project\\BF2l.mov';
    datapath = 'C:\\Users\\dan\\Documents\\HWU\\MSc\\Project\\BF1n.mov';
    #imagelist = [x for x in os.listdir(datapath) if x.endswith(".png")];
    
    #Total number of image frames
    nImg = 100;
    h_cent = 320;
    start_frame = 200;
    
    #Define how to get the frame image from the iterator
    def get_frame(cap):
        _,frame = cap.read();
        return frame;
    
    cap = cv2.VideoCapture(datapath);
    frame_uc = get_frame(cap);
    frame_c = ut.crop(get_frame(cap),256,h_cent);
    cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame);
    
    print("Getting keypoints for each frame...");
    #kpss = hg.
    #for i in range(nImg):
    #    for j in range(16):
    #        if kpss[i][j][2] < 0.15:
    #            kpss[i][j][0] = -1;
    #            kpss[i][j][1] = -1;
    
    cap.release();
    
    """kpss = np.zeros((100,16,3));
    
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
    output = eng.PoseFromVideo('heatmap',mlhms,'dict',bDict,'th_vis',0.1);
    
    print("Extracting 2D and 3D pose estimates...");
    preds_2d, preds_3d = mc.get_mc_pred(output,eng,frame_c,nImg);
    
    print("Scaling 3D points...");
    ps_3d = mc.get_3d_points(preds_3d);
    
    print("Calculating plots...");
    plts = [];
    for i in tqdm(range(nImg)):
        plts.append(ut.get_3d_lines(ps_3d[i])[16:-16,98:-78,:]);"""
        
    wrapper = op.get_wrapper();
    kpss = op.get_kpss(wrapper,(ut.crop(get_frame(cap),256,h_cent) for i in range(nImg)),nImg)
        
    f = h5py.File('man_tags2.h5','r');
    coords = np.array(f['coords']);
    f.close();
    
    coords[0] = ut.crop_kps(frame_uc, coords[0], 256, h_cent);
    coords[1] = ut.crop_kps(frame_uc, coords[1], 256, h_cent);

    print("Isolating head...");
    tls = [];
    brs = [];
    efs = [];
    for i in tqdm(range(nImg)):
        #tl,br = ut.crop_head(kpss[i][9],kpss[i][8],256);
        tl,br = ut.crop_head(coords[1][i],coords[0][i],256);
        tls.append(tl);
        brs.append(br);
        efs.append(face.est_face_box(coords[1][i],coords[0][i]));
    
    sefs = face.scale_face_boxes(efs, tls, brs, 256);
        
    cap = cv2.VideoCapture(datapath);
    frame = get_frame(cap);
    uc = ut.get_uncrop(frame_uc,256,h_cent);
    cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame);
    extra_data = {};
    
    """conf_ps = [[kps[8],kps[9],i] for i,kps in enumerate(kpss) if kps[8][2] > 0.1 and kps[8][2] > 0.1]
    
    tls = [];
    brs = [];
    for i in tqdm([p[2] for p in conf_ps]):
        tl,br = ut.crop_head(kpss[i][9],kpss[i][8],256);
        tls.append(tl);
        brs.append(br);
    
    fs = [];
    for i in range(len(conf_ps)):
        cap.set(cv2.CAP_PROP_POS_FRAMES,conf_ps[i][2]);
        fs.append(get_frame(cap));
    fbs,flms,has = face.id_faces(iter(fs),tls,brs,uc);
    
    extra_data['face_box'] = [];
    extra_data['face_landmarks'] = [];
    extra_data['head_angles'] = [];
    extra_data['frame_num'] = [];
    for i in range(len(conf_ps)):
        if fbs[i] is not None and flms[i] is not None and has[i] is not None:
            extra_data['face_box'].append(fbs[i]);
            extra_data['face_landmarks'].append(flms[i]);
            extra_data['head_angles'].append(has[i]);
            extra_data['frame_num'].append(conf_ps[i][2]);
    extra_data['scaled_points'] = extra_data['frame_num'];
    
    blender.prepare_render(extra_data,face_only=True,all_frames=False,end_frame=nImg);"""
    
    print("Idetifying facial features...");
    extra_data['face_box'], extra_data['face_landmarks'], extra_data['head_angles'] = face.id_faces((get_frame(cap) for i in range(nImg)),tls,brs,uc,sefs);
    cap.release();
        
    print("Preparing render...");
    #extra_data['scaled_points'] = blender.convert_points(ps_3d);
    extra_data['scaled_points'] = extra_data['face_box'];
    blender.prepare_render(extra_data,face_only=True,end_frame=nImg);
    
    print("Rendering output...");
    blender.render();
        
    #safe_frames= [];
    #for i in range(nImg):
    #    safe_frames.append(None);
        
    #for i,f in enumerate(extra_data['frame_num']):
    #    safe_frames[f] = i;
    
    print("Showing results...");
    cont = True;
    cap = cv2.VideoCapture(datapath);
    
    #out = cv2.VideoWriter('..\\videoOut5.mov',cv2.VideoWriter_fourcc(*'mjpa'), 20, (768*2,768));
    
    while cont:
        cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame);
        for i in range(nImg):
            render = cv2.imread(cf.renders_path + "{:04d}.png".format(i+1));
            
            frame = get_frame(cap);
            crop = ut.crop(frame,256,h_cent);
            frame1 = np.array(crop);
            frame2 = np.array(crop);
            frame3 = np.array(crop);
            frame4 = np.array(crop);
            plot = np.array(crop);#plts[i];
            frame9 = cv2.resize(render[20:180,330:490,:],(256,256));
            
            #ut.draw_stick_figure(frame3, preds_2d[i])   
            #For each point, highlight and label it on the original image
            #for p in range(0,2):
                #ut.highlightPoint(frame2,[kpss[i][p][0],kpss[i][p][1]],hg.parts['mpii'][p]);
                
            ut.highlightPoint(frame2,coords[0][i],hg.parts['mpii'][8]);
            ut.highlightPoint(frame2,coords[1][i],hg.parts['mpii'][9]);
                
            #Draw head box
            cv2.rectangle(frame4,tls[i],brs[i],(255,255,0));
            frame5 = ut.extract_head(frame,tls[i],brs[i],uc,256);
            
            frame7 = np.array(frame5);
            frame8 = np.array(frame5);
            
            cv2.rectangle(frame4, efs[i][0], efs[i][1], (0, 0, 0));
            
            if extra_data['face_box'][i] is not None:
                cv2.rectangle(frame7, extra_data['face_box'][i][0], extra_data['face_box'][i][1], (0, 255, 0));
                for p in extra_data['face_landmarks'][i]:
                    cv2.circle(frame8,p,3,(255,0,0));
            else:
                cv2.rectangle(frame7, *sefs[i], (0, 0, 255));
                for p in extra_data['face_landmarks'][i]:
                    cv2.circle(frame8,p,3,(255,0,200));

        
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
            #cont = False;
    cv2.destroyAllWindows();
    cap.release();
    #out.release();
