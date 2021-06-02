import numpy as np;
import utils.utils as ut;
import matlab.engine;
from tqdm import tqdm;
import config as cf;

#Indexes for HM36M data to reorder to MPII order
hm36m_to_mpii = [6, 5, 4, 1, 2, 3, 0, 8, 9, 10, 16, 15, 14, 11, 12, 13];

def get_ml_eng():
    eng = matlab.engine.start_matlab();
    eng.cd("C:\\Users\\dan\\Documents\\HWU\\MSc\\Project\\monocap",nargout=0);
    eng.startup(nargout=0);
    return eng;

def hm_to_mp(hm_data):
    mp_data = [];
    for i in hm36m_to_mpii:
        mp_data.append(hm_data[i]);
    return mp_data;

def hms_to_ml(hms):
    mlhms = [];
    for hm in hms:
        mlhm = np.transpose(hm,(1,2,0));
        mlhms.append(np.expand_dims(mlhm,3));
    mlhms = matlab.single(np.concatenate(mlhms,axis=3).tolist());
    return mlhms;

def get_mc_pred(output,eng,frame,nImg):
    preds_2d = [];
    preds_3d = [];
    
    c,s = ut.calc_cent_scale(frame);
    
    center = matlab.double([list(c)],(1,2));
    scale = matlab.double([s],(1,1));
    
    for i in tqdm(range(0,nImg)):
        preds_2d.append(eng.transformMPII(output["W_final"][2*i:2*(i+1)],center,scale,matlab.double([64,64],(1,2)),1));
        preds_3d.append(output["S_final"][3*i:3*i + 3]);
        
    print("Converting estimates to Python format...");
    preds_2d = np.array(preds_2d);
    preds_3d = np.array(preds_3d);
    
    preds_2d = preds_2d.swapaxes(1, 2);
    preds_3d = preds_3d.swapaxes(1, 2);
    
    return preds_2d, preds_3d;

def get_bone_data(eng):
    path = cf.pose_dict_path;
    b_dict = eng.load(path,'B');
    b = np.array(b_dict['B']);
    new_b = np.zeros((b.shape[0],16));
    for i,p in enumerate(b):
        new_b[i] = hm_to_mp(p);
    b_dict['B'] = matlab.double(new_b.tolist());
    return b_dict;

def get_3d_points(preds_3d):
    for i,p in enumerate(preds_3d):
        preds_3d[i] = preds_3d[i] - preds_3d[i].mean(0)*np.ones((16,1));
    return preds_3d;