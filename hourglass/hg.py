import os;
import argparse;
import numpy as np;
import torch;
from tqdm import tqdm;
from hourglass.hg_files.test import inference;
from hourglass import hg;
from hourglass.hg_files import dp;

import utils.utils as ut;

parts = {'mpii':['rank', 'rkne', 'rhip',
                 'lhip', 'lkne', 'lank',
                 'pelv', 'thrx', 'neck', 'head',
                 'rwri', 'relb', 'rsho',
                 'lsho', 'lelb', 'lwri']}

def load_model(hg_dir):
    opt = argparse.Namespace(continue_exp='pose',exp='pose',max_iters=250);
    
    from hourglass.hg_files import pose as task;
    exp_path = 'exp\\pose';
    
    config = task.__config__
    try: os.makedirs(exp_path)
    except FileExistsError: pass
    
    config['opt'] = opt;
    config['data_provider'] = dp;
    
    func = task.make_network(config)
    
    reload(config,hg_dir);
    
    return func, config;

def get_kp(frame,do,c,s):
    pred = do(frame,c,s);
    kps = pred[0]["keypoints"];
    return kps;

def get_kps(model, frame_it, total, get_frame):
    frame0 = get_frame(next(frame_it));
    c,s = ut.calc_cent_scale(frame0);
    
    do = hg.get_do(*model);
    kpss = [get_kp(frame0,do,c,s)]
    
    for f in tqdm(frame_it,initial=1,total=total):
        frame = get_frame(f);
        kpss.append(get_kp(frame,do,c,s));
        
    return kpss;

#Taken from test.py main()
def get_do(func,config):
    def runner(imgs):
        return func(0, config, 'inference', imgs=torch.Tensor(np.float32(imgs)))['preds']
    
    def do(img, c, s):
        ans = inference(img, runner, config, c, s)
        if len(ans) > 0:
            ans = ans[:,:,:3]
    
        ## ans has shape N,16,3 (num preds, joints, x/y/visible)
        pred = []
        for i in range(ans.shape[0]):
            pred.append({'keypoints': ans[i,:,:]})
        return pred
    return do;

"""def read_heatmap(fn):
    f = h5py.File(fn,'r');
    ds = f['heatmap'];
    ds = np.array(ds);
    f.close();
    ds = np.array(ds);
    #Equivilent to the reverse way ML reads data + the permute
    ds = np.transpose(ds,(1,2,0))
    return ds;"""

def convert_keypoints(kps,inSize,outSize):
    scale = inSize / outSize;    
    for kp in kps:
        kp[0] /= scale;
        kp[1] /= scale;
        
        kp[0] = min(kp[0],outSize);
        kp[0] = max(0,kp[0]);
        
        kp[1] = min(kp[1],outSize);
        kp[1] = max(0,kp[1]);
        
        kp[0] = int(kp[0]);
        kp[1] = int(kp[1]);
        
    return np.array([kps]);

def gen_heatmaps(kpss,in_size,out_size):
    conKps = [convert_keypoints(kps,in_size,out_size) for kps in kpss];
    
    hmg = dp.GenerateHeatmap(out_size,16);
    
    hms = [hmg(kps) for kps in conKps];
    
    for i,hm in tqdm(enumerate(hms)):
        kps = kpss[i];
        for j,m in enumerate(hm):
            p = kps[j][2];
            if j not in [9,10,11,12,13,14,15]: p = 0;
            m *= p;
            
    return hms;

def reload(config,hg_dir):
    """
    load or initialize model's parameters by config from config['opt'].continue_exp
    config['train']['epoch'] records the epoch num
    config['inference']['net'] is the model
    """
    opt = config['opt']

    if opt.continue_exp:
        resume = os.path.join(hg_dir + "\\exp", opt.continue_exp)
        resume_file = os.path.join(resume, 'checkpoint.pt')
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume_file)

            config['inference']['net'].load_state_dict(checkpoint['state_dict'])
            config['train']['optimizer'].load_state_dict(checkpoint['optimizer'])
            config['train']['epoch'] = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    if 'epoch' not in config['train']:
        config['train']['epoch'] = 0