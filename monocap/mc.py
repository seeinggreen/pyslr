import numpy as np;
import utils.utils as ut;
import matlab.engine;
from tqdm import tqdm;
import config as cf;

#Indexes for HM36M data to reorder to MPII order
hm36m_to_mpii = [6, 5, 4, 1, 2, 3, 0, 8, 9, 10, 16, 15, 14, 11, 12, 13];

def get_ml_eng():
    """
    Returns a started matlab engine.
    
    
    The engine is started and set to the correct directory for Monocap. The 
    startup code for Monocap is then run to ensure all necessary files are 
    added to the path.

    Returns
    -------
    eng : engine.matlabengine.MatlabEngine
        A Matlab engine that has been started and initialised.
    """
    eng = matlab.engine.start_matlab();
    eng.cd(cf.mc_dir,nargout=0);
    eng.startup(nargout=0);
    return eng;

def hm_to_mp(hm_data):
    """
    Converts bone data in HM36M format to MPII format.

    Parameters
    ----------
    hm_data : numpy.ndarray
        Bone data in HM36M format.

    Returns
    -------
    mp_data : list
        Bone data in MPII format.
    """
    mp_data = [];
    for i in hm36m_to_mpii:
        mp_data.append(hm_data[i]);
    return mp_data;

def hms_to_ml(hms):
    """
    Reorders and converts a Python style heatmap to a Matlab style.

    Parameters
    ----------
    hms : list of numpy.ndarray
        The heatmaps in Python format.

    Returns
    -------
    mlhms : mlarray.single
        The heatmaps in Matlab ordering stored in Matlab format.
    """
    mlhms = [];
    for hm in tqdm(hms):
        mlhm = np.transpose(hm,(1,2,0));
        mlhms.append(np.expand_dims(mlhm,3));
    mlhms = matlab.single(np.concatenate(mlhms,axis=3).tolist());
    return mlhms;

def get_mc_pred(output,eng,frame,nImg):
    """
    Extracts the predicted datapoints from the Matlab output.

    Parameters
    ----------
    output : dict
        The output from the Monocap Matlab code.
    eng : engine.matlabengine.MatlabEngine
        The Matlab engine.
    frame : numpy.ndarray
        A sample frame to calculate scale from.
    nImg : int
        The number of frames in the sequence.

    Returns
    -------
    preds_2d : numpy.ndarray
        The 2D PE from Monocap.
    preds_3d : numpy.ndarray
        The 3D PE from Monocap.
    """
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
    """
    Get the bone dictionary for Monocap and covert to MPII format.

    Parameters
    ----------
    eng : engine.matlabengine.MatlabEngine
        The Matlab engine.

    Returns
    -------
    b_dict : dict
        The bone data stored as a Matlab double, wrapped in a dict.
    """
    path = cf.pose_dict_path;
    b_dict = eng.load(path,'B');
    b = np.array(b_dict['B']);
    new_b = np.zeros((b.shape[0],16));
    for i,p in enumerate(b):
        new_b[i] = hm_to_mp(p);
    b_dict['B'] = matlab.double(new_b.tolist());
    return b_dict;

def get_3d_points(preds_3d):
    """
    Scales the 3D points.

    Parameters
    ----------
    preds_3d : numpy.ndarray
        The raw 3D points.

    Returns
    -------
    preds_3d : numpy.ndarray
        The scaled points.
    """
    for i,p in enumerate(preds_3d):
        preds_3d[i] = preds_3d[i] - preds_3d[i].mean(0)*np.ones((16,1));
    return preds_3d;

def get_2d_points(ps_3d):
    """
    Converts the 3D points to 3 sets of 2D projections in X,Y and Z.

    Parameters
    ----------
    ps_3d : numpy.ndarray
        The 3D points for each keypoint for each frame.

    Returns
    -------
    ps_2d : numpy.ndarray
        The projection along the X,Y and Z axes for each keypoint of each
        frame.
    """
    ps_2d = np.zeros((len(ps_3d),3,len(ps_3d[0]),2));
    for axis in range(3):
        if axis == 0:
            x = 1;
            y = 2;
            z = 0;
        if axis == 1:
            x = 0;
            y = 2;
            z = 1;
        if axis == 2:
            x = 0;
            y = 1;
            z = 2;
        
        add = [0,0,0];
        add[axis] = 5;
        s_3d = ps_3d + add;
        for i in range(100):
            for j in range(15):
                ps_2d[i][axis][j][0] = s_3d[i][j][x] * (2 / s_3d[i][j][z]);
                ps_2d[i][axis][j][1] = s_3d[i][j][y] * (2 / s_3d[i][j][z]);
    return ps_2d;