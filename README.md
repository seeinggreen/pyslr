# pyslr
A Python implementation of a Sign Language Recognition system produced as part of a thesis for a Data Science MSc at Heriot-Watt University.

Anaconda (https://www.anaconda.com/) is recomended as a Python environment for this system. Once Anaconda is installed, the commands in conda_env.txt can be used to set up a new environment with the correct dependencies.

To use the 3D pose estimation, you will need to clone the Monocap repo (https://github.com/daniilidis-group/monocap), download and install MATLAB and install the MATLAB Python engine (https://uk.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html). Alternatively, you can use the premade data provided in this repo.

Test data was taken from the BSL Corpus (https://bslcorpusproject.org/) and 'BF1n.mov' is the file used in much of the code as an example. Videos can be downloaded from the BSL Corpus website.

OpenPose will need to be complied from source in order to use the Python integration - https://cmu-perceptual-computing-lab.github.io/openpose/web/html/doc/md_doc_installation_0_index.html.

Blender is required for rendering the output and can be downloaded here - https://www.blender.org/.

Other requirements are bundled with Anaconda or are listed in the environment setup file.

The main file demonstrates much of the functionality of the system and all code is documented in Numpy style.
