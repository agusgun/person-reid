# Person Re-identification using Face Biometric Feature

The README has been updated. Please contact me or make an issue on this repository if you have any problem to install/reproduce the experiment. 

# How to Install Dependency and Run The Program

## Requirements
1. Darknet need to be installed (Refer to the source section)

## Install Dependency

1. [Optional] Create new conda environment (to make fbs command run smoothly)
```
conda create -n <NAME OF YOUR ENVIRONMENT> python=3
```

2. Install pip requirement
```
pip install -r requirements.txt
```

3. [Optional] Install opencv-python using conda command (if there is an error about opencv not found)
```
conda install -c conda-forge opencv
```

4. [Optional-GPU] Install yolo34py using pip (yolo34py is a wrapper of darknet so YOLO model can be run on GPU)
```
pip install yolo34py-gpu
```
NB: This wrapper doesn't support MacOS

5. Install tensorflow-gpu (or tensorflow if you don't have GPU) and keras on the environment
```
conda install -n <NAME OF YOUR NEW CONDA ENVIRONMENT> tensorflow keras
```

6. Install keras_vggface (if there is a problem keras_application then fix it manually -> edit site-packages)
```
pip install keras_vggface
```

7. Install facerec from bytefish repository
```
git clone https://github.com/bytefish/facerec
cd py
python setup.py install
```
8. Install mtcnn
```
pip install mtcnn
```

9. Export some environment variable
	- GPU environment variable to know whether you have GPU or not
		1. Only CPU: `EXPORT USE_GPU=`
		2. GPU support: `EXPORT USE_GPU=1` (GPU boost and Linux tested)
	- There are 2 versions of the system. The first is the full system that used all the pipeline meanwhile, the second only used face detection and re-identification. 
		1. Full system: `EXPORT PERSON_REID_DIRECT_REIDENTIFICATION=` 
		2. Partial system (faster): `EXPORT PERSON_REID_DIRECT_REIDENTIFICATION=1`

# How to Run the Experiment

1. Install jupyter notebook on the environment
2. Install the dependency
3. Pick one of the notebook
4. Run all the cell of the notebook
5. Please make an issue if there is any problem 


# Source
## Detection

### Install Darknet
Install YOLOv3 Darknet using this [link](https://github.com/AlexeyAB/darknet)

### Change Makefile
Change the Makefile before installation using make: `GPU=1`, `CUDNN=1`, `OPENCV=1`[Optional], `OPENMP=1`, and `LIBSO=1`

### Create symlink
Create symlink using this command:  
`sudo ln -s <DARKNET HOME DIRECTORY WITH .SO FILE> /usr/lib/x86_64-linux-gnu/libdarknet.so`

### Install Wrapper
1. Clone the wrapper:  `git clone https://github.com/madhawav/YOLO3-4-Py` 
2. Export CUDA_HOME path: `EXPORT CUDA_HOME=<CUDA_HOME_PATH>`
3. Export GPU option: `EXPORT GPU=1`
4. Export OpenCV option: `EXPORT OPENCV=1` [For this, install OpenCV first using the link given by README of the file]
5. Go to the root directory of the yolo34py and install using pip
- Install manually
```
cd YOLO3-4-Py
pip install .
```

- Install using pypi project
```
pip install yolo34py
```

## Detection
- [YOLOv3](https://pjreddie.com/darknet/)

## Tracking
- [Deep Sort](https://github.com/nwojke/deep_sort)

## Keyframe Extraction
### Face
#### Detection
- [x] [Haar Cascade Classifier](https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html)
- [x] [HOG using dlib](http://dlib.net/face_detector.py.html)
- [x] [MTCNN](https://github.com/ipazc/mtcnn)

#### Facial Landmark Extraction
- [x] [Dlib Kazemi Sullivan](http://dlib.net/dlib/image_processing/shape_predictor_trainer_abstract.h.html)
- [x] [MTCNN](https://github.com/ipazc/mtcnn)

#### Keyframe Extraction
Using thresholding on face tilt angle (can be measured using slope between two eyes position)

## Re-identification
- [x] [VGGFace]()
- [x] [FaceNet](https://github.com/davidsandberg/facenet)
- [x] [LBPH](https://github.com/bytefish/facerec)  

Please download the model weight from each link above and place it on `model_data` directory.

# Thesis Progress

- [x] Threading Input
- [x] Detection
- [x] Tracking
- [x] Detection & Tracking Threading
- [x] Keyframe Extraction
- [x] Re-identification Threading
- [x] Face Recognition (Re-identification Face)

# Source and Tutorial

- Other Github:
	1. https://github.com/Qidian213/deep_sort_yolov3
- Benhoff:
	1. http://benhoff.net/face-detection-opencv-pyqt.html
- LearnOpenCV:
	1. https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/
	2. https://www.learnopencv.com/facemark-facial-landmark-detection-using-opencv/
- PyImageSearch:  
	1. https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
- Stackoverflow
- Medium
- And a lot more tutorial and source that I can't list one by one (if I use your code and not listed here please contact me)