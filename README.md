# Person Re-identification using Face Biometric Feature

The system have already finished for the initial production, but the README not yet updated.

# Source
## Detection

### Install Darknet
Install YOLOv3 Darknet using this [link](https://github.com/AlexeyAB/darknet)

### Change Makefile
Change the Makefile: `GPU=1`, `CUDNN=1`, `OPENCV=1`[Optional], `OPENMP=1`, and `LIBSO=1`

### Create symlink
Create symlink using this command:  
`sudo ln -s <DARKNET HOME DIRECTORY WITH .SO FILE> /usr/lib/x86_64-linux-gnu/libdarknet.so`

### Install Wrapper
1. Clone the wrapper:  `git clone https://github.com/madhawav/YOLO3-4-Py` 
2. Export CUDA_HOME path: `EXPORT CUDA_HOME=<CUDA_HOME_PATH>`
3. Export GPU option: `EXPORT GPU=1`
4. Export OpenCV option: `EXPORT OPENCV=1` [For this, install OpenCV first using the link given by README of the file]

## Detection
- [YOLOv3]

## Tracking
- [Deep Sort](https://github.com/nwojke/deep_sort)
- [Optical Flow](TODO)

## Keyframe Extraction
### Face
#### Detection
- [x] [Haar Cascade Classifier]()
- [ ] [HOG using dlib]()
- [ ] [MTCNN](TODO)

#### Facial Landmark Extraction
- [x] [Dlib Kazemi Sullivan]()
- [ ] [MTCNN](TODO)

#### Keyframe Extraction
1. Using thresholding on facial landmark size (Face image need to be resized first for measure the threshold)
- Using heuristic to determine the threshold
- [TODO] Need to determine the necessary training sample for the keyfram extraction (Assumption: 5-10 keyframe per person)
- Assumption: Person Detection Accurate
- Assumption: Face Detection Accurate

2. Using confidence from MTCNN

### Apperance Model


# Thesis Progress

- [x] Threading Input
- [x] Detection
- [x] Tracking
- [x] Detection & Tracking Threading
- [ ] Keyframe Extraction
- [ ] Re-identification Threading
- [ ] Face Recognition (Re-identification Face)
- [ ] Appearance Model Re-identification

# Other Important Stuff
- Frame output resulting in about 11.444 person image until the end of the video.
- 