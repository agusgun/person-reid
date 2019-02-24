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

## Tracking
[Deep Sort](https://github.com/nwojke/deep_sort)