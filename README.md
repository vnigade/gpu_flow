GPU based optical flow extraction in OpenCV.
====================
The [original code](https://github.com/feichtenhofer/gpu_flow) is ported to Opencv 3.4.4 with some (or more) code changes. Please see the code diff for more information.

### Features:
* OpenCV wrapper for Real-Time optical flow extraction on GPU
* Automatic directory handling using Qt
* Allows saving of RGB frames to disk
* Allows saving of optical flow to disk,
** either with clipping large displacements 
** or by adaptively scaling the displacements to the radiometric resolution of the output image

### Download Dependencies
* [OpenCV 3.4.4](http://opencv.org/downloads.html)
* [FFMPEG 3.4.5](http://ffmpeg.org/download.html)
* [Qt 5.4](https://www.qt.io/qt5-4/)
* [cmake](https://cmake.org/)

### Or Build Dependencies
## FFMPEG
1. `git clone https://git.ffmpeg.org/ffmpeg.git`
2. `git checkout tags/n3.4.5`
3. `./configure --enable-shared --disable-programs --disable-doc --enable-gpl --prefix="$HOME/.local" --extra-cflags="-I$HOME/.local/include" --extra-cxxflags="-I$HOME/.local/include`

## OpenCV
1. `git clone git@github.com:opencv/opencv.git`
2. `git checkout tags/n3.4.5`
3. `export PKG_CONFIG_PATH="$HOME/.local/lib/pkgconfig"`
4. `export LD_LIBRARY_PATH=$$HOME/.local/lib:$HOME/.local/lib64:$LD_LIBRARY_PATH`
5. `cmake -DCMAKE_PREFIX_PATH="$HOME/.local" -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX="$HOME/.local" -DWITH_CUDA=ON -DENABLE_FAST_MATH=1 -DCUDA_FAST_MATH=1 -DWITH_CUBLAS=1 -DWITH_LIBV4L=ON -DWITH_FFMPEG=1 ../ > /tmp/diff 2>&1`

### Installation
1. `mkdir -p build`
2. `cd build`
3. ` cmake -DCMAKE_PREFIX_PATH="$HOME/.local" -DCMAKE_BUILD_TYPE=RELEASE ../`
4. `make`

### Configuration:
You should adjust the input and output directories by passing in `vid_path` and `out_path`. Note that `vid_path` must exist, Qt will create `out_path`. Use -h option t for more.
In the CMakeLists.txt there is an option called WARP. This selects if you want warped optical flow or not. The warped optical flow file also outputs optical flows as a single BGR image (red is the flow magnitude). In the `compute_flow_si_warp` file itself there is a warp variable that you can set to false to just compute normal flow. If you want grayscale for images (x and y) use `compute_flow`.

### Usage:
```
./compute_flow [OPTION] vid_path out_path
```
```
./compute_flow_si_warp [OPTION] ..
```

Available options:
* `gpuID`: use this GPU ID [0]
* `type`: use this flow method Brox = 0, TVL1 = 1, Farneback = 2 [1] 
* `skip`: the number of frames that are skipped between flow calcuation [1]

Additional features in `compute_flow.cpp`:
* `float MIN_SZ = 256`: defines the smallest side of the frame for optical flow computation
* `float OUT_SZ = 256`: defines the smallest side of the frame for saving as .jpeg 
* `bool clipFlow = true;`: defines whether to clip the optical flow larger than [-20 20] pixels and maps the interval [-20 20] to  [0 255] in grayscale image space. If no clipping is performed the mapping to the image space is achieved by finding the frame-wise minimum and maximum displacement and mapping to [0 255] via an adaptive scaling, where the scale factors are saved as a binary file to `out_path`.

### Example:
```
./compute_flow --gpuID=0 --type=1 ~/datasets/UCF101/test/ ~/datasets/UCF101/flow_output/ 
```
