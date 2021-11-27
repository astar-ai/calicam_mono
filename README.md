# CaliCam_Mono: Various Fisheye Rectification Modes

<p align="center">
  <img src="http://astar.support/dotai/calicam_mono.png">
</p>

CaliCam_Mono currently supports four rectification modes, these are the perspective, cylindrical, undistorted fisheye and longitude-and-latitude, respectively.

For more information see
[https://astar.ai](https://astar.ai).

The following steps have been tested and passed on Ubuntu **16.04.5**.

### 1. Theoretical Background

Fisheye Camera Model:
C. Mei and P. Rives, Single View Point Omnidirectional Camera Calibration From Planar Grids, ICRA 2007.

### 2. OpenCV Dependencies

Required at leat 3.0. Tested with OpenCV 3.4.0.

### 3. Run C++ Code
#### Compile

	mkdir build && cd build
	cmake ..
	make
#### Run
	./calicam_mono

### 4. Run Python Code

	python calicam_mono.py

### 5. Calibration Parameter File
To run CaliCam in the **LIVE** mode, you need to download the calibration parameter file from online.
Each CaliCam Mono camera has a **UNIQUE** parameter file. Please download the corresponding parameter file by following the instructions at [https://astar.ai/collections/astar-products](https://astar.ai/collections/astar-products).

### 6. Operation

#### 6.1 'Raw Image' window
There are 3 trackbars to adjust the vertical **FoV**, **width**, and **height** for the output image. **FoV** is only for the perspective image.

#### 6.2 'Rectified Image' window
Press number button 1~4 to switch the transformation mode.

	1: perspective mode
	2: cylindrical mode
	3: undistorted fisheye mode
	4: longitude-latitude mode

#### 6.3 Exit
Press 'q' or 'Esc' key to exit.

### 7. Live Mode
To run CaliCam in a live mode, please change the variable live to true:

	bool      live = true;

and run

	./calicam_mono YOUR_CALIBRATION_FILE.yml
