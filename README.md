# CaliCam Mono: Switching Between Perspective, Cylindrical, Undistorted Fisheye, and Longitude-Latitude Modes

<p align="center">
  <img src="http://astar.support/dotai/calicam_mono.png">
</p>

For more information see
[https://astar.ai](https://astar.ai).

The following steps have been tested and passed on Ubuntu **16.04.5**.

### 1. Theoretical Background

Fisheye Camera Model:
C. Mei and P. Rives, Single View Point Omnidirectional Camera Calibration From Planar Grids, ICRA 2007.

### 2. OpenCV Installation

Follow the steps in [CaliCam@GitHub](https://github.com/astar-ai/calicam).

### 3. Compile

	git clone https://github.com/astar-ai/calicam_mono.git
	cd calicam
	chmod 777 ./compile.sh
	./compile.sh

### 4. Run

	./calicam

### 5. Calibration Parameter File
To run CaliCam in the **LIVE** mode, you need to download the calibration parameter file from online.
Each CaliCam Mono camera has a **UNIQUE** parameter file. Please download the corresponding parameter file by following the instructions at [https://astar.ai/collections/astar-products](https://astar.ai/collections/astar-products).

### 6. Operation

#### 6.1 'Raw Image' window
There are 3 trackbars to adjust the vertical **FoV**, **width**, and **height** for the output image. **FoV** is only for the perspective image.

#### 6.2 'Disparity Image' window
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

	./calicam YOUR_CALIBRATION_FILE.yml

