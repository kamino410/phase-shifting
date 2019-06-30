# phase-shifting
This repository provides an implementation of robust 3-step phase-shifting method combined with graycode patterns.

## What is phase-shifting method ?

Phase-shifting method, also called sinusoidal patterns, is a kind of structured light pattern used for display-camera systems.
These methods provide algorithms to capture the correspondences from the camera pixels to the display pixels.

Phase-shifting method computes these correspondences by unwrapping phase map from some images which capture displayed sinusoidal patterns.

The main advantage of phase-shifting method is that it theoretically can estimate the corresponding display pixels at subpixel accuracy.

## Implementation in this repository

Although there are some derivative methods, I implemented the 3-step phase-shifting method which is simplest one.
Also, to obtain the global position of each cycles, I combined two sets of sinusoidal patterns whose frequencies are different and graycode patterns.
The periods of these patterns are defined as follows using user-defined parameter `step` \[pix\].

||period \[pix\]|
|----|----|
|Sinusoidal-1 (S1)|`step/3`|
|Sinusoidal-2 (S2)|`step/2`|
|Graycode (GC)|`step/2`|

The coordinate of corresponding display pixel for each camera pixels are estimated by the following process.

1. Unwrap the phases from S1 and S2.
    * `arctan2(sqrt(3)*(img1 - img3), 2*img2 - img1 - img3)`
2. Compute the phase difference (it is not a simple subtraction between unwrapped values).
3. Decode and interpolate the graycode coordinate in GC.
    * Interpolation is required for pixels on the border of graycode patterns.
    * Mask is computed by dilating and eroding (kernel size : `2*filter_size+1`).
    * The average of coordinates of neighbouring pixels is set (kernel size : `2*filter_size+1`).
4. Compute the coordinate of corresponding display pixel.
    * Phase is the value unwrapped from S1.
    * Global position is computed using the phase, phase difference and graycode coordinate complementarily.

## How to use

### 1. Generate patterns

You can generate pattern images by the following command.

```sh
python3 ./phase_shifting.py gen <display_width> <display_height> <step> <output_dir>

# example
python3 ./phase_shifting.py gen 1920 1080 400 ./patterns
```

This command saves pattern images (`pat00.png`~`patXX.png`) and a config file (`config.xml`) into the specified directory.

### 2. Gamma calibration

Before capture images, you have to set or calibrate gamma values of your display and camera.
The phase-shifting method requires that the input value to the display and the output value from the camera are linear.
However, most of imaging and display devices transform input values based on the gamma value.
Thus, you have to correct it by estimating the gamma value or creating look-up table.
Or, if your devices are gamma adjustable, set the gamma value to 1.0.

### 3. Capture displayed patterns

Display generated patterns on your display and capture it by your camera one by one.
Captured images must be saved as `xxxx00.png`~`xxxxXX.png` in a single directory.

### 4. Decode patterns

You can decode captured images by the following command.

```sh
python3 ./phase_shifting.py dec <input_prefix> <config_path> <output_path> [-black_thr BLACK_THR] [-white_thr WHITE_THR] [-filter_size FILTER_SIZE]

# sample
python3 ./phase_shifting.py dec ./captured/pat ./patterns/config.xml ./captured
```

This command saves the following files.

1. Visualized image (`visualized.png`)
    * B : decoded x coordinate of display pixel
    * G : decoded y coordinate of display pixel
    * R : 128 if decoded successfully
    * The colors are folded back at every 256 pixels
2. List of decoded coordinates (`camera2display.csv`)
    * `camera_y, camera_x, display_y, display_x`

