# Phase-shifting method
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


