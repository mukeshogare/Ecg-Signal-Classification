# ECG Signal Classification Project

## Overview
This project focuses on classifying Electrocardiogram (ECG) signals using machine learning techniques. The main objective is to develop a model capable of accurately categorizing ECG signals into different classes, such as normal rhythm and various arrhythmias.

## File Structure
- `ecg2cwtscg.m`: This MATLAB script converts ECG signals to scalogram images using Continuous Wavelet Transform (CWT) technique.
- `main.m`: This MATLAB script serves as the entry point for the project. It includes the main code for the classification of ECG signals.
- `usingalexnet.m`: This MATLAB script contains the code for utilizing the pre-trained AlexNet model for classification purposes.
- `squeeze.m`: This MATLAB function implements the SqueezeNet architecture, which can be used as a baseline model for classification.
- `usingcnn.m`: This MATLAB function demonstrates the implementation of a Convolutional Neural Network (CNN) for ECG signal classification.

## Usage
1. Run `ecg2cwtscg.m` to preprocess ECG signals and convert them into scalogram images.
2. Execute `main.m`, which utilizes the processed scalogram images for classification.
3. Alternatively, you can explore different classification approaches by utilizing `usingalexnet.m`, `squeeze.m`, or `usingcnn.m`.

## Dependencies
- MATLAB R2020a or later versions.
- Deep Learning Toolbox (for using pre-trained models and building neural networks).
- Signal Processing Toolbox (for signal preprocessing).

## References
- If you use pre-trained models like AlexNet or SqueezeNet, cite the original papers or documentation provided by MATLAB.
