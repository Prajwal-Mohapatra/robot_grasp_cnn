# robot_grasp_cnn

# Generative Robotic Grasp Prediction Pipeline

This project implements a state-of-the-art robotic grasp prediction pipeline based on a generative, fully convolutional neural network. It moves beyond simple regression to generate pixel-wise maps of grasp quality, angle, and width, allowing for more accurate and robust grasp detection in cluttered scenes.

The architecture and methodology are heavily inspired by the principles outlined in the provided technical report, particularly the GR-ConvNet model.

## Project Structure
'''bash
.
├── data/                     # Placeholder for Cornell Grasp Dataset
├── outputs/                  # Directory for saved models and visualizations
│   ├── models/
│   └── visualizations/
├── README.md                  # This file
├── dataset.py                 # New dataset loader and pre-processor
├── model.py                   # New Generative Residual Network (GR-ConvNet)
├── train.py                   # New training script for the generative model
├── predict.py                 # New script for inference and visualization
├── utils/
   └── data_processing.py      # Utilities for generating ground-truth maps
'''

## Core Concepts & Methodology

### 1. Paradigm Shift: Regression vs. Generation

* **Previous Approach (Regression):** The old pipeline used a standard ResNet-34 to regress a single 6-parameter vector representing one grasp. This is inefficient as it doesn't capture all possible grasps and relies on a simplistic loss function.
* **New Approach (Generative):** As recommended by the technical report, we now use a fully convolutional network to generate dense prediction maps. For every pixel in the input image, the model predicts:
    * **Grasp Quality (Q):** The probability of a successful grasp centered at that pixel.
    * **Grasp Angle ($\theta$):** The orientation of the gripper. To handle the periodicity of angles, we predict `cos(2θ)` and `sin(2θ)`.
    * **Grasp Width (W):** The required opening width of the gripper.

This generative approach provides a much richer training signal and elegantly handles scenes with multiple valid grasps.

### 2. Custom GR-ConvNet Architecture

The new `model.py` defines a **Generative Residual Convolutional Network (GR-ConvNet)**. Its key features are:

* **Encoder-Decoder Structure:** It progressively downsamples the input to learn abstract features (encoder) and then upsamples to reconstruct full-resolution output maps (decoder).
* **Residual Backbone:** At the core of the network, a series of residual blocks (the "ResNet" part) allows for a deeper, more powerful architecture that can be trained effectively without gradient degradation issues.
* **Skip Connections:** Features from the encoder are concatenated with corresponding layers in the decoder. This allows the network to combine high-level semantic information with low-level spatial details, crucial for precise pixel-wise prediction.
* **Multi-Channel Output:** The final layer is a 1x1 convolution that produces 4 output channels: Q, cos(2θ), sin(2θ), and W, each with an appropriate activation function (Sigmoid or Tanh).

### 3. Data Processing and Ground-Truth Generation

A generative model requires ground-truth "heatmaps" for training. The new `utils/data_processing.py` and `dataset.py` handle this:

1.  For each training image, we take the provided ground-truth grasp rectangles.
2.  For each rectangle, we generate a corresponding set of maps:
    * The **Quality map** has a value of 1.0 inside the grasp rectangle and 0.0 elsewhere.
    * The **Angle and Width maps** contain the respective values for each pixel within the rectangle.
3.  All ground-truth maps for an image are combined into a single set of target maps. This process is essential for training the network.
4.  Data augmentation (random rotations, zooms) is applied to both the input image and the target maps to create a more robust model.

### 4. Training with a Masked Loss

The `train.py` script is completely overhauled:

* It loads the new `GRConvNet` model and the `GraspDataset`.
* The loss function is a sum of the Mean Squared Error (MSE) between the predicted and ground-truth maps.
* **Crucially**, the loss for the angle and width maps is **masked**. We only calculate the loss for these parameters at pixels where a valid grasp exists (i.e., where the ground-truth quality map is 1.0). This prevents the network from being penalized for predicting random angles and widths in empty space.

### 5. Inference and Visualization

The `predict.py` script replaces the old visualization logic. It performs the full end-to-end process:

1.  Loads a trained model and a sample image.
2.  Performs inference to get the four output heatmaps.
3.  **Post-processes** the maps to find the best grasp: it finds the pixel with the maximum value in the Quality map, and then looks up the corresponding angle and width values from the other maps.
4.  Generates a comprehensive visualization showing the input RGB image, the predicted Quality and Angle maps, and the final best grasp overlaid on the image.

## How to Run the Pipeline

1.  **Setup:** Place the Cornell Grasp Dataset in the `data/` directory.
2.  **Training:** Run the training script. It will save the best model to `outputs/models/`.
    ```bash
    python train.py
    ```
3.  **Prediction:** Run the prediction script to see the results on random validation samples. Visualizations will be saved to `outputs/visualizations/`.
    ```bash
    python predict.py
    ```
