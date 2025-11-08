# **AC\_GRConvnet\_Main: Grasp Detection Project**

This project implements an **AC-GRConvNet** (Attention Complementary Generative Residual Convolutional Network) for robotic grasp detection, finetuned on the Cornell Grasp Dataset.

The model uses a two-stream encoder (for RGB and Depth) and an Attention Complementary Module (ACM) to fuse features and predict grasp poses (quality, angle, and width).

## **Project Structure**

This project consists of the following Python scripts:

* **model.py**: Defines the AC\_GRConvNet architecture, including the ResidualBlock and AttentionComplementaryModule (ACM).  
* **dataset.py**: Contains the GraspDataset class for loading and augmenting the Cornell Grasp Dataset.  
* **utils/data\_processing.py**: Provides utility functions for data processing, such as generate\_grasp\_maps\_gaussian to create ground-truth maps and normalization functions.  
* **train.py**: The main script for training the model. It includes:  
  * Data splitting (80/15/5 for train/val/test).  
  * A two-stage training process: main training and fine-tuning.  
  * LR schedulers with warmup and cosine annealing.  
  * Focal Loss for the quality map and weighted MSE for angle/width.  
  * Logging of training progress to outputs/training\_log.csv.  
* **evaluate.py**: Script to evaluate the trained models (main\_best.pth and finetune\_best.pth) on the test set. It calculates grasp accuracy based on IoU and angle thresholds and saves plots to outputs/evaluation/.  
* **predict.py**: A script to run inference on samples from the test set, visualize the model's predictions (quality map, angle map, best grasp), and save them to outputs/visualizations/.

## **How to Run**

### **1\. Data Setup**

1. Download the **Cornell Grasp Dataset**.  
2. Place the dataset files (e.g., pcd0100r.png, pcd0100cpos.txt, etc.) into a directory named ./data/.

### **2\. Training**

Run the training script to start the main training and fine-tuning process.  
Models will be saved in ./outputs/models/.  
python train.py

### **3\. Evaluation**

After training, evaluate the performance of the saved models on the test set.  
Results will be saved in ./outputs/evaluation/.  
python evaluate.py

### **4\. Prediction & Visualization**

To visualize the model's predictions on a few random test samples, run:  
Images will be saved in ./outputs/visualizations/.  
python predict.py

## **Key Dependencies**

* torch (PyTorch)  
* numpy  
* matplotlib  
* scikit-image (for skimage.draw.polygon)  
* opencv-python (for cv2.inpaint)  
* pillow (PIL)  
* tqdm  
* shapely (for evaluation)
