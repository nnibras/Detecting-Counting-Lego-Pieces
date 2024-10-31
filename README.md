# Detecting-Counting-Lego-Pieces
Counting &amp; Detecting Lego Pieces using Mask R-CNN (pytorch)

# LEGO Detection with Mask R-CNN

This project provides a comprehensive setup for detecting LEGO pieces in images using Mask R-CNN. It includes scripts for training, evaluation, data processing, and an interactive interface powered by Gradio for testing the model.

## Project Overview

- **Model**: Mask R-CNN, configured for LEGO piece detection.
- **Features**:
  - Detects and counts LEGO pieces in uploaded images.
  - Provides bounding boxes and pseudo-masks to highlight detected objects.
  - Includes Gradio-based app interfaces for interactive usage.
  - Supports training, evaluation, and data processing.

## Table of Contents

- [Setup Instructions](#setup-instructions)
- [Project Structure](#project-structure)
- [Data Processing](#data-processing)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Running the Application](#running-the-application)
- [License](#license)

---

## Setup Instructions
Download or Provide Model Weights
To run the Gradio app or perform evaluation, download the pretrained model weights (mask_rcnn_lego.pth) and place them in the project directory. This file contains the trained weights for detecting LEGO pieces.

Note: If you don’t have this model file, you may need to train it using train.py.

Install Dependencies
Install the required Python packages listed in requirements.txt:

pip install -r requirements.txt

## Project Structure
lego-detection-maskrcnn/
├── app.py                    # Main Gradio application script
├── app2.py                   # Alternate Gradio application script
├── All_In_One.py             # Consolidated script combining major functionality
├── data_processing.ipynb     # Notebook for data processing and conversions
├── evaluate.py               # Script for model evaluation
├── LegoDataset.py            # Custom dataset class for LEGO detection
├── lego_coco_annotations.json         # COCO format annotation file
├── lego_coco_annotations_train.json   # Training annotation JSON file
├── lego_coco_annotations_val.json     # Validation annotation JSON file
├── requirements.txt          # Dependencies file
├── train.py                  # Training script for Mask R-CNN model
├── visualize_predictions.py  # Script for visualizing model predictions
├── voc_to_coco.py            # VOC to COCO format converter script

## Data Processing
If your data annotations are in VOC format, you can convert them to COCO format using voc_to_coco.py:

python voc_to_coco.py

This will generate COCO format annotation files necessary for training the Mask R-CNN model.

You can also inspect and process data using data_processing.ipynb, which includes code for data exploration and format conversion.

## Training the Model
To train the Mask R-CNN model, use train.py. Ensure your data is organized and annotated in COCO format. Adjust the training hyperparameters in the script as needed.

python train.py

Training outputs include:
Updated model weights saved as mask_rcnn_lego.pth (or a specified filename).
Logs for monitoring loss and accuracy.

## Evaluating the Model
Evaluate the model using evaluate.py to see its performance on validation data.

python evaluate.py

Results include:
Evaluation metrics like mean Average Precision (mAP) and Intersection over Union (IoU).
Visualizations that help interpret detection results.

## Running the Application
Launch the Gradio application using app.py:

python app.py
or
python app2.py

This will start a Gradio interface at a local address, where you can:
Upload Image: Select an image containing LEGO pieces.
View Results: The app displays:
Bounding boxes around each detected LEGO piece.
Pseudo-masks to highlight detected areas.
The count of detected LEGO pieces.
Alternatively, you can experiment with app2.py (which has the pseudomask).

## License
This project is open-source under the MIT license.

### 1. Clone the Repository and Download the model

Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/lego-detection-maskrcnn.git
cd lego-detection-maskrcnn

Note: If you do not want to train the model you can download the trained model from:
https://drive.google.com/file/d/1Hbyj18F_mlAw3ZHYesZIrWrYPOvmBVWc/view?usp=sharing



