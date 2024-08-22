
Video Prediction Transformer (VPTR) for Surgical Video Prediction

This repository contains code and documentation for our project exploring future frame prediction in surgical video sequences using the Video Prediction Transformer (VPTR) model, with a specific focus on the JIGSAWS Suturing dataset.

Project Overview

Our research investigates the application of deep learning, particularly Transformer-based architectures, to predict future frames in medical videos. We focus on the challenging task of forecasting surgical tool movements and tissue interactions in robot-assisted surgery. We leverage the VPTR model, which combines the strengths of convolutional autoencoders and efficient transformer networks to capture both spatial and temporal information in video sequences.

Key Features

VPTR Implementation: Adapted and extended the VPTR model for the JIGSAWS Suturing dataset.

Preprocessing Pipeline: Implemented a custom preprocessing pipeline to extract frames, resize them, and organize them into a BAIR-like format for compatibility.

Training and Evaluation: Provides scripts for training and evaluating VPTR models on the JIGSAWS dataset, including both Non-Autoregressive (NAR) and Fully Autoregressive (FAR) variants, as well as a novel Partially Autoregressive (PAR) approach.

Quantitative Metrics: Calculates and reports PSNR, SSIM, and LPIPS metrics to objectively assess prediction accuracy.

Qualitative Visualization: Generates visualizations of predicted frames for visual inspection and comparison with ground truth.

Dataset

JIGSAWS Suturing Dataset: We use the Suturing subset of the JIGSAWS dataset, a benchmark for surgical activity recognition and skill assessment. The dataset is preprocessed and split into training and test sets based on user IDs, following the methodology of the TPG-VAE paper [2].

Code Structure

model/: Contains the definitions for the AutoEncoder, Transformer, and other model components.

utils/: Includes utility functions for data loading, preprocessing, training, evaluation, and visualization.

train_AutoEncoder.py: Script for training the AutoEncoder.

train_FAR.py: Script for training the FAR model.

train_NAR.py: Script for training the NAR model.

test_VPTR.py: Script for testing and evaluating trained models.

"
Dependencies

Python 3.8

PyTorch 1.9.0

OpenCV

Pillow (PIL)

TensorBoard

... (list other required packages from your requirements.txt)

"
Any versions of these dependencies work by the way. It's just that the packages have to exist, the newer the better.
Usage

Download the JIGSAWS Suturing dataset: https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/

Preprocess the dataset:

Extract frames, resize, and organize them using the provided scripts (e.g., preprocess_jigsaws.py, convert_to_bair.py).
Or as a shortcut: The dataset is loaded from \VPTR_jigsaws_modifications\jigsaws_suturing\bair_format_dir

Train the AutoEncoder: Run train_AutoEncoder.py to train the AutoEncoder on the preprocessed data.

Train the Transformer (FAR or NAR): Use either train_FAR.py or train_NAR.py to train the desired Transformer variant, loading the pretrained AutoEncoder weights.

Test and Evaluate: Run test_VPTR.py to load a trained model, perform inference, calculate metrics, and generate visualizations.

The Model files are available here 👉 https://leeds365-my.sharepoint.com/:f:/g/personal/sc23gd_leeds_ac_uk/ErcROKZKelFLtahFkWEbrNMBhLdT0Tkr34_DCOBTS__3Aw?e=t31Ucj

References

[1] Y. Gao et al., "JHU-ISI Gesture and Skill Assessment Working Set (JIGSAWS): A Surgical Activity Dataset for Human Motion Modeling," MICCAI, 2014.
[2] X. Gao et al., "Future Frame Prediction for Robot-assisted Surgery," ICRA, 2021.
[3] J. Johnson, et al., "Perceptual Losses for Real-Time Style Transfer and Super-Resolution", ECCV, 2016.

Acknowledgements

This project is inspired by the VPTR paper [1] and the TPG-VAE paper [2]. We acknowledge the contributions of the authors of these papers and the creators of the JIGSAWS dataset.

Feel free to modify this template with your specific details and add any additional sections or information that you think would be relevant for users of your repository.

