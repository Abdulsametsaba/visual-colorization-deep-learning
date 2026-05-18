
# Visual Colorization with Deep Learning

An AI-based image colorization project that transforms grayscale landscape images into realistic colored images using a convolutional autoencoder architecture built with PyTorch.

# 🚀 Project Overview

This project focuses on automatic image colorization using deep learning techniques. The model receives grayscale images as input and predicts color channels to generate realistic colored outputs.

The system is trained on landscape images and learns spatial and color relationships through an encoder-decoder architecture.

# 🧱 Model Architecture

The project uses a Convolutional Autoencoder architecture:

Encoder
Conv2D
ReLU
Downsampling
Decoder
ConvTranspose2D
Upsampling
Color reconstruction
📊 Technologies Used
Python
PyTorch
NumPy
Matplotlib
PIL
scikit-image
Jupyter Notebook

# 📁 Dataset

Dataset used:

Landscape Image Colorization Dataset

Source:

Kaggle

The dataset contains colored landscape images which are converted into LAB color space during preprocessing.
