# AI Content Moderation System

A machine vision project that automatically detects and classifies images and videos as Safe (Neutral) or Unsafe (NSFW) using deep learning.

## Model
- Architecture: EfficientNetB0 with Transfer Learning
- Training: Two-phase fine-tuning on 16,638 labeled images
- Dataset: NSFW Detection Dataset (Kaggle - jjeevanprakash/nsfw-detection)
- Validation Accuracy: 71% | AUC: 0.81

## Features
- Image classification (Safe / Unsafe)
- Video analysis with frame-by-frame detection
- Grad-CAM heatmap visualization
- Real-time confidence score
- Flask web interface

## Tech Stack
- Python, TensorFlow 2.19, Keras 3.12
- EfficientNetB0 pretrained on ImageNet
- Flask, OpenCV
- Grad-CAM for explainability

## Setup

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    python app.py

Open http://localhost:5000 in your browser.

## Project Structure

    content_moderation/
    app.py          - Flask backend and prediction logic
    train.ipynb     - Model training notebook
    templates/      - HTML frontend
    static/         - JavaScript
    model/          - Saved model (not included in repo)

## Machine Vision Concepts
- Convolutional Neural Networks
- Transfer Learning and Two-phase Fine-tuning
- Data Augmentation
- Grad-CAM Visualization
- Binary Image Classification
- Video Frame Sampling
