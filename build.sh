#!/usr/bin/env bash

# Update package repositories
apt-get update

# Install Tesseract OCR for text recognition
apt-get install -y tesseract-ocr

# Install Poppler for PDF to image conversion (required by pdf2image)
apt-get install -y poppler-utils

# Install OpenCV dependencies
apt-get install -y libgl1-mesa-glx

# Install any other required system libraries
apt-get install -y libsm6 libxext6 libxrender-dev

# Set up Tesseract language data (optional)
apt-get install -y tesseract-ocr-eng

# Print versions for debugging
echo "Tesseract version:"
tesseract --version

echo "Poppler version:"
pdftoppm -v

echo "System dependencies installed successfully"
