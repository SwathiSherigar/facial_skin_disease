# Skin Disease Classification Using CNN

This project demonstrates the use of a Convolutional Neural Network (CNN) for the classification of skin diseases using image data. The dataset includes images of different classes of skin diseases, and the model is trained to classify these images into their respective categories.

## Dataset

The dataset is organized in a directory structure where images are stored in subfolders corresponding to the disease classes. The dataset is preprocessed and augmented using `ImageDataGenerator`.


## Model Architecture

The CNN model includes the following layers:
- **Conv2D**: Extracts features using convolutional filters.
- **MaxPooling2D**: Reduces spatial dimensions.
- **Flatten**: Converts the 2D matrix into a 1D vector.
- **Dense (Fully Connected)**: Performs the classification task.

### Model Summary
- Input Shape: `(150, 150, 3)` (150x150 RGB images)
- Output: 5 classes corresponding to the diseases.

## Prerequisites

Ensure you have Python 3.8+ and the following libraries installed:
- `tensorflow`
- `numpy`
- `Pillow`
- `matplotlib`

Install dependencies using:
```bash
pip install tensorflow numpy pillow matplotlib
