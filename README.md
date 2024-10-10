# ColD-Model

# Project Title: Collagen D-banding Classification

A deep learning pipeline for collagen D-banding classification in atomic force microscope (AFM) images using MobileNetV2. This project includes training, evaluating, and visualizing results on sub-images of AFM data. 

## Table of Contents
- [Project Overview](#project-overview)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Displaying Model Recognition Results](#displaying-model-recognition-results)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Transfer learning of MobileNetV2 convolutional neural network trained on AFM images of collagen nanoscale D-banding patterns was performed. The model achieved 99% training and testing accuracy and has been validated on external datasets to ensure high robustness and generalizability.


## Usage

### Training the Model

The first script (`train_model.py`) is used to train the MobileNetV2 model on AFM images.

Run the following command to start training:

```bash
python train_model.py
```

### Displaying Model Recognition Results

The second script (`display_results.py`) is used to visualize model predictions on AFM images.

Run the following command to display results:

```bash
python display_results.py
```

Ensure that the paths to the model weights and input folder are correctly specified in the script before running.


## Features
- **Training Pipeline**: Train a MobileNetV2 model on AFM sub-images for binary classification.
- **Evaluation Metrics**: Calculate metrics like accuracy, precision, recall, F1 score, MCC, and Cohen's kappa.
- **Results Visualization**: Display sub-image classification results on original AFM images with color-coded bounding boxes.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
