
# Skin Cancer Detection (Neural Networks - CNN - TensorFlow)

This project demonstrates the use of a custom Convolutional Neural Network (CNN) architecture built from scratch using **TensorFlow** to detect skin cancer based on image data. The dataset contains images categorized as either **benign** or **malignant** skin lesions, and the model is trained to classify the images accordingly.

## Project Overview

The main goal of this project is to build a deep learning model to classify images of skin lesions as benign or malignant using a custom CNN. Techniques such as **regularization** and **early stopping** are used to prevent overfitting and improve model performance.

### Key Features:
- Custom CNN architecture with multiple convolutional blocks.
- L2 Regularization and Dropout for overfitting prevention.
- Early stopping to optimize training time and performance.
- Class balancing techniques to handle imbalanced data.

---

## Project Structure

- `skin_cancer_detection_using_tensorflow.py`: Main script containing the CNN model, data preprocessing, training, and evaluation logic.
- `README.md`: Documentation of the project (this file).

---

## Dataset

The dataset used for this project consists of images of skin lesions, labeled as either **benign** or **malignant**. It is expected to have the following directory structure:

```
train_cancer/
│
├── benign/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── malignant/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

The dataset is split into training and validation sets using an 85%-15% ratio. The images are resized to 224x224 pixels for model input.

---

## Model Architecture

The model is a **Convolutional Neural Network (CNN)** built from scratch using the following layers:

- **Convolutional Layers**: 4 convolutional blocks with increasing filter sizes (32, 64, 128, 256) to extract features from the images.
- **MaxPooling**: Used after each convolutional block to reduce the spatial dimensions.
- **Dense Layers**: Fully connected layers to make the final classification.
- **Dropout and L2 Regularization**: Applied to reduce overfitting.
- **Sigmoid Activation**: For binary classification (benign vs malignant).

### Regularization and Dropout
- L2 regularization (also known as weight decay) is used in the convolutional and dense layers to penalize large weights and prevent overfitting.
- Dropout layers with a rate of 0.5 are used after the dense layers to randomly disable neurons during training, further reducing overfitting.

### Early Stopping
- **EarlyStopping** is used to halt training when the validation loss stops improving. This ensures the model does not overfit the training data and prevents unnecessary computation.

---

## Requirements

To run this project, you need the following dependencies:

- `tensorflow`
- `keras`
- `numpy`
- `pandas`
- `matplotlib`
- `PIL` (Pillow)
- `seaborn`
- `scikit-learn`
- `glob`
- `imblearn` (optional, for handling imbalanced datasets)

You can install the dependencies using `pip`:

```bash
pip install tensorflow keras numpy pandas matplotlib Pillow seaborn scikit-learn imblearn
```

---

## Running the Project

1. **Dataset Preparation**: Place your image dataset in the `train_cancer/` directory with subdirectories for each class (benign and malignant).
   
2. **Run the Script**: Execute the Python script to preprocess the data, train the CNN model, and evaluate its performance.

```bash
python skin_cancer_detection_using_tensorflow.py
```

3. **Model Training**: The model will train for up to 50 epochs but will stop early if the validation loss stops improving for 3 consecutive epochs.

4. **Results Visualization**: After training, the script will plot the loss and AUC for both training and validation data.

---

## Model Evaluation

During training, the following metrics are tracked:
- **Binary Cross-Entropy Loss**: To measure the difference between true and predicted labels.
- **AUC (Area Under the Curve)**: A key metric for binary classification tasks that evaluates how well the model distinguishes between classes.

---

## License

This project is licensed under the MIT License.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue if you have any suggestions or improvements.

---

## Acknowledgements

- The project was developed using the **TensorFlow** deep learning library.
- Special thanks to the open-source community for providing useful resources for image classification tasks.
