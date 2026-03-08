# Emotion Recognition with CNN (FER2013 Dataset)

## Project Overview

This project implements a **Facial Emotion Recognition system** using a **Convolutional Neural Network (CNN)** trained on the **FER2013 dataset**.

The goal of the project is to build a deep learning model capable of recognizing human emotions from facial images. The model classifies facial expressions into seven emotion categories.

The project demonstrates a full machine learning pipeline including:

* data exploration
* data preprocessing
* model development
* training and evaluation
* prediction on new images

The model is implemented using **TensorFlow / Keras** and trained using **data augmentation** to improve performance and generalization.

---

# Dataset

The model uses the **FER2013 dataset**, which contains grayscale images of faces labeled with emotional expressions.

### Dataset characteristics

* **Total images:** 35,887
* **Image size:** 48 × 48 pixels
* **Color format:** grayscale
* **Emotion classes:** 7

Emotion categories:

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

The dataset is split into three subsets:

* **Training set**
* **Public test set (validation)**
* **Private test set (final evaluation)**

---

# Data Exploration

Before training the model, the dataset is explored and visualized.

The analysis includes:

* dataset structure and statistics
* distribution of emotions
* class imbalance analysis
* visualization of sample images for each emotion
* distribution of training and test data

Visualization tools used:

* Matplotlib
* Seaborn

These plots help understand the structure of the dataset and potential class imbalance.

---

# Data Preprocessing

The dataset requires several preprocessing steps before it can be used by a neural network.

### Pixel Processing

Each image is stored as a string of pixel values. These values are:

* converted into numerical arrays
* reshaped into **48 × 48 images**
* normalized by dividing pixel values by **255**

---

### Reshaping

Images are reshaped into the format required for CNN input:

```
48 × 48 × 1
```

This represents grayscale images with one channel.

---

### Label Encoding

Emotion labels are converted into **one-hot encoded vectors** using:

```
tf.keras.utils.to_categorical()
```

This allows the neural network to perform multi-class classification.

---

# Convolutional Neural Network (CNN)

A deep CNN architecture is built using **TensorFlow Keras**.

The architecture includes:

### Convolution Layers

Three convolution blocks are used:

* Conv2D
* BatchNormalization
* MaxPooling
* Dropout

These layers extract hierarchical visual features from facial images.

---

### Fully Connected Layers

After feature extraction, the model uses:

* Flatten layer
* Dense layer (256 neurons)
* Dropout (0.5)

The final output layer uses:

```
Softmax activation
```

to classify images into **7 emotion classes**.

---

# Data Augmentation

To improve generalization and reduce overfitting, **data augmentation** is applied using `ImageDataGenerator`.

The following transformations are used:

* small rotations
* horizontal flipping
* filling empty pixels after transformations

Data augmentation allows the model to see more variations of training images.

---

# Model Training

The model is trained using:

* **Adam optimizer**
* **Categorical crossentropy loss**
* **Accuracy metric**

Training parameters include:

* batch size: 64
* multiple training epochs
* validation using the public test dataset

Two callbacks are used:

### ModelCheckpoint

Saves the **best performing model** during training.

### EarlyStopping

Stops training if validation accuracy stops improving.

This prevents overfitting.

---

# Model Evaluation

The trained model is evaluated on the **Private Test dataset**, which represents unseen data.

Example result:

```
Test Accuracy ≈ 61%
```

Evaluation metrics include:

* accuracy
* loss
* confusion matrix
* classification report

The confusion matrix helps visualize which emotions are correctly or incorrectly predicted.

---

# Training Visualization

Training history is visualized using plots for:

* training accuracy vs validation accuracy
* training loss vs validation loss

These plots help evaluate the learning behavior of the neural network.

---

# Emotion Prediction on New Images

The trained model can also predict emotions for **new uploaded images**.

Steps:

1. Upload an image
2. Convert it to grayscale
3. Resize to **48 × 48**
4. Normalize pixel values
5. Run the model prediction

The predicted emotion is displayed together with the input image.

---

# Interactive Feature: Emotion-Based Music

An additional interactive feature is implemented:

After predicting the emotion from a face image, the system **plays a song corresponding to the detected emotion**.

Example:

* happiness → happy music
* sadness → sad music
* anger → angry music

This demonstrates a simple application of emotion recognition in interactive systems.

---

# Technologies Used

* Python
* Google Colab
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* SciPy
* PIL (Python Imaging Library)

---

# Project Structure

```
EmotionDetector

│
├── EmotionDetector.ipynb
├── best_emotion_model.keras
├── emotion_labels.pkl
├── fer2013 dataset
├── README.md
└── sample images / audio files
```

---

# Conclusion

This project demonstrates how deep learning can be used to recognize emotions from facial expressions.

The model successfully learns visual patterns in facial features and can classify emotions with reasonable accuracy.

Future improvements may include:

* deeper CNN architectures
* transfer learning with pretrained models
* better handling of class imbalance
* real-time emotion detection using webcam input
