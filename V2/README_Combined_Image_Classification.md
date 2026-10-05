# Image Classification Project

## Project Overview

This repository documents the development of two image classification models created while learning computer vision and deep learning.

The project evolved through two versions:

1. **Version 1:** a multiclass CIFAR 10 image classifier built with PyTorch
2. **Version 2:** a binary Happy vs Sad image classifier built with TensorFlow and Keras

The purpose of keeping both versions together is to show the progression from a standard benchmark classification task to building and training a custom image classifier on a separate image dataset.

---

# Version 1: CIFAR 10 Image Classification with PyTorch

## Overview

The first version implements a custom Convolutional Neural Network using PyTorch.

The model is trained on the CIFAR 10 dataset and predicts one of 10 image categories:

```text
plane
car
bird
cat
deer
dog
frog
horse
ship
truck
```

The recorded CIFAR 10 test accuracy was:

```text
67.91%
```

## Dataset

The model uses the CIFAR 10 dataset through `torchvision.datasets.CIFAR10`.

Each image has the shape:

```text
3 × 32 × 32
```

The data is normalized using:

```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])
```

Training and test images are loaded in batches of 64.

## CNN Architecture

The PyTorch model contains two convolutional layers followed by three fully connected layers.

```text
Input
3 × 32 × 32
    ↓
Conv2D
3 → 12 channels
    ↓
ReLU
    ↓
Max Pooling
    ↓
Conv2D
12 → 24 channels
    ↓
ReLU
    ↓
Max Pooling
    ↓
Flatten
    ↓
Linear
600 → 120
    ↓
ReLU
    ↓
Linear
120 → 84
    ↓
ReLU
    ↓
Linear
84 → 10 classes
```

The model is implemented manually using:

```python
nn.Conv2d
nn.MaxPool2d
nn.Linear
torch.nn.functional.relu
```

## Training

The loss function is:

```python
nn.CrossEntropyLoss()
```

The optimizer is Stochastic Gradient Descent:

```python
optim.SGD(
    net.parameters(),
    lr=0.001,
    momentum=0.9
)
```

Training configuration:

```text
Epochs: 50
Batch Size: 64
Learning Rate: 0.001
Momentum: 0.9
```

The recorded loss decreased from approximately:

```text
Epoch 0:  2.2989
Epoch 10: 1.2017
Epoch 20: 0.9130
Epoch 30: 0.7290
Epoch 40: 0.5813
Epoch 49: 0.4643
```

## Evaluation

The trained CNN achieved:

```text
Test Accuracy: 67.91%
```

The network was switched to evaluation mode using:

```python
net.eval()
```

and predictions were generated without gradient calculations.

## Saving the Model

The trained weights are stored in:

```text
trained_net.pth
```

using:

```python
torch.save(
    net.state_dict(),
    "trained_net.pth"
)
```

## Custom Image Prediction

The first version also supports classification of external images.

Images are resized to the CIFAR 10 input size:

```text
32 × 32
```

and passed through the trained model.

Recorded custom predictions included:

```text
image3.jpg → truck
image4.jpg → frog
```

---

# Version 2: Happy vs Sad Image Classification with TensorFlow

## Overview

The second version moves from a predefined multiclass benchmark dataset to a custom binary image classification problem.

The objective is to classify an image as:

```text
Happy
Sad
```

This version is implemented using TensorFlow and Keras.

The dataset loaded by the notebook contains:

```text
266 images
2 classes
```

stored in:

```text
data/
    happy_images/
    sad_images/
```

## Data Validation

Before training, the notebook checks image files and removes unsupported image formats.

Supported formats include:

```text
jpeg
jpg
bmp
png
```

OpenCV and Python image utilities are used during this stage.

## Loading the Dataset

The dataset is loaded with:

```python
tf.keras.utils.image_dataset_from_directory(
    "data"
)
```

The generated image batches have the shape:

```text
32 × 256 × 256 × 3
```

representing:

```text
Batch Size: 32
Image Width: 256
Image Height: 256
Color Channels: 3
```

## Image Preprocessing

Pixel values are scaled from the original 0 to 255 range to approximately 0 to 1:

```python
data = data.map(
    lambda x, y: (x / 255, y)
)
```

## Train, Validation and Test Split

The dataset is divided approximately into:

```text
70% Training
20% Validation
10% Testing
```

using TensorFlow dataset operations:

```python
train = data.take(train_size)

val = data.skip(train_size).take(val_size)

test = data.skip(
    train_size + val_size
).take(test_size)
```

## CNN Architecture

The second version uses a deeper TensorFlow/Keras CNN.

```text
Input
256 × 256 × 3
      ↓
Conv2D
16 filters
3 × 3 kernel
      ↓
ReLU
      ↓
Max Pooling
      ↓
Conv2D
32 filters
3 × 3 kernel
      ↓
ReLU
      ↓
Max Pooling
      ↓
Conv2D
16 filters
3 × 3 kernel
      ↓
ReLU
      ↓
Max Pooling
      ↓
Flatten
      ↓
Dense
256 neurons
      ↓
ReLU
      ↓
Dense
1 neuron
      ↓
Sigmoid
```

The recorded model summary contains:

```text
Total Parameters: 3,696,625
Trainable Parameters: 3,696,625
```

## Model Compilation

The model uses the Adam optimizer:

```python
model.compile(
    "adam",
    loss=tf.losses.BinaryCrossentropy(),
    metrics=["accuracy"]
)
```

Training configuration:

```text
Optimizer: Adam
Loss: Binary Cross Entropy
Metric: Accuracy
Requested Epochs: 20
```

The notebook's saved execution output shows training progress through the first several epochs.

For example:

```text
Epoch 1
Training Accuracy:   0.5365
Validation Accuracy: 0.6406

Epoch 3
Training Accuracy:   0.6927
Validation Accuracy: 0.7031

Epoch 4
Training Accuracy:   0.6927
Validation Accuracy: 0.8125

Epoch 5
Training Accuracy:   0.8125
Validation Accuracy: 0.7969
```

## TensorBoard Logging

The second version introduces TensorBoard logging.

```python
tensorboard_callback = (
    tf.keras.callbacks.TensorBoard(
        log_dir="logs"
    )
)
```

The repository includes TensorBoard event files generated during the experiments.

This allows training behaviour to be inspected visually, including metrics such as loss and accuracy.

## Evaluation

The model is evaluated using:

```python
Precision
Recall
BinaryAccuracy
```

The recorded notebook output for its test split was:

```text
Precision: 1.0
Recall:    1.0
Accuracy:  1.0
```

These results were produced on the project's small test subset, so they should be interpreted as an experimental result rather than evidence that the model will achieve perfect accuracy on a larger independent dataset.

## Predicting a New Image

A custom image is loaded using OpenCV:

```python
img = cv2.imread(
    "test_im_s.jpg"
)
```

It is resized to:

```text
256 × 256
```

and normalized before prediction.

```python
resize = tf.image.resize(
    img,
    (256, 256)
)

yhat = model.predict(
    np.expand_dims(
        resize / 255,
        0
    )
)
```

The model returned:

```text
0.5768254
```

Using the decision threshold:

```python
if yhat > 0.5:
    print("Predicted class is Sad")
else:
    print("Predicted class is Happy")
```

the image was classified as:

```text
Sad
```

## Saving and Reloading the Model

Unlike Version 1, which stores only the PyTorch state dictionary, Version 2 saves the complete Keras model.

```python
model.save(
    "models/happysadmodel.h5"
)
```

The saved model can be reloaded using:

```python
from tensorflow.keras.models import load_model

new_model = load_model(
    "models/happysadmodel.h5"
)
```

The reloaded model successfully reproduced the Sad prediction in the notebook.

The repository includes:

```text
happysadmodel.h5
```

---

# Version Comparison

| Feature | Version 1 | Version 2 |
| --- | --- | --- |
| Framework | PyTorch | TensorFlow / Keras |
| Task | Multiclass classification | Binary classification |
| Dataset | CIFAR 10 | Custom Happy / Sad dataset |
| Classes | 10 | 2 |
| Input Size | 32 × 32 | 256 × 256 |
| Convolution Layers | 2 | 3 |
| Output Layer | 10 neurons | 1 sigmoid neuron |
| Loss | Cross Entropy | Binary Cross Entropy |
| Optimizer | SGD with Momentum | Adam |
| Epochs | 50 | Configured for 20 |
| Test Metric | Accuracy | Precision, Recall, Accuracy |
| Recorded Test Result | 67.91% accuracy | 1.0 precision, recall and accuracy on the notebook test split |
| Saved Model | `trained_net.pth` | `happysadmodel.h5` |
| Training Logs | Console output | TensorBoard |

The accuracy values between the two versions are **not directly comparable** because the models solve different classification problems using different datasets and test sets.

---

# Project Evolution

The project progression can be summarized as:

```text
CIFAR 10
Multiclass Classification
        ↓
Custom PyTorch CNN
        ↓
Manual Training Loop
        ↓
Model Evaluation
        ↓
External Image Prediction
        ↓
Saved PyTorch Weights
        ↓
Custom Image Dataset
        ↓
Binary Classification
        ↓
TensorFlow / Keras CNN
        ↓
Train / Validation / Test Split
        ↓
TensorBoard Logging
        ↓
Precision and Recall Evaluation
        ↓
Saved Full Keras Model
```

Version 1 focused on understanding the fundamentals of CNN training in PyTorch.

Version 2 extended the project by working with a custom dataset, higher resolution images, a different deep learning framework, validation data, TensorBoard monitoring, additional evaluation metrics, and full model serialization.

---

# Technologies Used

Across both versions:

```text
Python
PyTorch
Torchvision
TensorFlow
Keras
NumPy
OpenCV
Pillow
Matplotlib
Jupyter Notebook
TensorBoard
```

---

# Skills Demonstrated

This project demonstrates practical experience with:

1. Image classification
2. Binary classification
3. Multiclass classification
4. Convolutional Neural Networks
5. Image preprocessing
6. Image normalization
7. Image resizing
8. Convolutional layers
9. ReLU activation
10. Max pooling
11. Dense neural network layers
12. Sigmoid output layers
13. Cross Entropy Loss
14. Binary Cross Entropy
15. SGD optimization
16. Momentum
17. Adam optimization
18. Backpropagation
19. Mini batch training
20. Train, validation and test splitting
21. Precision and recall
22. Model accuracy evaluation
23. TensorBoard logging
24. Model checkpoint saving
25. Model reloading
26. Inference on external images
27. PyTorch
28. TensorFlow and Keras

---

# Suggested Repository Structure

```text
image-classification-project/
│
├── README.md
│
├── version_1_pytorch/
│   ├── image_class.ipynb
│   ├── trained_net.pth
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   └── image4.jpg
│
└── version_2_tensorflow/
    ├── image_class2.0.ipynb
    ├── happysadmodel.h5
    ├── data/
    │   ├── happy_images/
    │   └── sad_images/
    └── logs/
```

The Version 2 notebook also references:

```text
test_im_s.jpg
```

for custom inference. Add that image to the repository if you want the notebook to run from beginning to end without changing the example prediction cell.

---

# Running Version 1

Install the required packages:

```bash
pip install torch torchvision numpy pillow jupyter
```

Open:

```text
image_class.ipynb
```

and run the notebook cells.

---

# Running Version 2

Install the required packages:

```bash
pip install tensorflow numpy opencv-python matplotlib jupyter
```

Arrange the custom dataset as:

```text
data/
    happy_images/
    sad_images/
```

Then open:

```text
image_class2.0.ipynb
```

and run the cells.

To inspect TensorBoard logs:

```bash
tensorboard --logdir logs
```

---

# Possible Improvements

## Version 1

Possible improvements include:

1. Add data augmentation.
2. Add Batch Normalization.
3. Add Dropout.
4. Use a validation set.
5. Track per class accuracy.
6. Add a confusion matrix.
7. Compare SGD with Adam.
8. Experiment with a deeper CNN.
9. Use GPU acceleration.
10. Compare the custom CNN with transfer learning.

## Version 2

Possible improvements include:

1. Increase the number of training images.
2. Create deterministic train, validation and test splits before shuffling.
3. Keep a larger independent test set.
4. Add data augmentation.
5. Add Dropout to reduce overfitting.
6. Add Batch Normalization.
7. Track confusion matrix and F1 score.
8. Add EarlyStopping.
9. Add ModelCheckpoint.
10. Save the model in the modern `.keras` format instead of legacy HDF5.
11. Plot both accuracy and loss curves.
12. Add prediction confidence to inference output.
13. Compare the custom CNN with transfer learning models such as MobileNet or EfficientNet.

---

# Conclusion

This repository shows the progression of an image classification project across two deep learning frameworks and two different computer vision problems.

**Version 1** established the fundamentals using a PyTorch CNN trained on CIFAR 10 and achieved:

```text
67.91% test accuracy
```

**Version 2** moved to TensorFlow/Keras and a custom Happy vs Sad image dataset, adding validation data, TensorBoard logging, binary classification metrics, custom image inference, and full model saving.

Together, the two versions demonstrate the evolution from a standard benchmark classifier to a more complete custom computer vision workflow.
