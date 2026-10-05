# CIFAR 10 Image Classification with PyTorch

## Project Overview

This project implements an image classification system using a custom Convolutional Neural Network built with PyTorch.

The model is trained on the CIFAR 10 dataset to classify images into 10 categories. After training, the network is evaluated on the CIFAR 10 test set and then used to classify external images.

The final recorded test accuracy is:

```text
67.91%
```

The trained model is saved as:

```text
trained_net.pth
```

## Image Classes

The network predicts one of the following 10 CIFAR 10 classes:

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

## Project Workflow

```text
Load CIFAR 10
      ↓
Normalize Images
      ↓
Create DataLoaders
      ↓
Build CNN Architecture
      ↓
Define Loss and Optimizer
      ↓
Train for 50 Epochs
      ↓
Save Model Weights
      ↓
Reload Trained Model
      ↓
Evaluate Test Accuracy
      ↓
Predict Custom Images
```

## Dataset

The project uses the CIFAR 10 dataset through `torchvision.datasets.CIFAR10`.

Each image has the shape:

```text
3 × 32 × 32
```

representing three color channels and a 32 by 32 pixel image.

Training and test datasets are loaded separately:

```python
train_data = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)
```

## Image Preprocessing

Images are converted to tensors and normalized:

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])
```

## DataLoaders

The project uses a batch size of 64. Training data is shuffled while test data is not.

```python
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=64,
    shuffle=True,
    num_workers=0
)
```

## Neural Network Architecture

A custom CNN is defined with two convolutional layers, max pooling, and three fully connected layers:

```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 24, 5)

        self.fc1 = nn.Linear(24 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
```

Architecture summary:

```text
Input: 3 × 32 × 32
      ↓
Conv2D: 3 → 12
      ↓
ReLU
      ↓
Max Pooling
      ↓
Conv2D: 12 → 24
      ↓
ReLU
      ↓
Max Pooling
      ↓
Flatten
      ↓
Linear: 600 → 120
      ↓
ReLU
      ↓
Linear: 120 → 84
      ↓
ReLU
      ↓
Linear: 84 → 10
```

## Loss Function

The project uses multiclass Cross Entropy Loss:

```python
loss_function = nn.CrossEntropyLoss()
```

## Optimizer

Training uses Stochastic Gradient Descent with momentum:

```python
optimizer = optim.SGD(
    net.parameters(),
    lr=0.001,
    momentum=0.9
)
```

Training configuration:

```text
Optimizer: SGD
Learning Rate: 0.001
Momentum: 0.9
Batch Size: 64
Epochs: 50
```

## Training

The network is trained for 50 epochs using the standard PyTorch training cycle:

```text
Forward Pass
      ↓
Loss Calculation
      ↓
Backpropagation
      ↓
Parameter Update
```

The training loop uses:

```python
for epoch in range(50):
    running_loss = 0.0

    for i, data in enumerate(train_loader):
        inputs, labels = data

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
```

## Training Progress

The recorded loss decreased steadily:

```text
Epoch 0     Loss: 2.2989
Epoch 10    Loss: 1.2017
Epoch 20    Loss: 0.9130
Epoch 30    Loss: 0.7290
Epoch 40    Loss: 0.5813
Epoch 49    Loss: 0.4643
```

## Saving and Loading the Model

The trained model parameters are saved with:

```python
torch.save(net.state_dict(), "trained_net.pth")
```

They can be loaded again with:

```python
net = NeuralNet()
net.load_state_dict(torch.load("trained_net.pth"))
```

## Model Evaluation

The network is evaluated in inference mode using the test dataset:

```python
net.eval()

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
```

The final recorded test accuracy is:

```text
67.91%
```

## Custom Image Classification

The project also performs inference on external images.

Custom images are resized to 32 by 32 pixels and normalized using the same statistics as the training data:

```python
new_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])
```

A helper function adds the required batch dimension:

```python
def load_image(image_path):
    image = Image.open(image_path)
    image = new_transform(image)
    image = image.unsqueeze(0)
    return image
```

## Recorded Custom Predictions

The notebook records the following predictions:

```text
image3.jpg → truck
image4.jpg → frog
```

The repository also includes additional custom images that can be passed through the same inference pipeline.

## Project Structure

```text
image-classification/
│
├── image_class.ipynb
├── trained_net.pth
├── image1.jpg
├── image2.jpg
├── image3.jpg
├── image4.jpg
└── README.md
```

## Technologies Used

```text
Python
PyTorch
Torchvision
NumPy
Pillow
Jupyter Notebook
```

## Deep Learning Concepts Demonstrated

This project demonstrates practical experience with:

1. Image classification
2. Convolutional Neural Networks
3. Convolutional layers
4. ReLU activation
5. Max pooling
6. Flattening feature maps
7. Fully connected layers
8. Multiclass classification
9. Cross Entropy Loss
10. Stochastic Gradient Descent
11. Momentum
12. Mini batch training
13. Backpropagation
14. Image normalization
15. Model evaluation
16. Saving and loading model parameters
17. Inference on unseen images
18. PyTorch DataLoaders

## How to Run

Install the required packages:

```bash
pip install torch torchvision numpy pillow jupyter
```

Place the project files in the same directory and start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
image_class.ipynb
```

and run the cells in order.

## Possible Improvements

Future improvements could include:

1. Add a validation dataset during training.
2. Track training and validation accuracy after every epoch.
3. Plot training and validation loss curves.
4. Add data augmentation such as random cropping and horizontal flipping.
5. Compare SGD with Adam.
6. Add more convolutional layers.
7. Add Batch Normalization.
8. Add Dropout to reduce overfitting.
9. Use a learning rate scheduler.
10. Train using GPU acceleration when available.
11. Add per class accuracy.
12. Generate a confusion matrix.
13. Calculate precision, recall, and F1 score.
14. Display prediction confidence for custom images.
15. Compare the custom CNN with transfer learning models.

## Conclusion

This project demonstrates an end to end computer vision workflow using PyTorch.

A Convolutional Neural Network was built and trained on CIFAR 10, then saved, reloaded, evaluated, and used to classify external images.

The training loss decreased from approximately:

```text
2.2989
```

to:

```text
0.4643
```

over 50 epochs.

The final recorded CIFAR 10 test accuracy was:

```text
67.91%
```
