# Neural Network with Convolutional Layers

This project implements a neural network with convolutional layers for image classification, specifically designed to work with the MNIST dataset. The implementation includes features like convolution, pooling, and backpropagation.

## Features

- Convolutional Neural Network (CNN) implementation
- Support for multiple convolution layers
- Max pooling implementation
- ReLU and Tanh activation functions
- Softmax output layer
- Dropout regularization
- Mini-batch gradient descent
- Weight saving and loading functionality

## Requirements

- Python 3.x
- NumPy
- idx2numpy (for MNIST dataset handling)

## Installation

1. Clone the repository
2. Install the required packages:
```bash
pip install numpy idx2numpy
```

## Project Structure

- `main.py`: Contains the main neural network implementation
- `MNIST_ORG/`: Directory containing MNIST dataset files
  - `train-images.idx3-ubyte`: Training images
  - `train-labels.idx1-ubyte`: Training labels
  - `t10k-images.idx3-ubyte`: Test images
  - `t10k-labels.idx1-ubyte`: Test labels

## Usage

The neural network is designed to work with the MNIST dataset. Here's a basic example of how to use it:

```python
# Initialize the neural network
nn = NeuralNetwork(784, 1, 0.01)  # input_size, batch_size, learning_rate

# Add a convolution layer
nn.add_convolution_layer(16, 3, 3, -0.01, 0.01)  # num_kernels, kernel_size_x, kernel_size_y, min_weight, max_weight

# Add a fully connected layer
nn.add_layer(10, len(flatten_pooled_image), -0.1, 0.1, 0)  # output_size, input_size, min_weight, max_weight, activation_function

# Train the network
# The code includes example training loops for the MNIST dataset
```

## Key Components

### Neural Network Class

The `NeuralNetwork` class provides the following main functionalities:

- **Initialization**: Create a new neural network with specified input size and learning rate
- **Layer Management**: Add convolution and fully connected layers
- **Training**: Implement backpropagation and mini-batch gradient descent
- **Prediction**: Process input data through the network
- **Weight Management**: Save and load network weights

### Convolution Operations

- Supports custom kernel sizes
- Implements padding and striding
- Includes max pooling for dimensionality reduction

### Activation Functions

- ReLU
- Tanh
- Softmax (for output layer)

## Training Process

The network is trained using:
1. Forward propagation through convolution and pooling layers
2. Backpropagation for weight updates
3. Mini-batch gradient descent for efficient training
4. Dropout for regularization

## Performance

The network is evaluated on the MNIST test dataset, with accuracy reported as a percentage of correctly classified images.

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues and enhancement requests! 

## Overview

![ai](https://github.com/user-attachments/assets/014c7618-b190-4142-aef6-7cda1311665e)
