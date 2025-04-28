import os

import numpy as np
import idx2numpy

np.set_printoptions(suppress=True)


class NeuralNetwork:
    def __init__(self, n, m, a):
        self.inputs = np.random.rand(n, m)
        self.weights_layers = []
        self.convolution_layers = []
        self.kernels = np.array([])
        self.alpha = a
        self.training = False

    def print_inputs(self):
        print(self.inputs)
        print("\n")

    def print_weights(self):
        print(self.weights_layers)
        print("\n")

    @staticmethod
    def neural_network(input, weights):
        sum = np.matmul(weights, input)
        return sum

    @staticmethod
    def layer_error(prediction, goal):
        neurons_number = len(prediction)
        sum = 0
        for i in range(0, neurons_number):
            sum += (prediction[i] - goal[i]) ** 2

        return (1 / neurons_number) * sum

    @staticmethod
    def layer_delta(prediction, goal, x):
        neurons_number = len(prediction)

        return (2 * (1 / neurons_number)) * (np.outer(np.subtract(prediction, goal), x))

    def add_layer(self, n, m, weight_min_value, weight_max_value, activation_function):
        if len(self.weights_layers) == 0:
            if m == 0:
                m = len(self.inputs)
            new_layer = np.random.uniform(weight_min_value, weight_max_value, (n, m))
        else:
            if m == 0:
                m = len(self.weights_layers[len(self.weights_layers) - 1])
            new_layer = np.random.uniform(weight_min_value, weight_max_value, (n, m))
        self.weights_layers.append(new_layer)

    def predict(self, inputs):
        sum = self.neural_network(inputs, self.weights_layers[0])
        sum = self.layer_relu_tanh(sum)

        if self.training:
            dropout = np.array([self.dropout(len(sum), 0.5)]).T
            sum = np.multiply(dropout, sum)
            sum *= 1 / (1 - 0.5)

        for i in range(1, len(self.weights_layers) - 1):
            sum = self.neural_network(sum, self.weights_layers[i])
            sum = self.layer_relu_tanh(sum)

            if self.training:
                dropout = np.array([self.dropout(len(sum), 0.5)]).T
                sum = np.multiply(dropout, sum)
                sum *= 1 / (1 - 0.5)

        sum = self.neural_network(sum, self.weights_layers[-1])
        sum = self.layer_softmax(sum)

        return sum

    @staticmethod
    def layer_relu(output):
        return np.maximum(0, output)

    @staticmethod
    def layer_relu_deriv(output):
        return np.where(output > 0, 1, 0)

    @staticmethod
    def layer_relu_sigmoid(output):
        return 1 / (1 + np.exp(-output))

    @staticmethod
    def layer_relu_sigmoid_deriv(output):
        return np.multiply(output, 1 - output)

    @staticmethod
    def layer_relu_tanh(output):
        return np.tanh(output)

    @staticmethod
    def layer_relu_tanh_deriv(output):
        return 1 - np.power(np.tanh(output), 2)

    @staticmethod
    def layer_softmax(output):
        return np.exp(output) / np.sum(np.exp(output))

    def dropout(self, size, neurons_percentage):
        vector = np.zeros(size)
        for i in range(0, int(len(vector) * neurons_percentage)):
            index = np.random.randint(len(vector))
            while vector[index] == 1:
                index = np.random.randint(len(vector))
            vector[index] = 1

        return vector

    def back_propagation(self, inputs, expected_outputs, batch_size):
        num_samples = inputs.shape[1]

        layer_outputs = [inputs]
        layer_inputs = [inputs]

        # Forward pass
        for i in range(len(self.weights_layers)):
            layer_input = self.neural_network(layer_outputs[-1], self.weights_layers[i])
            layer_output = self.layer_relu_tanh(layer_input)
            layer_outputs.append(layer_output)
            layer_inputs.append(layer_input)

        # Backward pass
        deltas = [(2 * (1 / num_samples) * np.subtract(layer_outputs[-1], expected_outputs.T)) / batch_size]
        for i in range(len(self.weights_layers) - 1, 0, -1):
            delta = np.matmul(self.weights_layers[i].T, deltas[0])
            delta = np.multiply(delta, self.layer_relu_tanh_deriv(layer_inputs[i]))
            deltas.insert(0, delta)

        # Update weights
        for i in range(len(self.weights_layers) - 1, -1, -1):
            weight_delta = np.matmul(deltas[i], layer_outputs[i].T)
            self.weights_layers[i] = np.subtract(self.weights_layers[i], np.multiply(self.alpha, weight_delta))

    def fit(self, input, expected_output):
        self.back_propagation(input, expected_output, 1)

    def mini_batch_gradient_descent(self, inputs, expected_outputs, batch_size):
        num_samples = inputs.shape[1]

        for i in range(0, num_samples, batch_size):
            input_batch = inputs[:, i:i + batch_size]
            output_batch = expected_outputs[:, i:i + batch_size]
            self.back_propagation(input_batch, output_batch, batch_size)

    @staticmethod
    def convolve(image, kernel, step, padding):
        image_height, image_width = image.shape
        kernel_height, kernel_width = kernel.shape

        output_height = (image_height - kernel_height + 2 * padding) // step + 1
        output_width = (image_width - kernel_width + 2 * padding) // step + 1

        output = np.zeros((output_height, output_width))

        padded_image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')

        # Perform convolution
        for i in range(0, output_height):
            for j in range(0, output_width):
                roi = padded_image[i * step:i * step + kernel_height, j * step:j * step + kernel_width]

                output[i, j] = np.sum(roi * kernel)

        return output

    def add_convolution_layer(self, kernels_number, kernels_x, kernels_y, weight_min_value, weight_max_value):
        for i in range(0, kernels_number):
            kernel_weights = np.random.uniform(weight_min_value, weight_max_value, (kernels_x, kernels_y))
            self.convolution_layers.append(kernel_weights)

        self.add_kernels()

    def add_kernels(self):
        new_kernels = [conv_layer.flatten() for conv_layer in self.convolution_layers]
        if len(self.kernels) == 0:
            self.kernels = np.array(new_kernels)
        else:
            self.kernels = np.concatenate((self.kernels, new_kernels), axis=0)

    @staticmethod
    def partition_image(image, mask_size, step=1):
        square_size = int(np.sqrt(mask_size))
        cut_size = square_size // 2

        start_x, end_x = cut_size, image.shape[1] - cut_size
        start_y, end_y = cut_size, image.shape[0] - cut_size
        cut_image = np.array(
            [
                image[
                row - cut_size: row + cut_size + 1,
                col - cut_size: col + cut_size + 1,
                ]
                for row in range(start_y, end_y, step)
                for col in range(start_x, end_x, step)
            ]
        )
        cut_image = cut_image.reshape(cut_image.shape[0], mask_size)

        return cut_image

    @staticmethod
    def max_pooling(input_image, mask_size=2, step=2):
        image_height, image_width = input_image.shape
        end_x = image_width - mask_size + 1
        end_y = image_height - mask_size + 1

        cut_image = np.zeros((image_height // 2, image_width // 2))
        binary_image = np.zeros(input_image.shape)

        for row in range(0, end_y, step):
            for col in range(0, end_x, step):
                smaller_image = input_image[
                                row: row + mask_size, col: col + mask_size
                                ]
                max_value = np.max(smaller_image)

                if max_value == 0:
                    continue

                cut_image[row // 2, col // 2] = max_value
                binary_image[row: row + mask_size, col: col + mask_size][
                    smaller_image == max_value
                    ] = 1

        return cut_image, binary_image

    def pooling_convolution_fit(self, image_sections, expected_output):
        kernel_layer = np.matmul(image_sections, nn.kernels.T)
        kernel_layer = self.layer_relu(kernel_layer)

        binary_image = []
        pooled_image = []

        for i in range(0, kernel_layer.shape[1]):
            # one pooled column, next column of kernel_layer image
            pooled_image_column, binary_kernel_layer = nn.max_pooling(np.array(kernel_layer[:, i:i + 1].reshape(26, 26)))

            pooled_image.append(pooled_image_column)
            binary_image.append(binary_kernel_layer.flatten())

        # transposing the array to have vertical column
        pooled_image = np.vstack(pooled_image).T
        binary_kernel_layer = np.vstack(binary_image).T

        # flattening the pooled image
        flatten_pooled_image = pooled_image.flatten()[:, np.newaxis]

        # output
        layer_output = np.matmul(self.weights_layers[0], flatten_pooled_image)
        layer_output = self.layer_softmax(layer_output)

        # delta of output layer
        layer_output_delta = 2 * (1 / len(layer_output)) * np.subtract(layer_output, expected_output)

        # filters' deltas
        kernel_layer_1_delta = np.matmul(self.weights_layers[0].T, layer_output_delta)
        kernel_layer_1_delta = np.repeat(kernel_layer_1_delta, 2**2)

        kernel_layer_1_delta_reshaped = kernel_layer_1_delta.reshape(kernel_layer.shape)
        kernel_layer_1_delta_reshaped *= binary_kernel_layer

        kernel_layer_1_delta_reshaped *= self.layer_relu_deriv(kernel_layer)

        # output weight delta
        layer_output_weight_delta = np.matmul(layer_output_delta, flatten_pooled_image.T)

        # kernel weight delta
        kernel_layer_1_weight_delta = np.matmul(kernel_layer_1_delta_reshaped.T, image_sections)

        # updating weights of neurons
        self.weights_layers[0] = np.subtract(self.weights_layers[0], np.multiply(self.alpha, layer_output_weight_delta))

        self.kernels = np.subtract(self.kernels, np.multiply(self.alpha, kernel_layer_1_weight_delta))


    def convolution_fit(self, image_sections, expected_output):
        kernel_layer = np.matmul(image_sections, self.kernels.T)
        kernel_layer = self.layer_relu(kernel_layer)

        # filters flatten transformed
        kernel_layer_flatten = np.array([kernel_layer.flatten()]).T

        # output
        layer_output = np.matmul(self.weights_layers[0], kernel_layer_flatten)

        # delta of output layer
        layer_output_delta = 2 * (1 / len(image_sections)) * np.subtract(layer_output, expected_output)

        # filters' deltas
        kernel_layer_1_delta = np.matmul(self.weights_layers[0].T, layer_output_delta)
        kernel_layer_1_delta *= self.layer_relu_deriv(kernel_layer_flatten)

        kernel_layer_1_delta_reshaped = kernel_layer_1_delta.reshape(kernel_layer.shape[0], kernel_layer.shape[1])

        # output weight delta
        layer_output_weight_delta = np.matmul(layer_output_delta, kernel_layer_flatten.T)

        # kernel weight delta
        kernel_layer_1_weight_delta = np.matmul(kernel_layer_1_delta_reshaped.T, image_sections)

        # updating weights of neurons
        self.weights_layers[0] = np.subtract(self.weights_layers[0], np.multiply(self.alpha, layer_output_weight_delta))

        self.kernels = np.subtract(self.kernels, np.multiply(self.alpha, kernel_layer_1_weight_delta))

    def save_weights(self, file_name):
        with open(file_name, 'w') as file:
            for layer_weights in self.weights_layers:
                np.savetxt(file, layer_weights, delimiter=',', fmt='%f')
                file.write('\n')

    def load_weights(self, file_name):
        if not os.path.exists(file_name):
            print(f"Error: File {file_name} not found.")
            return

        with open(file_name, 'r') as file:
            self.weights_layers = []
            lines = file.readlines()
            i = 0
            while i < len(lines):
                layer_weights = []
                while i < len(lines) and lines[i].strip():
                    weights_row = [float(val) for val in lines[i].split(',')]
                    layer_weights.append(weights_row)
                    i += 1
                self.weights_layers.append(np.array(layer_weights))
                i += 1


# column I = 0
# column II = 1
# column III = 2
# column IV = 3
# column V = 4
# column VI = 5
# column VII = 6
# column VIII = 7
# column IX = 8
# column X = 9
y = np.array([
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]
])

image_train_path = "./MNIST_ORG/train-images.idx3-ubyte"
label_train_path = "./MNIST_ORG/train-labels.idx1-ubyte"

images_train = idx2numpy.convert_from_file(image_train_path)
labels_train = idx2numpy.convert_from_file(label_train_path)

nn = NeuralNetwork(784, 1, 0.01)
nn.add_convolution_layer(16, 3, 3, -0.01, 0.01)

# partitioning the image
input_image = np.array(images_train[0]) / 255.0
image_sections = nn.partition_image(input_image, 3 ** 2, 1)

# this is for getting the number of output layer's weights columns
kernel_layer = np.matmul(image_sections, nn.kernels.T)
kernel_layer = nn.layer_relu(kernel_layer)

# print(kernel_layer.shape)

pooled_image = []
for i in range(0, kernel_layer.shape[1]):
    # one pooled column, next column of kernel_layer image
    pooled_image_column, _ = nn.max_pooling(np.array(kernel_layer[:, i:i+1].reshape(26, 26)))
    pooled_image.append(pooled_image_column)

# transposing the array to have vertical column
pooled_image = np.vstack(pooled_image).T

# flattening the pooled image
flatten_pooled_image = pooled_image.flatten()[:, np.newaxis]

nn.add_layer(10, len(flatten_pooled_image), -0.1, 0.1, 0)

# training
for i in range(0, 10):
    sum = 0
    for j in range(0, len(labels_train[:1000])):
        input = np.array(images_train[j]) / 255.0
        image_sections = nn.partition_image(input, 3 ** 2, 1)

        kernel_layer = np.matmul(image_sections, nn.kernels.T)
        kernel_layer = nn.layer_relu(kernel_layer)

        pooled_image = []

        for k in range(0, kernel_layer.shape[1]):
            # one pooled column, next column of kernel_layer image
            pooled_image_column, _ = nn.max_pooling(np.array(kernel_layer[:, k:k + 1].reshape(26, 26)))

            pooled_image.append(pooled_image_column)

        # transposing the array to have vertical column
        pooled_image = np.vstack(pooled_image).T

        # flattening the pooled image
        flatten_pooled_image = pooled_image.flatten()[:, np.newaxis]

        # output
        layer_output = np.matmul(nn.weights_layers[0], flatten_pooled_image)

        y_from = labels_train[j]
        y_to = labels_train[j] + 1

        if np.argmax(layer_output) == labels_train[j]:
            sum += 1

        nn.pooling_convolution_fit(image_sections, y[:, y_from:y_to])

    print("Epoch: " + str(i) + ", Accurancy: " + str((sum / len(labels_train[:1000])) * 100) + "%")


images_test_path = "./MNIST_ORG/t10k-images.idx3-ubyte"
labels_test_path = "./MNIST_ORG/t10k-labels.idx1-ubyte"

images_test = idx2numpy.convert_from_file(images_test_path)
labels_test = idx2numpy.convert_from_file(labels_test_path)

sum = 0
for i in range(0, len(labels_test[:1000])):
    input = np.array(images_test[i]) / 255.0
    image_sections = nn.partition_image(input, 3 ** 2, 1)

    kernel_layer = np.matmul(image_sections, nn.kernels.T)
    kernel_layer = nn.layer_relu(kernel_layer)

    pooled_image = []

    for k in range(0, kernel_layer.shape[1]):
        # one pooled column, next column of kernel_layer image
        pooled_image_column, _ = nn.max_pooling(np.array(kernel_layer[:, k:k + 1].reshape(26, 26)))

        pooled_image.append(pooled_image_column)

    # transposing the array to have vertical column
    pooled_image = np.vstack(pooled_image).T

    # flattening the pooled image
    flatten_pooled_image = pooled_image.flatten()[:, np.newaxis]

    # output
    layer_output = np.matmul(nn.weights_layers[0], flatten_pooled_image)

    if np.argmax(layer_output) == labels_test[i]:
        sum += 1

print("Test accurancy: " + str((sum / len(labels_train[:1000])) * 100) + "%")