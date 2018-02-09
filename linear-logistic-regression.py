import NeuralNetworkMBGD as ANN # stands for Artificial Neural Network
import numpy as np
import matplotlib.pyplot as plt
import random 
# 3D attempt
from mpl_toolkits.mplot3d import Axes3D

#####
#
# A logistic regression on (almost) linearly separable data
#
#####
NUMBER_OF_DATA_POINTS = 500

# Initialize the pseudorandom data, modelled in this case after the function y=x1 + x2
# Our neural network's realistic goal is to guess the output as accurately as possible given the input
# Since we will blur our data a little bit with noise, data_accuracy is the percentage of data points 
# whose noise did not affect the resulting category they should have been classified in
input_data = np.array([[random.uniform(-5,5), random.uniform(-5,5)] for _ in range(NUMBER_OF_DATA_POINTS)], dtype='float')
output_data = np.array([[ANN.threshold(np.sum(input_data[i]) + random.normalvariate(0,0.5), 0)] for i in range(NUMBER_OF_DATA_POINTS)])
model_data = np.array([[ANN.threshold(np.sum(input_data[i]), 0)] for i in range(NUMBER_OF_DATA_POINTS)])
data_accuracy = 1 - ANN.norm_squared( output_data - model_data )/NUMBER_OF_DATA_POINTS 

# Initialize the network
neural_net = ANN.Neural_Network(input_scales = np.float_([[-5,5],[-5,5]]),
                                output_scales = np.float_([[0,1]]),
                                output_layer_activation_function = "sigmoid",
                                hidden_layers_sizes = [20],
                                hidden_layers_activation_functions = ["sigmoid"],
                                biases=[True, False])


# Train the network with our pseudo-random data
training_cost, validation_cost, accuracy = neural_net.train_network(input_data,
                                                                     output_data, 
                                                                     training_split = 0.8, 
                                                                     validation_split = 0.2,
                                                                     test_split = 0,
                                                                     batch_size_split = 0.25,
                                                                     learning_rate = 0.1,
                                                                     number_of_epochs = 10000,
                                                                     display_performance_frequency = 1000,
                                                                     data_accuracy = data_accuracy)

# Show some statistics
plt.plot(training_cost, label="Training")
plt.plot(validation_cost, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.legend()
plt.show()

plt.plot(accuracy, label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")
x,y = list(list(i) for i in input_data.transpose())
ax.scatter(x, y, list(output_data.flatten()), c="r", marker="o") 
ax.scatter(x, y, list(neural_net.run_network(input_data).flatten()), c="b", marker="o") 
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")
plt.show()
