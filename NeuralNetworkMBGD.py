import matplotlib.pyplot as plt
import numpy as np

# This example models a binary classifier, but with little modification you can turn it into a neural network for other purposes.
# To compute the cost function efficiently, it is more useful to use the following than the np.linalg.norm squared:

def norm_squared(x):
    return (x**2).sum()

def add_ones_column(x): # This is for adding bias
    return np.append( x, np.ones((np.shape(x)[0], 1)), axis=1 )

#########
#
# Threshold function
#
#########

# (Also called the Heaviside Step function, here applicable coefficient-wise to numPy arrays)
# x is the numPy array of inputs of this function
# T is the threshold
# lower_bound is the value the function takes if you are below the threshold T
# upper_bound is the value the function takes if x is above the threshold T
def threshold(x,T, lower_bound = 0, upper_bound = 1): 
    return np.where(x < T, lower_bound, upper_bound)

# Note : this is the same function as the derivative of ReLU which we also define, but we wanted to avoid weird function names.

######################
#
# Activation functions (and their derivatives)
#
######################

# Note : these functions can be applied to numpy arrays directly and are applied coefficient-wise

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_prime(z):
    # We exploit the fact that the derivative of the sigmoid satisfies
    # sigmoid' = sigmoid * (1-sigmoid)
    # to avoid computing two exponentials and potentially accelerate the algorithm
    x = sigmoid(z)
    return x*(1-x)

def relu(z): 
    return np.where(z < 0, 0, z)

def relu_prime(z): # The derivative of the ReLU function
    return np.where(z < 0, 0, 1)

def leakyrelu(z, leak = 0.01): # Leak = 0 makes this the same as ReLU
    return np.where(z < 0, leak*z, z)

def leakyrelu_prime(z,leak = 0.01): # The derivative of the Leaky ReLU function
    return np.where(z < 0, leak, 1)

#####################
#
# Layer class
#
#####################

# This defines a fully connected layer object

class Layer(object):
    def __init__(self, number_of_neurons, 
                        activation_function = None, 
                        preceding_layer = None, # Only the input layer keeps this as None
                        bias = True):
        self.number_of_neurons = number_of_neurons
        self.activate_layer = None
        self.activate_layer_prime = None
        self.bias = bias

        if preceding_layer is not None: # Or in other words, if this is not the input layer
            self.A_gradient = np.zeros(self.number_of_neurons) 
            self.preceding_layer = preceding_layer
            if self.preceding_layer.bias: # The bias node is inserted in the previous layer, 
                                          # so the weight matrix of this layer has to adjust!
                self.weights = np.random.randn(self.preceding_layer.number_of_neurons + 1, self.number_of_neurons)
                # The partial derivatives of the cost function with respect to 
                # the coefficients of the weight matrices, also written dC/dw_{ij} in the notes
                self.weights_gradient = np.zeros([self.preceding_layer.number_of_neurons + 1, self.number_of_neurons]) 
                # The partial derivatives of the cost function with respect to 
                # the values of the activated layers, also written dC/dA in the notes

            else:
                self.weights = np.random.randn(self.preceding_layer.number_of_neurons, self.number_of_neurons)
                self.weights_gradient = np.zeros([self.preceding_layer.number_of_neurons, self.number_of_neurons]) 

            if activation_function == "relu":
                self.activate_layer = relu
                self.activate_layer_prime = relu_prime
            elif activation_function == "sigmoid":
                self.activate_layer = sigmoid
                self.activate_layer_prime = sigmoid_prime
            elif activation_function == "leakyrelu":
                self.activate_layer = leakyrelu
                self.activate_layer_prime = leakyrelu_prime

    ##########
    #
    # Forward-propagation
    #
    ##########

    # Recall that np.dot is matrix multiplication
    def forward_propagation(self, input_data = None): # think of the input_data here as the input from the previous layer
        product = np.dot( input_data, self.weights ) # The ones are added when passing the input to this method when the input has bias
        return product, self.activate_layer(product)

    ##########
    #
    # Update weights 
    #
    ##########

    def update_weights(self, learning_rate):
        self.weights -= learning_rate * self.weights_gradient
        self.weights_gradient = np.zeros(np.shape(self.weights_gradient))




#############
#
# Neural Network class
#
#############

class Neural_Network(object):

    ##########
    #
    # Initialization
    #
    ##########

    # input_scales and output_scales are two lists of two floats indicating how to rescale the input and output between 0 and 1
    # hidden_layers_sizes is a list containing the number of neurons in each layer
    def __init__(self, input_scales,                                # must be a list of tuples of length the number of features in the input data
                        output_scales,                              # must be a list of tuples of length the number of features in the output data
                        output_layer_activation_function="relu",   # For the moment, relu or sigmoid
                        hidden_layers_sizes = [],                        # must be a list of integers where each entry is the number of nodes in that layer
                        hidden_layers_activation_functions = [],   # must be a list of strings, either "relu" or "sigmoid" in each layer
                        biases = [True]):                          # must be a non-empty list of booleans

        self.input_scales = input_scales
        self.output_scales = output_scales
        input_layer_size = len(input_scales)
        output_layer_size = len(output_scales)

        # The 0th element of that list is the input layer
        # The activation function None stands for the identity, it's just to say that it belongs to the layer class too
        self.layers_list = [Layer(input_layer_size, bias = biases[0])] 
        if len(hidden_layers_sizes) != len(hidden_layers_activation_functions):
            raise ValueError("The list of layers sizes must have the same length as the list of layers' activation functions")
        if len(hidden_layers_sizes)+1 != len(biases):
            raise ValueError("The list of layers sizes must have the same length as the list of layers' biases booleans")

        # Building the computation graph
        for i in range(len(hidden_layers_sizes)):
            self.layers_list.append(Layer(hidden_layers_sizes[i], hidden_layers_activation_functions[i], self.layers_list[-1], biases[i]))

        # Note that the output layer never needs bias
        self.layers_list.append(Layer(output_layer_size, output_layer_activation_function, self.layers_list[-1]))

    ##########
    #
    # Rescaling/descaling (i.e. inverting the rescale) inputs/outputs method
    #
    ##########
    
    # Data is usually inputted in the form of a data frame, but numPy arrays are lists of lists where the sublists are rows, not columns.
    # If we want to rescale each column according to its appropriate scale, it makes more sense to work with the transpose of that data frame,
    # where the columns become observations and each row is a variable, i.e. given as a list. 
    def rescale_data(self, data, 
                            bounds_array, 
                            in_reverse=False):
        # Rescale (or invert the scaling) the data between 0 and 1
        # Transposition is only required to apply the operation on rows (observations), not columns
        # Check for correct scaling format, i.e. if bounds_array is a nx2 numPy array of floats
        
        if type(bounds_array) is list:
            raise TypeError('Please convert this scale to a numPy array, such as input_scales = np.array([[x1min, x1max],[x2min,x2max]], dtype="float32")')

        if not( bounds_array.dtype is np.array([]).dtype and len(bounds_array.T) == 2 ):
            raise TypeError('The variable "bounds_array" must be a numPy array of float64s of shape (*,2)')

        # This is an artifical function to check if each lower bound is smaller than its upper bound
        elif sum(1-(bounds_array.T[0] < bounds_array.T[1])):
            raise ValueError('The lower bounds in "bounds_array" must be smaller than the upper bounds')
        
        else:
            if in_reverse:
                return (bounds_array.T[1] - bounds_array.T[0])*data + bounds_array.T[0]
            else:
                return (data - bounds_array.T[0])/(bounds_array.T[1] - bounds_array.T[0])

    ##########
    #
    # Run the network on data
    #
    ##########

    def run_network(self, input_data):
        # We rescale the input_data, pass it through the network via 
        # the forward propagation method of the layers and invert the scaling
        data = self.rescale_data(input_data, self.input_scales)
        for layer in self.layers_list[1:]:
            if layer.bias:
                data = add_ones_column(data)
            data = layer.forward_propagation(data)[1]

        return self.rescale_data(data, self.output_scales, in_reverse=True)

    ##########
    #
    # Optimizer (Mini-batch Gradient Descent, somewhere between Batch Gradient Descent (BGD) and Stochastic Gradient Descent (SGD)
    #
    ##########

    # This defines pass of the gradient descent optimizer technique for a given collection of input/output observations
    # with a given learning rate. It does not determine which subset of the training set is used 
    # and since the learning rate is a parameter, we may adjust it through the epochs.
    def gradient_descent_optimizer(self, input_data, output_data, learning_rate):
        # We proceed in three steps:

        # Step 1 : Compute a forward pass but by recording the non-activated and the activated values of the layers
        number_of_inputs = input_data.shape[0]
        training_data = self.rescale_data(input_data, self.input_scales)

        N = [None] # Just to shift the indices and make them match with theory
        A = [training_data]

        for layer in self.layers_list[1:]: # The input layer cannot forward-propagate, it has no weight matrix!
            if layer.preceding_layer.bias:
                Nvalues, Avalues = layer.forward_propagation( add_ones_column(A[-1]) )
            else:
                Nvalues, Avalues = layer.forward_propagation( A[-1] )
            N.append(Nvalues)
            A.append(Avalues)

        # Step 2 : Compute the gradients (note that these are stored in the layers)
        self.layers_list[-1].A_gradient = A[-1] - self.rescale_data(output_data, self.output_scales)
        for l in reversed(range(1,len(self.layers_list))): 
            # Recall that np.dot is matrix multiplication,
            # np.einsum is tensor contraction via the Einstein summation notation
            # and * denotes the Hadamard product of two tensors of the same size
            # We dediced to average the gradients in the input data right here, which is why ai,aj->ij (and not aij as in theory)
            # Note : we actually need this index l to access the lists N and A
            layer = self.layers_list[l]
            if layer.preceding_layer.bias:
                layer.weights_gradient = np.einsum("ai,aj->ij",
                                                   add_ones_column(A[l-1]),
                                                   layer.activate_layer_prime(N[l]) * layer.A_gradient / number_of_inputs)
            else:
                layer.weights_gradient = np.einsum("ai,aj->ij",
                                                   A[l-1],
                                                   layer.activate_layer_prime(N[l]) * layer.A_gradient / number_of_inputs)
        # Step 3 : Update the weights 
        # (we are still within the loop, so in some sense Step 2 and 3 are done simultaneously since dC/dA has not been updated yet)
            layer.update_weights(learning_rate)

            if l > 1: # This updates the values of dC/dA if the preceding layer is not the input, because the input layer has no weight matrix
                if layer.preceding_layer.bias: 
                    # We remove the row of b's in the matrix W to back-propagate to not back-propagate to the constant 1 node
                    layer.preceding_layer.A_gradient = np.dot( layer.activate_layer_prime(N[l]) * layer.A_gradient ,
                                                               layer.weights[0:-1].T )
                else:
                    layer.preceding_layer.A_gradient = np.dot( layer.activate_layer_prime(N[l]) * layer.A_gradient ,
                                                               layer.weights.T )

    ##########
    #
    # Training network 
    #
    ##########

    # Pre-requisites :
    # - Data has to be tidy. This means, both data frames corresponding to input and output have to have rows corresponding to observations
    #           and columns to features of the data.
    # - The data has to be stored in numPy arrays filled with numerical data. 
    def train_network(self, input_data, 
                            output_data,
                            number_of_epochs,               # How many times should we run the optimizer
                            display_performance_frequency,  # How often should we display the performance of the network on training/validation data
                                                            # Ideally, the integer display_performance_frequency divides number_of_epochs
                            training_split,                 # Proportion of the input_data used for training
                            validation_split,               # Proportion used for validation and adjusting hyperparameters (prevent overfitting)
                            test_split,                     # Proportion used for testing, once hyperparameters are adjusted using the validation set 
                            batch_size_split = 1,           # This is the proportion of the training data used for each run of the optimizer. 
                                                            # The value 1 gives rise to the batch gradient descent algorithm
                                                            # A value strictly between 0 and 1 gives the mini-batch gradient descent algorithm
                                                            # (If we could afford 1/training_data_size, we would get stochastic gradient descent)
                            learning_rate = 0.01,           # The learning rate used in the optimizer              
                            data_accuracy=None):            # The percentage of the perfect model (the circle) correctly predicting the data, just omit it if you are modifying this model
        # Checking for value errors
        if training_split > 1 or training_split < 0:
            raise ValueError("The proportion training_split must lie between 0 and 1")
        if validation_split > 1 or validation_split < 0:
            raise ValueError("The proportion validation_split must lie between 0 and 1")
        if test_split > 1 or test_split < 0:
            raise ValueError("The proportion test_split must lie between 0 and 1")
        if training_split + validation_split + test_split != 1:
            raise ValueError("The proportions must add up to 1")

        # We now compute a list (np.array) of length equal to the number of our observations
        # which will indicate if the observation is for training (0), validation (1) or testing(2). 
        # We determine which one is which randomly. 
        # See https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.multinomial.html 
        number_of_observations = len(input_data)
        split_row_indices = np.dot( np.random.multinomial(1, 
                                                          [training_split, validation_split, test_split], 
                                                          number_of_observations),
                                    np.array([[0],[1],[2]]) ).flatten()
        
        # We now use split_row_indices to determine the training, validation and testing data sets, which are disjoint by definition
        training_input_data = input_data[split_row_indices == 0]
        training_output_data = output_data[split_row_indices == 0]
        validation_input_data = input_data[split_row_indices == 1]
        validation_output_data = output_data[split_row_indices == 1]
        testing_data = input_data[split_row_indices == 2]
        
        # We now train the network on the training data and track performance
        training_data_size = len(training_input_data)
        validation_data_size = len(validation_input_data)
        sample_size = int(batch_size_split * training_data_size)
        i = 1
        training_cost = []
        validation_cost = []
        accuracy = []
        for i in range(number_of_epochs):
            # Collect the mini-batch for one run of optimizer
            sampling_row_indices = np.random.randint(training_data_size, size=sample_size)
            input_batch = training_input_data[sampling_row_indices,:]
            output_batch = training_output_data[sampling_row_indices,:]
            
            self.gradient_descent_optimizer(input_batch, output_batch, learning_rate)
            training_cost.append( norm_squared(self.run_network(training_input_data) - training_output_data) / training_split )
            validation_cost.append( norm_squared(self.run_network(validation_input_data) - validation_output_data) / validation_split )
            # The np.linalg.norm in accuracy counts the number of input values for which our network was off 
            # (the prediction was 1 instead of 0, or 0 instead of 1) divided by the number of inputs, square-rooted. 
            # So we square it and subtract it from 1 to measure our accuracy. 
            accuracy.append( 1 - norm_squared( threshold( self.run_network(validation_input_data), 0.5 ) - validation_output_data ) / validation_data_size )

            if i % display_performance_frequency == display_performance_frequency-1 or i == 0:
                print("Epoch : ", i+1, "/", number_of_epochs)
                print("Cost function on training set : ", training_cost[-1])
                print("Cost function on validation set : ", validation_cost[-1])
                print("Accuracy on validation set : ", accuracy[-1])
                if data_accuracy is not None:
                    print("Data accuracy : ", data_accuracy)
                print("")

        return training_cost, validation_cost, accuracy # This is for plotting purposes
                                                        # namely to plot the graphs of the training cost/validation cost
