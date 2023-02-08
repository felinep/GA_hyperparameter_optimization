from pymoo.core.problem import ElementwiseProblem
from keras.callbacks import EarlyStopping
import numpy as np
import string
import tensorflow as tf
from keras.layers.core import Activation
from keras import backend

# activation function: tanh(x/2)
def tanh(x):
    return backend.tanh(x/2)

def create_MLP_from_parameter(parameter : np.array(int), metric : string = 'accuracy'):
    num_neurons_per_layer = parameter 
    fitness_value: float
    activation_function = Activation(tanh)

    num_neurons_per_layer_without_zeros = num_neurons_per_layer[num_neurons_per_layer != 0]

    # create model: 
    cnn_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            max(int(num_neurons_per_layer_without_zeros[0]), 1),
            input_shape = (27,),  
            activation=activation_function)
    ])

    if (len(num_neurons_per_layer_without_zeros != 0)):
        for layer in range(1, len(num_neurons_per_layer_without_zeros)):
            num_neurons = int(num_neurons_per_layer_without_zeros[layer])
            cnn_model.add(
                tf.keras.layers.Dense(
                    num_neurons,
                    activation = activation_function))

    cnn_model.add(tf.keras.layers.Dense(
        2, 
        activation = activation_function))

    # Specify the loss fuction, optimizer, metrics
    cnn_model.compile(
        loss = tf.keras.losses.MeanSquaredError(),
        # https://keras.io/api/optimizers/ 
        optimizer = "sgd",
        metrics = [metric]
    )
    return cnn_model
    
def evaluate_MLP(cnn_model, model_paramter, input_data, target_data, metric : string = 'accuracy', verbose: bool = False, weight_thresh = 0.05, min_acc = 0.92 ):

    # add early stopping callback to save time
    es = EarlyStopping(monitor = f'val_{metric}', mode ='max', patience = 5)

    # Train the model
    history = cnn_model.fit(input_data, target_data, epochs=500, validation_split = 0.2, batch_size=2, callbacks = [es], use_multiprocessing=True, verbose=0)
    
    # TODO choose the best
    last_result = history.history[f"val_{metric}"][-1:][0]

    weights = cnn_model.get_weights()
    amount_dead_weights = len(np.where(abs(weights[0]) < weight_thresh)[0])

    if (verbose):
        print (f"acc: {last_result:0.02f}, sum: {np.sum(model_paramter)}, deadW_c: {amount_dead_weights}, parameter: {model_paramter}")
    
    return -last_result, np.sum(model_paramter), (amount_dead_weights), (0.92 - (last_result))

class ProblemMLP_Optimization(ElementwiseProblem):
    def __init__(self, max_hiddenlayers : int, max_neuron_per_layer : int, input_data, target_data, verbose: bool = False):
        super().__init__(
            n_var=max_hiddenlayers, 
            n_obj=3,
            n_constr=1, 
            xl=np.zeros(max_hiddenlayers),
            xu=np.ones(max_hiddenlayers)*max_neuron_per_layer + 1,
            vtype = int
        )
        self.verbose = verbose
        self.max_hiddenlayers = max_hiddenlayers
        self.max_neuron_per_layer = max_neuron_per_layer
        self.input_data = input_data
        self.target_data = target_data
    
    def _evaluate(self, X, out, *args, **kwargs):
        cnn_model = create_MLP_from_parameter(X)
        model_evaluation = evaluate_MLP(cnn_model, X, self.input_data, self.target_data, verbose = self.verbose)
        out["F"] = [model_evaluation[0], model_evaluation[1], model_evaluation[2]]
        out["G"] = [model_evaluation[3]]

