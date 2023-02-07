from typing import *
from pymoo.core.sampling import Sampling
import random
import numpy as np

def create_initial_population(count: int, max_hiddenlayers : int, max_neuron_per_layer : int) -> List[Tuple[Type]]:
    pop: List[List[int, Type]] = []
    
    for individual in range (0, count):
        num_hiddenlayers = random.randrange(1, max_hiddenlayers + 1)
        neuron_array = np.zeros(max_hiddenlayers)
        for layer in range (num_hiddenlayers):
            neuron_array[layer] = random.randrange(1, max_neuron_per_layer + 1)

        pop.append(neuron_array)

    return pop    

class SamlingMLP_Optimization(Sampling):
    def __init__(self, max_hiddenlayers : int, max_neuron_per_layer : int) -> None:
        super().__init__()
        self.vtype = float
        self.max_hiddenlayers = max_hiddenlayers
        self.max_neuron_per_layer = max_neuron_per_layer
        
    def _do(self, problem, n_samples, **kwargs):
        return create_initial_population(n_samples, self.max_hiddenlayers, self.max_neuron_per_layer)