# basically a HalfUniformCrossover with the addtition that it will be sorted at the end: 
#from pymoo.operators.crossover.hux import HalfUniformCrossover
from pymoo.core.crossover import Crossover
from typing import *
import math
import numpy as np
from pymoo_implementation.population.population_sorting import sort_pop_array


class CrossoverMLP_Optimization(Crossover):
    def __init__(self, prob_hux=0.5, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.prob_hux = prob_hux

    def _do(self, _, X, **kwargs):
        _, n_matings, n_var = X.shape

        # the mask do to the crossover
        M = np.full((n_matings, n_var), False)

        not_equal = X[0] != X[1]
        # switch genes where both individuums have no zeros:
        for i in range(n_matings):
            I = np.where(not_equal[i])[0]
            n = math.ceil(len(I) / 2)
            if n > 0:
                _I = I[np.random.permutation(len(I))[:n]]
                M[i, _I] = True
        _X = crossover_mask(X, M)
        return _X


def crossover_mask(X, M):
    # convert input to output by flatting along the first axis
    _X = np.copy(X)
    _X[0][M] = X[1][M]
    _X[1][M] = X[0][M]
    # sort the array after the crossover so that all zeros are at the end!
    _X = sort_pop_array(_X)
    return _X
