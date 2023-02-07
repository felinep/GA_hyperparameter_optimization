import pandas as pd
import string
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from datetime import date
from typing import *
import pickle
import os

# import the initial population sample class(returns a population object where the individuals are MLP parameters,
# variant in the number of neurons and the amount of hidden layers).
from pymoo_implementation.population.sampling_mlp_optimization import SamlingMLP_Optimization

# import a crossover class to cross mlp-paramter indivduals without generating invalid parameter sets (
# includes a population sort call)
from pymoo_implementation.crossover.crossover_mlp_optimization import CrossoverMLP_Optimization

# import a mutation class to cross mlp-paramter indivduals without generating invalid parameter sets (
# includes a population sort call)
from pymoo_implementation.mutation.mutation_mlp_optimization import PolynomialMutationMLP_Optimization

# import the mlp optimzation problem, which includes a mlp model fitness calculation
from pymoo_implementation.problem.problem_mlp_optimization import ProblemMLP_Optimization

# imports visualization functionality for the optimazation progress
from pymoo_implementation.visualization.visualization_mlp_optimization import store_video

# Data constants
PC_NAME = "test"

DATA_COUNT = 100

# mlp setup constants
MAX_HIDDENLAYERS = 10
MAX_NEURON_PER_LAYER = 300
NUM_EPOCHS = 500 # uses early-stop anyways!

# genetic algorithm constants
GEN_STEP_SIZE = 1 #100 #min 2, step size instead of complete number of generations
POPULATION_SIZE = 5 #250

# Random seed
RANDOM_SEED = 42

METRIC = "accuracy"

VERBOSE = True

PATH = f"{os.getcwd()}/GA_hyperparameter_optimization/" 
DATA_PATH = PATH + "data/"

def define_dataset_constants(saveset_name : string, dataset_number : int):
    today = date.today()

    original_data = pd.read_csv(DATA_PATH+f"random{dataset_number}", delimiter='\t', keep_default_na=True);    

    pickle_name_result = PATH + f"pickled_data/res-{PC_NAME}-{saveset_name}-{POPULATION_SIZE}pop-{MAX_HIDDENLAYERS}ly-{today}.p"
    pickle_name_pop    = PATH + f"pickled_data/pop-{PC_NAME}-{saveset_name}-{POPULATION_SIZE}pop-{MAX_HIDDENLAYERS}ly-{today}.p"
    video_name         = PATH + f"videos/{PC_NAME}-{saveset_name}-{POPULATION_SIZE}pop-{MAX_HIDDENLAYERS}ly-{today}.mp4"

    data = original_data.iloc[:
    DATA_COUNT
    , 9:]
    input = data.iloc[:, :-9]
    target = data.iloc[:, -9:-7]
    
    return pickle_name_result, pickle_name_pop, video_name, target, input

def run_algorithm(population, input, target, verbose : bool = False):
    
    termination = get_termination("n_gen", GEN_STEP_SIZE + 1)

    algorithm = NSGA2(
        pop_size=POPULATION_SIZE,
        eliminate_duplicates=True, 
        sampling=population, 
        save_history = True, 
        crossover=CrossoverMLP_Optimization(), 
        mutation=PolynomialMutationMLP_Optimization(),
        verbose=verbose)

    res = minimize(ProblemMLP_Optimization(MAX_HIDDENLAYERS, MAX_NEURON_PER_LAYER, input, target, verbose),
                algorithm,
                seed=RANDOM_SEED,
                verbose=verbose,
                termination=termination)

    print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
    return res

def run_and_store(pickle_name_result, pickle_name_pop, video_name, target, input, verbose = False):

    initial_pop = SamlingMLP_Optimization(MAX_HIDDENLAYERS, MAX_NEURON_PER_LAYER)
    history_list = []

    result = run_algorithm(
        population= initial_pop,
        input= input, 
        target= target, 
        verbose= verbose)

    history_list.append(result.history)
    print ("GA done, storing data...")
    # save result and last population
    pickle.dump(history_list, open(pickle_name_result, "wb"))
    pickle.dump(result.pop, open(pickle_name_pop, "wb"))

    # store results as a video
    store_video(video_name, pickle_name_result)



def dataset_random(dataset_number : int):
    pickle_name_result, pickle_name_pop, video_name, target, input = define_dataset_constants(f"new{dataset_number}", dataset_number)

    run_and_store(
        pickle_name_result=pickle_name_result, 
        pickle_name_pop= pickle_name_pop, 
        video_name=video_name, 
        target=target,
        input=input, 
        verbose= VERBOSE
    )


def main(args=None):
    dataset_random(1)


if __name__ == '__main__':
    main()
