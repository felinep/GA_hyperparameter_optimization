import pandas as pd
import string
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from datetime import date
from typing import *
import pickle
import os
import argparse

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

# Global constants:
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
    pickle_name_result, pickle_name_pop, video_name, target, input = define_dataset_constants(f"set{dataset_number}", dataset_number)

    run_and_store(
        pickle_name_result=pickle_name_result, 
        pickle_name_pop= pickle_name_pop, 
        video_name=video_name, 
        target=target,
        input=input, 
        verbose= VERBOSE
    )
def parse_input_arguments(argv=None):
    parser = argparse.ArgumentParser(description='Genetic Algorithm for Hyperparameter Optimization')
    required = parser.add_argument_group('required arguments')

    required.add_argument("--dataset_number", "-ds", 
                        help="display a square of a given number",
                        type=int, 
                        required=True)

    required.add_argument("--num_generations", "-ng", 
                        help="display a square of a given number",
                        type=int, 
                        required=True)

    required.add_argument("--population_size", "-ps", 
                        help="display a square of a given number",
                        type=int, 
                        required=True)

    parser.add_argument("--data_count", "-dc", 
                        help="display a square of a given number",
                        type=int, 
                        default=5000)

    parser.add_argument("--max_neurons_per_layer", "-mnpl", 
                        help="display a square of a given number",
                        type=int, 
                        default = 300)

    parser.add_argument("--max_hiddenlayers", "-mh", 
                        help="display a square of a given number",
                        type=int, 
                        default = 10)

    parser.add_argument("--save_prefix", "-sp", 
                        help="display a square of a given number",
                        type=str, 
                        default = "GA")
    
    groups_order = {
        'positional arguments': 0,
        'required arguments': 1,
        'optional arguments': 2
    }
    parser._action_groups.sort(key=lambda g: groups_order[g.title])

    return parser.parse_args()

def main(args=None):
    dataset_random(DATASET_NUMBER)


if __name__ == '__main__':
    args = parse_input_arguments()

    DATASET_NUMBER =     args.dataset_number
    DATA_COUNT =         args.data_count
    GEN_STEP_SIZE =      args.num_generations
    POPULATION_SIZE =    args.population_size
    PC_NAME =            args.save_prefix
    MAX_HIDDENLAYERS =   args.max_hiddenlayers
    MAX_NEURON_PER_LAYER=args.max_neurons_per_layer

    # uses Early-Stopping, the training will most probably stop before reaching 500 epochs
    NUM_EPOCHS = 500
  
    print ("\n", "-" * 105, f"\nStarting the GA using {DATA_COUNT} instances of data from dataset {1}, a population size of {POPULATION_SIZE} and {GEN_STEP_SIZE} generations.")
    print (f"Each indivial MLP will have maximal {MAX_NEURON_PER_LAYER} neurons per layer and maximal {MAX_HIDDENLAYERS} hidden layers.\n", "-" * 105, "\n")

    main()
