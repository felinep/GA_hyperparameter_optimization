# GA_hyperparameter_optimization
Hyperparameter optimization using genetic algorihms.

## Usage:
Add data into the /data folder. 
Pickled data will be stored in the pickled_data folder and videos inside the video folder. 

The algorithm will be started by running:

python
~/GA_hyperparameter_optimization/scripts/main.py 
[-h] 
--dataset_number DATASET_NUMBER --num_generations NUM_GENERATIONS --population_size POPULATION_SIZE [--data_count DATA_COUNT] [--max_neurons_per_layer MAX_NEURONS_PER_LAYER]
               [--max_hiddenlayers MAX_HIDDENLAYERS] [--save_prefix SAVE_PREFIX]

## Required arguments:
dataset number: Number of the dataset that should be used (1 for random1 etc.).

num generations: The number of generations the algorithm should run for.

population size: The amount of MLP-indviduums in a population.

## Optional arguments:
help: show this help message and exit

data count: Amount of data used to train and validate the network individuums (default: 5000)

max neuros per layer: Maximal amount of neurons in a hiddenlayer.

max hiddenlayers: Maximal amount of hidden layers.

save prefix: Add a save-prefix for the generated data saves.

