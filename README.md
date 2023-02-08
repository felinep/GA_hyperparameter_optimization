# GA_hyperparameter_optimization
Hyperparameter optimization using genetic algorihms.

## Usage:
Add data into the /data folder. 
Pickled data will be stored in the pickled_data folder and videos inside the video folder. 

The algorithm will be started by running:

$ python ~/GA_hyperparameter_optimization/scripts/main.py [-h] --dataset_number DATASET_NUMBER --num_generations NUM_GENERATIONS --population_size POPULATION_SIZE [--data_count DATA_COUNT] [--max_neurons_per_layer MAX_NEURONS_PER_LAYER]
               [--max_hiddenlayers MAX_HIDDENLAYERS] [--save_prefix SAVE_PREFIX]

## required arguments:
  --dataset_number DATASET_NUMBER, -ds DATASET_NUMBER
                        Number of the dataset that should be used (1 for random1 etc.).
  --num_generations NUM_GENERATIONS, -ng NUM_GENERATIONS
                        The number of generations the algorithm should run for.
  --population_size POPULATION_SIZE, -ps POPULATION_SIZE
                        The amount of MLP-indviduums in a population.

## optional arguments:
  -h, --help            show this help message and exit
  --data_count DATA_COUNT, -dc DATA_COUNT
                        Amount of data used to train and validate the network individuums.
  --max_neurons_per_layer MAX_NEURONS_PER_LAYER, -mnpl MAX_NEURONS_PER_LAYER
                        Maximal amount of neurons in a hiddenlayer.
  --max_hiddenlayers MAX_HIDDENLAYERS, -mh MAX_HIDDENLAYERS
                        Maximal amount of hidden layers.
  --save_prefix SAVE_PREFIX, -sp SAVE_PREFIX

