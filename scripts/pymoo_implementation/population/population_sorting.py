def sort_pop(pop):
    for index in range (len(pop)):
        pop[index] = sorted(pop[index], key=lambda n: n==0)
    return pop

def sort_pop_array(pop_array):
    for index in range (len(pop_array)):
        pop_array[index] = sort_pop(pop_array[index])
    return pop_array