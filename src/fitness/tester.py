import random
from fitness.base_ff_classes.base_ff import base_ff
from algorithm.parameters import params
from stats.stats import stats
from utilities.stats import trackers


class tester(base_ff):
    
    maximise = True
    multi_objective = True

    def __init__(self):
        super().__init__()
        self.num_obj = 2
        fit = base_ff()
        fit.maximise = True
        self.fitness_functions = [fit, fit]
        self.default_fitness = [float('nan'), float('nan')]
    
    def evaluate(self, ind, **kwargs):
        print(ind.phenotype)
        print(stats['gen'], len(trackers.cache))
        if params['LOAD_STATE']:
            for i in trackers.state_individuals:
                print('\t', i.phenotype)
        return random.random(), random.random()

    @staticmethod
    def value(fitness_vector, objective_index):

        if not isinstance(fitness_vector, list):
            return float("inf")

        return fitness_vector[objective_index]