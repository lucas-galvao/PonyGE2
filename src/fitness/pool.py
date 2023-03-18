from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff
from stats.stats import stats
from tensorflow.keras import datasets, layers, models, callbacks, optimizers
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from datetime import datetime
from time import sleep
from training_pool import find_trained_phenotype
import numpy as np
import re, csv, os, requests
import tensorflow as tf


class pool(base_ff):

    maximise = True
    multi_objective = True

    def __init__(self):
        super().__init__()
        self.execution_id = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
        self.num_obj = 2
        fit = base_ff()
        fit.maximise = True
        self.fitness_functions = [fit, fit]
        self.default_fitness = [float('nan'), float('nan')]

    def save_step(self, phenotype, accuracy, f1_score, time):

        data = {
            'execution': self.execution_id,
            'grammar': params['GRAMMAR_NAME'],
            'dataset': params['DATASET_NAME'],
            'generation': stats['gen'],
            'phenotype': phenotype,
            'accuracy': accuracy,
            'f1_score': f1_score,
            'time': time,
        }
        try:
            r = requests.post('%s/api/steps/' % params['METRICS_URL'], json=data)
        except BaseException as ex:
            print('save_step error:', ex)
            

    def evaluate(self, ind, **kwargs):

        print('GENERATION:', stats['gen'])
        print('PHENOTYPE:', ind.phenotype)
        print('DATASET:', params['DATASET_NAME'])
        print('GRAMMAR:', params['GRAMMAR_NAME'])

        accuracy, f1_score, time = None, None, None

        while True:
            metrics = find_trained_phenotype(ind.phenotype)
            if metrics:
                print('Found:', metrics)
                accuracy = float(metrics['accuracy'])
                f1_score = float(metrics['f1_score'])
                time = float(metrics['time'])
                break
            else:
                sleep(15)

        self.save_step(ind.phenotype, accuracy, f1_score, time)

        return accuracy, f1_score

    @staticmethod
    def value(fitness_vector, objective_index):

        if not isinstance(fitness_vector, list):
            return float("inf")

        return fitness_vector[objective_index]
