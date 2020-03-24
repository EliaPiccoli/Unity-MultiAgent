seed = 1

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
from keras.models import Model
from keras.layers import Dense
from keras.models import clone_model, load_model
import keras
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Lambda, Input, add, GaussianNoise
import numpy as np
import gc
from memory_profiler import profile

import random
import tensorflow as tf


class GeneKeras:

    def __init__(self, load_compiled=False):
        self.state_size = 22
        self.action_size = 5
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size + 1, activation='linear'))
        model.add(Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], axis=1, keepdims=True), output_shape=(self.action_size,)))
        return model

    def set_genetic_parameters(self, crossover_enabled = True, mutation_enabled = True, mutation_prob = 0.1, mutation_rate = 0.5):
        self.crossover_enabled = crossover_enabled
        self.mutation_enabled = mutation_enabled
        self.mutation_prob = mutation_prob
        self.mutation_rate = mutation_rate


    def create_child(self, parent_weights):
        self.model.set_weights(parent_weights)

        '''
        if (self.crossover_enabled):
            self.crossover()
        '''
        
        if (self.mutation_enabled):
            self.mutation()

        return self.model.get_weights().copy()


    def mutation(self):
        layer_number = len(self.model.layers)
        for i in range(0, layer_number):
            if (isinstance(self.model.layers[i], Dense)):
                weights, biases = self.model.layers[i].get_weights().copy()
                for k in range (len(weights)):
                    for l in range (len(weights[k])):
                        if np.random.uniform() < self.mutation_prob:
                            weights[k][l] += np.random.uniform(-self.mutation_rate, self.mutation_rate)
                            #weights[k] = weights[k] * np.random.normal(0, self.mutation_rate)

                for k in range (len(biases)):
                    if np.random.uniform() < self.mutation_prob:
                        biases[k] += np.random.uniform(-self.mutation_rate, self.mutation_rate)
                        #biases[k] = biases[k] * np.random.normal(0, self.mutation_rate)

                self.model.layers[i].set_weights([weights, biases])

    '''
    def crossover(self, temp_model):
        available_layers_number = 0

        layer_number = len(temp_model.layers)

        for i in range (0, layer_number):
            if (isinstance(temp_model.layers[i], Dense)):
                available_layers_number += 1

        cut_layer = np.random.random_integers(0, available_layers_number)

        j = 0
        for i in range(0, layer_number):
            if (isinstance(temp_model.layers[i], Dense)):
                if(j < cut_layer):
                    temp_model.layers[i].set_weights(self.model_1.layers[i].get_weights())
                else:
                    temp_model.layers[i].set_weights(self.model_2.layers[i].get_weights())
                j += 1
    '''

    

    '''
    def check_models_shape(self):
        layer_numbers_1 = len(self.model_1.layers)
        layer_numbers_2 = len(self.model_2.layers)

        if(layer_numbers_1 != layer_numbers_2):
            return False

        for n in range (layer_numbers_1):
            if (isinstance(self.model_1.layers[n], Dense) and isinstance(self.model_1.layers[n], Dense)):
                if (self.model_1.layers[n].get_weights()[0].shape != self.model_2.layers[n].get_weights()[0].shape):
                    return False

        return True
    '''