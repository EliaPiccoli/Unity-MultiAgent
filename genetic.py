seed = 1

#import kegen as kg
from memory_profiler import profile

import os
import os.path
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
from keras.models import Sequential, load_model, clone_model
from keras.layers import Dense, GaussianNoise, Activation
import keras
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Lambda, Input, add, GaussianNoise
from genekeras import GeneKeras
from gym_unity.envs import UnityEnv

import numpy as np
import gc
import threading

import tensorflow as tf
import random

from keras import backend as K

class GeneticEvaluation:
    def __init__(self):
        self.state_size = 22
        self.action_size = 5

        crossover_enabled = False
        mutation_enabled = True
        mutation_p = 0.1
        mutation_v = 0.05

        self.genetic_lib = GeneKeras(load_compiled=True)  # keep the compiled model without recompiling
        self.genetic_lib.set_genetic_parameters(False, True, 0.75, 0.1)  # cross_enabled, mut_enabled, mut_p, mut_v

        self.n_agent = 10    # Number of genetic agents

        # Generating n_agents models to copy the children weights into them and avoid tensorflow graph overheads
        self.family = []
        for i in range(self.n_agent):
            self.family.append(self._build_model())

        '''
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution
        '''

    def _set_genetic_parameters(self, crossover, mutation, mutation_p, mutation_v):
        self.genetic_lib.set_genetic_parameters(crossover, mutation, mutation_p, mutation_v)  # cross_enabled, mut_enabled, mut_p, mut_v


    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size + 1, activation='linear'))
        model.add(Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], axis=1, keepdims=True), output_shape=(self.action_size,)))
        return model


    def create_unity_env(self, worker_id):
        env_name = "Unity/Games/gentics_maze_3x3.x86_64"
        self.env = UnityEnv(env_name, worker_id=worker_id, multiagent=True)


    def close_unity_env(self):
        self.env.close()
        gc.collect()


    # The idea is to fill the same children models with new weights, to limit the tf graph overheads
    def create_children(self, model):
        self.family[0].set_weights(model)   # Setting parent weights into the family
        for i in range(1, self.n_agent):
            self.family[i].set_weights(self.genetic_lib.create_child(model))


    def genetic_episode(self):
        best_model, switched, memory = self.evaluate_children(20)  # takes evaluation episodes as input

        return self.family[best_model].get_weights().copy(), switched, memory


    def evaluate_children(self, n_episodes=20):
        episode_counter = np.zeros(self.n_agent)
        total_reward = np.zeros(self.n_agent)
        normalized_reward = np.zeros(self.n_agent)

        stop_evaluation = False
        evaluated_agents = 0
        states = self.env.reset()
        
        memory = [[] for _ in range(self.n_agent)]

        while not stop_evaluation:
            #q_values = [self.family[i].predict(np.array([states[i]]))[0] for i in range(self.n_agent)]
            #actions = [[np.argmax(q_values[i][:5]), np.argmax(q_values[i][5:])] for i in range(self.n_agent)]
            actions = []
            for i in range(self.n_agent):
                q_values = self.family[i].predict(np.array([states[i]]))[0]
                actions.append(np.argmax(q_values))

            new_states, rewards, dones, info = self.env.step(actions)

            print()
            print("type(info):" + str(type(info)))
            print(info)
            print("type(info[brain_info]):" + str(type(info["brain_info"])))
            print(info["brain_info"].agents)
            print()

            for i in range (len(new_states)):
                if episode_counter[i] <= n_episodes:
                    if random.random() <= 0.10:
                        memory[i].append([states[i], actions[i], rewards[i], new_states[i], dones[i]])

                if dones[i]:
                    if episode_counter[i] <= n_episodes:

                        episode_counter[i] += 1
                        print(episode_counter)
                        
                        if(rewards[i] == 1 and episode_counter[i] <= n_episodes):
                            total_reward[i] += 1

            states = new_states

            stop_evaluation = all(element >= n_episodes for element in episode_counter)

            for i in range(len(episode_counter)):
                if episode_counter[i] > n_episodes * 2:
                    stop_evaluation = True     

        normalized_reward = [total_reward[i] / episode_counter[i] for i in range(len(total_reward))]
        print(normalized_reward)

        best_model = np.argmax(normalized_reward)
        switched = best_model != 0
        
        return best_model, switched, memory[best_model]


    '''
    def run(self):
        # Method that runs forever
        while True:
            # Do something
            if self.evaluate:

                self.create_env(self.worker_id)
                self.genetic_episode()
                self.destroy_env()
                self.worker_id += 1
                if self.worker_id >= 10:
                    self.worker_id = 2

                self.evaluate = False
                gc.collect()

            time.sleep(5)
    '''