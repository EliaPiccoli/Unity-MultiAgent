seed = 5000

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED']=str(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from gym_unity.envs import UnityEnv
from utils.memory import Memory
import utils.obs_wrapper as obs

import sys
import numpy as np
import random
import time
from collections import deque
from datetime import datetime

import keras
from keras import backend as K
from keras.optimizers import Adam, SGD, Adadelta, Nadam
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Lambda, Input, add, GaussianNoise
import tensorflow as tf

from genetic import GeneticEvaluation

'''
# https://towardsdatascience.com/optimize-your-cpu-for-deep-learning-424a199d7a87
NUM_PARALLEL_EXEC_UNITS = 4
config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=4,
                       allow_soft_placement=True, device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})
session = tf.Session(config=config)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

np.random.seed(seed)
random.seed(seed)

tf.set_random_seed(seed)

K.set_session(session)
'''
np.random.seed(seed)
random.seed(seed)
config = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
tf.set_random_seed(seed)
sess = tf.Session(graph=tf.get_default_graph(), config=config)
K.set_session(sess)

class RainbowAgent:
    def __init__(self):

        self.state_size = 22
        self.action_size = 5

        self.buffer = Memory(30000)
        self.batch_size = 64

        self.epsilon = 1.0  # eps-greedy exploration
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.995

        self.gamma = 0.99   # discount factor
        self.learning_rate = 0.001

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights().copy())
        self.tau = 0.05 # Soft update target_model weights (original paper value 0.001 seems not to work)

        self.model.save("models/dqn_model.h5")
        self.target_model.save("models/ddqn_model.h5")

        # Agent state variables required for execution
        self.state = None
        self.reward = None
        self.done = None
        self.action = None
        self.waiting = False
        self.reward_queue = deque(maxlen=100)
        self.genetic_e = GeneticEvaluation()

    def _build_model(self):
        '''
        network_input = Input(shape=(self.state_size, ))
        hidden = Dense(32, activation='relu')(network_input)
        hidden = Dense(32, activation='relu')(hidden)
        hidden = Dense(32, activation='relu')(hidden)

        state_value = Dense(1, init='uniform')(hidden)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0], axis=-1), output_shape=(self.action_size,))(state_value)
        
        action_advantage = Dense(self.action_size)(hidden)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(self.action_size,))(action_advantage)

        state_action_value = add([state_value, action_advantage])

        model = Model(input=network_input, output=state_action_value)
        '''
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size + 1, activation='linear'))
        model.add(Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], axis=1, keepdims=True), output_shape=(self.action_size,)))
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        return model

    # Transfer model weights to target model with a factor of Tau
    def _soft_target_update(self):
        model_weights, target_weights = self.model.get_weights().copy(), self.target_model.get_weights().copy()
        for i in range(len(model_weights)):
            target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)

    # Transfer best genetic weights to target model with a factor of Tau
    def _soft_target_update_genetic(self, genetic_weights):
        target_weights = self.target_model.get_weights().copy()
        for i in range(len(genetic_weights)):
            target_weights[i] = self.tau * genetic_weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)  

    def _target_update(self):
        self.target_model.set_weights(self.model.get_weights().copy())

    def _fit_model(self):
        state_list = []
        qvalue_list = []
        weights_list = []

        idxs, sampled_batch, weights = self.buffer.sample(self.batch_size)

        for index, experience in enumerate(sampled_batch):
            state = experience[0].copy()
            action = experience[1]
            reward = experience[2]
            state_ = experience[3].copy()
            done = experience[4]

            qvalue_ = reward

            if not done:
                action_ = np.argmax(self.model.predict(np.array([state_]))[0])
                qvalue_ += self.gamma * self.target_model.predict(np.array([state_]))[0][action_]

            qvalue = self.model.predict(np.array([state]))[0]

            self.buffer.batch_update(idxs[index], abs(qvalue_ - qvalue[action]))

            qvalue[action] = qvalue_

            state_list.append(state)
            qvalue_list.append(qvalue)

        self.model.train_on_batch(np.array(state_list), np.array(qvalue_list), sample_weight=np.array(weights))

    def get_action(self):
        if not self.waiting:
            if random.random() <= self.epsilon:
                self.action = random.randrange(0, self.action_size)
            else:
                self.action = np.argmax(self.model.predict(np.array([self.state]))[0])
            return self.action
        else:
            return self.action_size

    def set_state(self, state):
        self.state = state

    def is_waiting(self):
        return self.waiting

    def stop_waiting(self):
        self.waiting = False
    
    def step(self, new_state, reward, done):
        if not self.waiting:
            self.reward = reward
            self.done = done    
            
            if (self.done and self.reward != 1):
                self.reward = -1
                        
            self.buffer.push(self.state.copy(), self.action, self.reward, new_state.copy(), self.done)
            self.state = new_state.copy()

            '''
                The agent has finished his episode but he must wait that all the other agents finish their episode. In this state he will always send a fake signal to Unity. 
                Once all agents have finished the Agents will reset the waiting field and it will start the next episode.
            '''
            if self.done:
                self.waiting = True
                self.reward_queue.append(self.reward)

    def update_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
      
        self._fit_model()

        # Soft update the target model
        self._soft_target_update()

        success = int(self.reward_queue.count(1)/(len(self.reward_queue)+0.0)*100)

        if success > 50:
            self.genetic_e._set_genetic_parameters(False, True, 0.2, 0.1)

        if success > 70:
            self.genetic_e._set_genetic_parameters(False, True, 0.1, 0.05)

        return success, self.reward

    ''' old train method of the single AI reimplemented in Agents.py for handle multiple agents
    def train(self):
        reward_list = []
        success_list = []
        time_list = []
        genetic_list = []
        reward_e_list = []

        reward_queue = deque(maxlen=100)
        flag = 0
        genetic_worker_id = 200
        genetic_e = GeneticEvaluation()
        genetic_eval = 0
        genetic_switch = 0
        max_success = 0
        is_recording = False

        for e in range(self.episodes):
            time_list.append("Episode " + str(e) + ": " + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

            state = self.env.reset()
            done = False
            reward_e = 0

            while not done:
                if random.random() <= self.epsilon:
                    action = random.randrange(0, self.action_size)
                else:
                    action = np.argmax(self.model.predict(np.array([state]))[0])
   
                state_, reward, done, _ = self.env.step(action)
                
                if (done and reward != 1):
                    reward = -1
                    
                self.buffer.push(state.copy(), action, reward, state_.copy(), done)
                
                reward_e += reward
                state = state_.copy()   

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
      
            self._fit_model()

            # Soft update the target model
            self._soft_target_update()

            reward_e_list.append(reward_e)
            reward_list.append(reward)
            reward_queue.append(reward)
            success = int(reward_queue.count(1)/(len(reward_queue)+0.0)*100)
            success_list.append(success)
            
            if success > max_success:
                max_success = success

            if success > 50:
                genetic_e._set_genetic_parameters(False, True, 0.2, 0.1)

            if success > 70:
                genetic_e._set_genetic_parameters(False, True, 0.1, 0.05)

            if(e % 10 == 0 and e != 0):
                genetic_eval += 1

                genetic_e.create_unity_env(genetic_worker_id)
                genetic_e.create_children(self.model.get_weights().copy())
                genetic_weights, switched, genetic_memory = genetic_e.genetic_episode()
                genetic_worker_id += 1

                if (switched):
                    print("Found a genetic evaluation...")
                    genetic_switch += 1
                    self._soft_target_update_genetic(genetic_weights)
                    genetic_list.append(1)
                else:
                    genetic_list.append(0)

                genetic_e.close_unity_env()
            
            print("Episode: {:7.0f}, Success: {:3.0f}, Reward: {}, Genetic {}/{}, Highest: {}".format(e, success, reward, genetic_switch, genetic_eval, max_success))

            if(e % 199 == 0):
                np.savetxt("results/reward_list.txt", reward_list, fmt='%3i')
                np.savetxt("results/success_list.txt", success_list, fmt='%3i')
                np.savetxt("results/time_list.txt", time_list, fmt='%s')
                np.savetxt("results/reward_e.txt", reward_e_list, fmt='%s')
                np.savetxt("results/genetic.txt", genetic_list, fmt='%s')

            if(success >= 90):
                self.model.save("models/dqn_" + str(success) + "_" + str(e) + ".h5")
                self.target_model.save("models/ddqn_" + str(success) + "_" + str(e) + ".h5")
                if not is_recording:
                    obs.start_recording()
                    is_recording = True
            elif is_recording:
                    is_recording = False
                    obs.stop_recording()
    '''
