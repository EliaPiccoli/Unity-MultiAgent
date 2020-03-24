seed = 5000

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED']=str(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from gym_unity.envs import UnityEnv

from datetime import datetime

from rainbow_agent import RainbowAgent

class Agents:
    def __init__(self, env, n_agents):
        self.env = env
        self.episodes = 25000

        self.n_agents = n_agents;
        self.agents = self._build_agents()

    #creates and return a list of all the agents
    def _build_agents(self):
        return [RainbowAgent() for _ in range(self.n_agents)]

    def train(self):
        print("Start training")

        reward_list = [[] for _ in range(self.n_agents)]
        success_list = [[] for _ in range(self.n_agents)]
        time_list = []
        genetic_list = [[] for _ in range(self.n_agents)]
        reward_e_list = [[]  for _ in range(self.n_agents)]

        is_recording = False
        max_success = [0 for _ in range(self.n_agents)]
        genetic_eval = [0 for _ in range(self.n_agents)]
        genetic_switch = [0 for _ in range(self.n_agents)]

        for e in range(self.episodes):
            time_list.append("Episode " + str(e) + ": " + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

            state = self.env.reset()

            if e == 0:
                print("Mapping agents: (id_unity, index_python)")
                for i in range(self.n_agents):
                    print("({}, {})".format(state[i][-1], i))
                print()

            for i in range(self.n_agents):
                self.agents[i].set_state(state[i][:-1])
            
            done = False
            reward_e = [0 for _ in range(self.n_agents)]

            while not done:
                actions = [self.agents[i].get_action() for i in range(self.n_agents)]
                new_states, rewards, dones, _ = self.env.step(actions);

                for i in range(self.n_agents):
                    self.agents[i].step(new_states[i][:-1], rewards[i], dones[i])
                    reward_e[i] += rewards[i]

                done = all(self.agents[i].is_waiting() for i in range(self.n_agents))

            for i in range(self.n_agents):
                success_list[i], reward =  self.agents[i].update_episode()
                
                if success_list[i] > max_success[i]:
                    max_success[i] = success_list[i]
                reward_e_list[i].append(reward_e[i])
                reward_list[i].append(reward)

                self.agents[i].stop_waiting()

                print("Agent: {}, Episode: {:5.0f}, Success: {:3.0f}, Reward: {}, Genetic {}/{}, Highest: {}"
                        .format(i, e, success_list[i], reward, genetic_switch[i], genetic_eval[i], max_success[i])
                    )

            # Crossover
                    
            print()

            if e%4 == 0 and e!= 0:
                break


