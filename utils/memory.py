seed = 1

from collections import namedtuple, deque
import numpy as np
import random

np.random.seed(seed)
random.seed(seed)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'state_', 'done'))

class Memory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.memory_probabiliy = deque(maxlen=capacity)
        self.capacity = capacity

        self.small_epsilon = 0.0001
        self.alpha = 0.6
        self.beta = 0.4  # importance-sampling, from initial value increasing to 1 with the proceeding of the training
        self.beta_increment = 0.00005
        #0.00005

    def push(self, state, action, reward, state_, done):
        """Saves a transition."""
        if len(self.memory) > 0:
            max_probability = max(self.memory_probabiliy)
        else:
            max_probability = self.small_epsilon
        self.memory.append([state.copy(), action, reward, state_.copy(), done])
        self.memory_probabiliy.append(max_probability)

    def sample(self, batch_size):
        probability_sum = sum(self.memory_probabiliy)
        p = [probability / probability_sum for probability in self.memory_probabiliy]
        # print(len(self.memory_probabiliy))

        indexes = np.random.choice(np.arange(len(self.memory)), batch_size, p=p)
        transitions = [self.memory[idx] for idx in indexes]
        transitions_p = [p[idx] for idx in indexes]

        weights = [pow(self.capacity * p_j, -self.beta) for p_j in transitions_p]
        #weights = torch.Tensor(weights).to(device)
        # print(weights)
        weights = weights / np.max(weights)
        # print(weights)

        '''
        This implementation directly calculates td_error and update the priority of sampled experiences
        td_error = QNet.get_td_error(net, target_net, batch.state, batch.next_state, batch.action, batch.reward, batch.mask)

        td_error_idx = 0
        for idx in indexes:
            self.memory_probabiliy[idx] = pow(abs(td_error[td_error_idx]) + small_epsilon, alpha).item()
            # print(pow(abs(td_error[td_error_idx]) + small_epsilon, alpha).item())
            td_error_idx += 1
        '''
        self.beta = np.min([1., self.beta + self.beta_increment])  # max = 1

        return indexes, transitions, weights

    def n_entries(self):
        return len(self.memory)

    # Update the priorities on the tree
    def batch_update(self, idx, abs_errors):
        abs_errors += self.small_epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, 1.)
        #ps = np.power(abs_errors, self.alpha)
        self.memory_probabiliy[idx] = pow(abs_errors, self.alpha).item()

        #self.tree.update(tree_idx, ps)