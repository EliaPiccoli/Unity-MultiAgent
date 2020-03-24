import sys
import os
import math
import numpy as np

class SumTree(object):

    def __init__(self, capacity):
        self.capacity = capacity    # Number of leaf nodes that contains experiences and the priority score
        self.tree = np.zeros(2 * capacity - 1)  # Generate the tree with all nodes values = 0
        self.data = np.zeros(capacity, dtype=object)    # Contains the experiences (so the size of data is capacity)
        self.data_pointer = 0
        self.n_entries = 0

    # Add our priority score in the sumtree leaf and add the experience in data
    def add(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1  # Look at what index we want to put the experience (first one is the first leaf from left)

        self.data[self.data_pointer] = data # Update data frame
        self.data_pointer += 1

        self.update(tree_index, priority)  # Update the leaf
        
        if self.data_pointer >= self.capacity:  # If we're above the capacity, you go back to first index (we overwrite)
            self.data_pointer = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # Update the leaf priority score and propagate the change through tree
    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]   # new priority - former priority
        self.tree[tree_index] = priority
        
        while tree_index != 0:  # then propagate the change through tree
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    # Get the leaf_index, priority value of that leaf and experience associated with that index (on the root there is the total sum of the priorities)
    def get_leaf(self, v):
        parent_index = 0
        
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            if left_child_index >= len(self.tree):  # If we reach bottom, end the search
                leaf_index = parent_index
                break
            
            else: # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
            
        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0] # Returns the root node

    def get_entries(self):
        return self.n_entries # Returns the n_entries