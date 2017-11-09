import numpy as np
import math

class Node(object):

    def __init__(self, nodes_in):
        self.weights = np.random.rand(nodes_in)

    def dist(self, V):
        return np.linalg.norm(V-self.weights)

    def S(self, other_node):
        raise NotImplementedError

    def T(self, other_node, sigma):
        raise NotImplementedError

    def update_weight(self, center_node, datapoint, learn_rate, sigma):
        t = self.T(center_node, sigma)
        distance = datapoint - self.weights
        #print("T value {}".format(t))
        #print(learn_rate * t * distance)
        self.weights += learn_rate * t * distance

    def set_weight(self, weights):
        self.weights = weights

    def __lt__(self, other):
        raise NotImplementedError
