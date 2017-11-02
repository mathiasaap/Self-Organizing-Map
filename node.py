import numpy as np
import math

class Node(object):

    def __init__(self, nodes_in):
        self.weights = np.random.rand(nodes_in)

    def dist(self, V):
        sqdist = 0
        for v,w in zip(V,self.weights):
            sqdist += (v-w)**2
        return sqdist

    def S(self, other_node):
        raise NotImplementedError

    def T(self, other_node, sigma):
        raise NotImplementedError

    def update_weight(self, center_node, learn_rate, sigma):
        t = self.T(center_node, sigma)
        distance = center_node.weights - self.weights
        #print(learn_rate * t * distance)
        self.weights += learn_rate * t * distance