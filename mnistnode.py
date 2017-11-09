from node import Node
import math
import numpy as np

class MNISTNode(Node):
    def __init__(self, x, y, nodes_per_dim, dimensions):
        Node.__init__(self, dimensions)
        self.x = x
        self.y = y
        self.nodes_per_dim = nodes_per_dim
        self.labels_history = 20
        self.current_label_index = 0
        self.labels = np.zeros(self.labels_history, dtype = np.int)

    def __repr__(self):
        return "Position {} of {}".format(self.x, self.total_nodes)

    def S(self, other):
        d1 =  math.sqrt((self.x-other.x)**2 + (self.y-other.y)**2)
        return d1
        d2 =  math.sqrt((self.x + self.nodes_per_dim-other.x)**2 + (self.y-other.y)**2)
        d3 =  math.sqrt((self.x - other.x)**2 + (self.y + self.nodes_per_dim - other.y)**2)
        d4 =  math.sqrt((self.x + self.nodes_per_dim - other.x)**2 + (self.y + self.nodes_per_dim - other.y)**2)
        return min(d1,d2,d3,d4)
        return d1

    def T(self, other_node, sigma):
        return math.exp((-(self.S(other_node)**2)) / (2 * sigma**2))

    def add_label(self, label):
        self.labels[self.current_label_index] = label
        self.current_label_index = (self.current_label_index + 1) % self.labels_history

    def get_number(self):
        return np.bincount(self.labels).argmax()

    def serialize(self):
        element = {}
        element['weights'] = self.weights.tolist()
        element['x'] = self.x
        element['y'] = self.y
        element['labels'] = [float(n) for n in self.labels]
        return element
