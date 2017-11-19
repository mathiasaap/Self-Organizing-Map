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
        self.reset_labels()


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

    def reset_labels(self):
        self.labels = np.zeros(10,dtype=np.int)

    def T(self, other_node, sigma):
        return math.exp((-(self.S(other_node)**2)) / (2 * sigma**2))

    def add_label(self, label):
        self.labels[label] += 1

    def get_number(self):
        return self.labels.argmax()

    def serialize(self):
        element = {}
        element['weights'] = self.weights.tolist()
        element['x'] = self.x
        element['y'] = self.y
        element['labels'] = [float(n) for n in self.labels]
        return element
