from node import Node
import math

class TSPNode(Node):
    def __init__(self, x, total_cities):
        Node.__init__(self, 2)
        self.x = x
        self.total_nodes = total_cities

    def __repr__(self):
        return "Position {} of {}".format(self.x, self.total_nodes)

    def S(self, other_node):
        dist1 = abs(self.x - other_node.x)
        dist2 = abs(self.total_nodes - other_node.x + self.x)
        return min(dist1, dist2)

    def T(self, other_node, sigma):
        return math.exp((-self.S(other_node)**2) / 2 * sigma**2)
