import math
import random

class SOM(object):

    def __init__(self, dataset, nodes, sigma_0, learn_rate_0, total_iterations, sigma_timeconst):
        self.total_iterations = total_iterations
        self.iteration = 1
        self.dataset = dataset
        self.nodes = nodes
        self.sigma_0 = sigma_0
        self.learn_rate_0 = learn_rate_0
        self.sigma_timeconst = sigma_timeconst


    def train(self):
        while self.iteration <= self.total_iterations:
            #print("Iteration {}".format(self.iteration))
            datapoint = random.choice(self.dataset)
            closest_node = self.closest_node(datapoint)
            self.update_nodes(closest_node, datapoint)
            self.iteration += 1



    # Neighbourhood size
    def sigma(self):
        return self.sigma_0 * math.exp(-self.iteration / self.sigma_timeconst)

    def learn_rate(self):
        return self.learn_rate_0 * math.exp(-self.iteration / self.total_iterations)

    def closest_node(self, datapoint):
        mindist = float("inf")
        winning_node = None
        for node in self.nodes:
            dist = node.dist(datapoint)
            if dist < mindist:
                mindist = dist
                winning_node = node
        return winning_node


    def update_nodes(self, center_node, datapoint):
        learn_rate = self.learn_rate()
        sigma = self.sigma()

        for node in self.nodes:
            node.update_weight(center_node, datapoint, learn_rate, sigma)

    def report(self):
        raise NotImplementedError
