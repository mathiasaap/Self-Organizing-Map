import math
import random


class SOM(object):

    def __init__(self, dataset, nodes, sigma_0, learn_rate_0, total_iterations, sigma_timeconst, graphics):
        self.total_iterations = total_iterations
        self.iteration = 1
        self.dataset = dataset
        self.nodes = nodes
        self.sigma_0 = sigma_0
        self.learn_rate_0 = learn_rate_0
        self.sigma_timeconst = sigma_timeconst
        self.graphics = graphics


    def train(self):
        while self.iteration <= self.total_iterations:
            #print("Iteration {}".format(self.iteration))
            datapoint = random.choice(self.dataset)
            closest_node = self.closest_node(datapoint)
            self.update_nodes(closest_node, datapoint)
            self.iteration += 1
            if(self.iteration % 25 == 0):
                self.graphics.draw_frame(self, self.iteration)



    # Neighbourhood size
    def sigma(self):
        return self.sigma_0 * math.exp(-self.iteration / self.sigma_timeconst)

    def learn_rate(self):
        self.learnrate_timeconst = 100
        return self.learn_rate_0 * math.exp(-self.iteration / self.learnrate_timeconst)

    def closest_node(self, datapoint):
        mindist = float("inf")
        winning_node = None
        for node in self.nodes:
            dist = node.dist(datapoint)
            if dist < mindist:
                mindist = dist
                winning_node = node
        return winning_node

    def save(self, filename, type):
        import json
        nodes_dict = []
        for node in self.nodes:
            nodes_dict.append(node.serialize())
        output = {}
        output['nodes'] = nodes_dict
        output['type'] = type
        output['classes'] = self.nodes[0].nodes_per_dim
        with open(filename, 'w') as file:
            file.write(json.dumps(output))



    def update_nodes(self, center_node, datapoint):
        learn_rate = self.learn_rate()
        sigma = self.sigma()
        #print("Learn rate: {}\nSigma: {}".format(learn_rate,sigma))

        for node in self.nodes:
            node.update_weight(center_node, datapoint, learn_rate, sigma)

    def report(self):
        raise NotImplementedError
