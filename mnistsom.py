from som import SOM
import numpy as np
import math
from mnistgraphics import MNISTGraphics
import random

class MNISTSom(SOM):
    def __init__(self, dataset, labels, nodes, sigma_0, learn_rate_0, total_iterations, scaler, sigma_timeconst):
        SOM.__init__(self, dataset, nodes, sigma_0, learn_rate_0, total_iterations, sigma_timeconst, graphics = MNISTGraphics())
        self.scaler = scaler
        self.labels = labels


    def train(self):
        while self.iteration <= self.total_iterations:
            datanumber = random.randint(0,len(self.dataset) -1 )
            datapoint = self.dataset[datanumber]
            label = self.labels[datanumber]

            closest_node = self.closest_node(datapoint)
            closest_node.add_label(label)
            self.iteration += 1
            self.update_nodes(closest_node, datapoint)

            if(self.iteration % 100 == 0):
                self.graphics.draw_frame(self, self.iteration)

        self.graphics.wait()
        self.test()
        self.save("mnist-network.json", 'mnist')



    def test(self, cases = 1000):
        correct = 0
        for i in range(cases):
            datanumber = random.randint(0,len(self.dataset) -1 )
            datapoint = self.dataset[datanumber]
            label = self.labels[datanumber]

            closest_node = self.closest_node(datapoint)
            if closest_node.get_number() == label:
                correct += 1
        print("Accuracy: {}".format(correct/cases))


    def report(self):
        pass
