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
            datanumber = random.randint(0,len(self.dataset))
            datapoint = self.dataset[datanumber]
            label = self.labels[datanumber]

            closest_node = self.closest_node(datapoint)
            closest_node.add_label(label)
            self.update_nodes(closest_node, datapoint)
            self.iteration += 1
            if(self.iteration % 100 == 0):
                self.graphics.draw_frame(self)
                print(self.iteration)


    def report(self):
        pass
