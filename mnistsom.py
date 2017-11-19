from som import SOM
import numpy as np
import math
from mnistgraphics import MNISTGraphics
import random

class MNISTSom(SOM):
    def __init__(self, train_x, train_y, test_x, test_y, nodes, sigma_0, learn_rate_0, total_iterations, learn_rate_timeconst, scaler, sigma_timeconst):
        SOM.__init__(self, train_x, nodes, sigma_0, learn_rate_0, sigma_timeconst, learn_rate_timeconst, total_iterations, graphics = MNISTGraphics())
        self.scaler = scaler
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y


    def train(self):

        while self.iteration <= self.total_iterations:
            for data in self.dataset:
                closest_node = self.closest_node(data)
                self.update_nodes(closest_node, data)

            if(self.iteration % 10 == 0):
                self.update_labels()
                self.graphics.draw_frame(self, self.iteration)
                self.test()
                print("Learn rate {}".format(self.learn_rate()))
                print("Sigma {}".format(self.sigma()))


            self.iteration += 1
        self.graphics.wait()
        self.test()


        """
        while self.iteration <= self.total_iterations:
            datanumber = random.randint(0,len(self.dataset) -1 )
            datapoint = self.dataset[datanumber]
            label = self.labels[datanumber]

            closest_node = self.closest_node(datapoint)
            closest_node.add_label(label)
            self.iteration += 1
            self.update_nodes(closest_node, datapoint)



        self.graphics.wait()
        self.test()
        self.save("mnist-network.json", 'mnist')

        """



    def test(self):
        correct = 0
        for data, label in zip(self.dataset, self.train_y):
            closest_node = self.closest_node(data)
            if closest_node.get_number() == label:
                correct += 1
        print("Accuracy: {}".format(correct/len(self.dataset)))

    def update_labels(self):
        for node in self.nodes:
            node.reset_labels()

        for data, label in zip(self.dataset, self.train_y):
            closest_node = self.closest_node(data)
            closest_node.add_label(label)
        print("updating labels done")

    """
    def update_labels(self, samples = 10000):
        for node in self.nodes:
            node.reset_labels()
        print("updating labels")
        for i in range(samples):
            datanumber = random.randint(0,len(self.dataset) -1 )
            datapoint = self.dataset[datanumber]
            label = self.labels[datanumber]
            closest_node = self.closest_node(datapoint)
            closest_node.add_label(label)
        print("updating labels done")
    """

    def report(self):
        pass
