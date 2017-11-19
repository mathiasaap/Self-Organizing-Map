from som import SOM
import numpy as np
import math
from tspgraphics import TSPGraphics

class TSPSom(SOM):
    def __init__(self, dataset, nodes, sigma_0, learn_rate_0, total_iterations, scaler, sigma_timeconst, learn_t, plot_interval):
        SOM.__init__(self, dataset, nodes, sigma_0 = sigma_0, learn_rate_0 = learn_rate_0, total_iterations = total_iterations, sigma_timeconst = sigma_timeconst, learn_timeconst = learn_t, plot_interval = plot_interval, graphics = TSPGraphics())
        self.scaler = scaler

    def tsp_dist(self, weight1, weight2):
        return np.sqrt(np.sum((weight1 - weight2) ** 2))


    def report(self):
        self.find_path_nodes()
        dataset = np.empty([len(self.path), 2])
        for i, node in enumerate(self.path):
            dataset[i][0] = node.weights[0]
            dataset[i][1] = node.weights[1]
        dataset = self.scaler.inverse_transform(dataset)


        pathlen = 0
        for i in range(len(self.dataset)):
            pathlen += self.tsp_dist(dataset[i], dataset[(i+1)%len(self.dataset)])

        self.graphics.draw_frame(self, 'DONE')
        self.graphics.wait()
        print("Found path with length {}". format(pathlen))

    def find_path_nodes(self):
        path = []
        for data in self.dataset:
            distance = float('inf')
            node_index = 0
            for i, node in enumerate(self.nodes):
                dist = node.dist(data)
                if(dist < distance):
                    node_index = i
                    distance = dist
            path.append(self.nodes.pop(node_index))
            path[-1].set_weight(data)
        path.sort()
        self.path = path
        self.nodes = path
