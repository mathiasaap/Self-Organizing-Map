from som import SOM
import numpy as np
import math
class TSPSom(SOM):
    def __init__(self, dataset, nodes, sigma_0, learn_rate_0, total_iterations, scaler, sigma_timeconst):
        SOM.__init__(self, dataset, nodes, sigma_0, learn_rate_0, total_iterations, sigma_timeconst)
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


        print("Found path with length {}". format(pathlen))
