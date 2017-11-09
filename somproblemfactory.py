import numpy as np
from tspnode import TSPNode
from som import SOM
from tspsom import TSPSom
from sklearn.preprocessing import MinMaxScaler
import math

class SOMProblemFactory:

    def generate_TSP(self, filename):
        with open(filename, 'r') as file:
            cities = int(file.readline().strip().split(" ")[-1])
            file.readline() # Don't care
            dataset = np.empty([cities, 2])
            for i, line in enumerate(file):
                if "EOF" in line:
                    break
                line_data = line.split(" ")
                dataset[i][0] = line_data[1]
                dataset[i][1] = line_data[2]
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        nodes = []
        output_nodes = 2 * cities
        for i in range(output_nodes):
            nodes.append(TSPNode(i, output_nodes))

        total_iterations = 2000
        sigma_timeconst = total_iterations/math.log(cities)
        return TSPSom(dataset, nodes, sigma_0 = output_nodes, scaler = scaler,learn_rate_0 = 1, total_iterations= total_iterations, sigma_timeconst = sigma_timeconst)






    def generate_problem(self, filename):
        if "TSP" in filename:
            return self.generate_TSP(filename)
