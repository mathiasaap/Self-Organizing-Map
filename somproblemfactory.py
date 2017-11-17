import numpy as np
from tspnode import TSPNode
from mnistnode import MNISTNode
from som import SOM
from tspsom import TSPSom
from mnistsom import MNISTSom
from sklearn.preprocessing import MinMaxScaler
import math
import random

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
            nodes.append(TSPNode(i, output_nodes, cities))

        total_iterations = 50000
        sigma_timeconst = total_iterations/math.log(cities)
        return TSPSom(dataset, nodes, sigma_0 = cities*0.3, scaler = scaler,learn_rate_0 = 0.5, total_iterations= total_iterations, sigma_timeconst = sigma_timeconst)

    def generate_mnist_classifier(self, nodes = None):
        from mnist_basics import gen_flat_cases
        x,y = gen_flat_cases()
        #print(x,y)
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(x)
        classes = len(list(set(y))) +5
        print(classes)
        if nodes is None:
            nodes = []
            for i in range(classes):
                for j in range(classes):
                    nodes.append(MNISTNode(i, j, classes, len(x[0])))

        train_cases = 1500
        test_cases = 200
        samples = random.sample(range(0,len(dataset)), train_cases + test_cases)

        train_x, train_y, test_x, test_y = [], [], [], []
        for sample in samples[:train_cases]:
            train_x.append(dataset[sample])
            train_y.append(y[sample])
        for sample in samples[train_cases:]:
            test_x.append(dataset[sample])
            test_y.append(y[sample])
        total_iterations = 150
        #sigma_timeconst = total_iterations/math.log(classes)
        sigma_timeconst = 10
        return MNISTSom(train_x, train_y, test_x, test_y, nodes, sigma_0 = classes*2, scaler = scaler, learn_rate_0 = 0.3, total_iterations= total_iterations, sigma_timeconst = sigma_timeconst )

    def load_json(self, filename):
        import json
        with open(filename, 'r') as file:
            json_str = file.read()

        json = json.loads(json_str)
        if json['type'] == 'mnist':
            nodes = []
            for node in json['nodes']:
                mnistnode = MNISTNode(node['x'], node['y'], json['classes'], 0)
                mnistnode.weights = np.asarray(node['weights'])
                mnistnode.labels = np.asarray(node['labels'], dtype = np.int)
                nodes.append(mnistnode)
            return self.generate_mnist_classifier(nodes)



    def generate_problem(self, filename):
        if filename.endswith('.json'):
            return self.load_json(filename)
        if "TSP" in filename:
            return self.generate_TSP(filename)
        if "MNIST" in filename:
            return self.generate_mnist_classifier()
