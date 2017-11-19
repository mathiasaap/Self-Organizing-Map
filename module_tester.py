from tspnode import TSPNode
import numpy as np
from somproblemfactory import SOMProblemFactory

som_generator = SOMProblemFactory()
#som = som_generator.generate_problem("data/TSP/3.txt")
#som = som_generator.generate_problem("MNIST")
som = som_generator.read_config('configs/mnist_small.json')
#som = som_generator.generate_problem("mnist-network.json")
#noder = som.nodes

som.train()
som.report()
#som.test()
#print(noder[0].T(noder[47]))
#tspn = TSPNode(5,0,8)

#V = np.random.rand(8)

#print(tspn.dist(V))
