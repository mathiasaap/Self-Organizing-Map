from tspnode import TSPNode
import numpy as np
from somproblemfactory import SOMProblemFactory

som_generator = SOMProblemFactory()
som = som_generator.generate_problem("data/TSP/1.txt")
#noder = som.nodes

som.train()
som.report()
#print(noder[0].T(noder[47]))
#tspn = TSPNode(5,0,8)

#V = np.random.rand(8)

#print(tspn.dist(V))
