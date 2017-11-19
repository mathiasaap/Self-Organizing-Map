from tspnode import TSPNode
import numpy as np
from somproblemfactory import SOMProblemFactory
import sys
if len(sys.argv) > 1:
    som_generator = SOMProblemFactory()
    som = som_generator.read_config(sys.argv[1])
    som.train()
    som.report()
