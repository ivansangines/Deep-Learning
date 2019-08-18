import sys
import tensorflow as tf
from SimpleGraph import SimpleGraph
from NNGraph1 import NNGraph1
from NNGraph2 import NNGraph2

def main():
    #sim1 = SimpleGraph()
    #sim1.simpleComputation()
    #quad = QuadraticGraph();
    #quad.computeRoot();
    #nn1 = NNGraph1()
    #nn1.trainAndTestArchitecture()
    nn1 = NNGraph2()
    nn1.trainAndTestArchitectures()

if __name__ == "__main__":
    sys.exit(int(main() or 0))
