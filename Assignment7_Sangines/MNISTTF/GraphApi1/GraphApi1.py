import sys
import tensorflow as tf
from SimpleGraph import SimpleGraph
from QuadraticGraph import QuadraticGraph

def main():
 sim1 = SimpleGraph()
 sim1.simpleComputation()
 #quad = QuadraticGraph();
 #quad.computeRoot();

if __name__ == "__main__":
 sys.exit(int(main() or 0))



