from enum import Enum

class PoolingType(Enum):
    NONE
    MAXPOOLING
    AVGPOOLING

class ActivationType(Enum):
    SIGMOID
    TANH
    RELU
    SOFTMAX


