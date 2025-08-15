from core import *
import random

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, number, nonlinearity = True):
        self.w = [Value(random.uniform(-1,1)) for i in range(0,number)]
        self.b = Value(0)
        self.nonlin = nonlinearity

    def __call__(self,x):
        activation = sum((wi*xi for wi,xi in zip(self.w,x)))
        return activation.relu() if self.nonlin else activation
    
    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"
    
class Layer(Module):
    def __init__(self):
        pass
class MLP(Module):
    def __init__(self):
        pass