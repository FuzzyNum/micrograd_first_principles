from core import *
import random

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, number_input, nonlinearity = True):
        self.w = [Value(random.uniform(-2,2)) for _ in range(0,number_input)]
        self.b = Value(0)
        self.nonlin = nonlinearity

    def __call__(self,x):
        activation = sum((wi*xi for wi,xi in zip(self.w,x)))
        return activation.relu() if self.nonlin else activation
    
    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"
    
class Layer(Module):
    def __init__(self, number_input, number_output, **kwargs):
        self.neurons = [Neuron(number_input,**kwargs) for i in range(0,number_output)]
    
    def __call__(self,x):
        output = [n(x) for n in self.neurons]
        return output[0] if len(output) == 1 else output
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
      
        
class MLP(Module):
    def __init__(self,nin,nout_list):
        size = [nin] + nout_list
        self.layers = [Layer(size[i],size[i+1]) for i in range(len(nout_list))]

    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


class Experiment(Module):
    def __init__(self,nin,nouts):
        self.model = MLP(nin,nouts)
    
    def __call__(self,x,y,num_epochs=5):
        for i in range(num_epochs):
            y_pred = [self.model(i) for i in x]
            print(y_pred)
            print(y)
            loss = sum((yout-ygt)**2 for yout, ygt in zip(y_pred,y))
            loss.backward()
            for p in self.model.parameters():
                p.value += 0.01*p.grad
        return loss
        

    

    
    

