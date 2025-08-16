import math
from graphviz import Digraph
class Value:
    def __init__(self, value=0, _children=(), _op=""):
        self.value = value
        self.grad=0
        self._backward = lambda: None
        self.children = set(_children)
        self._op = _op
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value+other.value,(self,other),_op="+")
    
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = _backward
        return out
    
    def __sub__(self,other):
        return self+(other*(-1))

    def __mul__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value*other.value,(self,other),_op="*")

        def _backward():
            self.grad+=out.grad*other.value
            other.grad+=out.grad*self.value
        
        out._backward = _backward
        return out

    def __pow__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.value**other.value,(self,other), _op="^")

        def _backward():
            self.grad += out.grad * other.value * ((self.value)**(other.value - 1))
        
        out._backward = _backward
        return out
    
    def relu(self):
        out = Value(0 if self.value<0 else self.value, (self,), _op="ReLU")

        def _backward():
            self.grad += (0 if self.value<0 else 1)*out.grad

        out._backward = _backward
        return out

    def backward(self):
        order = []
        visited = set()
        def build_order(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_order(child)
                order.append(v)
        build_order(self)
        self.grad=1
        for v in reversed(order):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.value}, grad={self.grad})"


def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.children:
                edges.add((child,v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg',graph_attr={'rankdir':'LR'})
    nodes,edges = trace(root)
    for n in nodes:
        dot.node(name=str(id(n)), label = "{ data %.4f | grad %.4f }" % (n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot