import math
class Value:
    def __init__(self, value=0, _children=(), op=""):
        self.value = value
        self.grad=0
        self._backward = lambda: None
        self.children = set(_children)
        self.op = op
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value+other.value,(self,other),op="+")
    
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = _backward
        return out
    
    def __sub__(self,other):
        return self+(other*(-1))

    def __mul__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value*other.value,(self,other),op="*")

        def _backward():
            self.grad+=out.grad*other.value
            other.grad+=out.grad*self.value
        
        out._backward = _backward
        return out

    def __pow__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.value**other.value,(self,other), op="^")

        def _backward():
            self.grad += out.grad * other.value * ((self.value)**(other.value - 1))
            other.grad += out.grad * math.log(self.value) * out.value
        
        out._backward = _backward
        return out
    
    def relu(self):
        out = Value(0 if self.value<0 else self.value, (self,), op="ReLU")

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


        
