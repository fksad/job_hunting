# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2025/2/2 14:51
import numpy as np


class Value:
    def __init__(self, value, children=None, op=''):
        self._value = value
        self.grad = 0
        if children is not None:
            self._children = set(children)
        else:
            self._children = set()
        self._op = op
        self._backward = lambda : None

    @property
    def data(self):
        return self._value

    @data.setter
    def data(self, new_data):
        self._value = new_data

    @property
    def children(self):
        return self._children

    def backward(self):
        # topological order all of the children in the graph
        children = []
        visited = set()
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node.children:
                    build_topo(child)
                children.append(node)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for child in reversed(children):
            child._backward()

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        out = Value(1 / (1 + np.exp(-self.data)), (self,), 'Sigmoid')
        def _backward():
            self.grad += out.grad * out.data * (1 - out.data)
        out._backward = _backward
        return out

    def exp(self):
        out = Value(np.exp(self._value), [self], 'exp')
        def _backward():
            self.grad += out.grad * out.data
        out._backward = _backward
        return out

    def log(self):
        out = Value(np.log(self._value), [self], 'log')
        def _backward():
            if self.data > 0:
                self.grad += out.grad * 1 / self.data
            else:
                self.grad += out.grad * 1 / -self.data
        out._backward = _backward
        return out

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self._value + other._value, [self, other], '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self._value * other._value, [self, other], '*')
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward = _backward
        return out

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only support int or float exponential number"
        out = Value(self._value ** other, [self], f"**{other}**")
        def _backward():
            self.grad += out.grad * (other * self._value ** (other - 1))
        out._backward = _backward
        return out

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __rtruediv__(self, other):  # other / self
        return other * self ** -1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __str__(self):
        return repr(self)

