class Node:
    def __init__(self, value, parents=(), op=""):
        self.value = value
        self.parents = parents
        self.grad = 0.0
        self._backward = lambda : None
        self.op = op

    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(
            value=self.value + other.value,
            parents=(self, other),
            op="+"
        )
        def _backward():
            self.grad += self.out.grad
            other.grad += self.out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(
            value=self.value * other.value,
            parents=(self, other),
            op="*"
        )
        def _backward():
            self.grad += self.out.grad * other.value
            other.grad += self.out.grad * self.value
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

if __name__ == '__main__':
    x = Node(3.0)
    y = x * x + 2 * x
    y.backward()
    assert y.value == 15.0
    assert x.grad == 8.0