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
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(
            self.value - other.value,
            parents=(self, other),
            op="-"
        )
        def _backward():
            self.grad += out.grad
            other.grad += (-1.0) * out.grad
        out._backward = _backward
        return out

    def __rsub__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        return other - self

    def __neg__(self):
        out = Node(-self.value, (self, ), "-")
        def _backward():
            self.grad += -1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(
            value=self.value * other.value,
            parents=(self, other),
            op="*"
        )
        def _backward():
            self.grad += out.grad * other.value
            other.grad += out.grad * self.value
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(
            value=self.value / other.value,
            parents=(self, other),
            op="/"
        )

        def _backward():
            self.grad += (1.0 / other.value) * out.grad
            other.grad += (-self.value / other.value ** 2) * out.grad

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        return other / self

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Node(
            self.value ** other,
            parents=(self, ),
            op="**"
        )
        def _backward():
            self.grad += other * self.value ** (other - 1) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for parent in node.parents:
                    build_topo(parent)
                topo.append(node)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


if __name__ == '__main__':
    x = Node(3.0)
    a = x * x
    y = a * x
    y.backward()
    print(x.grad)
    y.backward()
    print(x.grad)

    #assert y.value == 15.0
    #assert x.grad == 8.0