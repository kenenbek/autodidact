from dataclasses import dataclass
from typing import Any, Callable, Tuple, Optional


class Node:
    def __init__(self, parents: Tuple["Node", ...], recipe: Optional["Recipe"]):
        self.parents = parents
        self.recipe = recipe

    @classmethod
    def new_root(cls):
        return cls(
            parents=(),
            recipe=None
        )

@dataclass
class Recipe:
    function: Callable
    result: Any
    args: tuple
    kwargs: dict
    argnums: tuple


class Box:
    def __init__(self, value: float, node: Node):
        self.value = value
        self.node = node

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        return multiply(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

def find_boxed_args(args):
    res = []
    for i, arg in enumerate(args):
        if type(arg) is Box:
            res.append((i, arg))
    return tuple(res)

def unbox_args(args):
    res = []
    for arg in args:
        if type(arg) is Box:
            res.append(arg.value)
        else:
            res.append(arg)
    return tuple(res)


def primitive(function):
    def wrapper(*args, **kwargs):
        boxed_args = find_boxed_args(args)
        if not boxed_args:
            return function(*args, **kwargs)

        unboxed_args = unbox_args(args)
        result = function(*unboxed_args, **kwargs)
        parents = tuple(box.node for _, box in boxed_args)
        argnums = tuple(i for i, _ in boxed_args)
        recipe = Recipe(
            function=function,
            result=result,
            args=unboxed_args,
            kwargs=kwargs,
            argnums=argnums
        )
        node = Node(parents, recipe)
        return Box(result, node)
    return wrapper

@primitive
def add(x, y):
    return x + y

@primitive
def multiply(x, y):
    return x * y



if __name__ == '__main__':
    # Ordinary calls remain ordinary.
    assert add(2.0, 3.0) == 5.0
    assert multiply(2.0, 3.0) == 6.0

    # One traced argument.
    root = Node.new_root()
    x = Box(2.0, root)
    y = add(x, 3.0)

    assert y.value == 5.0
    assert y.node.parents == (root,)
    assert y.node.recipe.argnums == (0,)

    # Two traced arguments.
    y = multiply(x, x)

    assert y.value == 4.0
    assert y.node.parents == (root, root)
    assert y.node.recipe.argnums == (0, 1)

    # Chained tracing.
    z = x * x + 3.0

    assert z.value == 7.0
    assert len(z.node.parents) == 1
    assert z.node.parents[0].parents == (root, root)

    # Stable forward values.
    x.value = 10.0
    assert z.node.parents[0].recipe.args == (2.0, 2.0)