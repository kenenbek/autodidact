from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Callable, Tuple, Optional
from functools import wraps


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
    @wraps(function)
    def wrapper(*args, **kwargs):
        boxed_args = find_boxed_args(args)
        if not boxed_args:
            return function(*args, **kwargs)

        unboxed_args = unbox_args(args)
        result = function(*unboxed_args, **kwargs)
        parents = tuple(box.node for _, box in boxed_args)
        argnums = tuple(i for i, _ in boxed_args)
        recipe = Recipe(
            function=wrapper,
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

@primitive
def subtract(x, y):
    return x - y

primitive_vjps = defaultdict(dict)

def defvjp(function, *vjps):
    for argnum, vjp in enumerate(vjps):
        primitive_vjps[function][argnum] =  vjp

def add_left_rule(g, result, x, y):
    return g

def add_right_rule(g, result, x, y):
    return g

def mul_left_rule(g, result, x, y):
    return g * y

def mul_right_rule(g, result, x, y):
    return g * x


def topological_sort(end_node):
    topo = []
    visited = set()

    def build_topo(node):
        if node not in visited:
            visited.add(node)
            for parent in node.parents:
                build_topo(parent)
            topo.append(node)

    build_topo(end_node)
    return topo


def add_outgrads(previous, contribution):
    if previous is None:
        return contribution
    return previous + contribution


def get_vjp_rule(function, argnum):
    function_rules = primitive_vjps.get(function)

    if function_rules is None:
        function_name = getattr(function, "__name__", repr(function))
        raise NotImplementedError(
            f"No VJPs registered for primitive {function_name!r}"
        )

    if argnum not in primitive_vjps[function]:
        function_name = getattr(function, "__name__", repr(function))
        raise NotImplementedError(
            f"No VJP registered for primitive {function_name!r} "
            f"argument {argnum}"
        )

    return primitive_vjps[function][argnum]


@primitive
def left_only(x, y):
    return x + y


defvjp(add, add_left_rule, add_right_rule)
defvjp(multiply, mul_left_rule, mul_right_rule)
defvjp(left_only, lambda g, result, x, y: g)



if __name__ == '__main__':
    root = Node.new_root()
    x = Box(2.0, root)
    result = multiply(x, 3.0)

    assert result.node.recipe.function is multiply