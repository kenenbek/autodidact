from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Callable, Tuple, Optional
from functools import wraps

import numpy as onp


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
    __array_ufunc__ = None
    def __init__(self, value: Any, node: Node):
        self.value = value
        self.node = node

    @property
    def shape(self):
        return onp.shape(self.value)

    @property
    def ndim(self):
        return onp.ndim(self.value)

    def __sub__(self, other):
        return subtract(self, other)

    def __rsub__(self, other):
        return subtract(other, self)

    def __truediv__(self, other):
        return divide(self, other)

    def __rtruediv__(self, other):
        return divide(other, self)

    def __neg__(self):
        return negative(self)

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

@primitive
def divide(x, y):
    return x / y

@primitive
def negative(x):
    return -x

@primitive
def sum_(x, axis=None, keepdims=False):
    return onp.sum(x, axis=axis, keepdims=keepdims)

@primitive
def reshape(x, shape):
    return onp.reshape(x, shape)

@primitive
def transpose(x):
    return onp.transpose(x)

@primitive
def exp(x):
    return onp.exp(x)

@primitive
def log(x):
    return onp.log(x)

@primitive
def dot(x, y):
    if not (1 <= onp.ndim(x) <= 2 and 1 <= onp.ndim(y) <= 2):
        raise ValueError("dot requires rank-1 or rank-2 operands")
    return onp.dot(x, y)


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

def div_left_rule(g, result, x, y):
    return g / y

def div_right_rule(g, result, x, y):
    return (-1) * g * x / y**2

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


def backward_pass(seed, end_node, start_node):
    order = topological_sort(end_node)
    outgrads = {
        end_node: seed
    }

    for node in reversed(order):
        if node.recipe is None:
            continue
        outgrad = outgrads[node]
        recipe = node.recipe

        for argnum, parent in zip(recipe.argnums, node.parents):
            rule = get_vjp_rule(
                function=recipe.function,
                argnum=argnum
            )
            parent_contribution = rule(
                outgrad,
                recipe.result,
                *recipe.args,
                **recipe.kwargs
            )
            previous_total = outgrads.get(parent)
            outgrads[parent] = add_outgrads(previous_total, parent_contribution)

    return outgrads[start_node]


def describe_backward_step(node, outgrad):
    if node.recipe is None:
        return "root"
    return {
        "function": node.recipe.function.__name__,
        "outgrad": outgrad,
        "argnums": node.recipe.argnums,
        "parent_count": len(node.parents)
    }



def trace(function, x):

    start_node = Node.new_root()
    start_box = Box(
        value=x,
        node=start_node,
    )
    output = function(start_box)

    if type(output) is Box:
        return output.value, output.node, start_node
    return output, None, start_node


def make_vjp(function, x):
    result, end_node, start_node = trace(function, x)

    if end_node is None:
        def vjp(seed):
            return onp.zeros_like(x, dtype=float)
    else:
        def vjp(seed):
            if onp.shape(seed) != onp.shape(result):
                raise ValueError(f"seed shape {onp.shape(seed)} != output shape {onp.shape(result)}")
            gradient_start_node = backward_pass(seed, end_node, start_node)
            return gradient_start_node

    return vjp, result


defvjp(add, add_left_rule, add_right_rule)
defvjp(multiply, mul_left_rule, mul_right_rule)
defvjp(divide, div_left_rule, div_right_rule)


if __name__ == '__main__':
    a = multiply(Box(2.0, Node.new_root()), 6.0)
    b = multiply(Box(3.0, Node.new_root()), 4.0)
