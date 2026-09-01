from dataclasses import dataclass
from typing import Any, Callable


class Node:
    def __init__(self, parents, recipe):
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
    def __init__(self, value, node):
        self.value = value
        self.node = node


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

if __name__ == '__main__':
    root = Node.new_root()
    # args = (Box(2.0, root), 3.0)

    x = Box(2.0, root)

    assert find_boxed_args((x, 3.0)) == ((0, x),)
    assert unbox_args((x, 3.0)) == (2.0, 3.0)