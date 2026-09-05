from collections import defaultdict
from third import Box, add, multiply, Node

primitive_vjps = defaultdict(dict)

def defvjp(function, *vjps):
    for argnum, vjp in enumerate(vjps):
        primitive_vjps[function][argnum] =  vjp

def add_left_rule(g, result, x, y):
    return g * 1

def add_right_rule(g, result, x, y):
    return g * 1

def mul_left_rule(g, result, x, y):
    return g * y

def mul_right_rule(g, result, x, y):
    return g * x


if __name__ == '__main__':
    defvjp(add, add_left_rule, add_right_rule)
    defvjp(multiply, mul_left_rule, mul_right_rule)

    left = primitive_vjps[multiply][0]
    right = primitive_vjps[multiply][1]

    assert left(1.0, 6.0, 2.0, 3.0) == 3.0
    assert right(1.0, 6.0, 2.0, 3.0) == 2.0

    # An incoming seed scales the result.
    assert left(10.0, 6.0, 2.0, 3.0) == 30.0
