# Why Section 3 Is Necessary

The current implementation in `first.py` is a good scalar autodiff engine for one backward pass. It successfully builds a DAG, visits nodes in reverse topological order, and accumulates gradient contributions from shared paths.

Its main limitation is architectural: one `Node` currently has three different jobs.

```text
Node
├── stores a numerical value
├── behaves like a number through __add__, __mul__, etc.
└── stores graph and backward state
```

Section 3 separates those jobs:

```text
Box                         Node
├── numerical value          ├── parent nodes
├── behaves like a value     └── recipe describing the operation
└── points to a Node
```

The following examples demonstrate where the current design becomes difficult or incorrect.

## Run all current failure demonstrations

Run this from the repository root without changing `first.py`:

```python
from my_autograd.first import Node
import math


def show(name, example):
    try:
        example()
    except Exception as error:
        print(f"{name}: {type(error).__name__}: {error}")


def repeated_backward():
    x = Node(3.0)
    a = x * x
    y = a * x

    y.backward()
    print("after first backward:", x.grad)

    y.backward()
    print("after second backward:", x.grad)


def mutate_after_forward():
    x = Node(2.0)
    y = x * x

    # The graph was built when x was 2, so y is still 4.
    x.value = 10.0
    y.backward()

    print("forward result:", y.value)
    print("gradient:", x.grad)


def constant_becomes_parent():
    x = Node(3.0)
    y = x + 2.0

    constant = y.parents[1]
    y.backward()

    print("number of parents:", len(y.parents))
    print("constant value:", constant.value)
    print("constant gradient:", constant.grad)


def unsupported_operation():
    x = Node(1.0)
    math.exp(x)


def constant_output():
    def f(x):
        return 7.0

    result = f(Node(3.0))
    result.backward()


show("repeated backward", repeated_backward)
show("mutation after forward", mutate_after_forward)
show("constant as parent", constant_becomes_parent)
show("math.exp", unsupported_operation)
show("constant output", constant_output)
```

Current output:

```text
after first backward: 27.0
after second backward: 72.0
forward result: 4.0
gradient: 20.0
number of parents: 2
constant value: 2.0
constant gradient: 1.0
math.exp: TypeError: must be real number, not Node
constant output: AttributeError: 'float' object has no attribute 'backward'
```

## Failure 1: backward closures read mutable values

Consider:

```python
x = Node(2.0)
y = x * x       # forward result is 4
x.value = 10.0
y.backward()

print(y.value)  # 4.0
print(x.grad)   # 20.0, but the derivative at x=2 is 4.0
```

The multiplication callback closes over `self` and `other`:

```python
def _backward():
    self.grad += out.grad * other.value
    other.grad += out.grad * self.value
```

It reads their values during the backward pass, not the values used during the forward pass. Because both operands are the same mutated `x`, it adds `10 + 10`.

Section 3 gives the graph node a stable forward recipe containing raw arguments:

```text
recipe.args == (2.0, 2.0)
```

The backward system can then calculate derivatives from the recorded forward operation instead of reading mutable wrapper objects.

## Failure 2: constants become fake differentiable parents

The current conversion logic wraps every constant in a `Node`:

```python
other = other if isinstance(other, Node) else Node(other)
```

Therefore:

```python
x = Node(3.0)
y = x + 2.0
```

produces this graph:

```text
x --------\
           + --> y
Node(2.0) -/
```

The constant receives `grad == 1.0`, even though nobody requested differentiation with respect to it. This does not make `x.grad` numerically wrong, but it creates unnecessary graph nodes and loses the distinction between traced inputs and ordinary constants.

With `Box` and `primitive`, only boxed arguments become parents:

```text
Box(x) ----> add ----> Box(y)
                 ^
                 |
             raw 2.0
```

The recipe may record both raw arguments, but the graph contains a parent edge only for the boxed `x`.

## Failure 3: tracing is hard-coded into every operator

Currently, graph construction is duplicated inside every method:

```python
def __mul__(self, other):
    # unwrap/coerce values
    # calculate the result
    # find parents
    # create a graph node
    # define backward behavior
    # return the node
```

Consequently, an operation that does not understand `Node` fails:

```python
import math

x = Node(1.0)
math.exp(x)  # TypeError
```

Adding every operation as another large `Node` method does not scale. A `primitive` decorator centralizes the tracing algorithm:

```python
@primitive
def add(x, y):
    return x + y


@primitive
def multiply(x, y):
    return x * y
```

The same wrapped function then has two modes:

```python
add(2.0, 3.0)       # no Box arguments: return raw 5.0
add(Box(2.0), 3.0)  # traced argument: return Box(5.0) with a Node
```

Later, new numerical operations can use the same tracing mechanism rather than reimplementing graph construction.

## Failure 4: the current object cannot represent nested tracing cleanly

Eventually, higher-order differentiation needs a value to participate in two roles simultaneously:

```text
inner trace: a numerical value being differentiated
outer trace: a traced value belonging to another graph
```

If `Node` is both the value and the graph record, these roles become entangled. A `Box` can instead wrap another boxed value while each trace owns separate graph nodes:

```text
Box(value=Box(value=2.0, outer_node), inner_node)
```

Section 3 does not implement higher derivatives yet. It creates the separation needed to support trace levels later without redesigning the entire engine.

## Related bug: repeated `backward()` calls

The repeated-backward example produces `72` instead of `54`:

```python
x = Node(3.0)
a = x * x
y = a * x

y.backward()  # x.grad = 27, a.grad = 3
y.backward()  # x.grad = 72, because a.grad starts at 3
```

On the second pass, `y` adds another `3` to the stale `a.grad`, making it `6`. Then `a = x*x` sends `6*3` through each of its two edges. Intermediate gradients describe one backward pass and must not accumulate across passes.

This bug can be fixed in the current design by resetting non-leaf gradients before traversal. It is related to mixed responsibilities in `Node`, but it is not the primary reason for Section 3. Later stages can make each backward pass keep intermediate gradients in its own local mapping rather than storing them permanently on graph nodes.

## What Section 3 should prove

Do not implement NumPy, broadcasting, VJPs, or higher-order gradients yet. The stage is complete when these structural checks work:

```python
# Raw calls behave like ordinary functions.
assert add(2.0, 3.0) == 5.0
assert multiply(2.0, 3.0) == 6.0

# A boxed argument triggers tracing.
root = Node.new_root()
x = Box(value=2.0, node=root)
y = add(x, 3.0)

assert isinstance(y, Box)
assert y.value == 5.0
assert y.node.parents == (root,)

# Only traced arguments become graph parents.
assert len(y.node.parents) == 1

# The recipe stores raw forward values, not mutable Box objects.
z = multiply(x, x)
assert z.node.recipe.args == (2.0, 2.0)
assert z.node.parents == (root, root)
```

The central lesson is:

```text
Box answers: "What value is flowing through the program?"
Node answers: "How was that value produced?"
primitive answers: "When should an ordinary operation be recorded?"
```

Keeping those answers separate makes later VJPs, NumPy support, and nested differentiation possible without embedding every feature directly into one class.
