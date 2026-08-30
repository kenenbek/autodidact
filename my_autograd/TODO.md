# My Autograd TODO

If your goal is to learn by rebuilding the whole library, don’t begin with NumPy wrapping. Start with the smallest possible reverse-mode autodiff engine and add machinery only when a test demands it.

Create a separate directory such as `my_autograd/` so the original remains available as a reference—but avoid reading it until you finish each milestone.

## 1. Build a scalar computation graph

Start with a `Node` representing one scalar value:

```python
class Node:
    def __init__(self, value, parents=()):
        self.value = value
        self.parents = parents
        self.grad = 0.0
```

Each parent connection needs to describe how an output gradient contributes to that parent:

```text
parent edge = (parent_node, local_derivative)
```

Implement only:

- Addition
- Multiplication
- Negation
- Division
- Power by a constant

Your first target:

```python
x = Node(3.0)
y = x * x + 2 * x
y.backward()

assert y.value == 15.0
assert x.grad == 8.0
```

This teaches the essence of reverse-mode autodiff:

```text
forward pass: calculate values and build a graph
backward pass: propagate gradients in reverse topological order
```

## 2. Write topological traversal

Implement a DFS that returns every ancestor before its children. Reverse that order during backpropagation.

Test shared graph paths carefully:

```python
x = Node(3.0)
y = x * x
```

Both operands point to the same `x`, so its two gradient contributions must be added. Getting this right is essential.

Also test:

```python
x = Node(2.0)
a = x * x
y = a + a
assert dy_dx == 8.0
```

## 3. Separate values from tracing

Once the scalar engine works, replace the all-in-one `Node` design with the architecture used by larger autodiff systems:

```text
Box
├── underlying numerical value
└── computation-graph Node

Node
├── parent nodes
└── recipe for backward propagation
```

Create a `primitive` decorator that:

1. Detects boxed arguments.
2. Extracts their raw values.
3. Calls the original function.
4. Creates a graph node.
5. Returns the result in another box.

Initially support just:

```python
@primitive
def add(x, y):
    return x + y

@primitive
def multiply(x, y):
    return x * y
```

## 4. Introduce VJPs

Don’t store scalar local derivatives on edges anymore. Register vector-Jacobian product functions by operation and argument number:

```python
defvjp(
    multiply,
    lambda g, ans, x, y: g * y,
    lambda g, ans, x, y: g * x,
)
```

Then implement:

```python
vjp, result = make_vjp(function, x)
gradient = vjp(1.0)
```

Test these functions before adding anything else:

```python
f = lambda x: x * x
f = lambda x: x * x + x
f = lambda x: x * x * x
f = lambda x: 7.0
```

The constant function should return a zero gradient.

## 5. Build `grad()`

`grad()` should be a small convenience layer over `make_vjp()`:

```text
grad(f)(x)
    = make_vjp(f, x)
    = seed output gradient with 1
    = return input gradient
```

Add support for `argnum`:

```python
def f(x, y):
    return x * y

assert grad(f, argnum=0)(3.0, 4.0) == 4.0
assert grad(f, argnum=1)(3.0, 4.0) == 3.0
```

## 6. Add NumPy last

Only after scalar tracing works:

1. Create an `ArrayBox`.
2. Overload arithmetic operators.
3. Wrap selected NumPy functions explicitly.
4. Register their VJPs.
5. Handle broadcasting.

Start with this tiny set:

- `add`
- `subtract`
- `multiply`
- `divide`
- `exp`
- `log`
- `sum`
- `reshape`
- `dot`

Avoid dynamically wrapping all of NumPy at first. Explicit wrappers are much easier to debug.

Broadcasting needs special attention. For example:

```python
x = np.array([1.0, 2.0, 3.0])
b = np.array([4.0])
y = np.sum(x * b)
```

The gradient with respect to `b` must have shape `(1,)`, not `(3,)`. You’ll need an `unbroadcast()` operation that sums gradients across dimensions introduced by broadcasting.

## 7. Add higher-order derivatives

Once VJPs themselves use wrapped operations, this should work:

```python
f = lambda x: x ** 3

assert grad(f)(2.0) == 12.0
assert grad(grad(f))(2.0) == 12.0
assert grad(grad(grad(f)))(2.0) == 6.0
```

This is where trace levels become necessary: nested calls to `grad()` must distinguish inner graph nodes from outer graph nodes.

Leave trace IDs until this milestone. They’re difficult to appreciate before you encounter the problem they solve.

## Recommended implementation order

```text
scalar Node
→ arithmetic operations
→ topological backward pass
→ Box and primitive tracing
→ VJP registry
→ make_vjp
→ grad
→ ArrayBox
→ NumPy primitives
→ broadcasting
→ higher-order gradients
→ nested trace levels
```

For every operation, compare its gradient with finite differences:

```python
def numerical_grad(f, x, eps=1e-6):
    return (f(x + eps) - f(x - eps)) / (2 * eps)
```

Your first session should stop after this passes:

```python
f = lambda x: x * x + 2 * x
assert abs(grad(f)(3.0) - numerical_grad(f, 3.0)) < 1e-5
```

That small result contains the fundamental mechanism behind the entire repository.
