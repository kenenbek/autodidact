# Section 6: NumPy Arrays and Broadcasting

## Goal and scope

Section 5 finishes the scalar derivative API. Section 6 teaches that the graph algorithm also works for arrays: the main new responsibility is making each VJP return the correct shape.

Create `my_autograd/fifth.py` as a checkpoint derived from the completed Section 5 engine. Keep the engine, wrappers, registry, and `grad` together initially, so every primitive uses the same Box class and registry. Importing an older `grad` that still calls an older engine is an easy integration mistake.

Use real floating-point arrays. Start with explicit wrappers rather than automatically wrapping every NumPy function. No in-place mutation, advanced indexing, complex derivatives, or batched matrix multiplication yet.

The intended public calls are:

```python
import numpy as onp

x = onp.array([1.0, 2.0, 3.0])
gradient = grad(lambda x: sum_(x*x))(x)
onp.testing.assert_allclose(gradient, [2.0, 4.0, 6.0])
```

`onp` means ordinary NumPy. Your wrapped functions have explicit names such as `sum_`, `reshape`, and `dot`. A wrapper calls ordinary NumPy only when calculating raw numerical results.

## 1. Mental model: a node can carry a whole tensor

The graph does not need one node per array element:

```text
one Box containing x with shape (3,)
    -> multiply node containing x*x with shape (3,)
    -> sum node containing a scalar
```

During the reverse pass, every message has the shape of the value receiving it:

```text
scalar seed 1
    -> sum sends [1, 1, 1]
    -> multiply sends two copies of x
    -> root receives [2, 4, 6]
```

The topological traversal and local gradient dictionary remain the same. Array arithmetic replaces scalar arithmetic inside the messages.

For an output shaped `(m,)` and input shaped `(n,)`, a VJP accepts an `(m,)` seed and returns an `(n,)` gradient. It does not build an `(m, n)` Jacobian.

## 2. Decide the scalar-output contract for `grad`

Keep `grad` restricted to zero-dimensional outputs: Python scalars and NumPy scalar/0-D values. Array inputs are allowed.

For array outputs, callers must use `make_vjp` with an explicitly shaped seed:

```python
x = onp.array([1.0, 2.0])
vjp, result = make_vjp(lambda x: x*x, x)
onp.testing.assert_allclose(result, [1.0, 4.0])
onp.testing.assert_allclose(vjp(onp.array([3.0, 4.0])), [6.0, 16.0])
```

Validate the seed shape against the raw output shape; require an exact match. A scalar seed for an array output should raise `ValueError`, rather than relying on accidental broadcasting.

In `grad`, inspect the output shape, reject non-scalar output with `ValueError`, and construct the scalar seed using `onp.ones_like`.

```python
def grad(function, argnum=0):
    # Keep the argument adapter and validation from Section 5.
    def gradfun(*args, **kwargs):
        # TODO: Build unary_function and selected_value.
        vjp, result = ...
        # TODO: Require raw output shape == ().
        ...
        # TODO: Seed with ones_like(result), then return the VJP result.
        ...
    return gradfun
```

For input-independent output, replace the scalar-only `return 0.0` with zeros having the input's shape. Return floating-point zeros for the floating inputs supported here.

## 3. Let boxes carry arrays

You can extend your existing `Box` rather than creating a second class hierarchy. In this tutorial, `ArrayBox` means the array-capable version of that class. Remove a scalar-only type restriction and let it contain a scalar or ndarray.

Add shape metadata access:

```python
class Box:
    def __init__(self, value, node):
        self.value = value
        self.node = node

    @property
    def shape(self):
        # TODO: Return the numerical value's shape.
        ...

    @property
    def ndim(self):
        # TODO: Return its number of dimensions.
        ...

    def __sub__(self, other):
        return subtract(self, other)

    def __rsub__(self, other):
        # TODO: Preserve operand order.
        ...

    def __truediv__(self, other):
        return divide(self, other)

    def __rtruediv__(self, other):
        ...

    def __neg__(self):
        return negative(self)
```

Retain addition and multiplication. Use your explicit functions such as `multiply(x, weights)` in initial tests. For mixed `ndarray * Box` syntax, add a high `__array_priority__` and test both operand orders. Avoid implicit conversion of a Box to ndarray, which can erase the trace.

Do not allow modification of inputs between tracing and VJP evaluation. Tuples of array references do not snapshot mutable array contents; Section 3's scalar stability test did not provide general mutation safety.

## 4. Wrap ordinary NumPy functions explicitly

Define numerical primitives with stable signatures:

```python
@primitive
def sum_(x, axis=None, keepdims=False):
    # TODO: Delegate to onp.sum.
    ...

@primitive
def reshape(x, shape):
    ...

@primitive
def transpose(x):
    # Restrict this convenience primitive to reversing axes.
    ...

@primitive
def exp(x):
    ...

@primitive
def log(x):
    ...

@primitive
def dot(x, y):
    # Initially require both operands to have rank 1 or 2.
    # Reject scalar operands and ranks > 2 clearly.
    ...
```

Also wrap `add`, `subtract`, `multiply`, `divide`, and `negative` with their NumPy counterparts. Only positional numerical arguments are differentiable. Shape, axis, and `keepdims` are fixed configuration and may be keyword arguments.

Keep decorated wrapper identity as the registry key. Rerun the scalar tests after converting the primitive implementations.

## 5. Mental model: broadcasting copies a dependency

Consider a scalar bias shared across three outputs:

```python
values = onp.array([1.0, 2.0, 3.0])
loss = sum_(values + bias)
```

The forward computation uses the same bias three times. Its backward messages must be added:

```text
loss gradient for each output = [1, 1, 1]
bias gradient = 1 + 1 + 1 = 3
```

Broadcasting is therefore similar to repeated parent edges in `x*x`: several uses of one value contribute to the same input gradient.

For input shape `(3,)` broadcast against `(2, 3)`, sum the two rows of the output gradient. For input shape `(2, 1)` broadcast against `(2, 3)`, sum each row's three columns, keeping the singleton dimension.

## 6. Implement `unbroadcast`

Choose this interface:

```python
def unbroadcast(target, gradient):
    target_shape = onp.shape(target)

    # TODO: While gradient has extra leading axes, sum over axis 0.
    ...

    # TODO: For each singleton axis in target_shape, sum that axis
    # with keepdims=True.
    ...

    # TODO: Restore the exact target shape and return.
    ...
```

The backward invariant is stronger than equal numbers: returned gradients must have the exact input shape.

Checkpoint:

```python
def test_unbroadcast():
    g = onp.ones((2, 3))
    onp.testing.assert_allclose(unbroadcast(onp.zeros(3), g), [2, 2, 2])
    onp.testing.assert_allclose(unbroadcast(onp.zeros((2, 1)), g), [[3], [3]])
    scalar_gradient = unbroadcast(0.0, g)
    assert onp.shape(scalar_gradient) == ()
    onp.testing.assert_allclose(scalar_gradient, 6.0)
```

At this stage, ordinary NumPy reductions inside this helper are acceptable. Section 7 will replace differentiable helper arithmetic with traced operations.

## 7. Extend the elementwise VJPs

First calculate a local contribution, then reduce it to the corresponding input shape:

| Primitive | Argument 0 contribution | Argument 1 contribution |
|---|---|---|
| `add(x, y)` | `unbroadcast(x, g)` | `unbroadcast(y, g)` |
| `subtract(x, y)` | `unbroadcast(x, g)` | `unbroadcast(y, -g)` |
| `multiply(x, y)` | `unbroadcast(x, g*y)` | `unbroadcast(y, g*x)` |
| `divide(x, y)` | `unbroadcast(x, g/y)` | `unbroadcast(y, -g*x/(y*y))` |

Unary rules:

```text
negative(x): -g
exp(x):      g * result
log(x):      g / x
```

Test `log` at strictly positive values and division away from zero. Domain-policy design is outside this milestone.

## 8. Reverse a reduction

`sum_` reduces many numerical values to fewer values. Its VJP spreads the incoming gradient back over all original positions.

```text
x shape (2, 3), sum over axis 1 -> result shape (2,)
incoming seed [10, 20]
input gradient [[10, 10, 10], [20, 20, 20]]
```

Skeleton:

```python
def sum_vjp(g, result, x, axis=None, keepdims=False):
    input_shape = onp.shape(x)
    # TODO: Normalize axis to a tuple of nonnegative axis numbers.
    # axis=None means all axes; a tuple may contain multiple axes.
    ...
    # TODO: If keepdims=False, reshape g to reinsert reduced axes of size 1.
    ...
    # TODO: Broadcast the reshaped g back to input_shape.
    ...
```

Hint: multiply an appropriately reshaped `g` by an array of ones having `input_shape`. Keep the original order of unreduced axes. Test `axis=None`, one axis, a negative axis, an axis tuple, `axis=()`, and both values of `keepdims`.

For `reshape`, the reverse rule reshapes `g` to the original input shape. For `transpose`, reverse the transpose again. These operations change arrangement, not the numerical sensitivity associated with each element.

## 9. Matrix products: reverse the linear map

Implement the four supported rank combinations separately. A transposed linear map sends gradients back to its input.

| Forward | VJP for x | VJP for y |
|---|---|---|
| vectors: `dot(x, y)` | `g*y` | `g*x` |
| matrix-vector: `dot(A, x)` | `outer(g, x)` | `dot(transpose(A), g)` |
| vector-matrix: `dot(x, B)` | `dot(g, transpose(B))` | `outer(x, g)` |
| matrix-matrix: `dot(A, B)` | `dot(g, transpose(B))` | `dot(transpose(A), g)` |

Represent `outer(u, v)` as multiplication of reshaped arrays of shapes `(m, 1)` and `(1, n)`. That avoids adding another primitive, and will remain differentiable in Section 7.

```python
def dot_left_vjp(g, result, x, y):
    x_rank, y_rank = onp.ndim(x), onp.ndim(y)
    if (x_rank, y_rank) == (1, 1):
        ...
    elif (x_rank, y_rank) == (2, 1):
        ...
    elif (x_rank, y_rank) == (1, 2):
        ...
    elif (x_rank, y_rank) == (2, 2):
        ...
    else:
        raise NotImplementedError("dot supports rank-1 and rank-2 operands")
```

Write the matching right-argument function and register both callbacks. Do not copy reference formulas without checking rectangular shapes; square matrices can hide transpose mistakes.

## 10. Acceptance checks

```python
def test_array_gradients():
    x = onp.array([1.0, 2.0, 3.0])
    onp.testing.assert_allclose(grad(lambda x: sum_(x*x))(x), 2*x)

    b = onp.array([4.0])
    gb = grad(lambda b: sum_(multiply(x, b)))(b)
    assert gb.shape == (1,)
    onp.testing.assert_allclose(gb, [6.0])

    matrix = onp.arange(6.0).reshape(2, 3)
    bias = onp.zeros(3)
    onp.testing.assert_allclose(
        grad(lambda b: sum_(add(matrix, b)))(bias), [2, 2, 2]
    )

    zeros = grad(lambda x: 7.0)(matrix)
    assert zeros.shape == matrix.shape
    onp.testing.assert_allclose(zeros, onp.zeros_like(matrix))


def test_reduction_seed():
    matrix = onp.arange(6.0).reshape(2, 3)
    vjp, result = make_vjp(lambda x: sum_(x, axis=1), matrix)
    onp.testing.assert_allclose(result, [3, 12])
    onp.testing.assert_allclose(vjp(onp.array([10.0, 20.0])),
                                [[10, 10, 10], [20, 20, 20]])


def test_elementwise_and_reshape():
    x = onp.array([1.0, 2.0, 3.0])
    onp.testing.assert_allclose(grad(lambda x: sum_(log(x)))(x), 1/x)
    onp.testing.assert_allclose(grad(lambda x: sum_(exp(x)))(x), onp.exp(x))
    onp.testing.assert_allclose(grad(lambda x: sum_(divide(x, 2.0)))(x),
                                onp.full_like(x, 0.5))
    onp.testing.assert_allclose(grad(lambda x: sum_(reshape(x, (3, 1))))(x),
                                onp.ones_like(x))


def test_rectangular_dot():
    A = onp.arange(6.0).reshape(2, 3)
    x = onp.array([1.0, 2.0, 3.0])
    gx = grad(lambda x: sum_(dot(A, x)))(x)
    onp.testing.assert_allclose(gx, A.T @ onp.ones(2))

    B = onp.arange(12.0).reshape(3, 4)
    gA = grad(lambda A: sum_(dot(A, B)))(A)
    onp.testing.assert_allclose(gA, onp.ones((2, 4)) @ B.T)
```

Also test the other argument of every dot rank combination and both arguments of broadcasting primitives. `grad(function, argnum=...)` gives you that capability without introducing multiple roots.

Write exception checks for array output passed to `grad`, incorrectly shaped VJP seeds, and unsupported dot ranks.

## 11. Independent numerical validation

For a scalar loss `L` and array input, choose a direction `d` with the same shape. Compare:

```text
sum(grad(L)(x) * d)
    approximately equals
(L(x + epsilon*d) - L(x - epsilon*d)) / (2*epsilon)
```

Use floating inputs, a small epsilon such as `1e-6`, and a tolerance such as `1e-5`. This directional check validates all coordinates together without constructing a full Jacobian. Use deterministic inputs and test rectangular matrices and broadcast biases.

## Implementation order and completion

1. Preserve all scalar tests and make boxes array-capable.
2. Add explicit numerical primitives and array-shaped zero gradients.
3. Implement `unbroadcast` and elementwise VJPs.
4. Implement reduction, reshape, and transpose VJPs.
5. Implement the four dot cases and their argument rules.
6. Enforce `grad` output and VJP seed shape contracts.
7. Run analytical, shape, error, and finite-difference checks.

Run your named tests from `python -B my_autograd/fifth.py`.

Section 6 is complete when scalar losses can accept scalar or array inputs, VJPs support array outputs with explicit seeds, and every returned gradient matches the input shape. Higher derivatives remain Section 7 work.
