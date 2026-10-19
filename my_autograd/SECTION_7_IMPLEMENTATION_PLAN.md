# Section 7: Higher-Order Derivatives and Nested Traces

## Goal and starting point

Section 6 differentiates scalar losses with array or scalar inputs. Section 7 makes the derivative computation itself differentiable.

Create `my_autograd/sixth.py` from your working Section 6 engine. Keep one consistent Box class, primitive registry, tracing implementation, and `grad` in this checkpoint. Preserve the earlier stage files.

Your first target uses existing primitives rather than adding power syntax:

```python
def cube(x):
    return x*x*x

assert grad(cube)(2.0) == 12.0
assert grad(grad(cube))(2.0) == 12.0
assert grad(grad(grad(cube)))(2.0) == 6.0
```

Use partial skeletons as exercises. This is a deeper architectural milestone; implement one concept and rerun a small test before adding the next.

## 1. Mental model: the backward pass is also a numerical program

For `f(x) = x*x`, your first derivative emerges from:

```text
left multiplication VJP  -> 1*x
right multiplication VJP -> 1*x
accumulate               -> x+x
```

That derivative is a program containing multiplication and addition. If an outer trace records those operations, it can differentiate the derivative program again.

```text
inner trace: differentiate f
outer trace: record the program that calculates f's derivative
```

You do not need symbolic differentiation or a special second-derivative registry. You need ordinary VJP arithmetic to remain visible to an outer trace.

## 2. Why boxes must sometimes contain boxes

During `grad(grad(f))(2.0)`, the outer trace first wraps `2.0`. The inner trace then receives that outer Box as its input value and wraps it again:

```text
inner Box
    value -> outer Box
                 value -> 2.0
                 node  -> outer root
    node  -> inner root
```

Unboxing once means removing one trace layer. Recursively unboxing everything means discarding the outer graph, which would destroy higher derivatives.

This updates a Section 3 invariant: recipes should contain arguments with the current trace layer removed. They may legitimately contain boxes from lower trace levels. An operation's result can also still be a lower-level Box.

## 3. Assign a level to each active trace

Add `trace_id` to each Box. It identifies the active nesting level owning that box.

```python
class Box:
    def __init__(self, value, node, trace_id):
        self.value = value
        self.node = node
        self.trace_id = trace_id
```

For a sequential teaching engine, a stack depth is sufficient. It is not a globally unique permanent graph ID. Saved VJPs must not be mixed with unrelated live boxes; thread and asynchronous isolation are outside this milestone.

```python
from contextlib import contextmanager


class TraceStack:
    def __init__(self):
        self.top = -1

    @contextmanager
    def new_trace(self):
        self.top += 1
        try:
            # TODO: Yield the current nesting level.
            ...
        finally:
            # TODO: Restore the previous level even if the user function raises.
            ...


trace_stack = TraceStack()
```

The `finally` matters: an exception must not leave the next derivative call at an incorrect level.

## 4. Find boxes belonging to the highest level

Replace the helper that selects every Box with one that selects only top-level arguments:

```python
def find_top_boxed_args(args):
    top_trace_id = -1
    top_boxes = []

    for argnum, arg in enumerate(args):
        # TODO: Ignore raw arguments.
        # TODO: If arg has a higher level, replace the collected group.
        # TODO: If arg has the same level, append it.
        ...

    return tuple(top_boxes), top_trace_id
```

Checkpoint:

```text
args: (outer_box level 0, inner_box level 1, raw constant)
selected positions: (1,)
selected level: 1
```

For two level-1 arguments, retain both positions, including duplicate edges. Each newly created node must have parents from only one trace level.

## 5. Update primitive execution: peel one layer, then recurse

Your old primitive called the raw numerical function immediately after unboxing. The new primitive must first give lower traces a chance to record the operation.

```python
def primitive(raw_function):
    @wraps(raw_function)
    def wrapper(*args, **kwargs):
        boxed_args, trace_id = find_top_boxed_args(args)

        if not boxed_args:
            return raw_function(*args, **kwargs)

        # TODO: Copy args, replacing ONLY selected top-level boxes
        # with their .value. Leave lower-level boxes intact.
        argvals = ...

        # TODO: Call wrapper recursively on argvals, rather than raw_function.
        result = ...

        # TODO: Construct parents and argnums from boxed_args only.
        # TODO: Store wrapper, result, argvals and kwargs in the recipe.
        node = ...

        return Box(value=result, node=node, trace_id=trace_id)

    return wrapper
```

Why recursion terminates: each recursive call removes the highest remaining Box layer. Eventually all arguments are raw and `raw_function` runs.

Why this records both graphs: the recursive call can return an outer Box. The inner node then records that outer Box as its numerical value, retaining the outer dependency.

Do not reuse the Section 3 helper that unboxes every boxed argument indiscriminately.

## 6. Update `trace` to recognize its own output

```python
def trace(function, x):
    start_node = Node.new_root()

    with trace_stack.new_trace() as trace_id:
        start_box = Box(value=x, node=start_node, trace_id=trace_id)
        output = function(start_box)

        # TODO: Require both that output is a Box and that its
        # trace_id equals this invocation's trace_id.
        if ...:
            return output.value, output.node, start_node

        # Output may be a lower-level Box, independent of THIS input.
        return output, None, start_node
```

A Box is not automatically evidence of dependence on the current input. It might belong only to an outer trace.

Example:

```python
def outer(x):
    return grad(lambda y: x)(5.0)
```

The inner function returns the outer `x` unchanged and never uses its own `y`. Its derivative with respect to `y` is zero. A test that checks only `isinstance(output, Box)` would mistakenly connect unrelated graphs.

## 7. Preserve tracing inside VJP rules and helpers

Audit every operation performed during backpropagation, not only the registered callbacks.

| Differentiable calculation | Use |
|---|---|
| Gradient addition | Wrapped `add` or Box `+` |
| Multiplication and division | Wrapped primitives or Box operators |
| Reduction in `unbroadcast` | Wrapped `sum_` |
| Shape restoration | Wrapped `reshape` |
| Matrix transpose | Wrapped `transpose` |
| Matrix product | Wrapped `dot` |
| `exp` and `log` in derivative formulas | Wrapped primitives |

Shape metadata itself is fixed for these derivatives. Reading rank, axis sizes, and shapes can inspect the ultimate raw value. Replacing numerical arithmetic with raw NumPy operations would remove the outer trace.

Use two distinct helpers deliberately:

```python
def raw_value(value):
    # TODO: Recursively remove Box layers for metadata inspection only.
    ...


def value_shape(value):
    return onp.shape(raw_value(value))
```

Recursive unboxing is appropriate here because shape inspection does not produce differentiable numerical values. It is inappropriate in primitive numerical evaluation.

For constant arrays of ones and zeros, use shapes derived from raw metadata. Multiply a reshaped traced gradient by raw ones to implement reduction broadcasting without introducing a new broadcasting primitive.

## 8. Keep `grad` and `make_vjp` compatible with nested values

Do not cast results or gradients using `float`, `onp.asarray`, or recursive unboxing. A first derivative evaluated inside an outer trace may return a Box that the outer trace needs.

Update the Section 6 output/seed checks to use `value_shape` rather than implicitly converting a Box. Scalar output still means shape `()`.

Seeds are constants of the appropriate shape:

```text
scalar output -> scalar one
independent input -> zeros with the input's raw shape
```

Zeros for a constant function have no numerical dependency on the outer value. It is correct for those results to remain raw constants; further differentiation will return zeros again.

Keep the Section 4 gradient dictionary local to every backward call. Allow its values to be lower-level boxes, because those messages may themselves be traced.

## 9. First acceptance tests: scalar nesting

```python
def test_polynomial_derivatives():
    def cube(x):
        return x*x*x

    first = grad(cube)
    second = grad(first)
    third = grad(second)
    fourth = grad(third)

    assert first(2.0) == 12.0
    assert second(2.0) == 12.0
    assert third(2.0) == 6.0
    assert fourth(2.0) == 0.0

    assert second(3.0) == 18.0
    assert second(2.0) == 12.0


def test_constant_derivatives():
    assert grad(grad(lambda x: 7.0))(3.0) == 0.0
    assert grad(grad(lambda x: x))(3.0) == 0.0


def test_exponential_derivatives():
    for derivative in (grad(exp), grad(grad(exp)), grad(grad(grad(exp)))):
        onp.testing.assert_allclose(derivative(0.5), onp.exp(0.5))
```

If first derivatives pass but second derivatives are zero, look for raw arithmetic or full unboxing inside primitive recursion or VJP helpers.

## 10. Test closure capture and mixed partial derivatives

This test exercises two boxes with different levels in the same multiplication:

```python
def test_closed_over_outer_input():
    def outer(x):
        def inner(y):
            return x*y
        return grad(inner)(x)

    # The inner derivative is x, so the outer derivative is 1.
    assert outer(5.0) == 5.0
    assert grad(outer)(5.0) == 1.0


def test_output_independent_of_inner_input():
    def outer(x):
        return grad(lambda y: x)(5.0)

    assert outer(3.0) == 0.0
    assert grad(outer)(3.0) == 0.0


def test_mixed_partial():
    def f(x, y):
        return x*x*y

    dx = grad(f, argnum=0)
    dxy = grad(dx, argnum=1)
    assert dxy(3.0, 4.0) == 6.0
```

The captured outer `x` is fixed for the inner derivative, yet remains a variable for the outer derivative. Trace levels express that distinction.

## 11. Test array curvature without constructing a Hessian

With array inputs, `grad(loss)` returns an array. Your `grad` API requires scalar outputs, so `grad(grad(loss))` is not directly valid for general array losses.

Instead, take a scalar projection of the gradient and differentiate that:

```python
def test_hessian_vector_product():
    def loss(x):
        return sum_(x*x*x)

    direction = onp.array([0.5, -1.0])
    x = onp.array([2.0, 3.0])

    def projected_gradient(x):
        return sum_(multiply(grad(loss)(x), direction))

    hv = grad(projected_gradient)(x)
    onp.testing.assert_allclose(hv, 6*x*direction)
```

This computes `H.T @ direction`. For a smooth scalar loss, the Hessian is symmetric, so this is the Hessian-vector product. No Hessian matrix is stored.

Also test a broadcast parameter and dot so helper operations are covered:

```python
def test_broadcast_second_derivative():
    coefficients = onp.array([1.0, 2.0, 3.0])

    def loss(bias):
        y = multiply(coefficients, bias)
        return sum_(y*y)

    # loss(b) = 14*b*b; loss''(b) = 28
    onp.testing.assert_allclose(grad(grad(loss))(2.0), 28.0)


def test_dot_curvature():
    A = onp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    direction = onp.array([0.5, -1.0])
    x = onp.array([2.0, 3.0])

    def loss(x):
        y = dot(A, x)
        return sum_(y*y)

    def projected_gradient(x):
        return sum_(multiply(grad(loss)(x), direction))

    expected = 2*A.T @ (A @ direction)
    onp.testing.assert_allclose(grad(projected_gradient)(x), expected)
```

## 12. Verify trace cleanup after errors

```python
def test_trace_stack_recovers():
    initial_level = trace_stack.top

    def broken(x):
        raise RuntimeError("deliberate test error")

    try:
        grad(broken)(2.0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected the deliberate error")

    assert trace_stack.top == initial_level
    assert grad(grad(lambda x: x*x))(2.0) == 2.0
    assert trace_stack.top == initial_level
```

Include an equivalent case where the failure happens inside a nested derivative call.

## 13. Optional syntax: constant powers

The original roadmap shows `x**3`. You can complete all conceptual checks using `x*x*x` first.

To add that syntax, introduce a `power(x, exponent)` primitive, support a fixed integer exponent of at least 1, and implement `Box.__pow__`. Its base VJP is:

```text
g * exponent * power(x, exponent - 1)
```

Special-case exponent 1 as the identity contribution so higher derivatives at zero do not evaluate problematic unnecessary powers. You may support exponent 0 as a separate constant result with a zero base VJP. Treat the exponent as configuration; do not imply that differentiating the exponent is implemented.

Broader real powers require explicit domain decisions and are outside the core milestone.

## Implementation order and completion

1. Add trace-level state and exception-safe cleanup.
2. Add `trace_id` to boxes and select top-level boxed arguments.
3. Make primitive calls peel one layer and recurse through the wrapper.
4. Make `trace` match output ownership to its own level.
5. Preserve lower-level boxes in recipes and backward messages.
6. Audit VJPs, unbroadcasting, reductions, reshape, and dot for traced arithmetic.
7. Make metadata checks and scalar seeds nesting-aware.
8. Run first-derivative regressions, scalar nesting, closure capture, mixed partials, array curvature, and error cleanup checks.

Run named tests from `python -B my_autograd/sixth.py`. For an independent curvature check, compare a Hessian-vector product against a finite difference of `grad(loss)` at `x + epsilon*direction` and `x - epsilon*direction`.

Section 7 completes the seven-stage learning roadmap when first derivatives remain correct and higher derivatives work for supported scalar and array operations. Completion means a small, tested autodiff implementation; it does not mean all NumPy functions or all features of production Autograd are supported.

Possible follow-up exercises are more primitive rules, indexing/scatter VJPs, numerical stability, package organization, and a small training example. They are optional extensions rather than missing roadmap sections.
