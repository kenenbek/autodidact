# Section 5: Build `grad()`

## Goal and starting point

Section 4 gave you a reusable reverse-pass function:

```python
vjp, value = make_vjp(function, x)
derivative = vjp(1.0)
```

Section 5 turns this into a convenient public interface:

```python
derivative_function = grad(function)
derivative = derivative_function(x)
```

Keep `third.py` as your Section 4 checkpoint. Create `my_autograd/fourth.py` for this stage, importing `make_vjp` from `third.py`. The existing graph engine does not need a redesign.

Use partial skeletons below as exercises: replace the `...` and TODOs yourself. They are not working implementations yet.

Before starting, run the Section 4 assertions:

```sh
python -B my_autograd/third.py
python -B my_autograd/testing.py
```

Remember that the current `testing.py` runner may only call three tests. Call every named test when checking the full stage. The earlier review found an unpacking typo (`**recipe.args` instead of `*recipe.args`) and mistakes in test inputs, expected argument positions, and imports. Confirm those are resolved rather than assuming the new guide means Section 4 has passed.

## 1. Mental model: `grad` builds a new function

There are two different moments:

```text
grad(f)           -> construct a derivative function
grad(f)(3.0)      -> evaluate that function at 3.0
```

Suppose:

```python
def square(x):
    return x*x

square_derivative = grad(square)
```

At this point, you have not selected an input or executed `square`.

Later:

```python
square_derivative(3.0)  # 6.0
square_derivative(5.0)  # 10.0
```

Each call builds a new graph at its own input. The returned function remembers `square`, but it must not remember one fixed evaluation point.

Compare the closures:

| Closure | What it remembers | What its caller supplies |
|---|---|---|
| `vjp` from Section 4 | A graph recorded at one input | An output seed |
| `gradfun` from Section 5 | A function and selected argument number | New function arguments |

## 2. Why the seed is `1.0`

For a scalar output `y = f(x)`, the VJP computes:

```text
seed * dy/dx
```

Choosing `seed = 1.0` returns `dy/dx`. You already implemented the calculus in Section 4. `grad` only prepares the function call and chooses the seed.

Start with a temporary single-argument version:

```python
from third import make_vjp


def grad(function):
    def gradfun(x):
        # TODO: Build a VJP for function at this x.
        vjp, value = ...

        # TODO: Evaluate the VJP with the scalar identity seed.
        return ...

    # TODO: Return the function, without evaluating it.
    ...
```

Checkpoint:

```python
def test_single_argument_grad():
    def square(x):
        return x*x

    derivative = grad(square)
    assert callable(derivative)
    assert derivative(3.0) == 6.0
    assert derivative(5.0) == 10.0
```

Do not calculate a gradient when the outer `grad(function)` runs. `make_vjp` belongs inside `gradfun`.

## 3. Mental model: `argnum` selects one independent variable

For a function with several arguments:

```python
def f(x, y):
    return x*y
```

there are two partial derivatives:

```text
df/dx = y
df/dy = x
```

The public interface selects one:

```python
grad(f, argnum=0)(3.0, 4.0)  # 4.0
grad(f, argnum=1)(3.0, 4.0)  # 3.0
```

For each evaluation, only the selected argument becomes a Box. The remaining arguments stay raw and act as constants for that particular derivative.

```text
Differentiate argument 0: f(Box(3), 4) -> derivative 4
Differentiate argument 1: f(3, Box(4)) -> derivative 3
```

This does not require multiple graph roots. Each derivative query still has exactly one root.

Do not confuse the public `argnum` with `Recipe.argnums`:

| Name | Meaning |
|---|---|
| `grad(f, argnum=1)` | Differentiate the user's function with respect to its second argument |
| `recipe.argnums` | Identify traced input positions of one primitive operation |

Those numbers need not match. A selected user argument can appear in many different primitive argument positions as the program runs.

## 4. Write a tuple-replacement helper

You need to reconstruct an argument tuple while replacing exactly one position:

```text
original args: (3.0, 4.0)
selected index: 1
replacement: Box(4.0)
new args: (3.0, Box(4.0))
```

Implement:

```python
def subval(values, index, replacement):
    # TODO: Make a mutable copy of values.
    copied = ...

    # TODO: Replace the selected element in the copy.
    ...

    # TODO: Return the reconstructed tuple.
    ...
```

Checkpoint:

```python
def test_subval():
    original = (3.0, 4.0, 5.0)
    marker = object()
    changed = subval(original, 1, marker)

    assert isinstance(changed, tuple)
    assert changed[0] == 3.0
    assert changed[1] is marker
    assert changed[2] == 5.0
    assert original == (3.0, 4.0, 5.0)
```

The copy prevents the adapter from modifying the caller's argument collection.

## 5. Adapt a multi-argument function to one traced input

`make_vjp` expects a function taking one traced argument. Build a small adapter to satisfy that interface.

For `argnum=1`, the adapter is conceptually:

```python
def unary_function(selected):
    return f(3.0, selected)
```

The forward tracer supplies a Box as `selected`. Your adapter places it into the chosen argument position before calling the original function.

Use this structure inside the final `gradfun`:

```python
def gradfun(*args, **kwargs):
    def unary_function(selected):
        # TODO: Substitute selected into args at argnum.
        replaced_args = ...

        # TODO: Call function with the reconstructed positional arguments
        # and the original keyword arguments.
        return ...

    # TODO: Obtain the raw evaluation point from args.
    selected_value = ...

    # TODO: Trace the adapter, not function directly.
    vjp, value = ...

    # TODO: Return the derivative using the scalar seed.
    return ...
```

The complete data flow is:

```text
grad(f, argnum=1)(3, 4)
    -> create unary_function(selected) = f(3, selected)
    -> make_vjp(unary_function, 4)
    -> tracer calls unary_function(Box(4))
    -> adapter calls f(3, Box(4))
    -> record graph
    -> apply VJP with seed 1
    -> return derivative for argument 1
```

## 6. Assemble the public interface

Upgrade your temporary single-argument version to:

```python
from functools import wraps
from third import make_vjp


def grad(function, argnum=0):
    # TODO: Validate argnum when constructing the derivative function.
    ...

    @wraps(function)
    def gradfun(*args, **kwargs):
        # TODO: Validate that args contains the selected position.
        ...

        def unary_function(selected):
            # TODO: Use subval and forward kwargs to function.
            ...

        # TODO: Create a VJP at args[argnum], then evaluate it with 1.0.
        ...

    return gradfun
```

For this tutorial, choose a small explicit contract:

- `argnum` is a nonnegative Python integer; reject booleans and other types with `TypeError`.
- Reject negative indices with `ValueError`.
- If the selected argument was not supplied positionally, raise `ValueError` when calling `gradfun`.
- Forward keyword arguments as fixed configuration. The differentiated argument must be positional.
- Inputs and outputs remain scalar numbers supported by the existing engine.

Hints: `type(argnum) is int` excludes booleans; `len(args)` tells you whether the selected position exists. You do not need signature inspection or a new argument-binding system.

Keyword configuration works because it enters the original function before that function calls primitives. You are not adding boxed keyword arguments to the primitive tracer.

## 7. Test partial derivatives with an asymmetric example

Use different formulas for different argument positions so accidentally selecting the wrong position is visible:

```python
def test_partial_derivatives():
    def f(x, y):
        return x*x*y + y

    # df/dx = 2*x*y; df/dy = x*x + 1
    assert grad(f, argnum=0)(3.0, 4.0) == 24.0
    assert grad(f, argnum=1)(3.0, 4.0) == 10.0
    assert grad(f)(3.0, 4.0) == 24.0
```

Then test fixed keyword configuration:

```python
def test_keyword_configuration():
    def scaled_square(x, *, scale=1.0):
        return scale*x*x

    assert grad(scaled_square)(3.0, scale=2.0) == 12.0
```

## 8. Check identity, constant output, and unused arguments

These three cases exercise different graph shapes:

```python
def test_graph_edge_cases():
    # Output is the input Box itself: start_node == end_node.
    assert grad(lambda x: x)(3.0) == 1.0

    # Raw output: no end_node exists.
    assert grad(lambda x: 7.0)(3.0) == 0.0

    # The selected argument is unused, although another argument is used.
    def f(x, y):
        return x*x

    assert grad(f, argnum=1)(3.0, 4.0) == 0.0
```

If identity fails, examine Section 4's handling of a graph containing only the root. If constant output fails, examine the no-end-node branch of `make_vjp`.

## 9. Verify that each derivative call traces at its own input

```python
def test_repeated_calls():
    calls = 0

    def cube(x):
        nonlocal calls
        calls += 1
        return x*x*x

    derivative = grad(cube)
    assert calls == 0

    assert derivative(2.0) == 12.0
    assert calls == 1

    assert derivative(3.0) == 27.0
    assert calls == 2

    assert derivative(2.0) == 12.0
    assert calls == 3
```

If the second result uses the first evaluation point, you created the VJP too early or cached it inside `grad`.

## 10. Check errors explicitly

Write assertions that the following calls raise the specified exception:

```python
grad(lambda x: x, argnum="0")  # TypeError
grad(lambda x: x, argnum=True) # TypeError
grad(lambda x: x, argnum=-1)   # ValueError

derivative = grad(lambda x, y: x*y, argnum=1)
derivative(3.0)               # ValueError: selected position is absent

grad(lambda x: x)(x=3.0)      # ValueError: selected input must be positional
```

Test these separately using `try`/`except` or a test framework's exception assertions. An unhandled expected exception should not stop the rest of your checks.

Let missing-VJP errors from Section 4 propagate. `grad` should not turn them into zero gradients.

## 11. Compare with finite differences

Use a numerical approximation as an independent check:

```python
def numerical_derivative(function, x, epsilon=1e-6):
    return (function(x + epsilon) - function(x - epsilon)) / (2*epsilon)


def test_against_finite_differences():
    def f(x):
        return x*x*x + 2.0*x + 1.0

    derivative = grad(f)

    for point in (-2.0, 0.0, 0.5, 3.0):
        expected = numerical_derivative(f, point)
        assert abs(derivative(point) - expected) < 1e-5
```

The subtraction and division here operate on raw numbers in the numerical checker. Your autodiff primitives can still be limited to addition and multiplication.

## 12. Connect the result to learning a weight

A scalar weight is just a function argument selected for differentiation:

```python
def loss(weight, input_value, target):
    prediction = weight*input_value
    error = prediction + (-1.0)*target
    return error*error

weight_gradient = grad(loss, argnum=0)
gradient = weight_gradient(2.0, 3.0, 10.0)

# prediction=6, error=-4, derivative=2*error*input_value=-24
assert gradient == -24.0

learning_rate = 0.01
updated_weight = 2.0 - learning_rate*gradient

assert loss(updated_weight, 3.0, 10.0) < loss(2.0, 3.0, 10.0)
```

Only the selected weight is boxed. The input and target remain fixed for that query. The weight update happens outside the differentiated function, using raw numbers.

## Implementation order and completion

1. Verify the Section 4 engine and its tests.
2. Create `fourth.py` and import `make_vjp`.
3. Implement single-argument `grad` and verify two evaluation points.
4. Implement and test `subval`.
5. Add `argnum`, the unary adapter, keyword forwarding, and validation.
6. Run the partial-derivative, edge-case, repeated-call, and error tests.
7. Compare against finite differences and try the scalar-weight example.

Call each named test from a runner under `if __name__ == "__main__":`. Run it from the repository root:

```sh
python -B my_autograd/fourth.py
```

You are finished when `grad(f)` creates a reusable derivative function, each evaluation traces once at its own input, and `argnum` selects the correct partial derivative using the existing single-root engine.

Keep NumPy arrays, differentiation by keyword name, multiple simultaneous differentiated inputs, and nested `grad(grad(f))` for later milestones. Higher derivatives require trace separation beyond this stage.
