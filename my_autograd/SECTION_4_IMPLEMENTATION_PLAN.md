# Section 4 Implementation Plan: Vector-Jacobian Products

## Starting point

Section 3 in `second.py` is complete: its assertions pass, primitives record computation graphs, constants do not become parents, and recipes retain raw forward values.

Keep `second.py` unchanged as the completed Section 3 checkpoint. Copy its design into `my_autograd/third.py` and implement Section 4 there.

Section 4 should add reverse-mode differentiation to the new graph architecture without putting `.grad` or `_backward` back onto `Node`.

The target flow is:

```text
make_vjp(function, x)
    |
    +-- Box x and execute function(x)
    +-- primitives record a graph of Node recipes
    +-- return a reusable vjp(seed) function

vjp(seed)
    |
    +-- traverse the recorded graph backward
    +-- look up a derivative rule for each primitive argument
    +-- accumulate contributions in a fresh local dictionary
    +-- return the gradient at x
```

Do not implement `grad()` yet. That is Section 5.

## Mental model: Section 3 records history, Section 4 interprets it

Section 3 built a recording system. When a boxed value passes through a primitive, the primitive records:

```text
which operation ran
which raw values it received
which arguments were traced
which earlier nodes produced those arguments
```

That is analogous to writing a receipt for every numerical operation. The graph does not yet know calculus; it only knows history.

Section 4 gives meaning to those receipts. A VJP rule says:

> If the final result is sensitive to this operation's output by amount `g`, how sensitive is it to each input?

The computation therefore has two distinct phases:

| Phase | Direction | Data being moved | Responsibility |
|---|---|---|---|
| Forward | input to output | Numerical values | Primitives calculate results and record recipes |
| Backward | output to input | Sensitivities/gradients | VJP rules interpret recipes |

Keeping these phases separate is the core design. The forward pass should not calculate gradients, and the backward pass should not rerun the original function.

## Mental model: a graph node is a question waiting to be answered

For this expression:

```python
t = x * x
y = t + x
```

the forward pass records:

```text
root(x=3)
  | \
  |  \
  v   v
multiply ----> t=9
                  \
                   \
root(x=3) ----------> add ----> y=12
```

Each non-root node contains enough information to answer a future question:

```text
multiply node:
    "If somebody tells me dL/dt, what are my contributions to dL/dx?"

add node:
    "If somebody tells me dL/dy, what should I send to t and x?"
```

The node does not answer the question itself. It identifies the primitive, and the VJP registry supplies the rule that answers it.

## Mental model: gradients are messages flowing backward

Think of the incoming gradient `g` as a message arriving at a node from everything downstream.

For addition:

```text
             incoming message g
                      |
                      v
x --------->       add
y --------->

message sent to x: g
message sent to y: g
```

For multiplication:

```text
             incoming message g
                      |
                      v
x --------->     multiply
y --------->

message sent to x: g * y
message sent to y: g * x
```

If a value influences the output through several paths, it receives several messages. Reverse-mode autodiff adds those messages together.

This is why `outgrads` is a mapping from node to accumulated message:

```python
outgrads[node] = total sensitivity received so far
```

It is not permanent state belonging to the node. It is temporary state for one particular backward query.

## Mental model: why the seed exists

A VJP does not start with a gradient automatically. It needs to know how the hypothetical final objective depends on the recorded output.

For a scalar function:

```text
y = f(x)
```

calling:

```python
vjp(1.0)
```

means:

```text
Assume dL/dy = 1. What is dL/dx?
```

If `L = y`, then `dL/dy = 1`, so the returned value is the ordinary derivative `dy/dx`.

Calling:

```python
vjp(2.0)
```

means:

```text
Assume dL/dy = 2. What is dL/dx?
```

Every backward message is scaled by two, so the answer is twice the ordinary derivative.

Example for `f(x) = x*x` at `x=3`:

```text
ordinary derivative: 2*x = 6

seed 1  -> 1 * 6  = 6
seed 2  -> 2 * 6  = 12
seed -1 -> -1 * 6 = -6
```

Section 5's `grad()` will simply be a convenience operation that calls `vjp(1.0)` for scalar outputs.

## Mental model: why it is called a vector-Jacobian product

For a general function with vector input and output:

```text
y = f(x)
```

the derivative is a Jacobian matrix `J`. Reverse mode does not normally construct the whole matrix. Instead, it accepts an output-side vector `g` and calculates:

```text
g J
```

That is a vector-Jacobian product, abbreviated VJP.

Your current engine is scalar-only, so the "vectors" and "matrix" each contain one number:

```text
[g] [df/dx] = [g * df/dx]
```

The scalar implementation may look like ordinary chain-rule multiplication, but the interface is already shaped for arrays later.

## Mental model: local rules compose into a global derivative

The registry knows only local calculus:

```text
add knows derivatives of x + y
multiply knows derivatives of x * y
```

It does not need a special rule for:

```python
x*x + x
```

or:

```python
x*x*x + x*x
```

The graph and reverse traversal compose the small rules into the derivative of the entire program. This is the central benefit of automatic differentiation:

```text
many simple local derivative rules
                 +
recorded computation graph
                 +
reverse accumulation
                 =
derivative of an arbitrary composed function
```

## Mental model: `argnums` labels parent edges

Recipes contain all raw arguments, but `parents` contains only traced arguments.

For:

```python
multiply(3.0, x)
```

the graph records:

```text
args     = (3.0, 2.0)
argnums  = (1,)
parents  = (x.node,)
```

`argnums == (1,)` says:

> The first and only parent corresponds to argument position 1.

That matters because the VJP rule for argument 0 and the rule for argument 1 are different:

```text
multiply argument 0 -> g * argument 1
multiply argument 1 -> g * argument 0
```

During backpropagation, this pairing restores information that was lost when boxes were replaced with raw values:

```python
for argnum, parent in zip(recipe.argnums, node.parents):
    rule = primitive_vjps[recipe.function][argnum]
```

One useful picture is:

```text
argnums:  (0,          2)
           |           |
parents: (x.node,     z.node)
```

They are parallel tuples. Never sort or deduplicate either one.

## Mental model: the registry is a calculus dispatch table

Python already dispatches the forward operation:

```python
multiply(x, y)
```

The VJP registry performs a second kind of dispatch during the backward pass:

```text
Which primitive created this node?
Which argument edge are we following?
Which derivative rule handles that pair?
```

Its key is the pair:

```text
(primitive function, argument number)
```

Conceptually:

```text
(add,      0) -> pass g through
(add,      1) -> pass g through
(multiply, 0) -> return g * y
(multiply, 1) -> return g * x
```

This is why exact function identity matters. A recipe containing the undecorated raw function cannot find a rule registered under the decorated wrapper, even if both functions happen to have the name `"multiply"`.

## Mental model: record once, ask backward questions many times

`make_vjp(function, x)` performs the forward trace once:

```text
function + input
      |
      v
recorded graph + forward result
```

It then returns a closure that remembers that graph:

```python
vjp, result = make_vjp(function, x)
```

You can ask several backward questions without rerunning the forward function:

```python
vjp(1.0)
vjp(2.0)
vjp(-1.0)
```

Each call uses:

- The same immutable forward recipes.
- A new seed.
- A new local `outgrads` dictionary.

That gives repeatable results and prevents the stale-intermediate problem from `first.py`.

## Worked example: `f(x) = x*x + x`

Use `x = 3`.

### Forward pass

```python
t = multiply(x, x)  # 9
y = add(t, x)       # 12
```

Recorded nodes:

```text
root:
    represents x=3

multiply node:
    result   = 9
    args     = (3, 3)
    argnums  = (0, 1)
    parents  = (root, root)

add node:
    result   = 12
    args     = (9, 3)
    argnums  = (0, 1)
    parents  = (multiply node, root)
```

### Start the backward pass

Seed the output:

```text
outgrads[add node] = 1
```

This represents `dy/dy = 1`.

### Process the add node

Addition sends its incoming message unchanged through both edges:

```text
outgrads[multiply node] += 1
outgrads[root]          += 1
```

State:

```text
multiply node: 1
root:          1
```

The direct `+ x` branch has already contributed `1` to the final derivative.

### Process the multiply node

The incoming message is `1`, and both recorded arguments are `3`:

```text
argument 0 contribution = 1 * 3 = 3
argument 1 contribution = 1 * 3 = 3
```

Both parent edges point to the same root:

```text
outgrads[root] = 1 + 3 + 3 = 7
```

Therefore:

```text
f'(3) = 7
```

This matches symbolic differentiation:

```text
f(x)  = x^2 + x
f'(x) = 2x + 1
f'(3) = 7
```

The important observation is that no component knew the formula `2x + 1` in advance. It emerged from local rules and graph structure.

## Invariants to keep in your head while coding

At every stage, ask whether these statements remain true:

1. `Box` carries a value; `Node` records history.
2. Recipes contain raw forward values, never mutable boxes.
3. Only boxed arguments create parent edges.
4. `argnums[i]` describes `parents[i]`.
5. Duplicate parent edges are meaningful and must remain duplicated.
6. A node is processed only after all downstream contributions have reached it.
7. Contributions to the same parent are added.
8. Backward state belongs to one VJP call, not permanently to the graph.
9. The primitive stored in a recipe is the same function object used as the registry key.
10. The forward function runs once per `make_vjp`, not once per seed.

When a test fails, identify which invariant was broken before changing code.

## Debugging mental model: inspect one node at a time

If a derivative is wrong, do not immediately inspect the whole graph. Pick the final node and ask:

```text
1. Is its incoming outgrad correct?
2. Does its recipe contain the correct raw args?
3. Do argnums and parents align?
4. Did registry lookup select the intended rule?
5. Is each local contribution mathematically correct?
6. Was each contribution added to the correct parent?
```

Then move one node toward the root. This mirrors the backward algorithm and usually reveals the first incorrect state quickly.

## 1. Understand the VJP contract

Suppose a primitive calculates:

```text
z = f(x, y)
```

During the backward pass, `g` is the gradient arriving from later computations:

```text
g = dL/dz
```

A VJP rule returns the contribution for one argument:

```text
VJP for x = g * dz/dx
VJP for y = g * dz/dy
```

Use one consistent callback signature:

```python
vjp(g, result, *args, **kwargs) -> parent_gradient
```

For multiplication:

```text
result = x * y
VJP for argument 0 = g * y
VJP for argument 1 = g * x
```

The `result` argument is unused for `add` and `multiply`, but keep it in the interface. Future rules such as `exp(x)` can use the already-computed result.

## 2. Fix primitive-function identity before building the registry

There is one important boundary issue in the current `second.py`.

Inside `primitive`, the recipe currently stores the undecorated function:

```python
recipe = Recipe(function=function, ...)
```

Outside the decorator, however, this name refers to the wrapper:

```python
@primitive
def multiply(x, y):
    ...

# `multiply` is now the wrapper, not the original function.
defvjp(multiply, ...)
```

If the registry uses the wrapper as its key but the recipe stores the original function, lookup will fail because they are different function objects.

In `third.py`, make recipes store the decorated wrapper:

```python
Recipe(
    function=wrapper,
    ...,
)
```

Also apply `functools.wraps` to preserve useful names and documentation:

```python
from functools import wraps

def primitive(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        ...
```

Checkpoint:

```python
root = Node.new_root()
x = Box(2.0, root)
y = multiply(x, 3.0)

assert y.node.recipe.function is multiply
```

Function identity must use `is`, not only matching names.

## 3. Create the VJP registry

The registry maps:

```text
primitive function
    -> argument number
        -> VJP callback
```

Its conceptual shape is:

```python
primitive_vjps = {
    multiply: {
        0: vjp_for_left_argument,
        1: vjp_for_right_argument,
    }
}
```

Start with an empty nested mapping. A normal dictionary with `setdefault`, or `defaultdict(dict)`, is sufficient.

Implement this interface:

```python
def defvjp(function, *vjps):
    ...
```

The position of each callback is its argument number:

```python
defvjp(function, vjp_arg0, vjp_arg1)
```

is equivalent to:

```text
registry[function][0] = vjp_arg0
registry[function][1] = vjp_arg1
```

Hint:

```python
for argnum, vjp in enumerate(vjps):
    ...
```

Checkpoint with temporary callbacks:

```python
def left_rule(g, result, x, y):
    return g * y

def right_rule(g, result, x, y):
    return g * x

defvjp(multiply, left_rule, right_rule)

assert primitive_vjps[multiply][0] is left_rule
assert primitive_vjps[multiply][1] is right_rule
```

Do not put these rules inside `Node` or `Recipe`. Recipes identify operations; the registry supplies their derivative behavior.

## 4. Register rules for `add` and `multiply`

Register both positional arguments of both primitives.

For addition:

```text
z = x + y
dz/dx = 1
dz/dy = 1
```

Therefore both VJPs return the incoming gradient unchanged.

For multiplication:

```text
z = x * y
dz/dx = y
dz/dy = x
```

Use the common callback signature even when parameters such as `result` are unused.

After registration, verify the rules directly before writing graph traversal:

```python
left = primitive_vjps[multiply][0]
right = primitive_vjps[multiply][1]

assert left(1.0, 6.0, 2.0, 3.0) == 3.0
assert right(1.0, 6.0, 2.0, 3.0) == 2.0

# An incoming seed scales the result.
assert left(10.0, 6.0, 2.0, 3.0) == 30.0
```

## 5. Reintroduce topological traversal for graph-only nodes

Port the topological-order idea from `first.py`, but make it a standalone function operating only on `Node` objects:

```python
def topological_sort(end_node):
    ...
```

Choose and document one ordering. For this plan, return nodes in forward topological order:

```text
root/ancestors first -> output node last
```

The backward pass will iterate over `reversed(topological_sort(end_node))`.

Requirements:

- Each unique node appears once.
- Parents appear before their children.
- Duplicate graph edges remain in `node.parents`; only the traversal list deduplicates nodes.
- The function does not calculate or store gradients.

Checkpoint:

```python
root = Node.new_root()
x = Box(2.0, root)
a = x * x
y = a + x

order = topological_sort(y.node)

assert order[0] is root
assert order[-1] is y.node
assert order.count(root) == 1
```

Do not treat the two edges in `a.node.parents == (root, root)` as duplicate mistakes. They represent the two uses of `x` in `x*x` and must both contribute during backpropagation.

## 6. Add a gradient-contribution helper

During one backward pass, multiple paths can reach the same parent. Add a small helper with this behavior:

```python
add_outgrads(None, 3.0) == 3.0
add_outgrads(3.0, 4.0) == 7.0
```

Suggested interface:

```python
def add_outgrads(previous, contribution):
    ...
```

This helper replaces mutation such as `parent.grad += contribution`.

## 7. Implement `backward_pass`

Use this interface:

```python
def backward_pass(seed, end_node, start_node):
    ...
```

The gradient state must live in a fresh local dictionary:

```python
outgrads = {end_node: seed}
```

For every non-root node in reverse topological order:

```text
1. Read its accumulated output gradient.
2. Read its recipe.
3. Pair each recipe.argnum with the corresponding parent.
4. Find registry[recipe.function][argnum].
5. Call that VJP with the recorded result and raw arguments.
6. Add the returned contribution to that parent's dictionary entry.
```

The critical alignment is:

```python
for argnum, parent in zip(recipe.argnums, node.parents):
    ...
```

For `x*x`, this loop intentionally runs twice:

```text
argnum 0 -> root -> contribution x
argnum 1 -> root -> contribution x
```

Return the gradient stored for `start_node`.

Pseudocode with the central expressions omitted:

```python
def backward_pass(seed, end_node, start_node):
    outgrads = {end_node: seed}

    for node in reversed(topological_sort(end_node)):
        if node.recipe is None:
            continue

        outgrad = ...
        recipe = node.recipe

        for argnum, parent in zip(recipe.argnums, node.parents):
            rule = ...
            contribution = rule(...)
            outgrads[parent] = add_outgrads(...)

    return outgrads[start_node]
```

Unlike `first.py`, this function never writes `node.grad`. Calling it twice creates two independent dictionaries and cannot reuse stale intermediate gradients.

## 8. Trace a single-input function

Implement a small helper:

```python
def trace(function, x):
    ...
```

Its flow is:

```text
create root Node
-> wrap x in Box(value=x, node=root)
-> call function(boxed_x)
-> inspect the result
```

Return enough information for `make_vjp`:

```python
result_value, end_node, start_node
```

If the function returns a `Box`, return its raw value and graph node. If it returns a raw value, it is independent of the input, so return:

```text
result_value = returned raw value
end_node = None
start_node = root
```

Checkpoints:

```python
value, end, start = trace(lambda x: x*x, 3.0)
assert value == 9.0
assert end is not None
assert start.recipe is None

value, end, start = trace(lambda x: 7.0, 3.0)
assert value == 7.0
assert end is None
```

At this stage, any returned `Box` is considered connected to the current input. Trace IDs for distinguishing nested traces belong to a later section.

## 9. Implement `make_vjp`

Use the roadmap interface:

```python
def make_vjp(function, x):
    ...
```

It returns two values:

```python
vjp, result = make_vjp(function, x)
```

Behavior:

- Trace `function` once at `x`.
- Return the raw forward result.
- Return a closure `vjp(seed)` that calls `backward_pass` on the recorded graph.
- If the result is independent of `x`, `vjp(seed)` returns `0.0` for this scalar-only stage.

Conceptual skeleton:

```python
def make_vjp(function, x):
    result, end_node, start_node = trace(function, x)

    if end_node is None:
        def vjp(seed):
            ...
    else:
        def vjp(seed):
            ...

    return vjp, result
```

The forward function must run only once when `make_vjp` is called, not again every time the returned VJP is evaluated.

## 10. Test the VJP seed explicitly

The seed is part of the VJP contract; it is not always `1.0`.

```python
vjp, result = make_vjp(lambda x: x*x, 3.0)

assert result == 9.0
assert vjp(1.0) == 6.0
assert vjp(2.0) == 12.0
assert vjp(-1.0) == -6.0
```

Section 5's `grad()` will later choose `1.0` automatically for scalar outputs.

## 11. Test argument-number lookup

Test both operand positions so an `argnums` or registry-indexing mistake cannot hide:

```python
left_vjp, _ = make_vjp(lambda x: x * 3.0, 2.0)
right_vjp, _ = make_vjp(lambda x: 3.0 * x, 2.0)

assert left_vjp(1.0) == 3.0
assert right_vjp(1.0) == 3.0
```

In the first graph, `argnums == (0,)`. In the second, `argnums == (1,)`.

## 12. Test shared and branched paths

These cases verify contribution accumulation:

```python
vjp, result = make_vjp(lambda x: x*x, 3.0)
assert result == 9.0
assert vjp(1.0) == 6.0

vjp, result = make_vjp(lambda x: x*x + x, 3.0)
assert result == 12.0
assert vjp(1.0) == 7.0

vjp, result = make_vjp(lambda x: x*x*x, 3.0)
assert result == 27.0
assert vjp(1.0) == 27.0
```

The last expression is parsed as `(x*x)*x`, so it contains an intermediate node and exercises a deeper graph.

## 13. Verify that VJPs are reusable

Unlike `Node.backward()` in `first.py`, a VJP evaluation should not retain gradients:

```python
vjp, _ = make_vjp(lambda x: x*x*x, 3.0)

assert vjp(1.0) == 27.0
assert vjp(1.0) == 27.0
assert vjp(2.0) == 54.0
```

Each call begins with a new local `outgrads` dictionary. Results are returned, not accumulated into leaf objects.

## 14. Test an input-independent result

```python
vjp, result = make_vjp(lambda x: 7.0, 3.0)

assert result == 7.0
assert vjp(1.0) == 0.0
assert vjp(100.0) == 0.0
```

The seed cannot create a dependency that did not exist in the forward computation.

## 15. Add a useful missing-rule failure

If a primitive participates in a trace but has no registered rule, fail at backward time with a message containing:

- The primitive's name.
- The missing argument number.

For example, a traced primitive named `subtract` without registered VJPs should fail with the equivalent meaning of:

```text
No VJP registered for primitive 'subtract' argument 0
```

Do not silently return zero. A missing derivative implementation is different from a mathematically zero derivative.

Hint: perform an explicit registry membership check before indexing the mapping, then raise `NotImplementedError`.

## Recommended implementation order

Work in small passing increments:

```text
1. Copy the completed Section 3 design into third.py
2. Make Recipe.function store the decorated primitive wrapper
3. Add the primitive_vjps registry
4. Implement defvjp
5. Register add and multiply rules
6. Test rules directly
7. Add topological_sort
8. Add add_outgrads
9. Implement backward_pass
10. Implement trace
11. Implement make_vjp
12. Add end-to-end assertions
13. Add the missing-rule error
```

Run the assertions after every step. When a step fails, keep the test small enough to determine whether the problem is graph construction, registry lookup, traversal order, or contribution accumulation.

## Section 4 completion criteria

Place these assertions under `if __name__ == "__main__":` in `third.py`, or translate them into test functions if you introduce a test suite:

```python
# Recipe and registry use the same function object.
root = Node.new_root()
x = Box(2.0, root)
recorded = multiply(x, 3.0)
assert recorded.node.recipe.function is multiply
assert 0 in primitive_vjps[multiply]
assert 1 in primitive_vjps[multiply]

# Raw primitive behavior is unchanged.
assert add(2.0, 3.0) == 5.0
assert multiply(2.0, 3.0) == 6.0

# Basic VJP and seed scaling.
vjp, result = make_vjp(lambda x: x*x, 3.0)
assert result == 9.0
assert vjp(1.0) == 6.0
assert vjp(2.0) == 12.0

# Both primitive argument positions are supported.
vjp, _ = make_vjp(lambda x: 3.0*x, 2.0)
assert vjp(1.0) == 3.0

# Shared and branched paths accumulate.
vjp, result = make_vjp(lambda x: x*x + x, 3.0)
assert result == 12.0
assert vjp(1.0) == 7.0

# A deeper graph differentiates correctly.
vjp, result = make_vjp(lambda x: x*x*x, 3.0)
assert result == 27.0
assert vjp(1.0) == 27.0

# Reusing a VJP does not reuse intermediate gradient state.
assert vjp(1.0) == 27.0
assert vjp(2.0) == 54.0

# An input-independent result has a zero VJP.
vjp, result = make_vjp(lambda x: 7.0, 3.0)
assert result == 7.0
assert vjp(1.0) == 0.0
```

Section 4 is finished when these checks pass without adding `.grad`, `_backward`, or `backward()` to `Node` or `Box`.

## Explicitly out of scope

Leave these for later sections:

- `grad()` and `argnum` selection for user functions.
- Subtraction, division, powers, `exp`, and other primitive rules.
- NumPy arrays and broadcasting.
- Keyword arguments containing boxes.
- Multiple independently traced inputs to `make_vjp`.
- Higher-order gradients and nested trace IDs.
- Mutation tracking.

The Section 4 result is deliberately narrow: a reusable scalar VJP for a single traced input, constructed from `add` and `multiply` primitives.
