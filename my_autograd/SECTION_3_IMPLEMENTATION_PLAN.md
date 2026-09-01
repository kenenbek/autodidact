# Section 3 Implementation Plan

## Goal of Section 3

Transform this combined design:

```text
Node = value + graph + arithmetic + gradients
```

into:

```text
Box = value flowing through the program
Node = record of how that value was produced
primitive = decides whether an operation should be recorded
```

Keep `first.py` unchanged as your completed stage-2 implementation. Build Section 3 in `my_autograd/second.py`.

## Step 1: Define a graph-only `Node`

The new `Node` should not behave like a number. Remove:

- `value`
- `grad`
- Arithmetic methods
- `_backward`
- `backward()`

It only needs:

```python
class Node:
    def __init__(self, parents, recipe):
        ...

    @classmethod
    def new_root(cls):
        ...
```

A root node has:

```text
parents = ()
recipe = None
```

Checkpoint:

```python
root = Node.new_root()

assert root.parents == ()
assert root.recipe is None
```

Hint: `Node` answers only:

> Which graph nodes produced this result, and which operation was used?

## Step 2: Define a `Recipe`

A recipe should retain enough forward-pass information for later backpropagation.

You can use a dataclass, named tuple, or normal class:

```python
Recipe(
    function=...,
    result=...,
    args=...,
    kwargs=...,
    argnums=...,
)
```

Meanings:

- `function`: operation that ran, such as `multiply`
- `result`: raw result produced during the forward pass
- `args`: raw argument values used during the operation
- `kwargs`: raw keyword arguments
- `argnums`: positions that originally contained boxes

For:

```python
multiply(Box(2.0), 3.0)
```

the recipe should describe:

```text
args = (2.0, 3.0)
argnums = (0,)
```

For:

```python
multiply(x, x)
```

it should contain:

```text
args = (2.0, 2.0)
argnums = (0, 1)
```

Do not remove duplicate parents. `x * x` has two edges to the same node.

## Step 3: Define `Box`

Start with:

```python
class Box:
    def __init__(self, value, node):
        ...
```

It stores:

```text
value: raw numerical value
node: graph node that produced the value
```

Checkpoint:

```python
root = Node.new_root()
x = Box(2.0, root)

assert x.value == 2.0
assert x.node is root
```

Do not add gradient state yet.

## Step 4: Write helpers for boxed arguments

Before writing `primitive`, solve this smaller problem:

```python
args = (Box(2.0, root), 3.0)
```

Produce:

```python
boxed_args = ((0, args[0]),)
raw_args = (2.0, 3.0)
parents = (root,)
argnums = (0,)
```

Suggested helper responsibilities:

```python
find_boxed_args(args)
unbox_args(args)
```

Important invariant:

```text
Only Box arguments create parent edges.
```

A raw constant such as `3.0` belongs in the recipe but is not a parent.

Checkpoint:

```python
x = Box(2.0, root)

assert find_boxed_args((x, 3.0)) == ((0, x),)
assert unbox_args((x, 3.0)) == (2.0, 3.0)
```

## Step 5: Implement the no-tracing path of `primitive`

Begin with a decorator that changes nothing when no arguments are boxed:

```python
@primitive
def add(x, y):
    return x + y
```

These calls must return ordinary numbers:

```python
assert add(2.0, 3.0) == 5.0
assert type(add(2.0, 3.0)) is float
```

Hint:

```text
wrapper receives args
→ find boxed args
→ if there are none, call the original raw function
```

Keep a reference to the original function inside the decorator. Otherwise, it is easy to accidentally call the wrapper recursively.

## Step 6: Implement the tracing path

When at least one argument is boxed:

```text
1. Find boxed arguments.
2. Replace boxes with raw values.
3. Call the original function with raw values.
4. Collect parent nodes from the boxes.
5. Create a Recipe.
6. Create a Node.
7. Return Box(result, node).
```

Pseudocode:

```python
def wrapper(*args, **kwargs):
    boxed_args = ...

    if not boxed_args:
        return raw_function(*args, **kwargs)

    raw_args = ...
    result = raw_function(*raw_args, **kwargs)

    parents = ...
    argnums = ...
    recipe = ...
    node = ...
    return Box(result, node)
```

For this milestone, support boxes in positional arguments only. `add` and `multiply` do not require boxed keyword arguments.

## Step 7: Implement the two primitives

Start with exactly:

```python
@primitive
def add(x, y):
    return x + y


@primitive
def multiply(x, y):
    return x * y
```

Test `add` first. Add `multiply` only after all `add` tests pass.

Checkpoint:

```python
root = Node.new_root()
x = Box(2.0, root)
y = add(x, 3.0)

assert isinstance(y, Box)
assert y.value == 5.0
assert y.node.parents == (root,)
assert y.node.recipe.args == (2.0, 3.0)
assert y.node.recipe.argnums == (0,)
```

## Step 8: Add arithmetic syntax to `Box`

Users should eventually write:

```python
x * x + 3.0
```

instead of:

```python
add(multiply(x, x), 3.0)
```

Make `Box` delegate to the primitives:

```python
class Box:
    def __add__(self, other):
        return add(self, other)

    def __mul__(self, other):
        return multiply(self, other)
```

Also implement:

```python
__radd__
__rmul__
```

Important: arithmetic methods belong to `Box`, not `Node`.

## Step 9: Test a multi-operation graph

```python
root = Node.new_root()
x = Box(2.0, root)

a = x * x
y = a + 3.0
```

Expected graph:

```text
root(x)
   | \
   |  \
   v   v
 multiply         raw arguments: (2.0, 2.0)
   |
   v
  add <--- 3.0     raw arguments: (4.0, 3.0)
   |
   v
   y
```

Assertions:

```python
assert a.value == 4.0
assert a.node.parents == (root, root)

assert y.value == 7.0
assert y.node.parents == (a.node,)
assert y.node.recipe.args == (4.0, 3.0)
```

Notice:

- `x * x` produces two parent edges.
- The constant `3.0` produces no parent edge.
- The recipe still records `3.0`.

## Step 10: Verify stable forward recipes

```python
root = Node.new_root()
x = Box(2.0, root)
y = x * x

x.value = 10.0

assert y.value == 4.0
assert y.node.recipe.args == (2.0, 2.0)
```

The recipe must contain the original raw values, not references to mutable boxes.

## Section 3 completion criteria

Section 3 is complete when all of these pass:

```python
# Ordinary calls remain ordinary.
assert add(2.0, 3.0) == 5.0
assert multiply(2.0, 3.0) == 6.0

# One traced argument.
root = Node.new_root()
x = Box(2.0, root)
y = add(x, 3.0)

assert y.value == 5.0
assert y.node.parents == (root,)
assert y.node.recipe.argnums == (0,)

# Two traced arguments.
y = multiply(x, x)

assert y.value == 4.0
assert y.node.parents == (root, root)
assert y.node.recipe.argnums == (0, 1)

# Chained tracing.
z = x * x + 3.0

assert z.value == 7.0
assert len(z.node.parents) == 1
assert z.node.parents[0].parents == (root, root)

# Stable forward values.
x.value = 10.0
assert z.node.parents[0].recipe.args == (2.0, 2.0)
```

Do not implement `backward()` in the new architecture yet. Section 4's VJP registry will determine how recipes are traversed backward. Section 3 is about recording a correct computation graph, not differentiating it.
